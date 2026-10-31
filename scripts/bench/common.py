"""Shared helpers for the voice benchmarks: paths, server lifecycle, memory probes, text scoring."""
from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RT = ROOT / ".runtime"
LLAMA_DIR = RT / "llama-vulkan" / "llama-b12017"
MODELS = RT / "models"
BENCH = RT / "bench"
RESULTS = BENCH / "results"
LOGS = RT / "logs"
FFMPEG = RT / "ffmpeg" / "bin" / "ffmpeg"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
RESULTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

LFM_Q8 = next((HF_CACHE / "models--LiquidAI--LFM3-Audio-2B-GGUF" / "snapshots").glob("*/LFM3-Audio-2B-Q8_0.gguf"))
LFM_Q8_MM = LFM_Q8.with_name("mmproj-LFM3-Audio-2B-Q8_0.gguf")
LFM_Q4 = MODELS / "LiquidAI__LFM3-Audio-2B-GGUF" / "LFM3-Audio-2B-Q4_K_M.gguf"
LFM_Q4_MM = LFM_Q4.with_name("mmproj-LFM3-Audio-2B-Q4_K_M.gguf")

GTT_USED = Path("/sys/class/drm/card1/device/mem_info_gtt_used")
VRAM_USED = Path("/sys/class/drm/card1/device/mem_info_vram_used")


def llama_env() -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{LLAMA_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def gpu_mem_gb() -> float:
    """GTT + carveout VRAM in use system-wide, GB. Shared with everything else using the GPU."""
    return (int(GTT_USED.read_text()) + int(VRAM_USED.read_text())) / 1e9


def rss_gb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    except FileNotFoundError:
        pass
    return 0.0


def free_port(start: int = 18080, end: int = 18999) -> int:
    for port in range(start, end):
        with contextlib.closing(socket.socket()) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port in 18xxx")


class LlamaServer:
    """Starts llama-server on a free 18xxx port and always kills it on exit."""

    def __init__(self, args: list[str], name: str = "llama", port: int | None = None, timeout: float = 300,
                 env: dict | None = None):
        self.args = args
        self.name = name
        self.port = port or free_port()
        self.timeout = timeout
        self.env = env or llama_env()
        self.proc: subprocess.Popen | None = None
        self.load_s = 0.0
        self.log_path = LOGS / f"{name}-{self.port}.log"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "LlamaServer":
        gpu0 = gpu_mem_gb()
        cmd = [str(LLAMA_DIR / "llama-server"), "--host", "127.0.0.1", "--port", str(self.port), "--no-webui", *self.args]
        self.cmd = cmd
        log = open(self.log_path, "w")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env, start_new_session=True)
        while time.perf_counter() - t0 < self.timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited {self.proc.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        self.load_s = time.perf_counter() - t0
                        self.gpu_delta_gb = gpu_mem_gb() - gpu0
                        return self
            except Exception:
                time.sleep(0.2)
        self.__exit__(None, None, None)
        raise TimeoutError(f"{self.name} did not become healthy")

    def rss(self) -> float:
        return rss_gb(self.proc.pid) if self.proc else 0.0

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=5)


def save_result(name: str, data) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


# text scoring

_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}


def _words_to_digits(words: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _TENS:
            val = _TENS[w]
            if i + 1 < len(words) and words[i + 1] in _ONES and 0 < _ONES[words[i + 1]] < 10:
                val += _ONES[words[i + 1]]
                i += 1
            out.append(str(val))
        elif w in _ONES:
            out.append(str(_ONES[w]))
        elif w == "oh" and out and out[-1].isdigit():
            out.append("0")
        else:
            out.append(w)
        i += 1
    return out


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, spoken numbers and written numbers become the same digit tokens."""
    t = text.lower()
    t = re.sub(r"\$\s*(\d+)\.(\d\d)", r"\1 dollars and \2 cents", t)
    t = re.sub(r"\$\s*(\d+)", r"\1 dollars", t)
    t = re.sub(r"(\d)[,](\d)", r"\1\2", t)
    t = t.replace("-", " ").replace("@", " at ").replace("/", " ")
    t = re.sub(r"(?<=\w)\.(?=\w)", " dot ", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"'", "", t)
    words = t.split()
    words = _words_to_digits(words)
    # split digit strings so "4419" and "4 4 1 9" compare equal, and letters of an ID too
    out: list[str] = []
    for w in words:
        if w.isdigit() and len(w) > 2:
            out.extend(list(w))
        else:
            m = re.fullmatch(r"([a-z]{1,3})(\d+)", w)
            if m:
                out.extend(list(m.group(1)))
                out.extend(list(m.group(2)) if len(m.group(2)) > 2 else [m.group(2)])
            else:
                out.append(w)
    return " ".join(out)


def wer(ref: str, hyp: str) -> float:
    """Word error rate over normalized text, as a fraction."""
    r = normalize(ref).split()
    h = normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def critical_hits(hyp: str, patterns: list[str]) -> tuple[int, int, list[str]]:
    """Counts critical-token regexes found in the normalized hypothesis. Returns hits, total, misses."""
    n = normalize(hyp)
    missed = [p for p in patterns if not re.search(p, n)]
    return len(patterns) - len(missed), len(patterns), missed
