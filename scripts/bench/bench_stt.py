"""STT candidates on the shared test set, clean and telephone copies: real-time factor, WER, critical-token hits.

Usage: bench_stt.py [parakeet-vulkan parakeet-cpu whisper-vulkan qwen-asr-vulkan parakeet-onnx ...]
Results go to .runtime/bench/results/stt-<engine>.json.
"""
from __future__ import annotations

import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import httpx

from common import BENCH, LOGS, MODELS, RT, LlamaServer, critical_hits, free_port, gpu_mem_gb, rss_gb, save_result, wer
from speech_backends import NEMO_DIR, PARAKEET, NemoServer

QWEN_ASR = MODELS / "ggml-org__Qwen3-ASR-0.6B-GGUF" / "Qwen3-ASR-0.6B-Q8_0.gguf"
QWEN_ASR_MM = QWEN_ASR.with_name("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf")
WHISPER_BIN = RT / "whisper.cpp" / "build" / "bin" / "whisper-server"
WHISPER_MODEL = MODELS / "whisper" / "ggml-large-v3-turbo-q5_0.bin"


def score(name: str, fn, meta: dict, conds=("clean", "tel"), warm: int = 1) -> dict:
    manifest = json.loads((BENCH / "manifest.json").read_text())
    fn(manifest[0]["clean"])
    rows = []
    for cond in conds:
        for m in manifest:
            t0 = time.perf_counter()
            text = fn(m[cond])
            dt = time.perf_counter() - t0
            rec = {"id": m["id"], "kind": m["kind"], "cond": cond, "dur": m["duration"], "secs": dt, "rtf": m["duration"] / dt, "text": text}
            if m["text"]:
                rec["wer"] = wer(m["text"], text)
            if m["critical"]:
                rec.update(zip(("crit_hits", "crit_total", "crit_missed"), critical_hits(text, m["critical"])))
            rows.append(rec)
    res = {"engine": name, **meta, "rows": rows, "summary": {}}
    for cond in conds:
        cur = [r for r in rows if r["cond"] == cond]
        scored = [r for r in cur if "wer" in r]
        crit = [r for r in scored if "crit_total" in r]
        long_row = [r for r in cur if r["id"] == "long30"]
        res["summary"][cond] = {
            "wer_all": st.mean(r["wer"] for r in scored), "wer_human": st.mean(r["wer"] for r in scored if r["kind"] == "human"),
            "wer_hard": st.mean(r["wer"] for r in scored if r["kind"] == "synth-hard"),
            "critical": f"{sum(r['crit_hits'] for r in crit)}/{sum(r['crit_total'] for r in crit)}",
            "rtf_median": st.median(r["rtf"] for r in cur), "ms_median": 1000 * st.median(r["secs"] for r in cur),
            "rtf_30s": long_row[0]["rtf"] if long_row else None}
        s = res["summary"][cond]
        print(f"{name:18s} {cond:5s} wer all {s['wer_all']:.3f} human {s['wer_human']:.3f} hard {s['wer_hard']:.3f} critical {s['critical']} median rtf {s['rtf_median']:.1f}x ({s['ms_median']:.0f} ms) 30s rtf {s['rtf_30s'] or 0:.1f}x", flush=True)
    save_result(f"stt-{name}", res)
    return res


def http_stt(url: str, model: str, extra: dict | None = None):
    client = httpx.Client(timeout=300)

    def go(path: str) -> str:
        with open(path, "rb") as f:
            r = client.post(url + "/v1/audio/transcriptions", files={"file": ("a.wav", f, "audio/wav")}, data={"model": model, "language": "en", **(extra or {})})
        r.raise_for_status()
        return r.json()["text"].strip()
    return go


def parakeet(name: str, device: str):
    gpu0 = gpu_mem_gb()
    with NemoServer(PARAKEET, device) as nemo:
        mid = httpx.get(nemo.url + "/v1/models").json()["data"][0]["id"]
        meta = {"load_s": nemo.load_s, "gpu_gb": gpu_mem_gb() - gpu0, "rss_gb": nemo.rss(), "model": mid}
        res = score(name, http_stt(nemo.url, mid), meta)
        # confidence check: are per-word confidences meaningful?
        client = httpx.Client(timeout=60)
        confs = []
        manifest = json.loads((BENCH / "manifest.json").read_text())
        for m in manifest[:14]:
            with open(m["tel"], "rb") as f:
                r = client.post(nemo.url + "/v1/audio/transcriptions", files={"file": ("a.wav", f, "audio/wav")},
                                data={"model": mid, "response_format": "verbose_json", "timestamp_granularities[]": "word"}).json()
            confs += [(w["word"], w.get("confidence")) for w in r.get("words", [])]
        vals = [c for _, c in confs if c is not None]
        res["confidence"] = {"n": len(vals), "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "share_below_0.9": sum(v < 0.9 for v in vals) / len(vals),
                             "lowest": sorted(confs, key=lambda x: x[1])[:12]}
        print("  confidence:", json.dumps(res["confidence"])[:400], flush=True)
        save_result(f"stt-{name}", res)


def whisper(model_file: Path = WHISPER_MODEL, name: str = "whisper-vulkan"):
    port = free_port(18700)
    log = open(LOGS / "whisper-server.log", "w")
    env = {"LD_LIBRARY_PATH": str(RT / "toolchain" / "lib"), "PATH": "/usr/bin:/bin"}
    gpu0 = gpu_mem_gb()
    proc = subprocess.Popen([str(WHISPER_BIN), "-m", str(model_file), "--port", str(port), "--host", "127.0.0.1", "-l", "en", "-t", "8", "-nt"],
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        t0 = time.perf_counter()
        client = httpx.Client(timeout=300)
        url = f"http://127.0.0.1:{port}"
        while True:
            try:
                if client.get(url + "/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None:
                raise RuntimeError("whisper-server exited, see whisper-server.log")
            if time.perf_counter() - t0 > 90:
                raise TimeoutError("whisper-server")
            time.sleep(0.3)
        meta = {"load_s": time.perf_counter() - t0, "gpu_gb": gpu_mem_gb() - gpu0, "rss_gb": rss_gb(proc.pid), "model": model_file.name}

        def go(path: str) -> str:
            with open(path, "rb") as f:
                r = client.post(url + "/inference", files={"file": ("a.wav", f, "audio/wav")}, data={"response_format": "json", "temperature": "0"})
            r.raise_for_status()
            return r.json()["text"].strip()
        res = score(name, go, meta)
        five = BENCH / "clean" / "long300.wav"
        if five.exists():
            t1 = time.perf_counter()
            text = go(str(five))
            dt = time.perf_counter() - t1
            res["long300"] = {"secs": dt, "rtf": 300 / dt, "words": len(text.split())}
            print(f"  300 s clip: {dt:.1f} s, {300 / dt:.1f}x", flush=True)
            save_result(f"stt-{name}", res)
    finally:
        proc.terminate()
        proc.wait(10)


def qwen_asr():
    gpu0 = gpu_mem_gb()
    with LlamaServer(["-m", str(QWEN_ASR), "-mm", str(QWEN_ASR_MM), "--temp", "0", "-c", "8192", "-np", "1", "-ngl", "999"], name="qwen-asr") as s:
        meta = {"load_s": s.load_s, "gpu_gb": gpu_mem_gb() - gpu0, "rss_gb": s.rss(), "model": QWEN_ASR.name}
        http = http_stt(s.url, "x")

        def go(path: str) -> str:
            import re
            text = http(path)
            # the model may prefix "language English<asr_text>"
            return re.sub(r"^.*?<asr_text>", "", text, flags=re.S).strip()
        score("qwen-asr-vulkan", go, meta)


def parakeet_onnx():
    import onnx_asr
    t0 = time.perf_counter()
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=["CPUExecutionProvider"])
    meta = {"load_s": time.perf_counter() - t0, "gpu_gb": 0, "model": "onnx-asr nemo-parakeet-tdt-0.6b-v3"}
    score("parakeet-onnx-cpu", lambda p: model.recognize(p), meta)


def main(names: list[str]) -> int:
    runners = {
        "parakeet-vulkan": lambda: parakeet("parakeet-vulkan", "vulkan:0"),
        "parakeet-cpu": lambda: parakeet("parakeet-cpu", "cpu"),
        "whisper-vulkan": whisper,
        "whisper-large-v3": lambda: whisper(MODELS / "whisper" / "ggml-large-v3-q5_0.bin", "whisper-large-v3-vulkan"),
        "qwen-asr-vulkan": qwen_asr,
        "parakeet-onnx": parakeet_onnx,
    }
    for name in names or list(runners):
        try:
            runners[name]()
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
            save_result(f"stt-{name}", {"engine": name, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
