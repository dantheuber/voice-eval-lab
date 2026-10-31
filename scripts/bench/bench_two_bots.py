"""Two bots on a live call. Real audio crosses a simulated 8 kHz mu-law line at wall-clock pace in both directions.

Each side runs its own VAD on what it hears and knows only that audio. Per turn the clock runs from the moment the other
side's last audio sample left the line to the moment this side's first audio sample enters it.

Usage:
  bench_two_bots.py cascade --stt parakeet --llm lfm3-1.7b --tts kokoro --turns 12
  bench_two_bots.py realtime --quant q8 --turns 12
Writes .runtime/bench/results/twobots-*.json and a stereo recording to .runtime/bench/out/twobots/.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import queue
import statistics as st
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import psutil
import torch
from scipy.signal import resample_poly
from silero_vad import VADIterator, load_silero_vad

from bench_e2e import LLMS, llm_args
from bench_lfm_realtime import PERSONA, TOOLS as RT_TOOLS, TOOLS_PROMPT, lfm_args
from common import BENCH, LFM_Q4, LFM_Q4_MM, LlamaServer, free_port, gpu_mem_gb, save_result
from fernhill_fixture import CALLER_SYSTEM, SUPPORT_SYSTEM, TOOLS, run_tool, tool_message
from llm_client import THINK, chat
from rt_client import Session, write_pcm16
from speech_backends import KokoroTTS, make_tts, open_stt

torch.set_num_threads(1)
WINDOW = 512
TICK = WINDOW / 16000
OUT = BENCH / "out" / "twobots"
OUT.mkdir(parents=True, exist_ok=True)
BUSY = Path("/sys/class/drm/card1/device/gpu_busy_percent")
_MU = 255.0


def phone(pcm16k: np.ndarray) -> np.ndarray:
    """16 kHz to 8 kHz, mu-law companding with 8-bit quantization, back to 16 kHz."""
    x = resample_poly(pcm16k.astype(np.float32) / 32768, 1, 2)
    y = np.sign(x) * np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    y = np.round(np.clip(y, -1, 1) * 127) / 127
    x = np.sign(y) * (np.power(1 + _MU, np.abs(y)) - 1) / _MU
    out = resample_poly(x, 2, 1)
    return (np.clip(out, -1, 1) * 32767).astype("<i2")


def to16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return pcm.astype("<i2")
    g = np.gcd(rate, 16000)
    return (np.clip(resample_poly(pcm.astype(np.float32), 16000 // g, rate // g), -32768, 32767)).astype("<i2")


class Line:
    """One direction of the call."""

    def __init__(self):
        self.buf = collections.deque()
        self.lock = threading.Lock()
        self.queued = 0
        self.t_last_real = 0.0
        self.recorded: list[np.ndarray] = []

    def write(self, pcm: np.ndarray) -> None:
        with self.lock:
            self.buf.append(pcm)
            self.queued += len(pcm)

    def take(self, n: int) -> tuple[np.ndarray, int]:
        out = np.zeros(n, dtype="<i2")
        got = 0
        with self.lock:
            while got < n and self.buf:
                head = self.buf[0]
                k = min(n - got, len(head))
                out[got:got + k] = head[:k]
                got += k
                if k == len(head):
                    self.buf.popleft()
                else:
                    self.buf[0] = head[k:]
            self.queued -= got
        self.recorded.append(out)
        return out, got


class Side:
    """Hears windows from its input line, decides when the other side finished, answers on its output line."""

    def __init__(self, name: str, vad_model, hang_ms: int):
        self.name = name
        self.vad = VADIterator(vad_model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=hang_ms, speech_pad_ms=30)
        self.heard: list[np.ndarray] = []
        self.in_speech = False
        self.pending_end = False
        self.turn_done = threading.Event()   # this side finished writing its current reply
        self.turn_done.set()
        self.jobs: queue.Queue = queue.Queue()
        self.turns: list[dict] = []
        self.premature = 0
        self.underruns = 0
        self.ready_to_end = False
        self.line_in: Line = None
        self.line_out: Line = None
        self.peer: "Side" = None
        self.t_turn_written = 0.0

    def on_window(self, window: np.ndarray, real: int) -> None:
        # speech end on the line: the last window with speech energy, so tail silence in TTS output does not count
        if real and self.line_in is not None and np.abs(window).mean() > 300:
            self.line_in.t_last_real = time.perf_counter()
        ev = self.vad(torch.from_numpy(window.astype(np.float32) / 32768), return_seconds=False)
        self.feed(window)
        if ev and "start" in ev:
            self.in_speech = True
            self.pending_end = False
        if self.in_speech:
            self.heard.append(window)
        if ev and "end" in ev:
            # the peer still talking means this VAD end fell in a pause, or in TTS tail silence still on the line
            if not self.peer.turn_done.is_set() or self.line_in.queued > 0:
                self.premature += 1
                self.pending_end = True
                return
            self._end_turn()
        elif self.pending_end and self.peer.turn_done.is_set() and self.line_in.queued == 0:
            self._end_turn()

    def _end_turn(self) -> None:
        self.pending_end = False
        self.in_speech = False
        utter = np.concatenate(self.heard) if self.heard else np.zeros(0, dtype="<i2")
        self.heard = []
        self.turn_done.clear()
        self.on_turn_end()
        self.jobs.put((utter, self.line_in.t_last_real, time.perf_counter()))

    def feed(self, window: np.ndarray) -> None:
        pass

    def on_turn_end(self) -> None:
        pass

    def say(self, pcm: np.ndarray, rate: int) -> None:
        self.line_out.write(phone(to16k(pcm, rate)))


class CascadeSide(Side):
    def __init__(self, name, vad_model, hang_ms, stt, llm_url, tts, system, tools):
        super().__init__(name, vad_model, hang_ms)
        self.stt, self.llm_url, self.tts, self.tools = stt, llm_url, tts, tools
        self.messages = [{"role": "system", "content": system}]
        self.client = httpx.Client(timeout=300)

    def worker(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                utter, t_end, t_det = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            text = self.stt.transcribe(utter)
            t_stt = time.perf_counter()
            self.messages.append({"role": "user", "content": text})
            rec = {"side": self.name, "heard": text, "vad_ms": 1000 * (t_det - t_end), "stt_ms": 1000 * (t_stt - t_det)}
            q: queue.Queue = queue.Queue()
            first_audio: list[float] = []

            def speaker() -> None:
                while True:
                    s = q.get()
                    if s is None:
                        return
                    rate = self.tts.rate
                    for pcm in self.tts.stream(s):
                        if not first_audio:
                            first_audio.append(time.perf_counter())
                        elif self.line_out.queued == 0:
                            self.underruns += 1
                        self.say(pcm, rate)

            th = threading.Thread(target=speaker)
            th.start()
            reply = ""
            first_sentence: list[float] = []

            def on_sentence(s: str, t: float) -> None:
                if not first_sentence:
                    first_sentence.append(t)
                q.put(s)

            calls = []
            for _ in range(5):
                r = chat(self.llm_url, self.messages, tools=self.tools, on_sentence=on_sentence, client=self.client, max_tokens=160)
                reply += THINK.sub("", r.content)
                if not r.tool_calls:
                    self.messages.append({"role": "assistant", "content": r.content})
                    break
                calls += [c["name"] for c in r.tool_calls]
                self.messages.append({"role": "assistant", "content": r.content or "", "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in r.tool_calls]})
                for c in r.tool_calls:
                    self.messages.append(tool_message(c["id"], c["name"], c["args"]))
            q.put(None)
            th.join()
            self.turn_done.set()
            fa = first_audio[0] if first_audio else time.perf_counter()
            rec.update(e2e_ms=1000 * (fa - t_end), llm_ms=1000 * ((first_sentence or [fa])[0] - t_stt), tts_ms=1000 * (fa - (first_sentence or [fa])[0]),
                       reply=reply, calls=calls)
            self.turns.append(rec)
            print(f"  {self.name:8s} e2e {rec['e2e_ms']:6.0f} ms  vad {rec['vad_ms']:4.0f} stt {rec['stt_ms']:4.0f} llm {rec['llm_ms']:5.0f} tts {rec['tts_ms']:5.0f} calls {calls}  heard {text[:50]!r} -> {reply[:70]!r}", flush=True)


class RealtimeSide(Side):
    def __init__(self, name, vad_model, hang_ms, url, instructions, tools, loop):
        super().__init__(name, vad_model, hang_ms)
        self.url, self.instructions, self.tools, self.loop = url, instructions, tools, loop
        self.s: Session | None = None
        self.resampled = 0

    async def open(self) -> None:
        self.s = Session(self.url)
        await self.s.__aenter__()
        cfg = {"instructions": self.instructions, "output_modalities": ["text", "audio"]}
        if self.tools:
            cfg.update(tools=self.tools, tool_choice="auto")
        await self.s.update(**cfg)

    def feed(self, window: np.ndarray) -> None:
        if self.in_speech or self.s is None:
            pass
        if self.in_speech:
            b64 = base64.b64encode(window.astype("<i2").tobytes()).decode()
            asyncio.run_coroutine_threadsafe(self.s.send(type="input_audio_buffer.append", audio=b64), self.loop)

    async def run_turn(self, t_end: float, t_det: float) -> None:
        s = self.s
        rec = {"side": self.name, "vad_ms": 1000 * (t_det - t_end), "calls": [], "reply": ""}
        first: list[float] = []

        def on_audio(pcm: np.ndarray, t: float) -> None:
            if not first:
                first.append(t)
            self.say(pcm, 24000)

        s.on_audio = on_audio
        turn = await s.respond()
        await s.done()
        rounds = 0
        while turn.calls and rounds < 5:
            rounds += 1
            rec["reply"] += turn.text + " "
            for c in turn.calls:
                rec["calls"].append(c["name"])
                try:
                    parsed = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    rec["malformed"] = rec.get("malformed", 0) + 1
                    await s.tool_output(c["call_id"], {"error": "malformed arguments"})
                    continue
                await s.tool_output(c["call_id"], run_tool(c["name"], parsed))
            turn = await s.respond()
            await s.done()
        rec["reply"] += turn.text
        self.turn_done.set()
        fa = first[0] if first else time.perf_counter()
        rec["e2e_ms"] = 1000 * (fa - t_end)
        rec["errors"] = list(s.errors)
        self.turns.append(rec)
        print(f"  {self.name:8s} e2e {rec['e2e_ms']:6.0f} ms  vad {rec['vad_ms']:4.0f} calls {rec['calls']} -> {rec['reply'][:90]!r} {s.errors[:1]}", flush=True)

    def worker(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                _, t_end, t_det = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            asyncio.run_coroutine_threadsafe(self.run_turn(t_end, t_det), self.loop).result(timeout=120)


def run_call(a: Side, b: Side, opener: np.ndarray, rate: int, turns: int, max_s: float, label: str) -> dict:
    a.line_out, b.line_out = Line(), Line()
    a.line_in, b.line_in = b.line_out, a.line_out
    a.peer, b.peer = b, a
    stop = threading.Event()
    for side in (a, b):
        threading.Thread(target=side.worker, args=(stop,), daemon=True).start()
    a.say(opener, rate)
    late_ms: list[float] = []
    busy: list[int] = []
    cpu: list[float] = []
    psutil.cpu_percent(None)

    def sampler() -> None:
        while not stop.is_set():
            busy.append(int(BUSY.read_text()))
            cpu.append(psutil.cpu_percent(None))
            time.sleep(0.5)

    threading.Thread(target=sampler, daemon=True).start()
    start = next_t = time.perf_counter()
    while time.perf_counter() - start < max_s and len(a.turns) + len(b.turns) < turns:
        next_t += TICK
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            late_ms.append(-1000 * delay)
        for line, listener in ((a.line_out, b), (b.line_out, a)):
            w, real = line.take(WINDOW)
            listener.on_window(w, real)
    stop.set()
    duration = time.perf_counter() - start
    stereo = np.zeros((max(sum(len(x) for x in a.line_out.recorded), 1), 2), dtype="<i2")
    stereo[:, 0] = np.concatenate(b.line_out.recorded)[:len(stereo)]
    stereo[:, 1] = np.concatenate(a.line_out.recorded)[:len(stereo)]
    import soundfile as sf
    sf.write(OUT / f"{label}.wav", stereo, 16000, subtype="PCM_16")
    turns_all = a.turns + b.turns
    e2e = [t["e2e_ms"] for t in turns_all]
    return {"label": label, "duration_s": duration, "turns": turns_all, "e2e_median_ms": st.median(e2e) if e2e else None, "e2e_max_ms": max(e2e) if e2e else None,
            "clock_late_windows": len(late_ms), "clock_late_max_ms": max(late_ms or [0]), "gpu_busy_mean": st.mean(busy or [0]), "gpu_busy_max": max(busy or [0]),
            "cpu_mean": st.mean(cpu or [0]), "cpu_max": max(cpu or [0]), "premature_vad_ends": a.premature + b.premature, "underruns": a.underruns + b.underruns}


def cascade(args) -> dict:
    vad_model = load_silero_vad(onnx=True)
    vad_model2 = load_silero_vad(onnx=True)
    label = f"cascade-{args.stt}-{args.llm}-{args.tts}-{args.tts_caller or args.tts}"
    gpu0 = gpu_mem_gb()
    servers = []
    try:
        lfm = None
        if "lfm3" in (args.tts, args.tts_caller):
            lfm = LlamaServer(["-m", str(LFM_Q4), "-mm", str(LFM_Q4_MM), "--temp", "0", "-c", "16384", "-np", "2", "-ngl", "999"], name="tb-lfm", port=free_port(18500)).__enter__()
            servers.append(lfm)
        stt, stt_servers = open_stt(args.stt, lfm.url if lfm else None)
        servers += stt_servers
        llm = LlamaServer(llm_args(args.llm, parallel=2), name="tb-llm", port=free_port(18550)).__enter__()
        servers.append(llm)
        tts_a = make_tts(args.tts, lfm.url if lfm else None, 0)
        tts_b = make_tts(args.tts_caller or args.tts, lfm.url if lfm else None, 1)
        tts_a.synth("Warm up.")
        tts_b.synth("Warm up.")
        support = CascadeSide("support", vad_model, args.hang, stt, llm.url, tts_a, SUPPORT_SYSTEM, TOOLS)
        caller = CascadeSide("caller", vad_model2, args.hang, stt, llm.url, tts_b, CALLER_SYSTEM, None)
        opener = tts_a.synth("Thanks for calling Fernhill Internet, this is Robin. How can I help you today?")
        support.messages.append({"role": "assistant", "content": "Thanks for calling Fernhill Internet, this is Robin. How can I help you today?"})
        res = run_call(support, caller, opener, tts_a.rate, args.turns, args.max_s, label)
        res["gpu_gb"] = gpu_mem_gb() - gpu0
        return res
    finally:
        for s in servers:
            s.__exit__(None, None, None)


def realtime(args) -> dict:
    vad_model = load_silero_vad(onnx=True)
    vad_model2 = load_silero_vad(onnx=True)
    label = f"realtime-{args.quant}"
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    gpu0 = gpu_mem_gb()
    with LlamaServer(lfm_args(args.quant, "vulkan"), name="tb-a") as sa, LlamaServer(lfm_args(args.quant, "vulkan"), name="tb-b", port=free_port(18600)) as sb:
        support = RealtimeSide("support", vad_model, args.hang, sa.url, PERSONA + TOOLS_PROMPT, RT_TOOLS, loop)
        caller_instr = ("You are a customer named Dana Reyes phoning Fernhill Internet. Your account number is AC9077. You were charged twice for invoice FH-4419 and FH-4420, "
                        "49 dollars and 99 cents each. Get the duplicate refunded. Say one short sentence per turn. Respond with interleaved text and audio.")
        caller = RealtimeSide("caller", vad_model2, args.hang, sb.url, caller_instr, None, loop)
        asyncio.run_coroutine_threadsafe(support.open(), loop).result(30)
        asyncio.run_coroutine_threadsafe(caller.open(), loop).result(30)
        kok = KokoroTTS("af_heart")
        opener = kok.synth("Thanks for calling Fernhill Internet, this is Robin. How can I help you today?")
        res = run_call(support, caller, opener, 24000, args.turns, args.max_s, label)
        res["gpu_gb"] = gpu_mem_gb() - gpu0
    loop.call_soon_threadsafe(loop.stop)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["cascade", "realtime"])
    p.add_argument("--stt", default="parakeet")
    p.add_argument("--llm", default="lfm3-1.7b", choices=list(LLMS))
    p.add_argument("--tts", default="kokoro")
    p.add_argument("--tts-caller", default=None)
    p.add_argument("--quant", default="q8")
    p.add_argument("--turns", type=int, default=12)
    p.add_argument("--max-s", type=float, default=180)
    p.add_argument("--hang", type=int, default=300)
    args = p.parse_args()
    res = cascade(args) if args.mode == "cascade" else realtime(args)
    save_result(f"twobots-{res['label']}", res)
    print(f"== {res['label']}: {len(res['turns'])} turns in {res['duration_s']:.0f}s, e2e median {res['e2e_median_ms']} max {res['e2e_max_ms']}, "
          f"late windows {res['clock_late_windows']} (max {res['clock_late_max_ms']:.1f} ms), premature VAD {res['premature_vad_ends']}, underruns {res['underruns']}, "
          f"gpu busy {res['gpu_busy_mean']:.0f}%/{res['gpu_busy_max']}%, cpu {res['cpu_mean']:.0f}%/{res['cpu_max']:.0f}%, gpu mem {res['gpu_gb']:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
