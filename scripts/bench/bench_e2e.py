"""End-to-end latency: end of caller speech to start of bot audio.

Cascade: clip is fed at wall-clock pace through Silero VAD, the utterance goes to STT, the transcript to a streaming LLM
with tools, tool calls run against the mock, and the first finished sentence goes to TTS. The clock stops at the first
audio sample available to send. A 4-turn call runs per repetition so the prompt grows as it would in a real call.

Usage: bench_e2e.py STACK [STACK...] [--reps 3] [--vad-hang-ms 300] [--cond tel|clean]
Stacks are stt+llm+tts, for example parakeet+lfm3-1.7b+kokoro.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics as st
import sys
import threading
import time

import httpx
import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from common import BENCH, LFM_Q4, LFM_Q4_MM, LFM_Q8, LFM_Q8_MM, MODELS, LlamaServer, free_port, gpu_mem_gb, save_result
from fernhill_fixture import SUPPORT_SYSTEM, TOOLS, run_tool, tool_message
from llm_client import THINK, chat
from rt_client import read_pcm16
from speech_backends import make_tts, open_stt

torch.set_num_threads(1)
WINDOW = 512  # Silero window at 16 kHz, 32 ms
LLMS = {
    "lfm3-1.7b": (MODELS / "LiquidAI__LFM3-1.7B-GGUF" / "LFM3-1.7B-Q4_K_M.gguf", ["--temp", "0.1", "--top-k", "50", "--repeat-penalty", "1.1"]),
    "qwen3.5-4b": (MODELS / "unsloth__Qwen3.5-4B-GGUF" / "Qwen3.5-4B-Q4_K_M.gguf", []),
    "qwen3.5-9b": (MODELS / "unsloth__Qwen3.5-9B-GGUF" / "Qwen3.5-9B-Q4_K_M.gguf", []),
    "bolt-e4b": (MODELS / "nvidia__NVIDIA-Nemotron-3.5-Bolt-E4B-4GB" / "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf", []),
    "gemma-e2b": (MODELS / "ggml-org__gemma-4-E2B-it-GGUF" / "gemma-4-E2B-it-Q4_0.gguf", []),
    "gemma-e4b": (MODELS / "ggml-org__gemma-4-E4B-it-GGUF" / "gemma-4-E4B-it-Q4_0.gguf", []),
}
CALL = ["caller-greeting", "caller-lookup", "caller-refund", "caller-question"]


def llm_args(name: str, parallel: int = 1) -> list[str]:
    model, sampling = LLMS[name]
    return ["-m", str(model), "--jinja", "--reasoning", "off", "-c", str(16384 * parallel), "-ngl", "999", "-np", str(parallel), *sampling]


def speech_end_index(pcm: np.ndarray, floor: float = 0.01) -> int:
    """Sample index where the last 20 ms frame above the floor ends. Ground truth for end of speech."""
    hop = 320
    frames = np.abs(pcm[: len(pcm) // hop * hop].reshape(-1, hop) / 32768).mean(axis=1)
    loud = np.nonzero(frames > floor)[0]
    return int((loud[-1] + 1) * hop) if len(loud) else 0


class Vad:
    def __init__(self, hang_ms: int):
        self.model = load_silero_vad(onnx=True)
        self.hang_ms = hang_ms

    def listen(self, pcm: np.ndarray, tail_s: float = 1.2, tap=None) -> dict:
        """Feeds clip plus trailing silence at wall-clock pace. Returns timing of the detected end of speech.

        VAD ends inside the utterance are counted but ignored, standing in for a turn-completion model. The turn ends at
        the first VAD end after the true end of speech. tap(window) receives each window as it is fed.
        """
        full = np.concatenate([pcm, np.zeros(int(16000 * tail_s), dtype="<i2")])
        it = VADIterator(self.model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=self.hang_ms, speech_pad_ms=30)
        it.reset_states()
        true_end = speech_end_index(pcm)
        start = time.perf_counter()
        begin = None
        cost = 0.0
        premature = 0
        for i, j in enumerate(range(0, len(full) - WINDOW, WINDOW)):
            target = start + (i + 1) * WINDOW / 16000
            time.sleep(max(0, target - time.perf_counter()))
            if tap:
                tap(full[j:j + WINDOW])
            t = time.perf_counter()
            ev = it(torch.from_numpy(full[j:j + WINDOW].astype(np.float32) / 32768), return_seconds=False)
            cost += time.perf_counter() - t
            if ev and "start" in ev and begin is None:
                begin = ev["start"]
            if ev and "end" in ev:
                if ev["end"] < true_end - 1600:
                    premature += 1
                    continue
                detected = time.perf_counter()
                return {"t_speech_end": start + true_end / 16000, "t_detected": detected, "begin": begin or 0, "end": ev["end"],
                        "vad_cost_per_window_ms": 1000 * cost / (i + 1), "premature_ends": premature}
        raise RuntimeError("VAD found no end of speech")


class Stack:
    def __init__(self, spec: str, reps: int, hang_ms: int, cond: str, voice: str = "af_heart"):
        self.spec = spec
        self.stt_name, self.llm_name, self.tts_name = spec.split("+")
        self.reps, self.cond = reps, cond
        self.hang_ms = hang_ms
        self.voice = voice

    def run(self) -> dict:
        args_lfm = ["-m", str(LFM_Q4), "-mm", str(LFM_Q4_MM), "--temp", "0", "-c", "8192", "-np", "1", "-ngl", "999"]
        need_lfm = self.stt_name == "lfm3" or self.tts_name == "lfm3"
        servers: dict = {}
        gpu0 = gpu_mem_gb()
        self.vad = Vad(self.hang_ms)
        try:
            if need_lfm:
                servers["lfm"] = LlamaServer(args_lfm, name="e2e-lfm", port=free_port(18400)).__enter__()
            stt, stt_servers = open_stt(self.stt_name, servers["lfm"].url if need_lfm else None)
            for i, srv in enumerate(stt_servers):
                servers[f"stt{i}"] = srv
            servers["llm"] = LlamaServer(llm_args(self.llm_name), name="e2e-llm", port=free_port(18450)).__enter__()
            tts = make_tts(self.tts_name, servers["lfm"].url if need_lfm else None)
            mem = gpu_mem_gb() - gpu0
            return self._calls(stt, servers["llm"].url, tts) | {"gpu_gb": mem}
        finally:
            for s in servers.values():
                s.__exit__(None, None, None)

    def _calls(self, stt, llm_url: str, tts) -> dict:
        caller = {c["id"]: c for c in json.loads((BENCH / "caller.json").read_text())}
        client = httpx.Client(timeout=300)
        tts.synth("Warming up.")
        turns = []
        for rep in range(self.reps):
            messages = [{"role": "system", "content": SUPPORT_SYSTEM}]
            for clip in CALL:
                pcm = read_pcm16(caller[clip][self.cond])
                turn = self._turn(stt, llm_url, tts, pcm, messages, client)
                turn.update(rep=rep, clip=clip)
                turns.append(turn)
                print(f"  r{rep} {clip:16s} e2e {turn['e2e_ms']:6.0f} ms = vad {turn['vad_ms']:4.0f} stt {turn['stt_ms']:4.0f} "
                      f"llm {turn['llm_ms']:5.0f} (calls {turn['n_tool_rounds']}) tts {turn['tts_ms']:5.0f}  heard {turn['heard'][:44]!r} say {turn['reply'][:60]!r}", flush=True)
        keep = [t for t in turns if t["rep"] > 0 or self.reps == 1]
        return {"stack": self.spec, "cond": self.cond, "vad_hang_ms": self.hang_ms, "turns": turns,
                "e2e_median_ms": st.median(t["e2e_ms"] for t in keep), "e2e_max_ms": max(t["e2e_ms"] for t in keep),
                "stt_median_ms": st.median(t["stt_ms"] for t in keep), "llm_median_ms": st.median(t["llm_ms"] for t in keep),
                "tts_median_ms": st.median(t["tts_ms"] for t in keep)}

    def _turn(self, stt, llm_url, tts, pcm, messages, client) -> dict:
        v = self.vad.listen(pcm)
        t_end = v["t_speech_end"]
        t_det = v["t_detected"]
        utter = pcm[int(v["begin"]): int(v["end"])]
        text = stt.transcribe(utter)
        t_stt = time.perf_counter()
        messages.append({"role": "user", "content": text})

        first_audio: list[float] = []
        first_sentence: list[float] = []
        q: list[str] = []
        cv = threading.Condition()
        finished = [False]

        def speaker() -> None:
            while True:
                with cv:
                    while not q and not finished[0]:
                        cv.wait()
                    if not q:
                        return
                    s = q.pop(0)
                tts.synth(s, on_first=lambda t: first_audio.append(t))

        def on_sentence(s: str, t: float) -> None:
            first_sentence.append(t)
            with cv:
                q.append(s)
                cv.notify()

        th = threading.Thread(target=speaker)
        th.start()
        rounds = 0
        reply = ""
        tcalls = []
        for _ in range(5):
            r = chat(llm_url, messages, tools=TOOLS, on_sentence=on_sentence, client=client, max_tokens=200)
            reply += THINK.sub("", r.content)
            tcalls += [{"name": c["name"], "args": c["args"]} for c in r.tool_calls]
            if not r.tool_calls:
                messages.append({"role": "assistant", "content": r.content})
                break
            rounds += 1
            messages.append({"role": "assistant", "content": r.content or "", "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in r.tool_calls]})
            for c in r.tool_calls:
                messages.append(tool_message(c["id"], c["name"], c["args"]))
        with cv:
            finished[0] = True
            cv.notify()
        t_llm_done = time.perf_counter()
        th.join()
        fa = first_audio[0] if first_audio else time.perf_counter()
        fs = first_sentence[0] if first_sentence else t_llm_done
        return {"e2e_ms": 1000 * (fa - t_end), "vad_ms": 1000 * (t_det - t_end), "stt_ms": 1000 * (t_stt - t_det),
                "llm_ms": 1000 * (fs - t_stt), "tts_ms": 1000 * (fa - fs), "n_tool_rounds": rounds, "tools": tcalls,
                "heard": text, "reply": reply, "last_timings": r.timings, "vad_cost_ms": v["vad_cost_per_window_ms"],
                "premature_ends": v["premature_ends"]}


async def realtime_call(url: str, vad: Vad, cond: str, reps: int, instructions: str) -> dict:
    """Same call and same VAD, through one /v1/realtime session per call with forced text+audio."""
    from bench_lfm_realtime import TOOLS as RT_TOOLS
    from rt_client import Session
    caller = {c["id"]: c for c in json.loads((BENCH / "caller.json").read_text())}
    loop = asyncio.get_running_loop()
    turns = []
    for rep in range(reps):
        async with Session(url) as s:
            await s.update(instructions=instructions, tools=RT_TOOLS, tool_choice="auto", output_modalities=["text", "audio"])
            for clip in CALL:
                pcm = read_pcm16(caller[clip][cond])

                def tap(window: np.ndarray) -> None:
                    b64 = base64.b64encode(window.astype("<i2").tobytes()).decode()
                    asyncio.run_coroutine_threadsafe(s.send(type="input_audio_buffer.append", audio=b64), loop)

                v = await asyncio.to_thread(vad.listen, pcm, 1.2, tap)
                turn = await s.respond()
                t_start = turn.t_end_input
                await s.done()
                fa = turn.t_first_audio
                rounds = 0
                tools = []
                malformed = 0
                last = turn
                while last.calls and rounds < 5:
                    rounds += 1
                    for c in last.calls:
                        tools.append({"name": c["name"], "args": c["arguments"]})
                        try:
                            parsed = json.loads(c["arguments"] or "{}")
                        except json.JSONDecodeError:
                            malformed += 1
                            await s.tool_output(c["call_id"], {"error": "malformed arguments"})
                            continue
                        await s.tool_output(c["call_id"], run_tool(c["name"], parsed))
                    last = await s.respond()
                    await s.done()
                    fa = fa or last.t_first_audio
                rec = {"rep": rep, "clip": clip, "e2e_ms": 1000 * (fa - v["t_speech_end"]) if fa else None, "vad_ms": 1000 * (v["t_detected"] - v["t_speech_end"]),
                       "model_ms": 1000 * (fa - t_start) if fa else None, "n_tool_rounds": rounds, "tools": tools, "reply": last.text, "audio_s": sum(len(a) for a in last.audio) / 24000,
                       "premature_ends": v["premature_ends"], "malformed_args": malformed, "errors": list(s.errors)}
                turns.append(rec)
                print(f"  r{rep} {clip:16s} e2e {rec['e2e_ms'] or -1:6.0f} ms = vad {rec['vad_ms']:4.0f} model {rec['model_ms'] or -1:5.0f} calls {[t['name'] for t in tools]} say {last.text[:70]!r}", flush=True)
    ok = [t["e2e_ms"] for t in turns if t["e2e_ms"] is not None]
    return {"stack": "lfm3-realtime", "cond": cond, "turns": turns, "e2e_median_ms": st.median(ok), "e2e_max_ms": max(ok)}


def run_realtime(args) -> None:
    from bench_lfm_realtime import PERSONA, TOOLS_PROMPT, lfm_args
    vad = Vad(args.vad_hang_ms)
    gpu0 = gpu_mem_gb()
    with LlamaServer(lfm_args("q4" if args.quant == "q4" else "q8", "vulkan"), name="e2e-rt") as s:
        res = asyncio.run(realtime_call(s.url, vad, args.cond, args.reps, PERSONA + TOOLS_PROMPT))
    res["gpu_gb"] = gpu_mem_gb() - gpu0
    save_result(f"e2e-lfm3-realtime-{args.quant}-{args.cond}", res)
    print(f"  median e2e {res['e2e_median_ms']:.0f} ms, max {res['e2e_max_ms']:.0f} ms", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stacks", nargs="+")
    p.add_argument("--quant", default="q8", choices=["q8", "q4"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--vad-hang-ms", type=int, default=300)
    p.add_argument("--cond", default="tel", choices=["tel", "clean"])
    p.add_argument("--voice", default="af_heart")
    args = p.parse_args()
    for spec in args.stacks:
        print(f"== {spec} cond={args.cond} hang={args.vad_hang_ms}", flush=True)
        if spec == "lfm3-realtime":
            run_realtime(args)
            continue
        res = Stack(spec, args.reps, args.vad_hang_ms, args.cond, args.voice).run()
        save_result(f"e2e-{spec.replace('+', '_')}-{args.cond}", res)
        print(f"  median e2e {res['e2e_median_ms']:.0f} ms, max {res['e2e_max_ms']:.0f} ms, stt {res['stt_median_ms']:.0f}, llm {res['llm_median_ms']:.0f}, tts {res['tts_median_ms']:.0f}, gpu {res['gpu_gb']:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
