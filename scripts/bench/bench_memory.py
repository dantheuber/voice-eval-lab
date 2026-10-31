"""Loads the whole recommended stack at once and records GPU and process memory after each step.

GPU numbers are the system-wide GTT plus VRAM counters, so the deltas assume nothing else loads a model during the run.
Usage: bench_memory.py [--judge PATH]
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx
import psutil

from bench_e2e import llm_args
from common import LFM_Q4, LFM_Q4_MM, LlamaServer, MODELS, free_port, gpu_mem_gb, rss_gb, save_result
from speech_backends import PhononTTS, PocketTTS, WhisperServer, open_stt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--judge", default=str(MODELS / "unsloth__Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-Q4_K_M.gguf"))
    args = p.parse_args()
    steps = []
    open_servers = []
    start_gpu = gpu_mem_gb()
    start_used = psutil.virtual_memory().used / 1e9

    def note(name: str, pid: int | None = None) -> None:
        time.sleep(1)
        steps.append({"step": name, "gpu_total_gb": gpu_mem_gb() - start_gpu, "rss_gb": rss_gb(pid) if pid else None,
                      "sys_used_gb": psutil.virtual_memory().used / 1e9 - start_used})
        print(f"{name:34s} gpu +{steps[-1]['gpu_total_gb']:6.2f} GB  rss {steps[-1]['rss_gb'] or 0:5.2f} GB  system +{steps[-1]['sys_used_gb']:6.2f} GB", flush=True)

    try:
        _, stt_servers = open_stt("qwen-asr")
        open_servers += stt_servers
        note("qwen3-asr 0.6b q8 -np 2", stt_servers[0].proc.pid)
        llm = LlamaServer(llm_args("qwen3.5-4b", parallel=2), name="mem-llm", port=free_port(18800)).__enter__()
        open_servers.append(llm)
        note("qwen3.5-4b q4_k_m -np 2", llm.proc.pid)
        lfm_a = LlamaServer(["-m", str(LFM_Q4), "-mm", str(LFM_Q4_MM), "--temp", "0", "-c", "16384", "-np", "1", "-ngl", "999"], name="mem-lfm-a", port=free_port(18810)).__enter__()
        open_servers.append(lfm_a)
        note("lfm3-audio q4_k_m", lfm_a.proc.pid)
        phonon = PhononTTS("Easton")
        list(phonon.stream("Warm up."))
        note("phonon in this process")
        pocket = PocketTTS("english_18m_2026-10", "marius")
        list(pocket.stream("Warm up."))
        note("pocket 18m in this process")
        whisper = WhisperServer(model="ggml-large-v3-q5_0.bin").__enter__()
        open_servers.append(whisper)
        note("whisper large-v3 q5", whisper.proc.pid)
        judge = LlamaServer(["-m", args.judge, "--jinja", "-c", "32768", "-np", "1", "-ngl", "999"], name="mem-judge", port=free_port(18820), timeout=600).__enter__()
        open_servers.append(judge)
        note("judge llm", judge.proc.pid)
        # one request to each so lazily allocated buffers exist
        httpx.post(judge.url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 4}, timeout=300)
        httpx.post(llm.url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 4}, timeout=300)
        note("after one request each")
        steps.append({"free_system_gb": psutil.virtual_memory().available / 1e9})
        save_result("memory-all-loaded", {"steps": steps})
        print(steps[-1], flush=True)
    finally:
        for s in reversed(open_servers):
            s.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
