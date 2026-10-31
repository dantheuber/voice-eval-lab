"""Qwen3-TTS CustomVoice on ROCm 7.2, run with .runtime/venvs/qwen-rocm/bin/python.

Bf16 on the Radeon, codec decoder on CPU in float32, as in ~/projects/cc-tts. The package has no streaming generation, so
time to first audio is the time for a whole chunk. Usage: bench_qwen_tts_rocm.py 0.6B 1.7B
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".runtime" / "bench" / "out" / "tts"
RESULTS = ROOT / ".runtime" / "bench" / "results"
SENTENCES = [
    ("short", "Thanks for calling Fernhill Internet. How can I help you today?"),
    ("sorry", "I'm sorry about the double charge. I can issue that refund for you right now."),
    ("invoice", "Your invoice number is F H dash 4 4 1 9 for forty nine dollars and ninety nine cents."),
]


def run(size: str) -> dict:
    name = f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
    t0 = time.perf_counter()
    tun = ROOT / ".runtime" / "bench" / f"tunableop-{size}.csv"
    torch.cuda.tunable.set_filename(str(tun))
    torch.cuda.tunable.enable(True)
    torch.cuda.tunable.tuning_enable(True)
    model = Qwen3TTSModel.from_pretrained(name, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    codec = model.model.speech_tokenizer
    codec.model = codec.model.to(device="cpu", dtype=torch.float32)
    codec.device = torch.device("cpu")
    load = time.perf_counter() - t0
    model.generate_custom_voice(text="Ready.", language="English", speaker="Ryan")
    out = OUT / f"qwen-rocm-{size}"
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for rep in range(3):
        for sid, text in SENTENCES:
            t = time.perf_counter()
            wavs, sr = model.generate_custom_voice(text=text, language="English", speaker="Ryan")
            dt = time.perf_counter() - t
            wav = np.asarray(wavs[0], dtype=np.float32)
            dur = len(wav) / sr
            sf.write(out / f"{sid}-{rep}.wav", wav, sr)
            runs.append({"id": sid, "rep": rep, "total_s": dt, "audio_s": dur, "rtf": dur / dt, "ttfa_s": dt, "n_chunks": 1, "rate": sr})
            print(f"  {size} {sid}: {dur:.1f}s audio in {dt:.1f}s, rtf {dur / dt:.2f}", flush=True)
    res = {"engine": f"qwen3-tts-{size}-rocm", "rate": runs[0]["rate"], "load_s": load, "runs": runs, "rtf_median": st.median(r["rtf"] for r in runs[3:]),
           "ttfa_median_s": st.median(r["ttfa_s"] for r in runs[3:]), "ttfa_max_s": max(r["ttfa_s"] for r in runs[3:]), "chunks_median": 1, "streams": False,
           "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    print(f"{res['engine']}: load {load:.1f}s rtf {res['rtf_median']:.2f} ttfa {res['ttfa_median_s']:.1f}s gpu {res['gpu_peak_gb']:.1f} GB", flush=True)
    (RESULTS / f"tts-{res['engine']}.json").write_text(json.dumps(res, indent=2))
    del model
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for size in sys.argv[1:] or ["0.6B", "1.7B"]:
        run(size)
