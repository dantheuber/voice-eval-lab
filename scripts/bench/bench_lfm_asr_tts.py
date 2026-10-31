"""LFM3-Audio ASR and TTS through llama-server, on Vulkan and CPU, Q8 and Q4_K_M.

ASR: POST /v1/audio/transcriptions per clip, clean and telephone copies.
TTS: POST /tts with stream=true and response_format=pcm. Time to first byte is time to first audio.
Usage: bench_lfm_asr_tts.py [vulkan-q8 vulkan-q4 cpu-q8 cpu-q4]
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
import wave

import httpx
import numpy as np

from common import (BENCH, LFM_Q4, LFM_Q4_MM, LFM_Q8, LFM_Q8_MM, LlamaServer, critical_hits, save_result, wer)

CONFIGS = {
    "vulkan-q8": (LFM_Q8, LFM_Q8_MM, ["-ngl", "999"]),
    "vulkan-q4": (LFM_Q4, LFM_Q4_MM, ["-ngl", "999"]),
    "cpu-q8": (LFM_Q8, LFM_Q8_MM, ["-ngl", "0", "--no-mmproj-offload", "--device", "none", "-t", "16"]),
    "cpu-q4": (LFM_Q4, LFM_Q4_MM, ["-ngl", "0", "--no-mmproj-offload", "--device", "none", "-t", "16"]),
}
TTS_TEXT = [
    "Your invoice number is F H dash 4 4 1 9 for forty nine dollars and ninety nine cents.",
    "Update your MX records and check IMAP settings at mail dot fernhill dot example.",
    "I'm sorry about the double charge. I can issue that refund for you right now.",
    "Thanks for calling Fernhill Internet. How can I help you today?",
]


def asr(url: str, path: str) -> tuple[str, float]:
    with open(path, "rb") as f:
        t0 = time.perf_counter()
        r = httpx.post(url + "/v1/audio/transcriptions", files={"file": ("a.wav", f, "audio/wav")},
                       data={"model": "x", "language": "en"}, timeout=300)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json()["text"], dt


def tts_stream(url: str, text: str, voice: str, out_path: str) -> dict:
    body = {"input": text, "voice": voice, "lang": "en", "stream": True, "response_format": "pcm"}
    t0 = time.perf_counter()
    first = None
    buf = bytearray()
    chunk_log = []
    with httpx.stream("POST", url + "/tts", json=body, timeout=300) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type")
        for chunk in r.iter_bytes():
            if chunk and first is None:
                first = time.perf_counter() - t0
            if chunk:
                chunk_log.append((time.perf_counter() - t0, len(chunk)))
            buf += chunk
    total = time.perf_counter() - t0
    pcm = np.frombuffer(bytes(buf[: len(buf) // 4 * 4]), dtype="<f4")
    rate = 24000
    chunks = chunk_log
    if "rate=" in (ctype or ""):
        rate = int(ctype.split("rate=")[1].split(";")[0])
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    dur = len(pcm) / rate
    # first time 20 ms of audio louder than -50 dBFS had arrived, so a leading silent chunk does not count as audio
    audible = float(np.argmax(np.abs(pcm) > 0.00316)) / rate if len(pcm) else 0.0
    done_bytes = 0
    ttfa_audible = None
    for t, n in chunks:
        done_bytes += n
        if done_bytes / 4 / rate >= audible + 0.02:
            ttfa_audible = t
            break
    return {"ttfa_s": first, "ttfa_audible_s": ttfa_audible, "lead_silence_s": audible, "n_chunks": len(chunks),
            "first_chunk_bytes": chunks[0][1] if chunks else 0, 
            "total_s": total, "audio_s": dur, "rtf": dur / total if total else 0, "rate": rate, "ctype": ctype}


def main(names: list[str]) -> int:
    manifest = json.loads((BENCH / "manifest.json").read_text())
    out_dir = BENCH / "out" / "lfm-tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        model, mm, flags = CONFIGS[name]
        args = ["-m", str(model), "-mm", str(mm), "--temp", "0", "-c", "16384", "-np", "1", *flags]
        with LlamaServer(args, name=f"lfm-{name}") as s:
            res: dict = {"config": name, "load_s": s.load_s, "gpu_gb": s.gpu_delta_gb, "rss_gb": s.rss(), "asr": [], "tts": []}
            print(f"== {name} load {s.load_s:.1f}s gpu+{s.gpu_delta_gb:.2f}GB rss {s.rss():.2f}GB", flush=True)
            for cond in ("clean", "tel"):
                for m in manifest:
                    text, dt = asr(s.url, m[cond])
                    rec = {"id": m["id"], "cond": cond, "kind": m["kind"], "dur": m["duration"], "secs": dt, "rtf": m["duration"] / dt, "text": text}
                    if m["text"]:
                        rec["wer"] = wer(m["text"], text)
                    if m["critical"]:
                        hits, total, missed = critical_hits(text, m["critical"])
                        rec.update(crit_hits=hits, crit_total=total, crit_missed=missed)
                    res["asr"].append(rec)
                scored = [r for r in res["asr"] if r["cond"] == cond and "wer" in r]
                crit = [r for r in scored if "crit_total" in r]
                print(f"  ASR {cond}: mean WER {st.mean(r['wer'] for r in scored):.3f} "
                      f"human {st.mean(r['wer'] for r in scored if r['kind'] == 'human'):.3f} "
                      f"critical {sum(r['crit_hits'] for r in crit)}/{sum(r['crit_total'] for r in crit)} "
                      f"median RTF {st.median(r['rtf'] for r in res['asr'] if r['cond'] == cond):.1f}x", flush=True)
            for rep in range(3):
                for voice in ("female", "male"):
                    for i, text in enumerate(TTS_TEXT):
                        rec = tts_stream(s.url, text, voice, str(out_dir / f"{name}-{voice}-{i}-{rep}.wav"))
                        rec.update(voice=voice, text=text, rep=rep)
                        res["tts"].append(rec)
            print(f"  TTS: median TTFA {st.median(r['ttfa_s'] for r in res['tts']):.3f}s "
                  f"(audible {st.median(r['ttfa_audible_s'] or 0 for r in res['tts']):.3f}s), "
                  f"first chunk {res['tts'][0]['first_chunk_bytes']} bytes, chunks {res['tts'][0]['n_chunks']}, "
                  f"median RTF {st.median(r['rtf'] for r in res['tts']):.2f}x, "
                  f"rate {res['tts'][0]['rate']}, ctype {res['tts'][0]['ctype']}", flush=True)
            res["rss_gb_after"] = s.rss()
            save_result(f"lfm-asr-tts-{name}", res)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(CONFIGS)))
