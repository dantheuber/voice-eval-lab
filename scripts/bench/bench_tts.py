"""TTS candidates: real-time factor, time to first audio, streaming behavior, and critical-token accuracy heard by Parakeet.

Usage: bench_tts.py [kokoro phonon pocket-default pocket-7m pocket-2m pocket-donation lfm3] [--threads N]
Each engine runs the same sentences. Saved WAVs land in .runtime/bench/out/tts/<engine>/. Transcribe afterward with
bench_tts.py --score, which loads Parakeet on Vulkan and scores every saved WAV.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from common import BENCH, LFM_Q4, LFM_Q4_MM, LlamaServer, critical_hits, save_result, wer
from rt_client import write_pcm16
from speech_backends import KokoroTTS, open_stt

OUT = BENCH / "out" / "tts"
SENTENCES = [
    ("invoice", "Your invoice number is F H dash 4 4 1 9 for forty nine dollars and ninety nine cents.", [r"f ?h (dash )?4 4 1 9", r"49 dollars and 99 cents|49 99"]),
    ("invoice-digits", "Your invoice FH-4419 for $49.99 was charged on March 14th.", [r"f ?h (dash )?4 4 1 9", r"49 dollars and 99 cents|49 99", r"march 14"]),
    ("mx", "Update your MX records and check IMAP settings at mail dot fernhill dot example.", [r"\bmx\b", r"\bimap\b", r"fernhill"]),
    ("url", "You can renew it at fernhill.example/ssl, or email help@fernhill.example.", [r"fernhill", r"\bssl\b", r"help"]),
    ("short", "Thanks for calling Fernhill Internet. How can I help you today?", [r"fernhill", r"help"]),
    ("sorry", "I'm sorry about the double charge. I can issue that refund for you right now.", [r"double charge", r"refund"]),
]


def run_engine(name: str, make, reps: int = 2) -> dict:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = make()
    load_s = time.perf_counter() - t0
    engine.warm()
    runs = []
    for rep in range(reps):
        for sid, text, _ in SENTENCES:
            t = time.perf_counter()
            first = None
            pieces = []
            for chunk in engine.stream(text):
                if first is None:
                    first = time.perf_counter() - t
                pieces.append(chunk)
            total = time.perf_counter() - t
            pcm = np.concatenate(pieces) if pieces else np.zeros(0, dtype="<i2")
            dur = len(pcm) / engine.rate
            write_pcm16(str(out / f"{sid}-{rep}.wav"), pcm, engine.rate)
            runs.append({"id": sid, "rep": rep, "ttfa_s": first, "total_s": total, "audio_s": dur, "rtf": dur / total if total else 0, "n_chunks": len(pieces)})
    res = {"engine": name, "rate": engine.rate, "load_s": load_s, "runs": runs, "rtf_median": st.median(r["rtf"] for r in runs),
           "ttfa_median_s": st.median(r["ttfa_s"] for r in runs), "ttfa_max_s": max(r["ttfa_s"] for r in runs),
           "chunks_median": st.median(r["n_chunks"] for r in runs), "streams": st.median(r["n_chunks"] for r in runs) > 1}
    print(f"{name:18s} load {load_s:5.1f}s rtf {res['rtf_median']:5.2f}x ttfa {res['ttfa_median_s']*1000:6.0f} ms (max {res['ttfa_max_s']*1000:.0f}) chunks {res['chunks_median']}", flush=True)
    save_result(f"tts-{name}", res)
    return res


class Kokoro:
    rate = 24000

    def __init__(self, voice="af_heart"):
        self.t = KokoroTTS(voice)

    def warm(self):
        self.t.synth("Warm up.")

    def stream(self, text):
        yield self.t.synth(text)


class Phonon:
    rate = 24000

    def __init__(self, voice="Easton", quant="q8"):
        import ptts
        self.tts = ptts.TTS("hf://gradium/phonon-31m-en", lang="en", quant=quant)
        self.voice = voice

    def warm(self):
        list(self.stream("Warm up."))

    def stream(self, text):
        with self.tts.stream(text, voice=self.voice) as audio:
            self.rate = audio.sample_rate
            for pcm in audio:
                yield (np.clip(np.asarray(pcm), -1, 1) * 32767).astype("<i2")


class Pocket:
    def __init__(self, language=None, voice="marius", threads=None):
        import torch
        from pocket_tts import TTSModel
        if threads:
            torch.set_num_threads(threads)
        self.model = TTSModel.load_model(language=language) if language else TTSModel.load_model()
        self.rate = self.model.sample_rate
        self.state = self.model.get_state_for_audio_prompt(voice)

    def warm(self):
        list(self.stream("Warm up."))

    def stream(self, text):
        for chunk in self.model.generate_audio_stream(self.state, text):
            yield (np.clip(chunk.numpy(), -1, 1) * 32767).astype("<i2")


class Lfm:
    rate = 24000

    def __init__(self, voice="female"):
        self.server = LlamaServer(["-m", str(LFM_Q4), "-mm", str(LFM_Q4_MM), "--temp", "0", "-c", "8192", "-np", "1", "-ngl", "999"], name="tts-lfm")
        self.server.__enter__()
        self.voice = voice

    def warm(self):
        list(self.stream("Warm up."))

    def stream(self, text):
        body = {"input": text, "voice": self.voice, "lang": "en", "stream": True, "response_format": "pcm"}
        with httpx.stream("POST", self.server.url + "/tts", json=body, timeout=120) as r:
            r.raise_for_status()
            left = b""
            for chunk in r.iter_bytes():
                data = left + chunk
                n = len(data) // 4 * 4
                left = data[n:]
                if n:
                    yield (np.clip(np.frombuffer(data[:n], dtype="<f4"), -1, 1) * 32767).astype("<i2")


def score() -> int:
    from common import RT
    import subprocess
    table = {sid: (text, crit) for sid, text, crit in SENTENCES}
    results = {}
    stt, servers = open_stt("qwen-asr")
    try:
        for d in sorted(OUT.iterdir()):
            rows = []
            for wav in sorted(d.glob("*-0.wav")):
                sid = wav.stem.rsplit("-", 1)[0]
                if sid not in table:
                    continue
                f = RT / "bench" / "tmp16k.wav"
                subprocess.run([str(RT / "ffmpeg/bin/ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-ar", "16000", "-ac", "1", str(f)], check=True)
                from rt_client import read_pcm16
                heard = stt.transcribe(read_pcm16(str(f)))
                hits, total, missed = critical_hits(heard, table[sid][1])
                rows.append({"id": sid, "heard": heard, "wer": wer(table[sid][0], heard), "hits": hits, "total": total, "missed": missed})
            if rows:
                results[d.name] = rows
                print(f"{d.name:18s} wer {st.mean(r['wer'] for r in rows):.3f} critical {sum(r['hits'] for r in rows)}/{sum(r['total'] for r in rows)}", flush=True)
                for r in rows:
                    if r["missed"]:
                        print(f"     {r['id']}: missed {r['missed']} heard {r['heard'][:110]!r}", flush=True)
    finally:
        for srv in servers:
            srv.__exit__(None, None, None)
    save_result("tts-score", results)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("engines", nargs="*")
    p.add_argument("--score", action="store_true")
    p.add_argument("--threads", type=int)
    p.add_argument("--voice-file")
    args = p.parse_args()
    if args.score:
        return score()
    makers = {
        "kokoro": Kokoro,
        "phonon": Phonon,
        "pocket-default": lambda: Pocket(threads=args.threads),
        "pocket-18m": lambda: Pocket("english_18m_2026-10", threads=args.threads),
        "pocket-7m": lambda: Pocket("english_7m_2026-10", threads=args.threads),
        "pocket-2m": lambda: Pocket("english_2m_2026-10", threads=args.threads),
        "pocket-donation": lambda: Pocket("english_7m_2026-10", voice=args.voice_file, threads=args.threads),
        "lfm3": Lfm,
    }
    for name in args.engines or list(makers):
        try:
            run_engine(name, makers[name])
        except Exception as e:  # keep going so one broken engine does not hide the rest
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            save_result(f"tts-{name}", {"engine": name, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
