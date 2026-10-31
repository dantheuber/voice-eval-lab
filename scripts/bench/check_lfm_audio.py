"""Transcribes saved LFM3 speech with Parakeet (an independent ASR) to confirm what the audio actually says.

Scores /tts samples against their input text, and prints realtime responses by mode so forced-audio output can be compared
with text+audio output. Writes .runtime/bench/results/lfm-audio-check.json.
"""
from __future__ import annotations

import json
import statistics as st
import subprocess
import sys

import httpx

from bench_lfm_asr_tts import TTS_TEXT
from bench_tts import SENTENCES
from common import BENCH, FFMPEG, critical_hits, save_result, wer
from rt_client import read_pcm16
from speech_backends import PARAKEET, HttpSTT, NemoServer

CRIT = {text: crit for _, text, crit in SENTENCES}
TMP = BENCH / "tmp16k.wav"


def hear(stt: HttpSTT, path) -> str:
    subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(TMP)], check=True)
    return stt.transcribe(read_pcm16(str(TMP)))


def main() -> int:
    res: dict = {"tts": [], "realtime": []}
    with NemoServer(PARAKEET, "vulkan:0") as nemo:
        stt = HttpSTT(nemo.url, httpx.get(nemo.url + "/v1/models").json()["data"][0]["id"])
        for cfg in ("vulkan-q8", "vulkan-q4", "cpu-q8", "cpu-q4"):
            for voice in ("female", "male"):
                for i, text in enumerate(TTS_TEXT):
                    for rep in range(3):
                        path = BENCH / "out" / "lfm-tts" / f"{cfg}-{voice}-{i}-{rep}.wav"
                        if not path.exists():
                            continue
                        heard = hear(stt, path)
                        rec = {"cfg": cfg, "voice": voice, "i": i, "rep": rep, "heard": heard, "wer": wer(text, heard)}
                        if text in CRIT:
                            rec.update(zip(("hits", "total", "missed"), critical_hits(heard, CRIT[text])))
                        res["tts"].append(rec)
        for cfg in sorted({(r["cfg"], r["voice"]) for r in res["tts"]}):
            rows = [r for r in res["tts"] if (r["cfg"], r["voice"]) == cfg]
            crit = [r for r in rows if "total" in r]
            print(f"{cfg[0]:10s} {cfg[1]:6s} WER {st.mean(r['wer'] for r in rows):.3f} critical {sum(r['hits'] for r in crit)}/{sum(r['total'] for r in crit)}", flush=True)
        for r in res["tts"]:
            if r["cfg"] == "vulkan-q8" and r["rep"] == 0:
                print(f"   {r['voice']:6s} {r['i']}: {r['heard'][:110]!r} wer {r['wer']:.2f}", flush=True)
        for path in sorted((BENCH / "out" / "realtime").glob("vulkan-q8-*.wav")):
            if "tools" in path.name:
                continue
            heard = hear(stt, path)
            res["realtime"].append({"file": path.name, "heard": heard})
            print(f"{path.name:44s} {heard[:150]!r}", flush=True)
    save_result("lfm-audio-check", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
