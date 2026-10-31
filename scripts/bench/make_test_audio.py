"""Builds the reproducible test set under .runtime/bench/.

Sources:
- Synthesized hard sentences (Kokoro ONNX), in one female and one male voice.
- LibriSpeech dummy subset (CC BY 4.0, hf-internal-testing/librispeech_asr_dummy), human speech with references.
- jfk.wav from whisper.cpp samples (public domain US government recording).

Every clip gets a 16 kHz PCM16 clean copy and a telephone copy: 8 kHz, mu-law encoded, decoded back to 16 kHz.
Writes .runtime/bench/manifest.json.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys

import numpy as np
import soundfile as sf

from common import BENCH, FFMPEG, MODELS

HARD = [
    {"id": "hard-invoice", "text": "Your invoice number is F H dash 4 4 1 9 for forty nine dollars and ninety nine cents.",
     "critical": [r"f ?h (dash )?4 4 1 9", r"49 dollars and 99 cents|49 99"]},
    {"id": "hard-mx", "text": "Update your MX records and check IMAP settings at mail dot fernhill dot example.",
     "critical": [r"\bmx\b", r"\bimap\b", r"mail dot fernhill dot example"]},
    {"id": "hard-refund", "text": "I have issued a refund of one hundred twenty dollars to the card ending in 7 3 0 2.",
     "critical": [r"1 2 0 dollars", r"7 3 0 2"]},
    {"id": "hard-ssl", "text": "Your SSL certificate for example dash shop dot net expires on March 14th, so renew it through the DNS panel.",
     "critical": [r"\bssl\b", r"example (dash )?shop dot net", r"march 14", r"\bdns\b"]},
    {"id": "hard-account", "text": "My account number is A C 9 0 7 7 and my email is dana dot reyes at fernhill mail dot example.",
     "critical": [r"a ?c 9 0 7 7", r"dana dot reyes", r"fernhill ?mail dot example"]},
    {"id": "hard-angry", "text": "I was charged twice last month and nobody has called me back. I want this fixed today.",
     "critical": [r"charged twice", r"fixed today"]},
]
VOICES = {"f": "af_heart", "m": "am_michael"}
KOKORO_ONLY_SPLIT = 8
# Caller turns for realtime and end-to-end tests. They exercise greeting, tool use, and a follow-up.
CALLER = [
    ("caller-greeting", "Hi, I have a question about my bill."),
    ("caller-lookup", "My name is Dana Reyes and my account number is A C 9 0 7 7. I was charged twice last month."),
    ("caller-refund", "Yes, please refund the duplicate charge on invoice F H dash 4 4 1 9 for forty nine dollars and ninety nine cents."),
    ("caller-question", "How long will the refund take to reach my card?"),
    ("caller-long", "I have been trying to get my email working for three days. The MX records look right to me but messages keep bouncing and I need this sorted out before my client meeting on Monday."),
]


def ffmpeg(*args: str) -> None:
    subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def telephone(src, dst) -> None:
    """16 kHz wav to 8 kHz mu-law and back to 16 kHz PCM. No band filter beyond the resampler."""
    mu = dst.with_suffix(".mulaw.wav")
    ffmpeg("-i", str(src), "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(mu))
    ffmpeg("-i", str(mu), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst))


def main() -> int:
    clean = BENCH / "clean"
    tel = BENCH / "tel"
    clean.mkdir(parents=True, exist_ok=True)
    tel.mkdir(parents=True, exist_ok=True)
    manifest = []

    from kokoro_onnx import Kokoro
    k = Kokoro(str(MODELS / "kokoro" / "kokoro-v1.0.onnx"), str(MODELS / "kokoro" / "voices-v1.0.bin"))
    for i, item in enumerate(HARD):
        for tag, voice in VOICES.items():
            cid = f"{item['id']}-{tag}"
            pcm, sr = k.create(item["text"], voice=voice, speed=1.0, lang="en-us")
            raw = clean / f"{cid}.24k.wav"
            sf.write(raw, pcm, sr, subtype="PCM_16")
            out = clean / f"{cid}.wav"
            ffmpeg("-i", str(raw), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out))
            raw.unlink()
            manifest.append({"id": cid, "kind": "synth-hard", "text": item["text"], "critical": item["critical"],
                             "clean": str(out), "duration": sf.info(out).duration})

    caller_dir = BENCH / "caller"
    caller_dir.mkdir(exist_ok=True)
    caller = []
    for cid, text in CALLER:
        pcm, sr = k.create(text, voice="af_sarah", speed=1.0, lang="en-us")
        raw = caller_dir / f"{cid}.24k.wav"
        sf.write(raw, pcm, sr, subtype="PCM_16")
        out = caller_dir / f"{cid}.wav"
        ffmpeg("-i", str(raw), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out))
        raw.unlink()
        tel_out = caller_dir / f"{cid}.tel.wav"
        telephone(out, tel_out)
        tel_out.with_suffix(".mulaw.wav").unlink()
        caller.append({"id": cid, "text": text, "clean": str(out), "tel": str(tel_out), "duration": sf.info(out).duration})
    (BENCH / "caller.json").write_text(json.dumps(caller, indent=2))

    # human speech
    import pyarrow.parquet as pq
    table = pq.read_table(BENCH / "src" / "libri" / "clean" / "validation-00000-of-00001.parquet").to_pylist()
    # one utterance per chapter first, so more than one speaker lands in the sample
    table = [r for r in table if 6 < len(r["text"].split()) < 40]
    seen: dict = {}
    for r in table:
        seen.setdefault(r["speaker_id"], []).append(r)
    print({k_: len(v) for k_, v in seen.items()})
    picked = []
    for group in seen.values():
        picked.extend(group[:KOKORO_ONLY_SPLIT // max(1, len(seen))])
    table = picked[:KOKORO_ONLY_SPLIT]
    for r in table:
        pcm, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
        cid = f"libri-{r['id']}"
        out = clean / f"{cid}.wav"
        sf.write(out, pcm, sr, subtype="PCM_16")
        manifest.append({"id": cid, "kind": "human", "text": r["text"], "critical": [], "clean": str(out),
                         "duration": len(pcm) / sr})

    jfk = BENCH / "src" / "jfk.wav"
    out = clean / "jfk.wav"
    ffmpeg("-i", str(jfk), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out))
    manifest.append({"id": "jfk", "kind": "human",
                     "text": "And so my fellow Americans ask not what your country can do for you ask what you can do for your country.",
                     "critical": [], "clean": str(out), "duration": sf.info(out).duration})

    # a 30 second clip from human utterances, for steady-state real-time factor
    parts = [sf.read(m["clean"], dtype="int16")[0] for m in manifest if m["kind"] == "human"]
    joined = np.concatenate(parts)[: 16000 * 30]
    out = clean / "long30.wav"
    sf.write(out, joined, 16000, subtype="PCM_16")
    manifest.append({"id": "long30", "kind": "long", "text": "", "critical": [], "clean": str(out), "duration": 30.0})

    for m in manifest:
        dst = tel / f"{m['id']}.wav"
        telephone(m["clean"], dst)
        m["tel"] = str(dst)
    (tel / "x").unlink(missing_ok=True)
    for f in tel.glob("*.mulaw.wav"):
        f.unlink()

    (BENCH / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} clips, {sum(m['duration'] for m in manifest):.1f} s of audio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
