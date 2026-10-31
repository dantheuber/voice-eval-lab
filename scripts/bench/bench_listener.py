"""Can an audio model hear delivery problems? Planted-defect trial with controls, scored against chance.

Trial A, defect class: 4 base clips x 6 conditions (normal, rushed, dragging, chopped, noisy, robotic). Six-way choice, chance 1/6.
Trial B, anything wrong: the same clips, yes or no. Reports hit rate on defects and false alarm rate on normals.
Trial C, intended text: 4 clips with a planted wrong word and 4 clean controls. Model gets the intended text and must say whether
the audio matches. Reports hit and false alarm rates, and whether it named the changed word.

Usage: bench_listener.py make                    build clips under .runtime/bench/listener/
       bench_listener.py run NAME MODEL MMPROJ   run one model through llama-server on Vulkan
       bench_listener.py signal                  plain signal checks on the same clips, as a baseline
       bench_listener.py report                  table of all results
"""
from __future__ import annotations

import base64
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from common import BENCH, FFMPEG, MODELS, LlamaServer, gpu_mem_gb, save_result, RESULTS

DIR = BENCH / "listener"
CHOICES = {"A": "normal", "B": "rushed", "C": "dragging", "D": "chopped", "E": "noisy", "F": "robotic"}
SYSTEM = "Answer questions about the provided audio clip. Base every answer only on what can be heard."
Q_CLASS = ("Listen to how this phone audio sounds, not to what the words mean. Pick the one best description:\n"
           "A. normal, clear speech at a normal pace\nB. rushed, speech is much too fast\nC. dragging, speech is much too slow\n"
           "D. chopped, the audio keeps cutting out\nE. noisy, loud background hiss\nF. robotic, metallic or synthetic distortion\n"
           "Answer with the letter only.")
Q_ANY = "Listen to how this phone audio sounds. Is there a problem with the audio quality or the speaker's pace? Answer yes or no only."
BASES = ["hard-invoice-f", "hard-mx-m", "hard-refund-f", "hard-angry-m"]
FX = {
    "normal": None,
    "rushed": ["-af", "atempo=1.7"],
    "dragging": ["-af", "atempo=0.65"],
    "chopped": ["-af", "aeval='val(0)*mod(floor(t*5),2)':c=same"],
    "robotic": ["-af", "afftfilt=real='hypot(re,im)*cos(0)':imag='0':win_size=512:overlap=0.75"],
}
WRONG = [
    ("invoice", "Your invoice number is F H dash 4 4 1 9 for forty nine dollars and ninety nine cents.", "Your invoice number is F H dash 4 4 1 9 for forty five dollars and ninety nine cents.", "forty five"),
    ("fernhill", "Thanks for calling Fernhill Internet. How can I help you today?", "Thanks for calling Fernhole Internet. How can I help you today?", "fernhole"),
    ("imap", "Please check your IMAP settings and your MX records.", "Please check your eye nap settings and your mix records.", "nap"),
    ("refund", "I have issued a refund to the card ending in seven three zero two.", "I have issued a robin to the card ending in seven three zero two.", "robin"),
]


def ff(*args: str) -> None:
    subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    trials = []
    for base in BASES:
        src = BENCH / "clean" / f"{base}.wav"
        for cond, af in FX.items():
            out = DIR / f"{base}-{cond}.wav"
            ff("-i", str(src), *(af or []), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out))
            trials.append({"clip": out.name, "base": base, "cond": cond})
        noisy = DIR / f"{base}-noisy.wav"
        dur = sf.info(src).duration
        ff("-f", "lavfi", "-i", f"anoisesrc=d={dur}:c=white:a=0.08:r=16000", "-i", str(src), "-filter_complex",
           "[1:a][0:a]amix=inputs=2:duration=first:normalize=0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(noisy))
        trials.append({"clip": noisy.name, "base": base, "cond": "noisy"})
    from kokoro_onnx import Kokoro
    k = Kokoro(str(MODELS / "kokoro" / "kokoro-v1.0.onnx"), str(MODELS / "kokoro" / "voices-v1.0.bin"))
    text_trials = []
    for wid, good, bad, word in WRONG:
        for label, text in (("good", good), ("bad", bad)):
            pcm, sr = k.create(text, voice="af_heart", speed=1.0, lang="en-us")
            raw = DIR / f"text-{wid}-{label}.24k.wav"
            sf.write(raw, pcm, sr, subtype="PCM_16")
            out = DIR / f"text-{wid}-{label}.wav"
            ff("-i", str(raw), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out))
            raw.unlink()
            text_trials.append({"clip": out.name, "intended": good, "planted": label == "bad", "word": word})
    (DIR / "trials.json").write_text(json.dumps({"class": trials, "text": text_trials}, indent=2))
    print(len(trials), "class clips,", len(text_trials), "text clips")


def ask(client: httpx.Client, url: str, path: Path, system: str | None, q: str, extra: dict) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}, {"type": "text", "text": q}]}]
    r = client.post(url + "/v1/chat/completions", json={"temperature": 0, "max_tokens": 80, "messages": msgs, **extra})
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def run(name: str, model: str, mmproj: str, flags: list[str]) -> None:
    t = json.loads((DIR / "trials.json").read_text())
    client = httpx.Client(timeout=300)
    gpu0 = gpu_mem_gb()
    with LlamaServer(["-m", model, "-mm", mmproj, "--temp", "0", "-c", "8192", "-np", "1", "-ngl", "999", "--reasoning", "off", *flags], name=f"listen-{name}", timeout=300) as s:
        res: dict = {"model": name, "gpu_gb": gpu_mem_gb() - gpu0, "class": [], "any": [], "text": []}
        for tr in t["class"]:
            a = ask(client, s.url, DIR / tr["clip"], SYSTEM, Q_CLASS, {})
            res["class"].append({**tr, "answer": a[:80], "pick": parse_pick(a)})
            a = ask(client, s.url, DIR / tr["clip"], SYSTEM, Q_ANY, {})
            res["any"].append({**tr, "answer": a[:60], "yes": bool(re.match(r"\W*yes", a, re.I))})
        for tr in t["text"]:
            q = (f'The speaker was supposed to say exactly: "{tr["intended"]}"\nListen carefully. Did the speaker say exactly that? '
                 "Answer yes or no first. If no, name the word that is different.")
            a = ask(client, s.url, DIR / tr["clip"], SYSTEM, q, {})
            res["text"].append({**tr, "answer": a[:140], "said_no": bool(re.match(r"\W*no", a, re.I)), "named": tr["word"].split()[-1] in a.lower()})
    save_result(f"listener-{name}", res)
    print(summary(res), flush=True)


def parse_pick(answer: str) -> str | None:
    """Accepts a letter, or the class word when a model answers in words."""
    m = re.match(r"\W*([A-F])\b", answer)
    if m:
        return CHOICES[m.group(1)]
    low = answer.lower()
    hits = [(low.find(w), w) for w in CHOICES.values() if w in low]
    return min(hits)[1] if hits else None


def binom_p(k: int, n: int, p: float) -> float:
    """One-sided P(X >= k) under chance."""
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def summary(res: dict) -> str:
    c = res["class"]
    for r in c:
        r["pick"] = parse_pick(r["answer"])
    ok = sum(r["pick"] == r["cond"] for r in c)
    defects = [r for r in c if r["cond"] != "normal"]
    def_ok = sum(r["pick"] == r["cond"] for r in defects)
    a = res["any"]
    hit = sum(r["yes"] for r in a if r["cond"] != "normal")
    fa = sum(r["yes"] for r in a if r["cond"] == "normal")
    na_def = sum(r["cond"] != "normal" for r in a)
    na_norm = len(a) - na_def
    t = res.get("text", [])
    t_hit = sum(r["said_no"] for r in t if r["planted"])
    t_fa = sum(r["said_no"] for r in t if not r["planted"])
    named = sum(r["named"] for r in t if r["planted"])
    t_n = sum(r["planted"] for r in t)
    return (f"{res['model']:14s} class {ok}/{len(c)} (defects {def_ok}/{len(defects)}, p={binom_p(ok, len(c), 1 / 6):.3f}) "
            f"any-problem hit {hit}/{na_def} false alarm {fa}/{na_norm} "
            f"wrong-word hit {t_hit}/{t_n} false alarm {t_fa}/{len(t) - t_n} named {named}/{t_n}")


def pitch_iqr(x: np.ndarray, sr: int) -> float:
    """Spread of pitch over voiced frames, in Hz. A monotone robotic voice has almost none."""
    n = int(0.04 * sr)
    out = []
    for i in range(0, len(x) - n, n):
        f = x[i:i + n]
        if np.abs(f).mean() < 0.02:
            continue
        f = f - f.mean()
        c = np.correlate(f, f, "full")[n - 1:]
        lo, hi = sr // 400, sr // 70
        lag = lo + int(np.argmax(c[lo:hi]))
        if c[lag] > 0.3 * c[0]:
            out.append(sr / lag)
    return float(np.subtract(*np.percentile(out, [75, 25]))) if len(out) > 3 else 0.0


def signal() -> None:
    """Plain signal checks. Thresholds come from the normal clips of the same base sentence, as a call baseline would."""
    t = json.loads((DIR / "trials.json").read_text())
    rows = {}
    for tr in t["class"]:
        x, sr = sf.read(DIR / tr["clip"], dtype="float32")
        hop = int(0.02 * sr)
        frames = x[: len(x) // hop * hop].reshape(-1, hop)
        rms = np.sqrt((frames**2).mean(axis=1)) + 1e-6
        db = 20 * np.log10(rms)
        floor = float(np.percentile(db, 10))
        loud = db > float(np.percentile(db, 90)) - 25
        idx = np.nonzero(loud)[0]
        inside = db[idx[0]:idx[-1]] if len(idx) > 1 else db
        gaps = np.diff(np.concatenate([[0], (inside < -60).astype(int), [0]]))
        dropouts = int((gaps == 1).sum())
        sxx = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) ** 2 + 1e-12
        flat = float(np.median(np.exp(np.log(sxx).mean(axis=1)) / sxx.mean(axis=1)))
        rows[tr["clip"]] = {**tr, "dur": len(x) / sr, "floor_db": floor, "dropouts": dropouts, "flatness": flat, "pitch_iqr": pitch_iqr(x, sr)}
    picks = []
    for tr in t["class"]:
        r = rows[tr["clip"]]
        n = rows[f"{tr['base']}-normal.wav"]
        ratio = r["dur"] / n["dur"]
        if r["dropouts"] >= n["dropouts"] + 5:
            pick = "chopped"
        elif r["floor_db"] > n["floor_db"] + 15:
            pick = "noisy"
        elif ratio < 0.75:
            pick = "rushed"
        elif ratio > 1.3:
            pick = "dragging"
        elif r["pitch_iqr"] < 0.5 * n["pitch_iqr"]:
            pick = "robotic"
        else:
            pick = "normal"
        picks.append({**r, "pick": pick})
    ok = sum(p["pick"] == p["cond"] for p in picks)
    print(f"signal checks: {ok}/{len(picks)} ({[(p['clip'], p['pick']) for p in picks if p['pick'] != p['cond']][:6]})", flush=True)
    save_result("listener-signal-checks", {"picks": picks, "ok": ok, "n": len(picks)})


def asrdiff() -> None:
    """Baseline for trial C: transcribe with an independent ASR and diff against the intended text."""
    from common import normalize
    from rt_client import read_pcm16
    from speech_backends import open_stt
    t = json.loads((DIR / "trials.json").read_text())
    res = {}
    for name in ("whisper", "parakeet"):
        stt, servers = open_stt(name)
        rows = []
        for tr in t["text"]:
            heard = stt.transcribe(read_pcm16(str(DIR / tr["clip"])))
            a, b = normalize(tr["intended"]).split(), normalize(heard).split()
            rows.append({**tr, "heard": heard, "flag": a != b, "named": tr["word"].split()[-1] in normalize(heard)})
        for srv in servers:
            srv.__exit__(None, None, None)
        hit = sum(r["flag"] for r in rows if r["planted"])
        fa = sum(r["flag"] for r in rows if not r["planted"])
        print(f"asr diff {name}: hit {hit}/4 false alarm {fa}/4 named {sum(r['named'] for r in rows if r['planted'])}/4", [r["heard"][:60] for r in rows if r["flag"] and not r["planted"]], flush=True)
        res[name] = rows
    save_result("listener-asrdiff", res)


def report() -> None:
    for f in sorted(RESULTS.glob("listener-*.json")):
        d = json.loads(f.read_text())
        if "class" in d and "model" in d:
            print(summary(d))
            by = {}
            for r in d["class"]:
                by.setdefault(r["cond"], []).append(r["pick"])
            print("   picks by true class:", {k: v for k, v in by.items()})


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "make":
        make()
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif cmd == "signal":
        signal()
    elif cmd == "asrdiff":
        asrdiff()
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
