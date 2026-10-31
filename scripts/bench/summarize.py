"""Prints markdown tables from .runtime/bench/results/*.json, for pasting into docs/research/benchmarks-2026-10.md."""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

from common import RESULTS


def load(pattern: str) -> dict:
    return {Path(p).stem: json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(RESULTS / f"{pattern}.json")))}


def table(head: list[str], rows: list[list]) -> None:
    print("| " + " | ".join(head) + " |")
    print("| " + " | ".join("---" for _ in head) + " |")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def f(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"


def stt() -> None:
    rows = []
    for name, d in load("stt-*").items():
        if "error" in d:
            rows.append([d["engine"], "failed", d["error"][:80], "", "", "", "", "", ""])
            continue
        c, t = d["summary"]["clean"], d["summary"]["tel"]
        rows.append([d["engine"], f(c["rtf_median"], 1), f(c["rtf_30s"], 1), f(c["wer_human"], 3), f(t["wer_human"], 3), f(c["wer_hard"], 3), f(t["wer_hard"], 3), c["critical"], t["critical"], f(d.get("gpu_gb"), 1)])
    for name, d in load("lfm-asr-tts-*").items():
        c = d["asr"]
        def s(cond):
            r = [x for x in c if x["cond"] == cond and "wer" in x]
            cr = [x for x in r if "crit_total" in x]
            return (st.mean(x["wer"] for x in r if x["kind"] == "human"), st.mean(x["wer"] for x in r if x["kind"] == "synth-hard"), f"{sum(x['crit_hits'] for x in cr)}/{sum(x['crit_total'] for x in cr)}")
        cl, te = s("clean"), s("tel")
        long_ = [x for x in c if x["id"] == "long30" and x["cond"] == "clean"]
        rows.append([f"lfm3-audio {d['config']}", f(st.median(x["rtf"] for x in c if x["cond"] == "clean"), 1), f(long_[0]["rtf"] if long_ else None, 1), f(cl[0], 3), f(te[0], 3), f(cl[1], 3), f(te[1], 3), cl[2], te[2], f(d.get("gpu_gb"), 1)])
    table(["Engine", "Median RTF", "30 s clip RTF", "WER human clean", "WER human tel", "WER hard clean", "WER hard tel", "Critical clean", "Critical tel", "GPU GB"], rows)


def tts() -> None:
    rows = []
    for name, d in load("tts-*").items():
        if name == "tts-score":
            continue
        if "error" in d:
            rows.append([d["engine"], "failed", d["error"][:90], "", "", ""])
            continue
        rows.append([d["engine"], f(d["rtf_median"]), f(d["ttfa_median_s"] * 1000, 0), f(d["ttfa_max_s"] * 1000, 0), d["streams"], f(d["load_s"], 1)])
    for name, d in load("lfm-asr-tts-*").items():
        t = d["tts"]
        rows.append([f"lfm3-audio {d['config']}", f(st.median(x["rtf"] for x in t)), f(st.median(x["ttfa_s"] for x in t) * 1000, 0), f(max(x["ttfa_s"] for x in t) * 1000, 0), True, f(d["load_s"], 1)])
    table(["Engine", "Median RTF", "TTFA median ms", "TTFA max ms", "Streams", "Load s"], rows)


def llm() -> None:
    rows = []
    for name, d in load("llm-*").items():
        if "error" in d:
            rows.append([d["model"], "failed", d["error"][:80], "", "", "", "", "", ""])
            continue
        rows.append([d["model"], f(d["cold_ttft_ms"], 0), f(d["warm_ttft_ms"], 0), f(d["tps"], 1), f(d["prompt_tps"], 0), f"{d['tool_ok']}/{d['tool_total']}", d["malformed"], d["leaks"], f(d["gpu_gb"], 1)])
    table(["Model", "Cold TTFT ms", "Warm TTFT ms", "t/s", "Prompt t/s", "Scenarios passed", "Malformed", "Leaks", "GPU GB"], rows)


def e2e() -> None:
    rows = []
    for name, d in load("e2e-*").items():
        t = [x for x in d["turns"] if x.get("e2e_ms") is not None and (x["rep"] > 0 or len(d["turns"]) <= 4)]
        no = [x["e2e_ms"] for x in t if x["n_tool_rounds"] == 0]
        yes = [x["e2e_ms"] for x in t if x["n_tool_rounds"] > 0]
        rows.append([d["stack"], d["cond"], f(st.median(no), 0) if no else "n/a", f(max(no), 0) if no else "n/a", f(st.median(yes), 0) if yes else "n/a", f(max(yes), 0) if yes else "n/a",
                     f(st.mean(x["vad_ms"] for x in t), 0), f(d.get("gpu_gb"), 1)])
    table(["Stack", "Cond", "No-tool median ms", "No-tool max", "Tool median ms", "Tool max", "VAD mean ms", "GPU GB"], rows)


def bots() -> None:
    rows = []
    for name, d in load("twobots-*").items():
        rows.append([d["label"], len(d["turns"]), f(d["duration_s"], 0), f(d["e2e_median_ms"], 0), f(d["e2e_max_ms"], 0), d["clock_late_windows"], d["underruns"], d["premature_vad_ends"], f(d["gpu_busy_mean"], 0), d["gpu_busy_max"], f(d["cpu_mean"], 0), f(d["gpu_gb"], 1)])
    table(["Call", "Turns", "Seconds", "E2E median ms", "E2E max ms", "Late windows", "Underruns", "Premature VAD ends", "GPU busy mean %", "GPU busy max %", "CPU mean %", "GPU GB"], rows)


if __name__ == "__main__":
    for title, fn in (("STT", stt), ("TTS", tts), ("LLM", llm), ("End to end", e2e), ("Two bots", bots)):
        print(f"### {title}\n")
        fn()
