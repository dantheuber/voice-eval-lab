"""Tool-call accuracy from speech: cascade (Parakeet then a text LLM) against LFM3 realtime, on the same telephone clips.

Usage: bench_toolacc.py cascade LLM [LLM...] [--reps 12] [--cond tel]
       bench_toolacc.py score     (prints a table from saved results, including rt-toolacc-vulkan-q8.json)
Scoring per session: lookup used account AC9077, refund used a real invoice with amount 49.99, arguments parsed as JSON and held no template syntax.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from bench_e2e import LLMS, llm_args
from common import BENCH, RESULTS, LlamaServer, save_result
from fernhill_fixture import SUPPORT_SYSTEM, TOOLS, tool_message
from llm_client import chat
from rt_client import read_pcm16
from speech_backends import open_stt

GARBAGE = re.compile(r"</?(tool_call|function|parameter)|\[customer_id\]|\[.*_id\]")
CALLER = {c["id"]: c for c in json.loads((BENCH / "caller.json").read_text())}


def score_run(calls: list[dict], malformed: int) -> dict:
    lookups = [c for c in calls if c["name"] == "lookup_customer"]
    refunds = [c for c in calls if c["name"] == "issue_refund"]
    def clean(c):
        return str(c.get("args", {}).get("account_number", "")).upper().replace(" ", "").replace("-", "")
    garbage = sum(1 for c in calls if GARBAGE.search(json.dumps(c.get("args", c.get("raw", "")))))
    lookup_ok = any(clean(c) == "AC9077" for c in lookups)
    refund_ok = any(c.get("args", {}).get("invoice_id") in ("FH-4419", "FH-4420") and abs(float(c.get("args", {}).get("amount", 0) or 0) - 49.99) < 0.005 for c in refunds)
    invoices = [c for c in calls if c["name"] == "get_invoices"]
    invoices_ok = any(c.get("args", {}).get("customer_id") == "cus_1042" for c in invoices)
    return {"lookup_ok": lookup_ok, "invoices_ok": invoices_ok, "n_invoice_calls": len(invoices), "refund_ok": refund_ok, "malformed": malformed, "garbage": garbage, "n_calls": len(calls), "n_refunds": len(refunds)}


def cascade(args) -> None:
    stt, servers = open_stt(args.stt)
    heard = {c: stt.transcribe(read_pcm16(CALLER[c][args.cond])) for c in ("caller-lookup", "caller-refund")}
    for srv in servers:
        srv.__exit__(None, None, None)
    print(heard, flush=True)
    for name in args.models:
        if not LLMS[name][0].exists():
            print(f"{name}: not downloaded", flush=True)
            continue
        runs = []
        with LlamaServer(llm_args(name), name=f"toolacc-{name}") as s:
            for rep in range(args.reps):
                msgs = [{"role": "system", "content": SUPPORT_SYSTEM}]
                calls: list[dict] = []
                malformed = 0
                for clip in ("caller-lookup", "caller-refund"):
                    msgs.append({"role": "user", "content": heard[clip]})
                    for _ in range(5):
                        r = chat(s.url, msgs, tools=TOOLS, temperature=0.3, max_tokens=256, extra={"seed": rep})
                        malformed += len(r.malformed)
                        if not r.tool_calls:
                            msgs.append({"role": "assistant", "content": r.content})
                            break
                        msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": [
                            {"id": c["id"] or f"c{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for i, c in enumerate(r.tool_calls)]})
                        for i, c in enumerate(r.tool_calls):
                            calls.append({"name": c["name"], "args": c["args"]})
                            msgs.append(tool_message(c["id"] or f"c{i}", c["name"], c["args"]))
                runs.append({"rep": rep, "calls": calls, "malformed": malformed, **score_run(calls, malformed)})
        out = {"llm": name, "stt": args.stt, "cond": args.cond, "heard": heard, "runs": runs}
        save_result(f"toolacc-cascade-{args.stt}-{name}-{args.cond}", out)
        print(f"{name}: lookup {sum(r['lookup_ok'] for r in runs)}/{len(runs)} invoices {sum(r['invoices_ok'] for r in runs)}/{sum(r['n_invoice_calls'] > 0 for r in runs)} refund {sum(r['refund_ok'] for r in runs)}/{len(runs)} malformed {sum(r['malformed'] for r in runs)} garbage {sum(r['garbage'] for r in runs)}", flush=True)
        print("   sample:", [(c["name"], c["args"]) for c in runs[0]["calls"]], flush=True)


def score() -> None:
    p = RESULTS / "rt-toolacc-vulkan-q8.json"
    if p.exists():
        runs = json.loads(p.read_text())["runs"]
        s = [score_run(r["calls"], r["malformed"]) for r in runs]
        print(f"lfm3-realtime: lookup {sum(x['lookup_ok'] for x in s)}/{len(s)} invoices {sum(x['invoices_ok'] for x in s)}/{sum(x['n_invoice_calls'] > 0 for x in s)} refund {sum(x['refund_ok'] for x in s)}/{len(s)} malformed {sum(x['malformed'] for x in s)} garbage {sum(x['garbage'] for x in s)} calls {sum(x['n_calls'] for x in s)}")
        save_result("toolacc-realtime-summary", s)
    for f in sorted(RESULTS.glob("toolacc-cascade-*.json")):
        d = json.loads(f.read_text())
        r = d["runs"]
        print(f"{d.get('stt', 'parakeet')}+{d['llm']} ({d['cond']}): lookup {sum(x['lookup_ok'] for x in r)}/{len(r)} invoices {sum(x['invoices_ok'] for x in r)}/{sum(x['n_invoice_calls'] > 0 for x in r)} refund {sum(x['refund_ok'] for x in r)}/{len(r)} malformed {sum(x['malformed'] for x in r)} garbage {sum(x['garbage'] for x in r)} calls {sum(x['n_calls'] for x in r)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["cascade", "score"])
    p.add_argument("models", nargs="*", default=["lfm3-1.7b", "qwen3.5-4b"])
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--cond", default="tel")
    p.add_argument("--stt", default="parakeet")
    args = p.parse_args()
    cascade(args) if args.mode == "cascade" else score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
