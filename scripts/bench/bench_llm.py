"""Support-bot and caller LLM candidates on Vulkan through llama-server's OpenAI-compatible API.

Measures cold and warm time to first token, tokens per second, and tool calling over scripted scenarios.
A tool call counts as well-formed if the server parsed it into tool_calls with valid JSON arguments, and raw call
syntax did not leak into content.
Usage: bench_llm.py [model ...] [--reps 3]
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import sys

from bench_e2e import LLMS, llm_args
from common import LlamaServer, gpu_mem_gb, save_result
from fernhill_fixture import SUPPORT_SYSTEM, TOOLS, tool_message
from llm_client import chat

LEAK = re.compile(r"<tool_call|</tool_call|<\|tool|\[TOOL_CALLS|<function|\"name\"\s*:\s*\"(lookup_customer|get_invoices|issue_refund)")
SCENARIOS = [
    {"id": "lookup", "user": "Hi, this is Dana Reyes, account number AC9077. I was charged twice last month.",
     "expect": [("lookup_customer", {"account_number": "AC9077"}), ("get_invoices", {"customer_id": "cus_1042"})], "forbid": ["issue_refund"]},
    {"id": "refund", "history": [("user", "Hi, this is Dana Reyes, account number AC9077. I was charged twice last month."),
                                 ("assistant", "I see two charges for forty nine dollars and ninety nine cents, invoices FH-4419 and FH-4420. FH-4420 is the duplicate. Would you like me to refund it?")],
     "user": "Yes, please refund invoice FH-4420 for 49.99.",
     "expect": [("issue_refund", {"invoice_id": "FH-4420", "amount": 49.99})], "forbid": []},
    {"id": "no-tool", "user": "How long do refunds usually take to reach my card?", "expect": [], "forbid": ["issue_refund", "get_invoices"],
     "must_say": r"five|5"},
    {"id": "policy-limit", "history": [("user", "Hi, this is Dana Reyes, account number AC9077."), ("assistant", "Thanks Dana, I found your account. How can I help?")],
     "user": "Refund invoice FH-4419 for three hundred and fifty dollars right now.", "expect": [], "forbid": ["issue_refund"]},
]


def run_scenario(url: str, sc: dict) -> dict:
    msgs = [{"role": "system", "content": SUPPORT_SYSTEM}]
    for role, text in sc.get("history", []):
        msgs.append({"role": role, "content": text})
    msgs.append({"role": "user", "content": sc["user"]})
    called: list[dict] = []
    malformed = 0
    leaks = 0
    first = None
    final = ""
    for _ in range(5):
        r = chat(url, msgs, tools=TOOLS, temperature=0.1, max_tokens=256)
        first = first or r
        malformed += len(r.malformed)
        leaks += 1 if LEAK.search(r.content) else 0
        final = r.content
        if not r.tool_calls:
            break
        msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": [
            {"id": c["id"] or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for i, c in enumerate(r.tool_calls)]})
        for i, c in enumerate(r.tool_calls):
            called.append({"name": c["name"], "args": c["args"]})
            msgs.append(tool_message(c["id"] or f"call_{i}", c["name"], c["args"]))
    ok = True
    for name, want in sc["expect"]:
        match = [c for c in called if c["name"] == name]
        if not match or not all(_close(match[0]["args"].get(k), v) for k, v in want.items()):
            ok = False
    if any(c["name"] in sc["forbid"] for c in called):
        ok = False
    if "must_say" in sc and not re.search(sc["must_say"], final, re.I):
        ok = False
    return {"id": sc["id"], "ok": ok, "called": called, "malformed": malformed, "leaks": leaks, "final": final,
            "ttft_ms": first.ttft_ms, "tps": first.tps}


def _close(got, want) -> bool:
    if isinstance(want, float):
        try:
            return abs(float(got) - want) < 0.005
        except (TypeError, ValueError):
            return False
    return str(got).upper().replace(" ", "") == str(want).upper().replace(" ", "")


def measure(name: str, reps: int) -> dict:
    gpu0 = gpu_mem_gb()
    with LlamaServer(llm_args(name), name=f"llm-{name}") as s:
        res: dict = {"model": name, "load_s": s.load_s, "gpu_gb": gpu_mem_gb() - gpu0, "rss_gb": s.rss()}
        msgs = [{"role": "system", "content": SUPPORT_SYSTEM}, {"role": "user", "content": "Hi, I have a question about my bill."}]
        cold = chat(s.url, msgs, tools=TOOLS, max_tokens=64, extra={"cache_prompt": False})
        res["cold_ttft_ms"] = cold.ttft_ms
        res["cold_prompt_tokens"] = cold.timings.get("prompt_n")
        res["prompt_tps"] = cold.prompt_tps
        warm = []
        for i in range(5):
            m = msgs + [{"role": "assistant", "content": "Of course, what would you like to know?"}, {"role": "user", "content": f"My last invoice looks wrong, question {i}."}]
            warm.append(chat(s.url, m, tools=TOOLS, max_tokens=64))
        res["warm_ttft_ms"] = st.median(r.ttft_ms for r in warm)
        long_ = chat(s.url, [{"role": "system", "content": SUPPORT_SYSTEM}, {"role": "user", "content": "Explain in about 200 words how email MX records work."}], max_tokens=300, temperature=0.7)
        res["tps"] = long_.tps
        res["long_tokens"] = long_.timings.get("predicted_n")
        runs = [run_scenario(s.url, sc) for _ in range(reps) for sc in SCENARIOS]
        res["scenarios"] = runs
        res["tool_ok"] = sum(r["ok"] for r in runs)
        res["tool_total"] = len(runs)
        res["malformed"] = sum(r["malformed"] for r in runs)
        res["leaks"] = sum(r["leaks"] for r in runs)
        res["by_scenario"] = {sc["id"]: sum(r["ok"] for r in runs if r["id"] == sc["id"]) for sc in SCENARIOS}
        res["rss_gb_after"] = s.rss()
    print(f"{name:12s} load {res['load_s']:5.1f}s gpu {res['gpu_gb']:5.1f}GB cold {res['cold_ttft_ms']:6.0f} ms warm {res['warm_ttft_ms']:5.0f} ms "
          f"{res['tps'] or 0:6.1f} t/s prompt {res['prompt_tps'] or 0:6.0f} t/s tools {res['tool_ok']}/{res['tool_total']} {res['by_scenario']} malformed {res['malformed']} leaks {res['leaks']}", flush=True)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", default=list(LLMS))
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    for name in args.models:
        if not LLMS[name][0].exists():
            print(f"{name}: not downloaded, skipping", flush=True)
            continue
        try:
            save_result(f"llm-{name}", measure(name, args.reps))
        except Exception as e:
            print(f"{name}: failed: {e}", flush=True)
            save_result(f"llm-{name}", {"model": name, "error": str(e)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
