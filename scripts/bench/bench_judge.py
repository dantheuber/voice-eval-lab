"""Judge LLM: tokens per second and schema-constrained JSON on a call transcript.

Usage: bench_judge.py llama --model PATH [--label NAME]
       bench_judge.py ollama --model qwen3.8:27b-mtp-q8_0     (API only, read-only speed reference)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

from common import LlamaServer, gpu_mem_gb, save_result

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
TRANSCRIPT = """[00:00] AGENT: Thanks for calling Fernhill Internet, this is Robin. How can I help you today?
[00:05] CALLER: Hi, I have a question about my bill. I think I was charged twice.
[00:11] AGENT: I'm sorry about that. Can I get your account number please?
[00:15] CALLER: Sure, it's A C 9 0 7 7. The name is Dana Reyes.
[00:21] AGENT: Thanks Dana. I see two charges of forty nine dollars and ninety nine cents on September thirtieth, invoices F H 4 4 1 9 and F H 4 4 2 0.
[00:34] CALLER: Right, that's what I saw. Can you refund the second one?
[00:38] AGENT: I can do that. Just to confirm, you'd like invoice F H 4 4 2 0 refunded for forty nine ninety nine?
[00:46] CALLER: Yes please.
[00:48] TOOL: issue_refund(invoice_id="FH-4420", amount=49.99, reason="duplicate charge") -> {"refund_id": "re_881", "status": "issued"}
[00:50] AGENT: Done. Your refund is issued and should reach your card in two to three business days.
[00:57] CALLER: Okay, and while you're on, my MX records are bouncing mail. Can you look at that too?
[01:04] AGENT: I'm not able to help with that today, please check our website.
[01:09] CALLER: That's not very helpful, but fine. Thanks.
[01:12] AGENT: You're welcome. Have a great day."""
RUBRIC = """Policies: duplicate charges are refunded in full. Refunds reach the card in five to seven business days. The agent must confirm the invoice and amount before refunding. The agent must try to help with email hosting issues before ending the call."""
SYSTEM = "You evaluate recorded support calls. Judge each criterion from the transcript only. Quote exact evidence."
USER = f"""Policies:
{RUBRIC}

Transcript:
{TRANSCRIPT}

Evaluate these criteria: refund_correct (right invoice and amount), confirmed_before_refund, refund_timing_accurate, helped_with_email_issue, polite_tone."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["criteria", "overall"], "properties": {
    "criteria": {"type": "array", "minItems": 5, "maxItems": 5, "items": {"type": "object", "additionalProperties": False,
                 "required": ["name", "verdict", "evidence"], "properties": {
                     "name": {"type": "string", "enum": ["refund_correct", "confirmed_before_refund", "refund_timing_accurate", "helped_with_email_issue", "polite_tone"]},
                     "verdict": {"type": "string", "enum": ["pass", "fail"]},
                     "evidence": {"type": "string"}}}},
    "overall": {"type": "string", "enum": ["pass", "fail", "needs_review"]}}}
EXPECT = {"refund_correct": "pass", "confirmed_before_refund": "pass", "refund_timing_accurate": "fail", "helped_with_email_issue": "fail", "polite_tone": "pass"}


def score(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return {"valid_json": False, "error": str(e), "tail": text[-200:]}
    got = {c["name"]: c["verdict"] for c in obj.get("criteria", [])}
    return {"valid_json": True, "correct": sum(got.get(k) == v for k, v in EXPECT.items()), "of": len(EXPECT), "verdicts": got, "overall": obj.get("overall")}


def llama(args) -> dict:
    gpu0 = gpu_mem_gb()
    with LlamaServer(["-m", args.model, "--jinja", "-c", "32768", "-ngl", "999", "-np", "1", "--temp", "0.1", *args.flags.split()], name=f"judge-{args.label}", timeout=600) as s:
        res: dict = {"model": args.label, "runtime": "llama.cpp b12017 Vulkan", "load_s": s.load_s, "gpu_gb": gpu_mem_gb() - gpu0, "rss_gb": s.rss(), "runs": []}
        client = httpx.Client(timeout=900)
        for rep in range(args.reps):
            for think in (False, True):
                body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}], "temperature": 0.1, "max_tokens": 4000,
                        "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA}},
                        "chat_template_kwargs": {"enable_thinking": think}, "cache_prompt": False}
                t0 = time.perf_counter()
                r = client.post(s.url + "/v1/chat/completions", json=body)
                dt = time.perf_counter() - t0
                r.raise_for_status()
                j = r.json()
                msg = j["choices"][0]["message"]
                t = j.get("timings", {})
                rec = {"thinking": think, "wall_s": dt, "tps": t.get("predicted_per_second"), "prompt_tps": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
                       "tokens": t.get("predicted_n"), **score(msg.get("content") or ""), "reasoning_chars": len(msg.get("reasoning_content") or "")}
                res["runs"].append(rec)
                print(f"  {args.label} think={think}: {dt:6.1f}s {rec['tps'] or 0:6.1f} t/s prompt {rec['prompt_tps'] or 0:6.0f} t/s tokens {rec['tokens']} {rec.get('correct')}/{rec.get('of')} json={rec['valid_json']} {rec.get('overall')}", flush=True)
    return res


def ollama(args) -> dict:
    ps = httpx.get(OLLAMA + "/api/ps", timeout=5).json()
    if ps.get("models") and not args.force:
        raise SystemExit(f"Ollama busy with {[m['name'] for m in ps['models']]}, not disturbing it")
    res: dict = {"model": args.model, "runtime": "Ollama API, read-only", "runs": []}
    client = httpx.Client(timeout=900)
    for rep in range(args.reps):
        for think in (False, True):
            body = {"model": args.model, "stream": False, "think": think, "format": SCHEMA, "keep_alive": 0 if rep == args.reps - 1 and think else "5m",
                    "options": {"temperature": 0.1, "num_predict": 4000},
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]}
            t0 = time.perf_counter()
            r = client.post(OLLAMA + "/api/chat", json=body)
            dt = time.perf_counter() - t0
            r.raise_for_status()
            j = r.json()
            rec = {"thinking": think, "wall_s": dt, "tps": j["eval_count"] / (j["eval_duration"] / 1e9) if j.get("eval_duration") else None,
                   "prompt_tps": j.get("prompt_eval_count", 0) / (j.get("prompt_eval_duration", 1) / 1e9), "tokens": j.get("eval_count"),
                   "load_s": j.get("load_duration", 0) / 1e9, **score(j["message"]["content"])}
            res["runs"].append(rec)
            print(f"  {args.model} think={think}: {dt:6.1f}s {rec['tps'] or 0:6.1f} t/s prompt {rec['prompt_tps']:6.0f} t/s tokens {rec['tokens']} {rec.get('correct')}/{rec.get('of')} json={rec['valid_json']} {rec.get('overall')}", flush=True)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("runtime", choices=["llama", "ollama"])
    p.add_argument("--model", required=True)
    p.add_argument("--label", default="judge")
    p.add_argument("--flags", default="--reasoning auto")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    res = llama(args) if args.runtime == "llama" else ollama(args)
    save_result(f"judge-{(args.label if args.runtime == 'llama' else args.model).replace(':', '_').replace('/', '_')}", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
