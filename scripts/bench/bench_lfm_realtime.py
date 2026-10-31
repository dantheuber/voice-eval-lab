"""LFM3-Audio /v1/realtime over WebSocket with a Python client.

Usage: bench_lfm_realtime.py [protocol latency tools cancel concurrent] [--quant q8|q4] [--device vulkan|cpu]
Each scenario writes .runtime/bench/results/rt-<scenario>-<config>.json, including raw event logs.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics as st
import sys
import time

import httpx
import numpy as np

from common import BENCH, LFM_Q4, LFM_Q4_MM, LFM_Q8, LFM_Q8_MM, LlamaServer, free_port, gpu_mem_gb, save_result
from rt_client import Session, read_pcm16, write_pcm16

CALLER = {c["id"]: c for c in json.loads((BENCH / "caller.json").read_text())}
OUT = BENCH / "out" / "realtime"
OUT.mkdir(parents=True, exist_ok=True)

VOICE_PROMPT = "Respond with interleaved text and audio."
TOOLS_PROMPT = "Respond with function calls, and interleaved text and audio.\n\nList of tools:"
PERSONA = ("You are a phone support agent for Fernhill Internet, which sells email hosting, website hosting, domains, and SSL. "
           "Keep replies to one or two short sentences. ")
TOOLS = [
    {"type": "function", "name": "lookup_customer", "description": "Find a customer by name or account number.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Customer full name"},
                                                      "account_number": {"type": "string", "description": "Account number such as AC9077"}},
                    "required": ["account_number"]}},
    {"type": "function", "name": "get_invoices", "description": "List recent invoices for a customer.",
     "parameters": {"type": "object", "properties": {"customer_id": {"type": "string", "description": "Customer id from lookup_customer"}},
                    "required": ["customer_id"]}},
    {"type": "function", "name": "issue_refund", "description": "Refund an invoice, in full or in part.",
     "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}, "amount": {"type": "number", "description": "Refund amount in dollars"},
                                                      "reason": {"type": "string"}},
                    "required": ["invoice_id", "amount", "reason"]}},
]
TOOL_RESULTS = {
    "lookup_customer": {"customer_id": "cus_1042", "name": "Dana Reyes", "plan": "Email Pro", "balance_due": 0},
    "get_invoices": {"invoices": [{"id": "FH-4419", "amount": 49.99, "status": "paid", "date": "2026-09-30"},
                                  {"id": "FH-4420", "amount": 49.99, "status": "paid", "date": "2026-09-30", "note": "duplicate of FH-4419"}]},
    "issue_refund": {"refund_id": "re_881", "status": "issued", "amount": 49.99},
}


def lfm_args(quant: str, device: str, parallel: int = 1, ctx: int = 16384) -> list[str]:
    model, mm = (LFM_Q8, LFM_Q8_MM) if quant == "q8" else (LFM_Q4, LFM_Q4_MM)
    flags = ["-ngl", "999"] if device == "vulkan" else ["-ngl", "0", "--no-mmproj-offload", "--device", "none", "-t", "16"]
    return ["-m", str(model), "-mm", str(mm), "--temp", "0", "-c", str(ctx * parallel), "-np", str(parallel), *flags]


async def scenario_protocol(url: str, cfg: str) -> dict:
    res: dict = {}
    pcm = read_pcm16(CALLER["caller-greeting"]["clean"])
    silence = np.zeros(16000 * 2, dtype="<i2")
    async with Session(url, keep_payloads=True) as s:
        res["session_created"] = s.session
        await s.update(instructions=VOICE_PROMPT, output_modalities=["text", "audio"])
        res["session_updated"] = s.session
        # 1. no VAD: speech, 2 s of silence, nothing committed. Any server event here would be server-side turn detection.
        mark = len(s.log)
        await s.stream_audio(pcm)
        await s.stream_audio(silence)
        await asyncio.sleep(1.0)
        res["events_during_speech_and_silence_without_commit"] = [e["type"] for e in s.log[mark:]]
        # 2. unsupported or no-op client events
        mark = len(s.log)
        await s.send(type="input_audio_buffer.clear")
        await s.send(type="conversation.item.truncate", item_id="x", content_index=0, audio_end_ms=100)
        await s.send(type="conversation.item.delete", item_id="x")
        await s.send(type="conversation.item.retrieve", item_id="x")
        await s.send(type="output_audio_buffer.clear")
        await asyncio.sleep(0.5)
        res["probe_events"] = [(e["type"], e.get("event", {}).get("error", {}).get("message")) for e in s.log[mark:]]
        # 3. the audio stayed in the conversation after clear? ask for a reply.
        mark = len(s.log)
        turn = await s.respond()
        await s.done()
        res["reply_after_clear"] = turn.metrics()
        res["full_turn_event_sequence"] = [e["type"] for e in s.log[mark:] if not e["type"].endswith("audio.delta")]
        res["sample_events"] = {}
        for e in s.log[mark:]:
            res["sample_events"].setdefault(e["type"], e.get("event"))
        # 4. config locked after the first audio
        mark = len(s.log)
        await s.send(type="session.update", session={"instructions": "new"})
        await s.send(type="session.update", session={"tools": TOOLS})
        await s.send(type="session.update", session={"output_modalities": ["audio"]})
        await asyncio.sleep(0.5)
        res["update_after_start"] = [(e["type"], e.get("event", {}).get("error", {}).get("message")) for e in s.log[mark:]]
        res["errors"] = s.errors
    # 5. session open with a held slot blocks a second connection on -np 1?
    return res


async def one_turn(url: str, clip: dict, modalities: list[str] | None, instructions: str, cond: str = "clean") -> dict:
    pcm = read_pcm16(clip[cond])
    async with Session(url) as s:
        cfg: dict = {"instructions": instructions}
        if modalities:
            cfg["output_modalities"] = modalities
        await s.update(**cfg)
        await s.stream_audio(pcm)
        turn = await s.respond()
        await s.done()
        m = turn.metrics()
        m.update(clip=clip["id"], cond=cond, errors=s.errors)
        m["_pcm"] = turn.pcm()
        return m


async def scenario_latency(url: str, cfg: str) -> dict:
    res: dict = {"runs": []}
    modes = {"text+audio": (["text", "audio"], VOICE_PROMPT), "audio": (["audio"], VOICE_PROMPT), "auto": (None, VOICE_PROMPT),
             "text": (["text"], "")}
    for mode, (mods, prompt) in modes.items():
        for clip in ("caller-greeting", "caller-lookup", "caller-question", "caller-long"):
            for rep in range(2):
                m = await one_turn(url, CALLER[clip], mods, prompt)
                pcm = m.pop("_pcm")
                if len(pcm) and rep == 0:
                    write_pcm16(str(OUT / f"{cfg}-{mode.replace('+', '_')}-{clip}.wav"), pcm, 24000)
                m["mode"] = mode
                res["runs"].append(m)
                print(f"  {mode:11s} {clip:16s} first audio {m['first_audio_ms'] or 0:6.0f} ms  text {m['first_text_ms'] or 0:6.0f} ms "
                      f"total {m['total_ms']:6.0f} ms audio {m['audio_s']:5.1f}s rtf {m['gen_rtf'] or 0:.2f} lead {m['min_lead_s'] if m['min_lead_s'] is not None else 0:.2f} "
                      f"{m['text'][:70]!r}", flush=True)
    return res


async def run_tool_turn(s: Session, pcm: np.ndarray | None, log: list, max_calls: int = 4) -> dict:
    """One user turn: audio goes in, calls run, ends when the model answers without calls."""
    if pcm is not None:
        await s.stream_audio(pcm)
    turn = await s.respond()
    await s.done()
    first = turn.metrics()
    steps = [first]
    calls_left = max_calls
    while turn.calls and calls_left:
        for c in turn.calls:
            await s.tool_output(c["call_id"], TOOL_RESULTS.get(c["name"], {"error": "unknown tool"}))
        turn = await s.respond()
        await s.done()
        steps.append(turn.metrics())
        calls_left -= 1
    return {"steps": steps, "last_pcm": turn.pcm()}


async def scenario_tools(url: str, cfg: str) -> dict:
    res: dict = {"runs": []}
    for label, instr in (("card-prompt", TOOLS_PROMPT), ("persona+card-prompt", PERSONA + TOOLS_PROMPT)):
        for rep in range(3):
            async with Session(url) as s:
                await s.update(instructions=instr, tools=TOOLS, tool_choice="auto")
                run: dict = {"prompt": label, "rep": rep, "turns": []}
                for clip in ("caller-lookup", "caller-refund", "caller-question"):
                    pcm = read_pcm16(CALLER[clip]["clean"])
                    out = await run_tool_turn(s, pcm, [])
                    last = out.pop("last_pcm")
                    if len(last):
                        write_pcm16(str(OUT / f"{cfg}-tools-{label}-{rep}-{clip}.wav"), last, 24000)
                    out["clip"] = clip
                    run["turns"].append(out)
                    for st_ in out["steps"]:
                        print(f"  {label} r{rep} {clip}: call_ms {st_['first_call_ms']} audio_ms {st_['first_audio_ms']} "
                              f"calls {[(c['name'], c['arguments']) for c in st_['calls']]} text {st_['text'][:90]!r}", flush=True)
                run["errors"] = s.errors
                res["runs"].append(run)
    return res


async def scenario_persona(url: str, cfg: str) -> dict:
    """Which prompt and modality settings keep both tool calls and speech when a persona is in the system prompt?"""
    res: dict = {"variants": []}
    variants = {
        "persona-before-card/auto": (PERSONA + TOOLS_PROMPT, None, None),
        "persona-before-card/force-mixed": (PERSONA + TOOLS_PROMPT, ["text", "audio"], None),
        "persona-before-card/auto-then-mixed-after-tool": (PERSONA + TOOLS_PROMPT, None, ["text", "audio"]),
        "card-first-persona-note/auto": ("Respond with function calls, and interleaved text and audio. " + PERSONA + "\n\nList of tools:", None, None),
        "card-first-persona-note/force-mixed": ("Respond with function calls, and interleaved text and audio. " + PERSONA + "\n\nList of tools:", ["text", "audio"], None),
    }
    for label, (instr, session_mods, after_tool_mods) in variants.items():
        for rep in range(2):
            async with Session(url) as s:
                cfg_: dict = {"instructions": instr, "tools": TOOLS, "tool_choice": "auto"}
                if session_mods:
                    cfg_["output_modalities"] = session_mods
                await s.update(**cfg_)
                await s.stream_audio(read_pcm16(CALLER["caller-lookup"]["clean"]))
                turn = await s.respond()
                await s.done()
                steps = [turn.metrics()]
                for _ in range(4):
                    if not turn.calls:
                        break
                    for c in turn.calls:
                        await s.tool_output(c["call_id"], TOOL_RESULTS.get(c["name"], {"error": "unknown"}))
                    turn = await (s.respond(output_modalities=after_tool_mods) if after_tool_mods else s.respond())
                    await s.done()
                    steps.append(turn.metrics())
                res["variants"].append({"label": label, "rep": rep, "steps": steps, "errors": s.errors})
                print(f"  {label} r{rep}:", [([c["name"] for c in x["calls"]], round(x["audio_s"], 1), round(x["first_audio_ms"] or 0)) for x in steps],
                      repr(steps[-1]["text"][:100]), s.errors[:1], flush=True)
    return res


async def scenario_heard(url: str, cfg: str) -> dict:
    """After a barge-in the context holds text nobody heard. Can a text note correct it, since truncate is unsupported?"""
    res: dict = {"runs": []}
    pcm = read_pcm16(CALLER["caller-long"]["clean"])
    prompt = "Answer in detail with several sentences. " + VOICE_PROMPT
    for note in (None, "SYSTEM NOTE: I interrupted you. I only heard your reply up to the words: \"{heard}\". I did not hear anything after that."):
        for rep in range(2):
            async with Session(url) as s:
                await s.update(instructions=prompt, output_modalities=["text", "audio"])
                await s.stream_audio(pcm)
                turn = await s.respond()
                while turn.t_first_audio is None or time.perf_counter() - turn.t_first_audio < 1.0:
                    await asyncio.sleep(0.01)
                await s.send(type="response.cancel")
                await s.done()
                full = turn.text
                # by 1 s of wall clock the listener heard about 1 s of speech, roughly the first 15 characters per second
                heard = " ".join(full.split()[:3])
                await s.user_text((note.format(heard=heard) + " " if note else "") + "What was the last thing you told me?")
                t2 = await s.respond(output_modalities=["text"])
                await s.done()
                res["runs"].append({"note": bool(note), "rep": rep, "generated": full, "heard": heard, "answer": t2.text, "errors": s.errors})
                print(f"  note={bool(note)} r{rep}: generated {len(full)} chars, heard {heard!r}, answer {t2.text[:160]!r}", flush=True)
    return res


async def scenario_workaround(url: str, cfg: str) -> dict:
    """Unheard-text workaround: on barge-in, close the session and reopen with the heard history in instructions."""
    res: dict = {"runs": []}
    pcm = read_pcm16(CALLER["caller-long"]["clean"])
    base = "Answer in detail with several sentences. "
    for rep in range(4):
        async with Session(url) as s:
            await s.update(instructions=base + VOICE_PROMPT, output_modalities=["text", "audio"])
            await s.stream_audio(pcm)
            turn = await s.respond()
            while turn.t_first_audio is None or time.perf_counter() - turn.t_first_audio < 1.0:
                await asyncio.sleep(0.01)
            await s.send(type="response.cancel")
            await s.done()
            full = turn.text
        heard = " ".join(full.split()[:3])
        t0 = time.perf_counter()
        history = ("Conversation so far:\nCaller: My email has been bouncing for three days and I need it fixed before Monday.\n"
                   f"You: {heard}\nThe caller cut you off there. They did not hear anything else you meant to say.\n\n")
        async with Session(url) as s2:
            t_open = time.perf_counter()
            await s2.update(instructions=history + VOICE_PROMPT, output_modalities=["text", "audio"])
            t_ready = time.perf_counter()
            await s2.user_text("What was the last thing you told me?")
            t2 = await s2.respond(output_modalities=["text"])
            await s2.done()
            rec = {"rep": rep, "generated": full, "heard": heard, "answer": t2.text, "reopen_ms": 1000 * (t_open - t0), "ready_ms": 1000 * (t_ready - t0),
                   "first_text_ms": 1000 * (t2.t_first_text - t2.t_end_input) if t2.t_first_text else None, "errors": s2.errors}
            leaked = [w for w in full.split()[3:] if len(w) > 5 and w.lower().strip(".,") in t2.text.lower()]
            rec["unheard_words_in_answer"] = leaked
            res["runs"].append(rec)
            print(f"  rep {rep}: reopen {rec['reopen_ms']:.0f} ms ready {rec['ready_ms']:.0f} ms heard {heard!r} answer {t2.text[:120]!r} leaked {leaked[:6]}", flush=True)
    return res


async def scenario_probes(url: str, cfg: str) -> dict:
    """Live probes of what the protocol refuses or ignores."""
    res: dict = {}
    async with Session(url) as s:
        await s.update(instructions=VOICE_PROMPT, output_modalities=["text", "audio"])
        mark = len(s.log)
        await s.send(type="conversation.item.create", item={"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello from the agent."}]})
        await s.send(type="conversation.item.create", item={"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "Hello from the agent."}]})
        await s.send(type="conversation.item.create", item={"type": "message", "role": "system", "content": [{"type": "input_text", "text": "Be brief."}]})
        await asyncio.sleep(0.4)
        res["assistant_injection"] = [(e["type"], e.get("event", {}).get("error", {}).get("message")) for e in s.log[mark:]]
    res["voice"] = []
    for label, cfg_ in (("no-voice", {}), ("voice-field-male", {"audio": {"output": {"voice": "male"}}}), ("voice-field-top", {"voice": "male"})):
        async with Session(url) as s:
            await s.update(instructions=VOICE_PROMPT, output_modalities=["text", "audio"], **cfg_)
            echoed = s.session.get("audio", {}).get("output", {})
            await s.user_text("Say hello and tell me the weather is nice.")
            t = await s.respond()
            await s.done()
            pcm = t.pcm()
            write_pcm16(str(OUT / f"voice-{label}.wav"), pcm, 24000)
            res["voice"].append({"label": label, "session_output": echoed, "f0_hz": f0(pcm), "errors": s.errors})
    for label, prompt in (("prompt-male", "Respond with interleaved text and audio using a male voice."), ("prompt-female", "Respond with interleaved text and audio using a female voice.")):
        async with Session(url) as s:
            await s.update(instructions=prompt, output_modalities=["text", "audio"])
            await s.user_text("Say hello and tell me the weather is nice.")
            t = await s.respond()
            await s.done()
            pcm = t.pcm()
            write_pcm16(str(OUT / f"voice-{label}.wav"), pcm, 24000)
            res["voice"].append({"label": label, "f0_hz": f0(pcm), "errors": s.errors, "text": t.text[:80]})
    print("  voice:", [(v["label"], v["f0_hz"]) for v in res["voice"]], flush=True)
    print("  injection:", res["assistant_injection"], flush=True)
    # a second connection while the only slot is held
    async with Session(url) as s:
        t0 = time.perf_counter()
        s2 = Session(url)
        try:
            await asyncio.wait_for(s2.__aenter__(), 3)
            res["second_connection"] = {"opened_ms": 1000 * (time.perf_counter() - t0)}
        except TimeoutError:
            res["second_connection"] = {"blocked_over_s": 3}
        # a queued connection keeps its place, so it must be closed or it takes the slot when this session ends
        await s2.ws.close()
        s2.reader.cancel()
    print("  second connection with -np 1:", res["second_connection"], flush=True)
    # transcript of the user in a realtime session
    async with Session(url, keep_payloads=True) as s:
        await s.update(instructions=VOICE_PROMPT, output_modalities=["text", "audio"])
        await s.stream_audio(read_pcm16(CALLER["caller-greeting"]["clean"]), realtime=False)
        await s.send(type="input_audio_buffer.commit")
        await asyncio.sleep(0.5)
        res["user_item_after_commit"] = [e.get("event") for e in s.log if e["type"] == "conversation.item.done"]
        res["transcript_events"] = [e["type"] for e in s.log if "transcription" in e["type"]]
    return res


def f0(pcm: np.ndarray, rate: int = 24000) -> float | None:
    """Median pitch by autocorrelation over voiced 40 ms frames, enough to tell a male voice from a female one."""
    x = pcm.astype(np.float32) / 32768
    n = int(0.04 * rate)
    out = []
    for i in range(0, len(x) - n, n):
        f = x[i:i + n]
        if np.abs(f).mean() < 0.02:
            continue
        f = f - f.mean()
        c = np.correlate(f, f, "full")[n - 1:]
        lo, hi = rate // 400, rate // 70
        lag = lo + int(np.argmax(c[lo:hi]))
        if c[lag] > 0.3 * c[0]:
            out.append(rate / lag)
    return round(float(np.median(out)), 1) if out else None


async def scenario_long(url: str, cfg: str) -> dict:
    """Context growth over a long call. Audio is sent faster than real time; generated audio is real length."""
    res: dict = {"turns": []}
    clips = ["caller-greeting", "caller-lookup", "caller-question", "caller-long"]
    async with Session(url) as s:
        await s.update(instructions=PERSONA + VOICE_PROMPT, output_modalities=["text", "audio"], max_output_tokens=256)
        total_audio = 0.0
        for i in range(60):
            clip = CALLER[clips[i % len(clips)]]
            await s.stream_audio(read_pcm16(clip["clean"]), realtime=False)
            turn = await s.respond()
            await s.done(180)
            m = turn.metrics()
            total_audio += clip["duration"] + m["audio_s"]
            res["turns"].append({"i": i, "call_s": total_audio, "first_audio_ms": m["first_audio_ms"], "audio_s": m["audio_s"], "rtf": m["gen_rtf"], "status": m["status"], "errors": list(s.errors)})
            if i % 10 == 0 or s.errors:
                print(f"  turn {i}: call {total_audio:.0f}s first audio {m['first_audio_ms'] or -1:.0f} ms rtf {m['gen_rtf'] or 0:.2f} status {m['status']} errors {s.errors[:1]}", flush=True)
            if s.errors:
                break
    return res


async def scenario_toolacc(url: str, cfg: str) -> dict:
    """Tool accuracy on telephone audio, forced text+audio, scored call by call over repeated sessions."""
    res: dict = {"runs": []}
    n = 12
    instr = PERSONA + "Confirm the invoice and amount before you refund. " + TOOLS_PROMPT
    for rep in range(n):
        async with Session(url) as s:
            await s.update(instructions=instr, tools=TOOLS, tool_choice="auto", output_modalities=["text", "audio"])
            run = {"rep": rep, "calls": [], "malformed": 0, "replies": []}
            for clip in ("caller-lookup", "caller-refund"):
                await s.stream_audio(read_pcm16(CALLER[clip]["tel"]), realtime=False)
                turn = await s.respond()
                await s.done()
                for _ in range(5):
                    if not turn.calls:
                        break
                    for c in turn.calls:
                        try:
                            args = json.loads(c["arguments"] or "{}")
                            run["calls"].append({"name": c["name"], "args": args})
                            await s.tool_output(c["call_id"], TOOL_RESULTS.get(c["name"], {"error": "unknown"}))
                        except json.JSONDecodeError:
                            run["malformed"] += 1
                            run["calls"].append({"name": c["name"], "raw": c["arguments"]})
                            await s.tool_output(c["call_id"], {"error": "malformed arguments"})
                    turn = await s.respond()
                    await s.done()
                run["replies"].append(turn.text[:160])
            run["errors"] = s.errors
            res["runs"].append(run)
            print(f"  r{rep}: {[(c['name'], c.get('args', c.get('raw'))) for c in run['calls']]} malformed {run['malformed']}", flush=True)
    return res


async def scenario_listen(url: str, cfg: str) -> dict:
    """Can the model describe tone or delivery of a clip? Same words, different delivery, spoken by synthesized voices."""
    res: dict = {"runs": []}
    client = httpx.AsyncClient(timeout=120)
    prompts = ["Describe the speaker's tone of voice and delivery in one sentence.", "Is the speaker calm, upset, or excited? Answer with one word, then explain briefly."]
    clips = {p.stem: p for p in sorted((BENCH / "tone").glob("*.wav"))}
    for cid, path in clips.items():
        data = base64.b64encode(path.read_bytes()).decode()
        for q in prompts:
            r = await client.post(url + "/v1/chat/completions", json={"temperature": 0, "max_tokens": 120, "messages": [
                {"role": "system", "content": "Answer questions about the provided audio clip. Base every answer only on what can be heard."},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}, {"type": "text", "text": q}]}]})
            text = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else r.text[:200]
            res["runs"].append({"clip": cid, "q": q, "a": text})
            print(f"  {cid}: {text[:150]!r}", flush=True)
    return res


async def scenario_cancel(url: str, cfg: str) -> dict:
    res: dict = {}
    pcm = read_pcm16(CALLER["caller-long"]["clean"])
    # A. response.cancel after 1 s of audio. How fast does it stop, and what is in context afterward?
    cancels = []
    for rep in range(4):
        async with Session(url, keep_payloads=False) as s:
            await s.update(instructions="Answer in detail with several sentences. " + VOICE_PROMPT, output_modalities=["text", "audio"])
            await s.stream_audio(pcm)
            turn = await s.respond()
            while turn.t_first_audio is None or time.perf_counter() - turn.t_first_audio < 1.0:
                await asyncio.sleep(0.01)
            t_cancel = time.perf_counter()
            await s.send(type="response.cancel")
            await s.done()
            got_after = [a for a in turn.audio_arrivals if a[0] > t_cancel]
            rec = {"cancel_to_done_ms": 1000 * (turn.t_done - t_cancel), "status": turn.status,
                   "audio_generated_s": sum(n for _, n in turn.audio_arrivals) / 24000,
                   "deltas_after_cancel": len(got_after), "text_so_far": turn.text, "usage": turn.usage}
            # B. follow-up: does the model remember text it generated, including text no listener heard?
            await s.user_text("Please repeat word for word the last sentence you said.")
            turn2 = await s.respond(output_modalities=["text"])
            await s.done()
            rec["followup_text"] = turn2.text
            rec["errors"] = s.errors
            cancels.append(rec)
            print(f"  cancel r{rep}: done in {rec['cancel_to_done_ms']:.0f} ms, status {turn.status}, "
                  f"gen {rec['audio_generated_s']:.1f}s, after-cancel deltas {len(got_after)}, followup {turn2.text[:100]!r}", flush=True)
    res["cancel"] = cancels
    # C. barge-in without cancel: audio appended while a response is running.
    async with Session(url) as s:
        await s.update(instructions=VOICE_PROMPT, output_modalities=["text", "audio"])
        await s.stream_audio(pcm)
        turn = await s.respond()
        while turn.t_first_audio is None:
            await asyncio.sleep(0.01)
        mark = len(s.log)
        await s.stream_audio(read_pcm16(CALLER["caller-question"]["clean"]))
        await s.send(type="response.cancel")
        await s.done()
        res["append_during_response"] = {"events": sorted({e["type"] for e in s.log[mark:]}), "status": turn.status, "errors": s.errors}
        turn2 = await s.respond()
        await s.done()
        res["append_during_response"]["reply_after"] = turn2.metrics()
    print("  append during response:", json.dumps(res["append_during_response"])[:400], flush=True)
    # D. cancel with no response active
    async with Session(url) as s:
        mark = len(s.log)
        await s.send(type="response.cancel")
        await asyncio.sleep(0.3)
        res["cancel_idle"] = {"events": [e["type"] for e in s.log[mark:]], "errors": s.errors}
    return res


async def concurrent_pass(urls: list[str], label: str) -> dict:
    """Both sessions listen and answer at the same time, as two bots on a call would in the worst case."""
    async def side(url: str, clip: dict, idx: int) -> dict:
        m = await one_turn(url, clip, ["text", "audio"], VOICE_PROMPT)
        pcm = m.pop("_pcm")
        write_pcm16(str(OUT / f"concurrent-{label}-{idx}.wav"), pcm, 24000)
        return m
    runs = []
    for rep in range(3):
        a, b = await asyncio.gather(side(urls[0], CALLER["caller-long"], 0), side(urls[-1], CALLER["caller-long"], 1))
        runs.append([a, b])
        print(f"  {label} r{rep}: first audio {a['first_audio_ms']:.0f}/{b['first_audio_ms']:.0f} ms rtf {a['gen_rtf']:.2f}/{b['gen_rtf']:.2f} "
              f"lead {a['min_lead_s']:.2f}/{b['min_lead_s']:.2f} audio {a['audio_s']:.1f}/{b['audio_s']:.1f}s", flush=True)
    return {"runs": runs}


async def scenario_concurrent(args) -> dict:
    res: dict = {}
    # baseline: one session alone
    gpu0 = gpu_mem_gb()
    with LlamaServer(lfm_args(args.quant, args.device), name="rt-solo") as s:
        solo = [await one_turn(s.url, CALLER["caller-long"], ["text", "audio"], VOICE_PROMPT) for _ in range(3)]
        for m in solo:
            m.pop("_pcm")
        res["solo"] = solo
        res["solo_gpu_gb"] = gpu_mem_gb() - gpu0
        print("  solo:", [(round(m["first_audio_ms"]), round(m["gen_rtf"], 2)) for m in solo], flush=True)
    await asyncio.sleep(1)
    # one server, two slots. Slots share one context: they "respond in turns" per server-realtime.cpp
    gpu0 = gpu_mem_gb()
    with LlamaServer(lfm_args(args.quant, args.device, parallel=2), name="rt-2slot") as s:
        res["two_slots"] = await concurrent_pass([s.url], "two-slots")
        res["two_slots_gpu_gb"] = gpu_mem_gb() - gpu0
    await asyncio.sleep(1)
    # two server processes on the same GPU
    gpu0 = gpu_mem_gb()
    with LlamaServer(lfm_args(args.quant, args.device), name="rt-a") as a, LlamaServer(lfm_args(args.quant, args.device), name="rt-b", port=free_port(18200)) as b:
        res["two_procs"] = await concurrent_pass([a.url, b.url], "two-procs")
        res["two_procs_gpu_gb"] = gpu_mem_gb() - gpu0
        res["two_procs_rss_gb"] = [a.rss(), b.rss()]
    return res


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scenarios", nargs="*", default=["protocol", "latency", "tools", "cancel", "concurrent"])
    p.add_argument("--quant", default="q8", choices=["q8", "q4"])
    p.add_argument("--device", default="vulkan", choices=["vulkan", "cpu"])
    args = p.parse_args()
    cfg = f"{args.device}-{args.quant}"
    funcs = {"protocol": scenario_protocol, "latency": scenario_latency, "tools": scenario_tools, "cancel": scenario_cancel,
             "persona": scenario_persona, "heard": scenario_heard, "workaround": scenario_workaround, "probes": scenario_probes,
             "long": scenario_long, "toolacc": scenario_toolacc, "listen": scenario_listen}
    for name in args.scenarios:
        print(f"== {name} ({cfg})", flush=True)
        if name == "concurrent":
            res = await scenario_concurrent(args)
        else:
            with LlamaServer(lfm_args(args.quant, args.device, ctx=32768 if name == "long" else 16384), name=f"rt-{name}") as s:
                res = await funcs[name](s.url, cfg)
        save_result(f"rt-{name}-{cfg}", res)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
