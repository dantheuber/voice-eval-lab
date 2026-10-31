"""Tools for writing tests: scenario drafting, text dry-runs, voice previews, the customer seed."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from voicelab.agents import (
    CallerBot,
    DraftError,
    SupportBot,
    TextCallResult,
    draft_scenario,
    run_text_call,
)
from voicelab.company.db import SEED_FILE, create_call_db, customer_notes
from voicelab.company.tools import ToolDispatcher
from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    Beat,
    LineConditions,
    Persona,
    Scenario,
    Utterance,
    VoiceRef,
)
from voicelab.core.profiles import ProviderStack
from voicelab.providers.base import LLM, ChatMessage
from voicelab.services.errors import Invalid, ServiceError
from voicelab.telephony.line import PhoneLine

_PROMPT_PLACEHOLDER = "(from the scenario)"
_BENCHMARK_LINE = "Hi, I think I was charged twice for my last invoice."
_BENCHMARK_REPS = 5


class DraftFailed(ServiceError):
    status = 502


# Customers -------------------------------------------------------------------------------------


def list_customers(seed_file: Path = SEED_FILE) -> list[dict[str, Any]]:
    seed = yaml.safe_load(Path(seed_file).read_text())
    notes = customer_notes(seed_file)
    plan_names = {plan["id"]: plan["name"] for plan in seed["plans"]}
    items = []
    for customer in seed["customers"]:
        customer_id = customer["id"]
        subscriptions = [s for s in seed["subscriptions"] if s["customer_id"] == customer_id]
        domains = [d["name"] for d in seed["domains"] if d["customer_id"] == customer_id]
        items.append(
            {
                "id": customer_id,
                "name": customer["name"],
                "note": notes.get(customer_id, ""),
                "plans": [plan_names[s["plan_id"]] for s in subscriptions],
                "domain": domains[0] if domains else None,
            }
        )
    return items


def customer_detail(customer_id: str) -> dict[str, Any] | None:
    """Raw seed rows for one customer. Keys are column names, so they stay snake_case."""
    db = create_call_db()
    try:
        customer = db.row("SELECT * FROM customers WHERE id = ?", customer_id)
        if customer is None:
            return None
        owned = {
            table: db.rows(f"SELECT * FROM {table} WHERE customer_id = ?", customer_id)
            for table in (
                "subscriptions",
                "invoices",
                "domains",
                "mailboxes",
                "websites",
                "tickets",
            )
        }
        payments = db.rows(
            "SELECT p.* FROM payments p JOIN invoices i ON p.invoice_id = i.id "
            "WHERE i.customer_id = ?",
            customer_id,
        )
        refunds = db.rows(
            "SELECT r.* FROM refunds r JOIN invoices i ON r.invoice_id = i.id "
            "WHERE i.customer_id = ?",
            customer_id,
        )
        dns_records = db.rows(
            "SELECT r.* FROM dns_records r JOIN domains d ON r.domain = d.name "
            "WHERE d.customer_id = ?",
            customer_id,
        )
    finally:
        db.close()
    return {
        "customer": customer,
        **owned,
        "payments": payments,
        "refunds": refunds,
        "dnsRecords": dns_records,
    }


# Drafting --------------------------------------------------------------------------------------


def _seed_values(detail: dict[str, Any]) -> set[str]:
    rows = [
        detail["customer"],
        *(row for key, v in detail.items() if isinstance(v, list) for row in v),
    ]
    return {str(value).lower() for row in rows for value in row.values() if value is not None}


def guessed_fields(scenario: Scenario, requested_customer: str | None) -> list[str]:
    """Which drafted fields didn't come from the request or the customer's seed rows."""
    guessed = []
    if scenario.customer_id and scenario.customer_id != requested_customer:
        guessed.append("customerId")
    detail = customer_detail(scenario.customer_id) if scenario.customer_id else None
    known = _seed_values(detail) if detail else set()
    guessed += [
        f"facts.{key}" for key, value in scenario.facts.items() if value.lower() not in known
    ]
    return guessed


def _draft_request(description: str, customer_id: str | None, base: Scenario | None) -> str:
    parts = [description]
    if customer_id:
        parts.append(f"The caller is the seeded customer {customer_id}.")
    if base is not None:
        fields = base.model_dump(mode="json", exclude={"id", "created_at", "updated_at"})
        parts.append(f"Start from this existing scenario:\n{json.dumps(fields)}")
    return "\n\n".join(parts)


async def draft(
    llm: LLM, description: str, customer_id: str | None, base: Scenario | None
) -> tuple[Scenario, list[str], str]:
    try:
        scenario = await draft_scenario(llm, _draft_request(description, customer_id, base))
    except DraftError as exc:
        raise DraftFailed(str(exc)) from exc
    if customer_id:
        scenario.customer_id = scenario.customer_id or customer_id
    notes = f"Based on {base.id}." if base is not None else ""
    return scenario, guessed_fields(scenario, customer_id), notes


# Dry runs --------------------------------------------------------------------------------------


def neutral_persona() -> Persona:
    return Persona(name="Neutral caller", voice=VoiceRef(voice_id="default"))


async def dry_run(
    support_llm: LLM, caller_llm: LLM, scenario: Scenario, persona: Persona
) -> TextCallResult:
    db = create_call_db()
    try:
        support = SupportBot(support_llm, ToolDispatcher(db))
        return await run_text_call(support, CallerBot(caller_llm, persona, scenario))
    finally:
        db.close()


def dry_run_turns(result: TextCallResult) -> list[dict[str, str]]:
    return [
        {"speaker": event.speaker.value, "text": event.spoken_text or event.intended_text or ""}
        for event in result.timeline
        if isinstance(event, Utterance)
    ]


def persona_prompt(persona: Persona) -> str:
    """The caller's instructions with the scenario's parts marked, since a persona has none."""
    placeholder = Scenario(
        title=_PROMPT_PLACEHOLDER,
        goal=_PROMPT_PLACEHOLDER,
        facts={"fact sheet": _PROMPT_PLACEHOLDER},
        beats=[Beat(id="beat", instruction=_PROMPT_PLACEHOLDER)],
    )
    return CallerBot(None, persona, placeholder).system_prompt()  # type: ignore[arg-type]


# Voices ----------------------------------------------------------------------------------------


async def list_voices(stack: ProviderStack) -> list[dict[str, str]]:
    """What a persona can sound like. Calls always speak a persona's voice through `tts_caller`,
    so a bot voice in this list would preview fine and then fail the call."""
    tts = stack.tts_caller
    return [
        {
            "provider": tts.name,
            "voiceId": voice.id,
            "label": voice.label,
            "language": voice.language,
        }
        for voice in await tts.voices()
    ]


async def preview_voice(
    stack: ProviderStack, text: str, voice: VoiceRef, line: LineConditions | None
) -> bytes:
    try:
        clip = await stack.tts_caller.synthesize(text, voice_id=voice.voice_id, style=voice.style)
    except ValueError as exc:  # the adapter refused the voice: not on the license-safe list
        raise Invalid(str(exc)) from exc
    return await asyncio.to_thread(_through_line, clip, line)


def _through_line(clip: AudioClip, line: LineConditions | None) -> bytes:
    if line is not None:
        clip = PhoneLine(line).process(clip)
    return clip.to_wav_bytes()


# Benchmark -------------------------------------------------------------------------------------


async def benchmark_reply(stack: ProviderStack) -> float:
    """Median seconds from a finished caller turn to the bot's first audio.

    This is the number that decides live against turn-based: STT on the turn, the LLM's first
    tokens, and TTS's first chunk, run in sequence the way one bot turn does.
    """
    voices = await stack.tts_caller.voices()
    if not voices:
        raise RuntimeError("the caller TTS offers no voices to benchmark with")
    clip = await stack.tts_caller.synthesize(_BENCHMARK_LINE, voice_id=voices[0].id)
    bot_voices = await stack.tts_bot.voices()
    samples = []
    for _ in range(_BENCHMARK_REPS):
        started = time.perf_counter()
        heard = await stack.stt_live.transcribe(clip)
        reply = ""
        messages = [ChatMessage(role="user", content=heard.text or _BENCHMARK_LINE)]
        async for delta in stack.llm_support.stream_chat(messages, max_tokens=24):
            reply += delta.text
            if delta.final is not None or len(reply) > 20:
                break
        async for _chunk in stack.tts_bot.stream(reply or "One moment.", voice_id=bot_voices[0].id):
            break
        samples.append(time.perf_counter() - started)
    return round(statistics.median(samples), 3)
