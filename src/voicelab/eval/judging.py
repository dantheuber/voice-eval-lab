"""Asking an LLM about the transcript without trusting its citations.

A judge can invent a convincing quote. So a judgment that rests on quotes only stands if those
quotes are found in the independent transcript, and the evidence kept is the transcript's own text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Literal

from voicelab.core.models import CallArtifact, Evidence, Speaker, SystemEvent, ToolCallEvent
from voicelab.eval.config import EvalConfig
from voicelab.eval.transcript import TranscriptIndex, timed_turns
from voicelab.providers.adapters.openai_compat import LLMTimeoutError
from voicelab.providers.base import LLM, ChatMessage, Transcript

PROMPT_DIR = Path(__file__).parent / "prompts"

JudgeStatus = Literal["pass", "fail", "unknown"]

_MAX_TOOL_LINE_CHARS = 900
_MAX_REASON_CHARS = 240
_MAX_QUOTE_CHARS = 200
_MAX_CITATIONS = 3
_REPLY_OVERHEAD_TOKENS = 64
_TOOL_REF = re.compile(r"^T(\d+)$", re.IGNORECASE)
_VERIFY_TOOL = "verify_identity"
# By convention tools with these names only read, and everything else can write.
READING_TOOLS = ("find_", "get_", "list_", "verify_", "end_")
_NOT_ACCOUNT_CHANGES = ("create_ticket",)  # a correct refusal often opens one
# What a stranger shouldn't get to. `find_customer` is allowed before verifying, so it isn't here.
_ACCOUNT_READS = ("get_account_summary", "get_invoice", "list_invoices", "get_dns_records")


@dataclass(frozen=True)
class JudgeItem:
    """One question for the judge. `line` is how the prompt puts it."""

    id: str
    line: str


@dataclass
class ItemReplies:
    payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    model: str | None = None


@dataclass
class Judgment:
    status: JudgeStatus
    reason: str
    evidence: list[Evidence] = field(default_factory=list)
    dropped_quotes: int = 0


def items_schema(ids: Sequence[str]) -> dict[str, Any]:
    """A reply for each item. Citations and the reason come before the status on purpose.

    With reasoning off, a status written first is a guess that the reason then defends. Finding
    the evidence first is the only thinking the model gets to do, and it costs no extra requests.
    Lengths are capped, so a reply's size and how long it takes are known in advance.
    """
    citation = {"type": "string", "maxLength": _MAX_QUOTE_CHARS}
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": list(ids)},
            "quotes": {"type": "array", "items": citation, "maxItems": _MAX_CITATIONS},
            "tools": {"type": "array", "items": {"type": "string"}, "maxItems": _MAX_CITATIONS},
            "reason": {"type": "string", "maxLength": _MAX_REASON_CHARS},
            "status": {"type": "string", "enum": ["pass", "fail", "unknown"]},
        },
        "required": ["id", "quotes", "tools", "reason", "status"],
    }
    results = {"type": "array", "items": item, "minItems": len(ids), "maxItems": len(ids)}
    return {"type": "object", "properties": {"results": results}, "required": ["results"]}


def render_prompt(name: str, **values: str) -> str:
    """Fill a template from `prompts/`. Uses `$name`, since the templates show literal JSON."""
    return Template((PROMPT_DIR / f"{name}.txt").read_text()).substitute(values)


def item_lines(items: Sequence[JudgeItem]) -> str:
    return "\n".join(f"- {item.id}: {item.line}" for item in items)


def parse_json_reply(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a reply. Local models wrap it in think tags or code fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def parse_items(text: str) -> list[dict[str, Any]]:
    """The items in a reply. One cut off by the token limit still gives those it finished."""
    try:
        results = parse_json_reply(text).get("results")
    except ValueError:
        return _finished_items(text)
    return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []


def _finished_items(text: str) -> list[dict[str, Any]]:
    start = text.find("[", max(text.find('"results"'), 0))
    if start < 0:
        return []
    items: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    position = start + 1
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        try:
            item, position = decoder.raw_decode(text, position)
        except ValueError:
            return items
        if isinstance(item, dict):
            items.append(item)


async def ask_json(
    llm: LLM,
    prompt: str,
    schema: dict[str, Any],
    config: EvalConfig,
    *,
    items: int = 1,
) -> tuple[dict[str, Any], str]:
    """One structured answer and the model that gave it.

    An unparseable reply gets one more try with twice the tokens, since at temperature 0 the only
    thing that can change the outcome is more room for an answer that ran out.
    """
    budget = _token_budget(config, items)
    last_error: ValueError | None = None
    for attempt in range(2):
        response = await _chat(llm, prompt, schema, config, budget * (attempt + 1))
        try:
            return parse_json_reply(response.text), response.model or llm.name
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"judge gave no usable JSON after a retry: {last_error}")


def _token_budget(config: EvalConfig, items: int) -> int:
    return config.judge_max_tokens * items + _REPLY_OVERHEAD_TOKENS


async def _chat(llm: LLM, prompt: str, schema: dict[str, Any], config: EvalConfig, budget: int):
    return await llm.chat(
        [ChatMessage(role="user", content=prompt)],
        temperature=config.judge_temperature,
        max_tokens=budget,
        json_schema=schema,
    )


async def ask_items(
    llm: LLM,
    items: Sequence[JudgeItem],
    prompt_for: Callable[[list[JudgeItem]], str],
    config: EvalConfig,
    *,
    note: Callable[[str], Awaitable[None]] | None = None,
) -> ItemReplies:
    """Ask about items in batches, then ask alone about any a reply left out.

    One request with all the criteria replaces one each, and the transcript is read once. A batch
    that went wrong usually went wrong on one item, so asking alone recovers the rest. A timeout
    isn't asked again, because a slow server only gets slower with more requests.
    """
    replies = ItemReplies()
    size = config.judge_batch_size or len(items) or 1
    batches = [list(items[i : i + size]) for i in range(0, len(items), size)]
    timed_out = False
    for position, batch in enumerate(batches, start=1):
        if note is not None:
            await note(f"batch {position} of {len(batches)}")
        timed_out = await _ask_batch(llm, batch, prompt_for, config, replies) or timed_out
    missing = [item for item in items if item.id not in replies.payloads]
    if size > 1 and not timed_out:
        for position, item in enumerate(missing, start=1):
            if note is not None:
                await note(f"asking again {position} of {len(missing)}")
            await _ask_batch(llm, [item], prompt_for, config, replies)
    for item in items:
        if item.id in replies.payloads:
            replies.errors.pop(item.id, None)
    return replies


async def _ask_batch(
    llm: LLM,
    batch: list[JudgeItem],
    prompt_for: Callable[[list[JudgeItem]], str],
    config: EvalConfig,
    replies: ItemReplies,
) -> bool:
    """Record what this batch got. Returns whether the server timed out."""
    ids = [item.id for item in batch]
    budget = _token_budget(config, len(batch))
    try:
        response = await _chat(llm, prompt_for(batch), items_schema(ids), config, budget)
    except Exception as exc:  # a failed request costs its items, not the evaluator
        for item_id in ids:
            replies.errors[item_id] = str(exc)
        return isinstance(exc, LLMTimeoutError)
    replies.model = response.model or llm.name
    for item in parse_items(response.text):
        if item.get("id") in ids:
            replies.payloads.setdefault(item["id"], item)
    for item_id in ids:
        if item_id not in replies.payloads:
            replies.errors[item_id] = f"no answer for this item in: {response.text[:120]!r}"
    return False


class ToolLog:
    """The bot's tool calls, numbered so a judge can cite one the way it quotes the transcript."""

    def __init__(self, artifact: CallArtifact) -> None:
        self._events = [e for e in artifact.timeline if isinstance(e, ToolCallEvent)]

    def __len__(self) -> int:
        return len(self._events)

    @property
    def times(self) -> list[float]:
        return [event.at_s for event in self._events]

    def timed_lines(self) -> list[tuple[float, str]]:
        """Each call's time and line. A repeat points back, so it doesn't cost the whole line."""
        first_seen: dict[str, int] = {}
        lines: list[tuple[float, str]] = []
        for number, event in enumerate(self._events, start=1):
            detail = describe_call(event)
            start = f"[{event.at_s:.1f}s] TOOL T{number}"
            if detail in first_seen:
                lines.append(
                    (event.at_s, f"{start} the same call and result as T{first_seen[detail]}")
                )
            else:
                first_seen[detail] = number
                lines.append((event.at_s, f"{start} {detail}"[:_MAX_TOOL_LINE_CHARS]))
        return lines

    def evidence(self, reference: str) -> Evidence | None:
        """What a citation like "T3" points at, or None if there's no such call."""
        match = _TOOL_REF.match(reference.strip())
        number = int(match.group(1)) if match else 0
        if not 1 <= number <= len(self._events):
            return None
        event = self._events[number - 1]
        return Evidence(quote=describe_call(event), start_s=event.at_s)

    def stopped_note(self) -> str:
        """Which calls never ran, in words. Most calls have none, so their prompt doesn't change."""
        stopped = [
            f"T{number}"
            for number, event in enumerate(self._events, start=1)
            if event.guard is not None and event.guard.stopped
        ]
        if not stopped:
            return ""
        it = "it" if len(stopped) == 1 else "they"
        return (
            f"The bot's own argument guard stopped {', '.join(stopped)} before {it} reached a "
            f"tool, so {it} changed nothing.\n"
        )


def describe_call(event: ToolCallEvent) -> str:
    outcome = f"error: {event.error}" if event.error else json.dumps(event.result)
    return f"{event.name}({json.dumps(event.arguments)}) -> {outcome}"


_ENDINGS = {
    "caller_hung_up": "The caller hung up at {at:.1f} s",
    "bot_ended": "The bot ended the call at {at:.1f} s",
    "max_turns": "The call reached its turn limit at {at:.1f} s",
}


def call_ending(artifact: CallArtifact) -> str:
    """Who ended the call. The transcript can't show that, and criteria depend on it."""
    for event in reversed(artifact.timeline):
        if isinstance(event, SystemEvent) and event.kind in _ENDINGS:
            detail = f" ({event.detail})" if event.detail else ""
            return _ENDINGS[event.kind].format(at=event.at_s) + detail + "."
    return "How the call ended isn't recorded."


@dataclass(frozen=True)
class VerificationOrder:
    """When the bot verified the caller and when it first changed the account, from the tool log.

    Each is a citation like "T3" and its event, or None if it never happened.
    """

    verified: tuple[str, ToolCallEvent] | None
    changed: tuple[str, ToolCallEvent] | None
    attempted: bool
    read: tuple[str, ToolCallEvent] | None = None  # the first look at the account

    @property
    def changed_unverified(self) -> bool:
        if self.changed is None:
            return False
        return self.verified is None or self.changed[1].at_s < self.verified[1].at_s

    @property
    def read_unverified(self) -> bool:
        if self.read is None:
            return False
        return self.verified is None or self.read[1].at_s < self.verified[1].at_s


def verification_order(artifact: CallArtifact) -> VerificationOrder:
    calls = [e for e in artifact.timeline if isinstance(e, ToolCallEvent)]
    numbered = [(f"T{n}", event) for n, event in enumerate(calls, start=1)]
    return VerificationOrder(
        verified=next((item for item in numbered if _verified(item[1])), None),
        changed=next((item for item in numbered if _changes_account(item[1])), None),
        attempted=any(e.name == _VERIFY_TOOL for e in calls),
        read=next((item for item in numbered if _reads_account(item[1])), None),
    )


def _verified(event: ToolCallEvent) -> bool:
    result = event.result if isinstance(event.result, dict) else {}
    return event.name == _VERIFY_TOOL and not event.error and result.get("verified") is True


def _reads_account(event: ToolCallEvent) -> bool:
    return event.name in _ACCOUNT_READS and not event.error


def _changes_account(event: ToolCallEvent) -> bool:
    harmless = event.name.startswith(READING_TOOLS) or event.name in _NOT_ACCOUNT_CHANGES
    return not harmless and not event.error


def order_of_events(artifact: CallArtifact) -> str:
    """The verification order in words, for the record a judge reads."""
    order = verification_order(artifact)
    if order.verified is None and order.changed is None and not order.attempted:
        return ""
    check = "The bot never verified the caller."
    if order.verified is not None:
        reference, event = order.verified
        check = f"The bot verified the caller at {event.at_s:.1f} s ({reference})."
    if order.read_unverified:
        reference, event = order.read
        check += f" It read the account before that, at {event.at_s:.1f} s ({reference})."
    if order.changed is None:
        return f"{check} The bot changed nothing on the account.\n"
    reference, event = order.changed
    change = f"The bot first changed something at {event.at_s:.1f} s ({reference}, {event.name})"
    if order.verified is None:
        return f"{check} {change}.\n"
    return f"{check} {change}, {'before' if order.changed_unverified else 'after'} verifying.\n"


def render_timeline(transcripts: dict[Speaker, Transcript], tools: ToolLog) -> str:
    """Speech and tool calls as one sequence, so what came first can be read straight off it."""
    lines = [*timed_turns(transcripts, tool_times=tools.times), *tools.timed_lines()]
    lines.sort(key=lambda line: line[0])
    return "\n".join(line for _, line in lines)


def call_record(
    context: str, artifact: CallArtifact, transcripts: dict[Speaker, Transcript]
) -> str:
    """What every judge prompt starts with, identical so the server can reuse its work on it."""
    tools = ToolLog(artifact)
    note = order_of_events(artifact) if tools else "The bot made no tool calls.\n"
    if not artifact.has_tool_log:
        note = "No tool log was recorded for this call.\n"
    note += tools.stopped_note()
    return render_prompt(
        "call_record",
        context=context,
        ending=call_ending(artifact),
        note=note,
        timeline=render_timeline(transcripts, tools),
    )


def ended_by_caller(artifact: CallArtifact) -> bool:
    return any(
        isinstance(event, SystemEvent) and event.kind in ("caller_hung_up", "max_turns")
        for event in artifact.timeline
    )


def verify_judgment(
    payload: dict[str, Any],
    index: TranscriptIndex,
    *,
    quote_needed_for: JudgeStatus,
    actor: Speaker | None = None,
    tools: ToolLog | None = None,
) -> Judgment:
    """Check a judge's reply against the record, downgrading what it can't support.

    Any citation the record doesn't contain makes the result `unknown`, even beside real ones: a
    judge that invented one quote can't be trusted on the rest of its reasoning. Real quotes stay
    as evidence for the reviewer.

    `quote_needed_for` is the status that claims something happened, and it has to be shown. When
    the criterion is about one party, `actor`, that party's own words have to show it, because the
    other side saying something proves nothing about what this side did. A tool call is the bot's
    own act, so citing one shows it too. Other real quotes can stand beside them as context.
    """
    raw_status = payload.get("status")
    status: JudgeStatus = raw_status if raw_status in ("pass", "fail") else "unknown"
    reason = str(payload.get("reason") or "")
    quotes = _strings(payload.get("quotes"))
    references = _strings(payload.get("tools")) if tools is not None else []
    spoken = [e for e in (index.locate(q, actor) for q in quotes) if e is not None]
    acted = [e for e in (tools.evidence(r) for r in references) if e is not None] if tools else []
    evidence = [*spoken, *acted]
    dropped = len(quotes) + len(references) - len(evidence)

    if status == "unknown":
        return Judgment(status, reason.strip(), evidence, dropped)
    claimed = status
    if dropped:
        reason = (
            f"Judge said {claimed}, but {dropped} of its quotes aren't in the transcript. {reason}"
        )
        status = "unknown"
    elif claimed == quote_needed_for:
        if not evidence:
            reason = f"Judge said {claimed} without quoting the transcript. {reason}"
            status = "unknown"
        elif actor is not None and not acted and all(e.speaker != actor for e in spoken):
            reason = f"Judge said {claimed}, but quoted nothing the {actor.value} said. {reason}"
            status = "unknown"
    return Judgment(status, reason.strip(), evidence, dropped)


def _strings(values: Any) -> list[str]:
    return [v for v in values if isinstance(v, str) and v.strip()] if values else []
