"""Asking an LLM about the transcript without trusting its citations.

A judge can invent a convincing quote. So a judgment that rests on quotes only stands if those
quotes are found in the independent transcript, and the evidence kept is the transcript's own text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Literal

from voicelab.core.models import CallArtifact, Evidence, Speaker, ToolCallEvent
from voicelab.eval.config import EvalConfig
from voicelab.eval.transcript import TranscriptIndex
from voicelab.providers.base import LLM, ChatMessage

PROMPT_DIR = Path(__file__).parent / "prompts"

JudgeStatus = Literal["pass", "fail", "unknown"]

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail", "unknown"]},
        "reason": {"type": "string"},
        "quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "reason", "quotes"],
}

_MAX_TOOL_RESULT_CHARS = 1500


@dataclass
class Judgment:
    status: JudgeStatus
    reason: str
    evidence: list[Evidence] = field(default_factory=list)
    dropped_quotes: int = 0


def render_prompt(name: str, **values: str) -> str:
    """Fill a template from `prompts/`. Uses `$name`, since the templates show literal JSON."""
    return Template((PROMPT_DIR / f"{name}.txt").read_text()).substitute(values)


def parse_json_reply(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a reply. Local models wrap it in think tags or code fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


async def ask_json(
    llm: LLM, prompt: str, schema: dict[str, Any], config: EvalConfig
) -> tuple[dict[str, Any], str]:
    """One structured answer and the model that gave it. An unparseable reply gets one retry."""
    messages = [ChatMessage(role="user", content=prompt)]
    last_error: ValueError | None = None
    for _ in range(2):
        response = await llm.chat(
            messages,
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens,
            json_schema=schema,
        )
        try:
            return parse_json_reply(response.text), response.model or llm.name
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"judge gave no usable JSON after a retry: {last_error}")


def verify_judgment(
    payload: dict[str, Any],
    index: TranscriptIndex,
    *,
    quote_needed_for: JudgeStatus,
    prefer: Speaker | None = None,
) -> Judgment:
    """Check a judge's reply against the transcript, downgrading what it can't support.

    `quote_needed_for` is the status that claims something happened, which must be shown. Quotes
    the transcript doesn't contain are dropped, and if that leaves nothing behind a claim, or the
    judge cited only made-up text, the result is `unknown`, whatever the judge concluded.
    """
    raw_status = payload.get("status")
    status: JudgeStatus = raw_status if raw_status in ("pass", "fail") else "unknown"
    reason = str(payload.get("reason") or "")
    raw_quotes = payload.get("quotes")
    quotes = [q for q in raw_quotes if isinstance(q, str) and q.strip()] if raw_quotes else []
    located = [index.locate(q, prefer) for q in quotes]
    evidence = [e for e in located if e is not None]
    dropped = len(quotes) - len(evidence)

    if status != "unknown" and not evidence:
        if dropped:
            reason = f"Judge said {status}, but its quotes aren't in the transcript. {reason}"
            status = "unknown"
        elif status == quote_needed_for:
            reason = f"Judge said {status} without quoting the transcript. {reason}"
            status = "unknown"
    elif dropped:
        reason = f"{reason} ({dropped} quote(s) not in the transcript were dropped.)"
    return Judgment(status, reason.strip(), evidence, dropped)


def render_tool_log(artifact: CallArtifact) -> str:
    lines = []
    for event in artifact.timeline:
        if isinstance(event, ToolCallEvent):
            outcome = f"error: {event.error}" if event.error else json.dumps(event.result)
            arguments = json.dumps(event.arguments)
            lines.append(f"[{event.at_s:.1f}s] {event.name}({arguments}) -> {outcome}")
    if not lines:
        return "The bot made no tool calls."
    return "\n".join(line[:_MAX_TOOL_RESULT_CHARS] for line in lines)
