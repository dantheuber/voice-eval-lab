"""Tool calls the bot's argument guard objected to, as findings. No model is involved.

A stopped call changes nothing, so no other evaluator would report it, yet the model made the
mistake all the same. Each objection becomes a finding, sorted by what it shows.

The guard only knows what live STT gave the bot, so "not heard" isn't proof of a guess. Here that's
checked against what the caller really said by then: its script, and the independent transcript.
A value in neither came from the model, and that's the bot's mistake. A value in either was lost by
STT, and so were digits the guard counted short, which are details and not the bot's doing.

A mistake that was stopped is advisory and minor, on purpose. Nothing reached a tool and the caller
lost nothing, while 13 of the 16 calls passing on the 36-call set held one. A non-advisory finding
would have failed them all for something no caller saw, and a major one would have left 3 passes
and 18 reviews, with nothing left for a person to decide. It scores 1 toward review, and a bot that
keeps guessing reaches the review score on its own. A mistake the guard was switched off for really
ran, and it could have used a try or verified a stranger, so that one counts against the call.
"""

from __future__ import annotations

from voicelab.agents.argument_guard import DIGIT_COUNT, NOT_HEARD, said_exactly
from voicelab.core.models import Evidence, Finding, Speaker, ToolCallEvent, Utterance
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.judging import describe_call
from voicelab.eval.rollup import roll_up

BLOCKED_TOOL_CALL = "blocked_tool_call"
UNGUARDED_TOOL_CALL = "unguarded_tool_call"
_MADE_UP = {"verify_identity": "guessed", "find_customer": "invented"}


def _caller_turns(ctx: EvalContext, by_s: float) -> list[Utterance]:
    return [
        event
        for event in ctx.artifact.timeline
        if isinstance(event, Utterance)
        and event.speaker == Speaker.CALLER
        and event.start_s <= by_s
    ]


def _really_said(ctx: EvalContext, by_s: float) -> str:
    """The caller's words by this time, from what it was scripted to say and what ASR heard.

    Later words don't count: a value the bot used first and the caller said after was a guess.
    """
    scripted = [u.spoken_text or u.intended_text or "" for u in _caller_turns(ctx, by_s)]
    transcript = ctx.transcripts.get(Speaker.CALLER)
    words = [w.text for w in transcript.words if w.start_s <= by_s] if transcript else []
    return " ".join([*scripted, *words])


def _made_up(event: ToolCallEvent, ctx: EvalContext) -> bool:
    if event.guard.rule != NOT_HEARD:
        return False
    said = _really_said(ctx, event.at_s)
    return any(
        not said_exactly(str(event.arguments.get(f) or ""), said) for f in event.guard.fields
    )


def _title(event: ToolCallEvent, made_up: bool) -> str:
    lead = f"Blocked {event.name}" if event.guard.stopped else f"{event.name} ran"
    fields = ", ".join(event.guard.fields)
    if event.guard.rule == DIGIT_COUNT:
        return f"{lead}: wrong number of digits in {fields}"
    if made_up:
        return f"{lead} with {_MADE_UP.get(event.name, 'made up')} {fields}"
    return f"{lead}: live STT lost {fields}"


def _detail(event: ToolCallEvent, made_up: bool) -> str:
    fields = ", ".join(event.guard.fields)
    if event.guard.rule == DIGIT_COUNT:
        what = f"Live STT gave the bot the wrong number of digits for {fields}."
    elif made_up:
        what = f"Not in anything live STT gave the bot, or in what the caller said: {fields}."
    else:
        what = f"The caller said {fields}, but live STT never gave the bot those words."
    if event.guard.stopped:
        return f"{what} The guard stopped the call, so no tool ran and no try was used."
    return f"{what} The guard is off, so the call reached the tool."


def _evidence(event: ToolCallEvent, ctx: EvalContext) -> list[Evidence]:
    """The attempt, and the bot's last hearing of the caller, as `heard_text` has it."""
    evidence = [Evidence(quote=describe_call(event), start_s=event.at_s)]
    heard = [u for u in _caller_turns(ctx, event.at_s) if u.heard_text]
    if heard:
        last = heard[-1]
        evidence.append(
            Evidence(
                quote=last.heard_text,
                speaker=Speaker.CALLER,
                start_s=last.start_s,
                end_s=last.end_s,
            )
        )
    return evidence


def _finding(event: ToolCallEvent, ctx: EvalContext) -> Finding:
    made_up = _made_up(event, ctx)
    ran = not event.guard.stopped
    return Finding(
        evaluator=ToolAttemptsEvaluator.name,
        category=UNGUARDED_TOOL_CALL if ran else BLOCKED_TOOL_CALL,
        severity=("major" if ran else "minor") if made_up else "info",
        advisory=not (ran and made_up),
        title=_title(event, made_up),
        detail=_detail(event, made_up),
        evidence=_evidence(event, ctx),
    )


def _rollup_title(group: list[Finding]) -> str:
    if group[0].category == UNGUARDED_TOOL_CALL:
        return f"{len(group)} tool calls the guard would have blocked"
    return f"{len(group)} blocked tool calls"


class ToolAttemptsEvaluator:
    name = "tool_attempts"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        if not ctx.artifact.has_tool_log:
            raise Skip("no tool log, so there are no tool calls to read")
        attempts = [
            event
            for event in ctx.artifact.timeline
            if isinstance(event, ToolCallEvent) and event.guard is not None
        ]
        found = [_finding(event, ctx) for event in attempts]
        mistakes = [finding for finding in found if finding.severity != "info"]
        blocked = [finding for finding in mistakes if finding.advisory]
        if len(blocked) >= ctx.config.blocked_attempts_major:
            for finding in blocked:
                finding.severity = "major"
        out.findings.extend(
            roll_up(found, key=lambda f: f"{f.category}:{f.severity}", title=_rollup_title)
        )
        out.metrics.update(
            {
                "attempts": len(attempts),
                "mistakes": len(mistakes),
                "stopped": sum(event.guard.stopped for event in attempts),
                "argument_guard": ctx.artifact.metadata.get("argument_guard"),
            }
        )
