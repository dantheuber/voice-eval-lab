"""Scenario rules and general quality criteria, judged against the transcript and tool log."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voicelab.core.models import CriterionResult, Finding, Speaker
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.judging import (
    JudgeItem,
    JudgeStatus,
    ToolLog,
    ask_items,
    call_record,
    ended_by_caller,
    item_lines,
    render_prompt,
    verify_judgment,
)

CriterionKind = Literal["must", "must_not", "general"]


@dataclass(frozen=True)
class _Item:
    id: str
    kind: CriterionKind
    text: str
    severity: str
    quote_needed_for: JudgeStatus


def _items(ctx: EvalContext) -> tuple[list[_Item], list[str], list[_Item]]:
    """What to judge, the general criteria this call can't support, and those it already meets."""
    items: list[_Item] = []
    settled: list[_Item] = []
    if ctx.scenario is not None:
        # A `must` passing and a `must_not` failing are the claims that something happened.
        items += [_Item(r.id, "must", r.text, r.severity, "pass") for r in ctx.scenario.must]
        items += [
            _Item(r.id, "must_not", r.text, r.severity, "fail") for r in ctx.scenario.must_not
        ]
    omitted: list[str] = []
    for criterion in ctx.config.general_criteria:
        if criterion.needs_tool_log and not ctx.artifact.has_tool_log:
            omitted.append(criterion.text)
            continue
        item = _Item(
            criterion.id,
            "general",
            criterion.text,
            criterion.severity,
            criterion.quote_needed_for,
        )
        if criterion.passes_if_caller_ended and ended_by_caller(ctx.artifact):
            settled.append(item)
        else:
            items.append(item)
    return items, omitted, settled


def _criterion_line(item: _Item) -> str:
    prefix = {"must": "The bot must: ", "must_not": "The bot must not: ", "general": ""}
    return prefix[item.kind] + item.text


def _call_context(ctx: EvalContext) -> str:
    scenario = ctx.scenario
    if scenario is None:
        return "No scenario: an unscripted call to support."
    return f"{scenario.title}. The caller wants: {scenario.goal}"


def _unknown(item: _Item, reason: str) -> CriterionResult:
    return CriterionResult(
        criterion_id=item.id, kind=item.kind, text=item.text, status="unknown", reason=reason
    )


class TranscriptJudgeEvaluator:
    name = "transcript_judge"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        items, omitted, settled = _items(ctx)
        if omitted:
            out.metrics["not_checked"] = omitted
        out.criteria.extend(
            CriterionResult(
                criterion_id=item.id,
                kind=item.kind,
                text=item.text,
                status="pass",
                reason="The caller ended the call.",
            )
            for item in settled
        )
        judge = ctx.deps.judge
        missing = None if judge else "no judge LLM configured"
        if not any(t.words for t in ctx.transcripts.values()):
            missing = missing or "no independent transcript"
        if missing:
            out.criteria.extend(_unknown(item, missing) for item in items)
            raise Skip(missing)
        out.model = judge.name

        record = call_record(_call_context(ctx), ctx.artifact, ctx.transcripts)

        def prompt_for(batch: list[JudgeItem]) -> str:
            return render_prompt("judge_criteria", record=record, criteria=item_lines(batch))

        asked = [JudgeItem(item.id, _criterion_line(item)) for item in items]
        replies = await ask_items(judge, asked, prompt_for, ctx.config, note=ctx.note_progress)
        out.model = replies.model or out.model
        tools = ToolLog(ctx.artifact)
        dropped_quotes = 0
        for item in items:
            payload = replies.payloads.get(item.id)
            if payload is None:
                out.criteria.append(
                    _unknown(item, f"judge failed: {replies.errors.get(item.id, 'no answer')}")
                )
                continue
            judgment = verify_judgment(
                payload,
                ctx.index,
                quote_needed_for=item.quote_needed_for,
                actor=Speaker.BOT,
                tools=tools,
            )
            dropped_quotes += judgment.dropped_quotes
            out.criteria.append(
                CriterionResult(
                    criterion_id=item.id,
                    kind=item.kind,
                    text=item.text,
                    status=judgment.status,
                    reason=judgment.reason,
                    evidence=judgment.evidence,
                )
            )
            if judgment.status == "fail":
                out.findings.append(
                    Finding(
                        evaluator=self.name,
                        category=item.kind,
                        severity=item.severity,
                        title=f"Failed: {item.text}",
                        detail=judgment.reason,
                        evidence=judgment.evidence,
                        criterion_id=item.id,
                    )
                )
        out.metrics["dropped_quotes"] = dropped_quotes
        out.metrics["judged"] = len(items)
        if items and len(replies.payloads) == 0:
            raise RuntimeError(f"judge failed on all {len(items)} criteria")
