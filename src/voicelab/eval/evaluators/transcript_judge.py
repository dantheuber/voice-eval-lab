"""Scenario rules and general quality criteria, judged one at a time against the transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voicelab.core.models import CriterionResult, Finding, Speaker
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.judging import (
    JUDGE_SCHEMA,
    JudgeStatus,
    ask_json,
    render_prompt,
    render_tool_log,
    verify_judgment,
)
from voicelab.eval.transcript import render_transcript

CriterionKind = Literal["must", "must_not", "general"]


@dataclass(frozen=True)
class _Item:
    id: str
    kind: CriterionKind
    text: str
    severity: str
    quote_needed_for: JudgeStatus


def _items(ctx: EvalContext) -> tuple[list[_Item], list[str]]:
    """What to judge, and the general criteria this call can't support."""
    items: list[_Item] = []
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
        items.append(
            _Item(
                criterion.id,
                "general",
                criterion.text,
                criterion.severity,
                criterion.quote_needed_for,
            )
        )
    return items, omitted


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
        items, omitted = _items(ctx)
        if omitted:
            out.metrics["not_checked"] = omitted
        judge = ctx.deps.judge
        missing = None if judge else "no judge LLM configured"
        if not any(t.words for t in ctx.transcripts.values()):
            missing = missing or "no independent transcript"
        if missing:
            out.criteria.extend(_unknown(item, missing) for item in items)
            raise Skip(missing)
        out.model = judge.name

        transcript = render_transcript(ctx.transcripts)
        tools = render_tool_log(ctx.artifact) if ctx.artifact.has_tool_log else "Not available."
        context = _call_context(ctx)
        errors = 0
        dropped_quotes = 0
        for position, item in enumerate(items, start=1):
            await ctx.note_progress(f"criterion {position} of {len(items)}")
            prompt = render_prompt(
                "judge_criterion",
                context=context,
                tool_log=tools,
                transcript=transcript,
                criterion=_criterion_line(item),
            )
            try:
                payload, model = await ask_json(judge, prompt, JUDGE_SCHEMA, ctx.config)
            except Exception as exc:  # a bad reply costs one criterion, not the whole judge
                errors += 1
                out.criteria.append(_unknown(item, f"judge failed: {exc}"))
                continue
            out.model = model
            judgment = verify_judgment(
                payload, ctx.index, quote_needed_for=item.quote_needed_for, prefer=Speaker.BOT
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
        if items and errors == len(items):
            raise RuntimeError(f"judge failed on all {errors} criteria")
