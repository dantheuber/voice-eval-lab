"""Scenario rules and general quality criteria, judged against the transcript and tool log."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voicelab.core.models import (
    VERIFIED_BEFORE_CHANGE,
    CriterionResult,
    Evidence,
    Finding,
    Rule,
    Speaker,
)
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.evaluators.state import check_expectation
from voicelab.eval.judging import (
    JudgeItem,
    JudgeStatus,
    ToolLog,
    ask_items,
    call_record,
    ended_by_caller,
    item_lines,
    render_prompt,
    verification_order,
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
    decided_by: str | None = None
    fails_if_changed_unverified: bool = False
    passes_if_caller_ended: bool = False


@dataclass(frozen=True)
class _Settled:
    """An item the record answers, so no judge is asked."""

    item: _Item
    status: JudgeStatus
    reason: str
    evidence: list[Evidence]
    report: bool = True  # False when the state evaluator already reported the same fact


def _rule_item(rule: Rule, kind: CriterionKind, quote_needed_for: JudgeStatus) -> _Item:
    return _Item(
        rule.id, kind, rule.text, rule.severity, quote_needed_for, decided_by=rule.decided_by
    )


def _items(ctx: EvalContext) -> tuple[list[_Item], list[str]]:
    """What to judge, and the general criteria this call can't support."""
    items: list[_Item] = []
    if ctx.scenario is not None:
        # A `must` passing and a `must_not` failing are the claims that something happened.
        items += [_rule_item(r, "must", "pass") for r in ctx.scenario.must]
        items += [_rule_item(r, "must_not", "fail") for r in ctx.scenario.must_not]
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
                fails_if_changed_unverified=criterion.fails_if_changed_unverified,
                passes_if_caller_ended=criterion.passes_if_caller_ended,
            )
        )
    return items, omitted


def _settled(ctx: EvalContext, item: _Item) -> _Settled | None:
    """The answer if the record has it. Without a tool log there's no record to read."""
    if item.passes_if_caller_ended and ended_by_caller(ctx.artifact):
        return _Settled(item, "pass", "The caller ended the call.", [])
    if not ctx.artifact.has_tool_log:
        return None
    if item.decided_by == VERIFIED_BEFORE_CHANGE or item.fails_if_changed_unverified:
        return _by_verification_order(ctx, item)
    return _by_state(ctx, item) if item.decided_by else None


def _by_verification_order(ctx: EvalContext, item: _Item) -> _Settled | None:
    order = verification_order(ctx.artifact)
    tools = ToolLog(ctx.artifact)
    if order.changed_unverified:
        reference, event = order.changed
        reason = f"The bot used {event.name} at {event.at_s:.1f} s before verifying the caller."
        return _Settled(item, "fail", reason, [tools.evidence(reference)])
    if item.decided_by != VERIFIED_BEFORE_CHANGE:
        return None  # a clean order proves nothing about what the bot said aloud
    if order.changed is None:
        return _Settled(item, "pass", "The bot changed nothing on the account.", [])
    reference, _ = order.verified
    reason = "The bot verified the caller before changing anything."
    return _Settled(item, "pass", reason, [tools.evidence(reference)])


def _by_state(ctx: EvalContext, item: _Item) -> _Settled | None:
    expectation = next((e for e in ctx.scenario.expected_state if e.id == item.decided_by), None)
    if expectation is None:
        return None
    met, evidence = check_expectation(expectation, ctx.artifact)
    reason = f"The database shows this: {expectation.description or expectation.id}"
    # The state evaluator reports a failed expectation, and one fact is one finding.
    return _Settled(item, "pass" if met else "fail", reason, evidence, report=False)


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
        decided = {item.id: found for item in items if (found := _settled(ctx, item))}
        for settled in decided.values():
            _record(out, self.name, settled)
        out.metrics["decided_by_the_record"] = len(decided)
        items = [item for item in items if item.id not in decided]

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
            result = _Settled(item, judgment.status, judgment.reason, judgment.evidence)
            _record(out, self.name, result)
        out.metrics["dropped_quotes"] = dropped_quotes
        out.metrics["judged"] = len(items)
        if items and not replies.payloads:
            raise RuntimeError(f"judge failed on all {len(items)} criteria")


def _record(out: EvaluatorOutput, evaluator: str, result: _Settled) -> None:
    item = result.item
    out.criteria.append(
        CriterionResult(
            criterion_id=item.id,
            kind=item.kind,
            text=item.text,
            status=result.status,
            reason=result.reason,
            evidence=result.evidence,
        )
    )
    if result.status == "fail" and result.report:
        out.findings.append(
            Finding(
                evaluator=evaluator,
                category=item.kind,
                severity=item.severity,
                title=f"Failed: {item.text}",
                detail=result.reason,
                evidence=result.evidence,
                criterion_id=item.id,
            )
        )
