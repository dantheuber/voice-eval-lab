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
    Utterance,
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
    checks_stated_facts: bool = False


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
                checks_stated_facts=criterion.checks_stated_facts,
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
        # One fact is one finding, so a rule saying the same thing is the one that reports it.
        report = item.decided_by == VERIFIED_BEFORE_CHANGE or not _has_order_rule(ctx)
        return _Settled(item, "fail", reason, [tools.evidence(reference)], report=report)
    if item.decided_by != VERIFIED_BEFORE_CHANGE:
        return None  # a clean order proves nothing about what the bot said aloud
    if order.changed is None:
        return _Settled(item, "pass", "The bot changed nothing on the account.", [])
    reference, _ = order.verified
    reason = "The bot verified the caller before changing anything."
    return _Settled(item, "pass", reason, [tools.evidence(reference)])


def _has_order_rule(ctx: EvalContext) -> bool:
    return ctx.scenario is not None and any(
        rule.decided_by == VERIFIED_BEFORE_CHANGE for rule in ctx.scenario.must
    )


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
            status, reason = judgment.status, judgment.reason
            if status == "fail" and item.checks_stated_facts:
                slip = _slip_behind(judgment.evidence, ctx)
                if slip is not None:
                    status, reason = "unknown", f"{slip} {reason}"
            result = _Settled(item, status, reason, judgment.evidence)
            _record(out, self.name, result)
        out.metrics["dropped_quotes"] = dropped_quotes
        out.metrics["judged"] = len(items)
        if items and not replies.payloads:
            raise RuntimeError(f"judge failed on all {len(items)} criteria")


def _slip_behind(evidence: list[Evidence], ctx: EvalContext) -> str | None:
    """Why a failed fact check may be an ASR slip, if nothing else holds the failure up.

    The judge reads one ASR's words, and a real call failed for "priority 2" when the bot said
    "priority ten". What the bot meant doesn't clear it, since the audio could be wrong too, so
    the call goes to a person and pronunciation reports the word.

    That's only so while the failure rests on the number. Evidence that agrees with what the bot
    meant can't show it wrong, and a judge cites its own earlier words and the tools that way. Any
    other citation is a second claim, or a tool that's wrong for both numbers, and that stands.
    """
    slips = [_misheard_number(item, ctx) for item in evidence]
    disputed = [slip for slip in slips if slip is not None]
    if not disputed:
        return None
    meant = {number for _, number in disputed}
    for item, slip in zip(evidence, slips, strict=True):
        if slip is not None or item.speaker == Speaker.CALLER:
            continue
        cited = [number for _, number in _numbers(ctx.index.norms(item.quote))]
        if not any(value in number for value in meant for number in cited):
            return None
    heard, number = disputed[0]
    return f"The transcript has {heard} where the bot meant to say {number}."


def _misheard_number(quote: Evidence, ctx: EvalContext) -> tuple[str, str] | None:
    """A number the transcript gives the bot here, and the one it meant to say, if they differ."""
    if quote.speaker != Speaker.BOT or quote.start_s is None:
        return None
    said = next(
        (
            event
            for event in ctx.artifact.timeline
            if isinstance(event, Utterance)
            and event.speaker == Speaker.BOT
            and event.intended_text
            and event.start_s <= quote.start_s <= event.end_s
        ),
        None,
    )
    if said is None:
        return None
    meant = _numbers(ctx.index.norms(said.intended_text))
    # By the word before it, because "two" elsewhere in the turn says nothing about this one.
    for before, heard in _numbers(ctx.index.norms(quote.quote)):
        meant_here = [number for word, number in meant if word == before]
        if meant_here and heard not in meant_here:
            return heard, meant_here[0]
    return None


def _numbers(tokens: list[str]) -> list[tuple[str, str]]:
    """Each whole number and the word before it. "Fifty" comes down to the tokens "5" and "0"."""
    found: list[tuple[str, str]] = []
    before, number = "", ""
    for token in [*tokens, ""]:
        if token.isdigit():
            number += token
            continue
        if number:
            found.append((before, number))
        before, number = token, ""
    return found


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
