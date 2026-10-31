"""Scenario state assertions against the call's database diff. No model is involved."""

from __future__ import annotations

import json
from typing import Any

from voicelab.core.models import (
    CallArtifact,
    CriterionResult,
    DbChange,
    Evidence,
    Finding,
    StateExpectation,
    ToolCallEvent,
)
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.rollup import roll_up
from voicelab.eval.verdict import UNEXPECTED_CHANGE

# By convention tools with these names only read. Everything else can write.
_READING_TOOLS = ("find_", "get_", "list_", "verify_", "end_")


def same_value(a: Any, b: Any) -> bool:
    """Equality that survives YAML and SQLite disagreeing about types, like "12.50" and 12.5."""
    if a == b or str(a) == str(b):
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _matches(fields: dict[str, Any], where: dict[str, Any]) -> bool:
    return all(k in fields and same_value(fields[k], v) for k, v in where.items())


def row_matches(change: DbChange, where: dict[str, Any]) -> bool:
    """`where` may describe the row before or after the change, as when it names a changed field."""
    return _matches({**(change.before or {}), **change.key}, where) or _matches(
        {**(change.after or {}), **change.key}, where
    )


def _applies(change: DbChange, expected: dict[str, Any]) -> bool:
    return _matches(change.after or {}, expected)


def satisfying_changes(expectation: StateExpectation, changes: list[DbChange]) -> list[DbChange]:
    """The changes that show this expectation being met, or breached for `no_change`."""
    in_table = [c for c in changes if c.table == expectation.table]
    if expectation.op == "row_added":
        return [
            c for c in in_table if c.kind == "insert" and _matches(c.after or {}, expectation.where)
        ]
    matched = [c for c in in_table if row_matches(c, expectation.where)]
    if expectation.op == "changed":
        return [c for c in matched if _applies(c, expectation.changes)]
    return matched


def _describe(expectation: StateExpectation) -> str:
    if expectation.description:
        return expectation.description
    if expectation.op == "row_added":
        return f"A row is added to {expectation.table} with {expectation.where}"
    if expectation.op == "no_change":
        return f"{expectation.table} rows matching {expectation.where} are left alone"
    return f"{expectation.table} rows matching {expectation.where} change to {expectation.changes}"


def _writing_calls(artifact: CallArtifact) -> list[ToolCallEvent]:
    return [
        e
        for e in artifact.timeline
        if isinstance(e, ToolCallEvent) and not e.error and not e.name.startswith(_READING_TOOLS)
    ]


def _caused_by(change: DbChange, calls: list[ToolCallEvent]) -> ToolCallEvent | None:
    """The tool call behind a change. The database diff has no times, but the call does.

    The earliest call to mention the row's key is it. A row keyed by a bare number, which tools
    never mention, falls back to the first write of the call, which at worst is too early.
    """
    keys = [str(value) for value in change.key.values()]

    def mentions(call: ToolCallEvent) -> bool:
        said = json.dumps([call.arguments, call.result], default=str)
        return any(key in said for key in keys)

    return next(filter(mentions, calls), calls[0] if calls else None)


def _change_evidence(change: DbChange, calls: list[ToolCallEvent]) -> list[Evidence]:
    call = _caused_by(change, calls)
    if call is None:
        return []
    return [Evidence(quote=f"{call.name}({json.dumps(call.arguments)})", start_s=call.at_s)]


def _breach_evidence(hits: list[DbChange], calls: list[ToolCallEvent]) -> list[Evidence]:
    """When a forbidden change happened, earliest first, once per tool call."""
    found = [e for change in hits for e in _change_evidence(change, calls)]
    unique = {(e.quote, e.start_s): e for e in found}
    return sorted(unique.values(), key=lambda e: e.start_s or 0.0)


def _failure_reason(expectation: StateExpectation, hits: list[DbChange]) -> str:
    if expectation.op == "no_change":
        return f"{len(hits)} matching row(s) changed: {[c.key for c in hits]}"
    return f"no change to {expectation.table} matched"


class StateEvaluator:
    name = "state"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        scenario = ctx.scenario
        if scenario is None:
            raise Skip("no scenario, so there is no expected state")
        if not ctx.artifact.has_tool_log:
            # Left unknown, not absent: the scenario did expect these, and they went unchecked.
            out.criteria.extend(
                CriterionResult(
                    criterion_id=e.id,
                    kind="state",
                    text=_describe(e),
                    status="unknown",
                    reason="no tool log",
                )
                for e in scenario.expected_state
            )
            raise Skip("no tool log, so database changes are unknown")

        changes = ctx.artifact.db_changes
        writes = _writing_calls(ctx.artifact)
        claimed: list[DbChange] = []
        for expectation in scenario.expected_state:
            hits = satisfying_changes(expectation, changes)
            met = not hits if expectation.op == "no_change" else bool(hits)
            text = _describe(expectation)
            claimed.extend(hits)
            reason = "" if met else _failure_reason(expectation, hits)
            evidence = _breach_evidence(hits, writes) if not met else []
            out.criteria.append(
                CriterionResult(
                    criterion_id=expectation.id,
                    kind="state",
                    text=text,
                    status="pass" if met else "fail",
                    reason=reason,
                    evidence=evidence,
                )
            )
            if not met:
                out.findings.append(
                    Finding(
                        evaluator="state",
                        category="state_expectation",
                        severity="major",
                        title=f"State expectation failed: {text}",
                        detail=f"{text}: {reason}",
                        evidence=evidence,
                        criterion_id=expectation.id,
                    )
                )
        ignored = set(ctx.config.ignored_state_tables)
        unexpected = [
            c
            for c in changes
            if c.table not in ignored and not any(c is claim for claim in claimed)
        ]
        out.findings.extend(
            _unexpected_finding(c, scenario.customer_id, writes) for c in unexpected
        )
        out.findings[:] = roll_up(out.findings, key=_rollup_key, title=_rollup_title)
        out.metrics.update(
            {
                "expectations": len(scenario.expected_state),
                "failed": sum(c.status == "fail" for c in out.criteria),
                "db_changes": len(changes),
                "unexpected_changes": len(unexpected),
            }
        )


def _rollup_key(finding: Finding) -> str:
    # Each failed expectation still has its own criterion. A serious change stands alone.
    return (
        finding.id
        if finding.severity == "major" and not _is_expectation(finding)
        else (finding.category)
    )


def _is_expectation(finding: Finding) -> bool:
    return finding.category == "state_expectation"


def _rollup_title(group: list[Finding]) -> str:
    if _is_expectation(group[0]):
        return f"{len(group)} state expectations failed"
    return f"{len(group)} unexpected side effects"


def _touches_other_customer(change: DbChange, customer_id: str | None) -> bool:
    row = change.after or change.before or {}
    owner = change.key.get("id") if change.table == "customers" else row.get("customer_id")
    return customer_id is not None and owner is not None and owner != customer_id


def _unexpected_finding(
    change: DbChange, customer_id: str | None, writes: list[ToolCallEvent]
) -> Finding:
    # Advisory when minor, because scenarios don't list every side effect: paying an invoice also
    # reactivates the customer, and a correct refusal often opens a ticket. A deletion or a write
    # to someone else's account has no innocent reading, so that's a commission and fails the call.
    serious = change.kind == "delete" or _touches_other_customer(change, customer_id)
    return Finding(
        evaluator="state",
        category=UNEXPECTED_CHANGE,
        severity="major" if serious else "minor",
        advisory=not serious,
        title=f"Unexpected {change.kind} on {change.table}",
        detail=f"key={change.key} before={change.before} after={change.after}",
        evidence=_change_evidence(change, writes),
    )
