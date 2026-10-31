"""Scenario state assertions against the call's database diff. No model is involved."""

from __future__ import annotations

from typing import Any

from voicelab.core.models import CriterionResult, DbChange, Finding, StateExpectation
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip


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
        claimed: list[DbChange] = []
        for expectation in scenario.expected_state:
            hits = satisfying_changes(expectation, changes)
            met = not hits if expectation.op == "no_change" else bool(hits)
            text = _describe(expectation)
            claimed.extend(hits)
            reason = "" if met else _failure_reason(expectation, hits)
            out.criteria.append(
                CriterionResult(
                    criterion_id=expectation.id,
                    kind="state",
                    text=text,
                    status="pass" if met else "fail",
                    reason=reason,
                )
            )
            if not met:
                out.findings.append(
                    Finding(
                        evaluator="state",
                        category="state_expectation",
                        severity="major",
                        title=f"State expectation failed: {text}",
                        detail=reason,
                        criterion_id=expectation.id,
                    )
                )
        ignored = set(ctx.config.ignored_state_tables)
        unexpected = [
            c
            for c in changes
            if c.table not in ignored and not any(c is claim for claim in claimed)
        ]
        out.findings.extend(_unexpected_finding(c) for c in unexpected)
        out.metrics.update(
            {
                "expectations": len(scenario.expected_state),
                "failed": sum(c.status == "fail" for c in out.criteria),
                "db_changes": len(changes),
                "unexpected_changes": len(unexpected),
            }
        )


def _unexpected_finding(change: DbChange) -> Finding:
    # Advisory because scenarios don't list every harmless write, such as opening a ticket. An
    # unlisted write still deserves a person's eyes, so it counts toward review at major.
    return Finding(
        evaluator="state",
        category="unexpected_change",
        severity="major",
        advisory=True,
        title=f"Unexpected {change.kind} on {change.table}",
        detail=f"key={change.key} before={change.before} after={change.after}",
    )
