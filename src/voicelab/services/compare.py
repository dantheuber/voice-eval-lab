"""Compares two runs. Pure: takes loaded calls and returns the /api/compare body.

Invalid calls test the caller, not the bot, so they leave every denominator here and show up only
as their own measure. Only scenarios both runs have are compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any

from voicelab.core.models import CallArtifact, CriterionResult, EvalResult, Verdict
from voicelab.store.calls import reply_p50


@dataclass
class ComparedCall:
    row: dict[str, Any]
    artifact: CallArtifact
    result: EvalResult


def _valid(calls: list[ComparedCall]) -> list[ComparedCall]:
    return [c for c in calls if c.result.verdict != Verdict.INVALID]


def _metric(call: ComparedCall, evaluator: str, key: str) -> Any:
    run = next((r for r in call.result.evaluators if r.name == evaluator), None)
    return run.metrics.get(key) if run else None


def _criterion_rate(calls: list[ComparedCall]) -> dict[str, dict[str, Any]]:
    # Beats measure the caller, and Compare is about the bot.
    tally: dict[str, dict[str, Any]] = {}
    for call in calls:
        for criterion in call.result.criteria:
            if criterion.kind == "beat" or criterion.status == "unknown":
                continue
            entry = tally.setdefault(
                criterion.criterion_id, {"criterion": criterion, "pass": 0, "n": 0, "failed": []}
            )
            entry["n"] += 1
            if criterion.status == "pass":
                entry["pass"] += 1
            else:
                entry["failed"].append(call.row["id"])
    return tally


def _fraction(entry: dict[str, Any] | None) -> dict[str, int]:
    return {"pass": entry["pass"], "n": entry["n"]} if entry else {"pass": 0, "n": 0}


def _criteria(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for criterion_id in sorted(a.keys() | b.keys()):
        found: CriterionResult = (a.get(criterion_id) or b[criterion_id])["criterion"]
        rows.append(
            {
                "criterionId": criterion_id,
                "text": found.text,
                "kind": found.kind,
                "a": _fraction(a.get(criterion_id)),
                "b": _fraction(b.get(criterion_id)),
            }
        )
    return rows


def _worst(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    worst: dict[str, Any] | None = None
    for criterion_id in a.keys() & b.keys():
        before, after = a[criterion_id], b[criterion_id]
        delta = round(100 * (after["pass"] / after["n"] - before["pass"] / before["n"]))
        if delta < 0 and (worst is None or delta < worst["deltaPts"]):
            worst = {
                "criterionId": criterion_id,
                "deltaPts": delta,
                "callIds": after["failed"],
            }
    return worst


def _cells(calls: list[ComparedCall]) -> dict[tuple[str, str], dict[str, int]]:
    cells: dict[tuple[str, str], dict[str, int]] = {}
    for call in calls:
        key = (call.row["scenario_id"], call.row["persona_id"])
        cell = cells.setdefault(key, {"pass": 0, "n": 0})
        cell["n"] += 1
        cell["pass"] += call.result.verdict == Verdict.PASS
    return cells


def _median_reply(calls: list[ComparedCall]) -> float | None:
    values = [reply_p50(c.artifact, c.result) for c in calls]
    measured = [value for value in values if value is not None]
    return round(median(measured), 3) if measured else None


def _mean_mos(calls: list[ComparedCall]) -> float | None:
    values = [_metric(c, "audio_quality", "mos") for c in calls]
    measured = [value for value in values if value is not None]
    return round(mean(measured), 2) if measured else None


def _sum_metric(calls: list[ComparedCall], evaluator: str, key: str) -> int:
    return sum(_metric(c, evaluator, key) or 0 for c in calls)


def _measures(
    a: list[ComparedCall],
    b: list[ComparedCall],
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    invalid: tuple[int, int],
) -> list[dict[str, Any]]:
    turn_based = "turn_based" in (run_a["mode"], run_b["mode"])
    reply_a, reply_b = _median_reply(a), _median_reply(b)
    reply: dict[str, Any] = {
        "key": "replyP50S",
        "label": "Bot reply latency p50",
        "a": reply_a,
        "b": reply_b,
        "comparable": not turn_based and reply_a is not None and reply_b is not None,
        "lowerIsBetter": True,
    }
    if turn_based:
        reply["reason"] = "turn-based"
    elif not reply["comparable"]:
        reply["reason"] = "not measured"
    mos_a, mos_b = _mean_mos(a), _mean_mos(b)
    mos: dict[str, Any] = {
        "key": "mos",
        "label": "MOS, bot channel",
        "a": mos_a,
        "b": mos_b,
        "comparable": mos_a is not None and mos_b is not None,
        "lowerIsBetter": False,
    }
    if not mos["comparable"]:
        mos["reason"] = "not measured"
    return [
        reply,
        {
            "key": "criticalMispronunciations",
            "label": "Critical token mispronunciations",
            "a": _sum_metric(a, "pronunciation", "critical_mismatches"),
            "b": _sum_metric(b, "pronunciation", "critical_mismatches"),
            "comparable": True,
            "lowerIsBetter": True,
        },
        {
            "key": "missingStateChanges",
            "label": "Expected db changes missing",
            "a": _sum_metric(a, "state", "failed"),
            "b": _sum_metric(b, "state", "failed"),
            "comparable": True,
            "lowerIsBetter": True,
        },
        mos,
        {
            "key": "advisoryNotes",
            "label": "Advisory listener notes",
            "a": sum(f.advisory for c in a for f in c.result.findings),
            "b": sum(f.advisory for c in b for f in c.result.findings),
            "comparable": False,
            "reason": "advisory",
        },
        {
            "key": "invalid",
            "label": "Invalid runs",
            "a": invalid[0],
            "b": invalid[1],
            "comparable": False,
            "reason": "test, not bot",
        },
    ]


def compare_runs(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    calls_a: list[ComparedCall],
    calls_b: list[ComparedCall],
) -> dict[str, Any]:
    ids_a, ids_b = run_a["scenarioIds"], run_b["scenarioIds"]
    shared = [scenario for scenario in ids_a if scenario in ids_b]
    dropped = sorted(set(ids_a) ^ set(ids_b))
    in_both = [c for c in calls_a if c.row["scenario_id"] in shared]
    other = [c for c in calls_b if c.row["scenario_id"] in shared]
    valid_a, valid_b = _valid(in_both), _valid(other)
    rates_a, rates_b = _criterion_rate(valid_a), _criterion_rate(valid_b)
    cells_a, cells_b = _cells(valid_a), _cells(valid_b)
    empty = {"pass": 0, "n": 0}
    invalid = (len(in_both) - len(valid_a), len(other) - len(valid_b))
    return {
        "a": run_a,
        "b": run_b,
        "scenarioIds": shared,
        "droppedScenarioIds": dropped,
        "criteria": _criteria(rates_a, rates_b),
        "measures": _measures(valid_a, valid_b, run_a, run_b, invalid),
        "cells": [
            {
                "scenarioId": scenario_id,
                "personaId": persona_id,
                "a": cells_a.get((scenario_id, persona_id), empty),
                "b": cells_b.get((scenario_id, persona_id), empty),
            }
            for scenario_id, persona_id in sorted(cells_a.keys() | cells_b.keys())
        ],
        "worst": _worst(rates_a, rates_b),
    }
