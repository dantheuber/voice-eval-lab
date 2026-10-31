"""Runs, presets, the launcher matrix, and Compare."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from voicelab.api import mapping
from voicelab.api.routes.calls import since
from voicelab.api.schemas import PresetBody, RunBody, Svc
from voicelab.core.models import LineConditions
from voicelab.services import authoring
from voicelab.services.compare import ComparedCall, compare_runs
from voicelab.services.errors import Invalid, NotFound
from voicelab.services.runs import RunPlan
from voicelab.store.calls import CallFilters

router = APIRouter(prefix="/api")


def _line(value: str | dict[str, Any]) -> LineConditions | None:
    if value == "persona":
        return None
    if isinstance(value, dict):
        return mapping.from_api(LineConditions, value)
    raise Invalid(f"line must be 'persona' or line conditions, not {value!r}")


def _require_run(svc: Svc, run_id: str) -> dict[str, Any]:
    run = svc.runs.get(run_id)
    if run is None:
        raise NotFound(f"no run {run_id}")
    return run


@router.get("/matrix")
def matrix(svc: Svc, profile: str | None = None, window: str = "7d") -> dict[str, Any]:
    profile = profile or svc.profiles.active_id
    filters = CallFilters(profile=[profile] if profile else [], since=since(window))
    groups = svc.calls.counts_by("scenario_id", "persona_id", filters=filters)
    return {
        "cells": [
            {
                "scenarioId": group["scenario_id"],
                "personaId": group["persona_id"],
                "counts": mapping.counts(group["counts"]),
            }
            for group in groups
            if group["scenario_id"] and group["persona_id"]
        ]
    }


@router.post("/runs", status_code=201)
async def create_run(body: RunBody, svc: Svc) -> dict[str, Any]:
    profile = body.profile or svc.profiles.active_id
    if profile is None:
        raise Invalid("no model profile in config/profiles")
    personas = [svc.catalog.persona(pair.persona_id) for pair in body.pairs]
    stack = svc.profiles.stack(svc.profiles.require(profile))
    voiceless = await authoring.voiceless_personas(stack, [p for p in personas if p is not None])
    if voiceless:
        names = ", ".join(sorted(set(voiceless)))
        raise Invalid(f"{profile} has no voice for {names}. Pick one of its voices on the persona.")
    plan = RunPlan(
        pairs=[(pair.scenario_id, pair.persona_id) for pair in body.pairs],
        reps=body.reps,
        profile=profile,
        mode=body.mode,
        line=_line(body.line),
        concurrency=body.concurrency,
        label=body.label,
    )
    run_id, mode = await svc.worker.submit(plan)
    summary = svc.worker.summary(run_id)
    return {"run": summary, "effectiveMode": mode.value, "estimateS": summary["etaS"]}


@router.get("/runs")
def list_runs(
    svc: Svc,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    runs, next_cursor = svc.runs.page(cursor, limit)
    return {"items": [svc.worker.summary(run["id"]) for run in runs], "nextCursor": next_cursor}


@router.get("/runs/{run_id}")
def get_run(run_id: str, svc: Svc) -> dict[str, Any]:
    _require_run(svc, run_id)
    return {
        "run": svc.worker.summary(run_id),
        "calls": [mapping.call_summary(row) for row in svc.calls.in_batch(run_id)],
    }


@router.post("/runs/{run_id}/pause")
async def pause_run(run_id: str, svc: Svc) -> dict[str, Any]:
    _require_run(svc, run_id)
    svc.worker.pause(run_id)
    return svc.worker.summary(run_id)


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, svc: Svc) -> dict[str, Any]:
    _require_run(svc, run_id)
    svc.worker.resume(run_id)
    return svc.worker.summary(run_id)


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str, svc: Svc) -> dict[str, Any]:
    _require_run(svc, run_id)
    svc.worker.stop(run_id)
    return svc.worker.summary(run_id)


# Presets -----------------------------------------------------------------------------------------


def _preset_from_run(run: dict[str, Any]) -> dict[str, Any]:
    line = mapping.to_api(LineConditions.model_validate(run["line"])) if run["line"] else "persona"
    return {
        "name": run["label"],
        "pairs": [
            {"scenarioId": pair["scenario_id"], "personaId": pair["persona_id"]}
            for pair in run["pairs"]
        ],
        "reps": run["reps"],
        "profile": run["profile"],
        "mode": run["mode"],
        "line": line,
        "concurrency": run["concurrency"],
    }


@router.get("/runs/{run_id}/presets")
def run_presets(run_id: str, svc: Svc) -> dict[str, Any]:
    """This run's own settings in preset form, so the launcher can repeat it."""
    return {"items": [_preset_from_run(_require_run(svc, run_id))]}


@router.get("/presets")
def list_presets(svc: Svc) -> dict[str, Any]:
    return {"items": svc.catalog.presets()}


@router.post("/presets", status_code=201)
def create_preset(body: PresetBody, svc: Svc) -> dict[str, Any]:
    preset = body.model_dump(by_alias=True)
    return svc.catalog.add_preset(preset)


# Compare -----------------------------------------------------------------------------------------


def _evaluated_calls(svc: Svc, run_id: str) -> list[ComparedCall]:
    calls = []
    for row in svc.calls.in_batch(run_id):
        result = svc.store.read_eval(row["id"])
        if result is not None and row["error"] is None:
            calls.append(ComparedCall(row, svc.store.read_artifact(row["id"]), result))
    return calls


@router.get("/compare")
def compare(a: str, b: str, svc: Svc) -> dict[str, Any]:
    _require_run(svc, a)
    _require_run(svc, b)
    return compare_runs(
        svc.worker.summary(a),
        svc.worker.summary(b),
        _evaluated_calls(svc, a),
        _evaluated_calls(svc, b),
    )
