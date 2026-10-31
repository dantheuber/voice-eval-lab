"""Personas, scenarios, voices, customers: what tests are built from."""

from __future__ import annotations

import logging
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Body, Response

from voicelab.agents.seeds import scenario_problems
from voicelab.api import mapping
from voicelab.api.schemas import DraftBody, DryRunBody, Svc, TtsPreview
from voicelab.core.models import LineConditions, Persona, Scenario, VoiceRef
from voicelab.services import authoring
from voicelab.services.errors import Conflict, Invalid, ModelNotReady, NotFound

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

JsonBody = Annotated[dict[str, Any], Body()]
_SERVER_ASSIGNED = {"id", "createdAt", "updatedAt"}


def _body_fields(body: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key not in _SERVER_ASSIGNED}


def _persona(svc: Svc, persona_id: str) -> Persona:
    persona = svc.catalog.persona(persona_id)
    if persona is None:
        raise NotFound(f"no persona {persona_id}")
    return persona


def _scenario(svc: Svc, scenario_id: str) -> tuple[Scenario, int]:
    found = svc.catalog.scenario(scenario_id)
    if found is None:
        raise NotFound(f"no scenario {scenario_id}")
    return found


def _check_references(scenario: Scenario) -> None:
    problems = scenario_problems(scenario)
    if problems:
        raise Invalid("; ".join(problems), {"problems": problems})


def _recent(svc: Svc, column: str, value: str) -> list[dict[str, Any]]:
    return [mapping.call_summary(row) for row in svc.calls.recent(column, value)]


# Personas ----------------------------------------------------------------------------------------


@router.get("/personas")
def list_personas(svc: Svc) -> dict[str, Any]:
    usage = svc.calls.usage("persona_id")
    return {
        "items": [
            {**mapping.to_api(p), "usage": mapping.usage(usage.get(p.id))}
            for p in svc.catalog.personas()
        ]
    }


@router.post("/personas", status_code=201)
def create_persona(svc: Svc, body: JsonBody) -> dict[str, Any]:
    persona = mapping.from_api(Persona, _body_fields(body))
    return mapping.to_api(svc.catalog.add_persona(persona))


@router.post("/personas/prompt")
def preview_persona_prompt(body: JsonBody) -> dict[str, str]:
    """The same text as the saved route, for a form that hasn't been saved yet."""
    return {"prompt": authoring.persona_prompt(mapping.from_api(Persona, _body_fields(body)))}


@router.get("/personas/{persona_id}")
def get_persona(persona_id: str, svc: Svc) -> dict[str, Any]:
    persona = _persona(svc, persona_id)
    return {
        **mapping.to_api(persona),
        "usage": mapping.usage(svc.calls.usage("persona_id").get(persona_id)),
        "recentCalls": _recent(svc, "persona_id", persona_id),
    }


@router.put("/personas/{persona_id}")
def replace_persona(persona_id: str, svc: Svc, body: JsonBody) -> dict[str, Any]:
    _persona(svc, persona_id)
    persona = mapping.from_api(Persona, _body_fields(body), id=persona_id)
    return mapping.to_api(svc.catalog.replace_persona(persona))


@router.delete("/personas/{persona_id}", status_code=204)
def delete_persona(persona_id: str, svc: Svc) -> Response:
    _persona(svc, persona_id)
    if svc.runs.uses("persona_id", persona_id):
        raise Conflict("a run that hasn't finished uses this persona")
    svc.catalog.delete_persona(persona_id)
    return Response(status_code=204)


@router.get("/personas/{persona_id}/prompt")
def persona_prompt(persona_id: str, svc: Svc) -> dict[str, str]:
    return {"prompt": authoring.persona_prompt(_persona(svc, persona_id))}


# Voices ------------------------------------------------------------------------------------------


@router.get("/voices")
async def list_voices(svc: Svc) -> dict[str, Any]:
    profile = svc.profiles.items.get(svc.profiles.active_id or "")
    if profile is None:
        return {"items": []}
    try:
        return {"items": await authoring.list_voices(svc.profiles.stack(profile))}
    except (ModelNotReady, httpx.HTTPError, OSError) as exc:
        # The persona form stays usable: voice ids can still be typed while TTS loads.
        logger.warning("can't list voices: %s", exc)
        return {"items": []}


@router.post("/tts/preview")
async def tts_preview(body: TtsPreview, svc: Svc) -> Response:
    stack = await svc.profiles.require_ready(svc.profiles.resolve(None), ("tts_caller",))
    voice = mapping.from_api(VoiceRef, body.voice)
    line = mapping.from_api(LineConditions, body.line) if body.line else None
    wav = await authoring.preview_voice(stack, body.text, voice, line)
    return Response(wav, media_type="audio/wav")


# Scenarios ---------------------------------------------------------------------------------------


def _scenario_json(
    scenario: Scenario, version: int, usage: dict[str, Any] | None
) -> dict[str, Any]:
    return {**mapping.to_api(scenario), "usage": mapping.usage(usage), "version": version}


@router.get("/scenarios")
def list_scenarios(svc: Svc) -> dict[str, Any]:
    usage = svc.calls.usage("scenario_id")
    return {
        "items": [
            _scenario_json(scenario, version, usage.get(scenario.id))
            for scenario, version in svc.catalog.scenarios()
        ]
    }


@router.post("/scenarios", status_code=201)
def create_scenario(svc: Svc, body: JsonBody) -> dict[str, Any]:
    scenario = mapping.from_api(Scenario, _body_fields(body))
    _check_references(scenario)
    return mapping.to_api(svc.catalog.add_scenario(scenario))


@router.post("/scenarios/draft")
async def draft_scenario(body: DraftBody, svc: Svc) -> dict[str, Any]:
    stack = await svc.profiles.require_ready(svc.profiles.resolve(None), ("llm_judge",))
    base = _scenario(svc, body.base_scenario_id)[0] if body.base_scenario_id else None
    scenario, guessed, notes = await authoring.draft(
        stack.llm_judge, body.description, body.customer_id, base
    )
    shaped = mapping.to_api(scenario)
    for assigned in _SERVER_ASSIGNED:
        shaped.pop(assigned)
    return {
        "scenario": shaped,
        "guessedFields": guessed,
        "notes": notes,
        "model": stack.llm_judge.name,
    }


@router.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str, svc: Svc) -> dict[str, Any]:
    scenario, version = _scenario(svc, scenario_id)
    usage = svc.calls.usage("scenario_id").get(scenario_id)
    return {
        **_scenario_json(scenario, version, usage),
        "recentCalls": _recent(svc, "scenario_id", scenario_id),
    }


@router.put("/scenarios/{scenario_id}")
def replace_scenario(scenario_id: str, svc: Svc, body: JsonBody) -> dict[str, Any]:
    _scenario(svc, scenario_id)
    scenario = mapping.from_api(Scenario, _body_fields(body), id=scenario_id)
    _check_references(scenario)
    saved, _ = svc.catalog.replace_scenario(scenario)
    return mapping.to_api(saved)


@router.delete("/scenarios/{scenario_id}", status_code=204)
def delete_scenario(scenario_id: str, svc: Svc) -> Response:
    _scenario(svc, scenario_id)
    if svc.runs.uses("scenario_id", scenario_id):
        raise Conflict("a run that hasn't finished uses this scenario")
    svc.catalog.delete_scenario(scenario_id)
    return Response(status_code=204)


@router.post("/scenarios/{scenario_id}/dry-run")
async def dry_run(scenario_id: str, body: DryRunBody, svc: Svc) -> dict[str, Any]:
    scenario, _ = _scenario(svc, scenario_id)
    persona = _persona(svc, body.persona_id) if body.persona_id else authoring.neutral_persona()
    profile = svc.profiles.resolve(None)
    stack = await svc.profiles.require_ready(profile, ("llm_support", "llm_caller"))
    result = await authoring.dry_run(
        stack.llm_support,
        stack.llm_caller,
        scenario,
        persona,
        argument_guard=profile.support_bot.argument_guard,
    )
    return {
        "turns": authoring.dry_run_turns(result),
        "reachedCheckpoints": result.completed_beats,
        "model": stack.llm_support.name,
    }


# Customers ---------------------------------------------------------------------------------------


@router.get("/customers")
def list_customers() -> dict[str, Any]:
    return {"items": authoring.list_customers()}


@router.get("/customers/{customer_id}")
def get_customer(customer_id: str) -> dict[str, Any]:
    detail = authoring.customer_detail(customer_id)
    if detail is None:
        raise NotFound(f"no customer {customer_id}")
    return detail
