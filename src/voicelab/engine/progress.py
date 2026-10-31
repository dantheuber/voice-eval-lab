"""Messages a running call sends to whoever is watching, shaped as in docs/API.md.

The engine has no verdict or batch position, so those `CallSummary` fields stay empty here and the
API layer fills them in. Only keys of our own models become camelCase, because tool arguments and
results are data with their keys intact.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic.alias_generators import to_camel

from voicelab.core.models import CallArtifact, Speaker, Utterance


def _dump(model: BaseModel) -> dict[str, Any]:
    return {to_camel(key): value for key, value in model.model_dump(mode="json").items()}


def call_summary(artifact: CallArtifact) -> dict[str, Any]:
    persona, scenario = artifact.persona_snapshot, artifact.scenario_snapshot
    return {
        "id": artifact.id,
        "batchId": artifact.batch_id,
        "rep": None,
        "scenarioId": artifact.scenario_id,
        "scenarioTitle": scenario.title if scenario else None,
        "personaId": artifact.persona_id,
        "personaName": persona.name if persona else None,
        "mode": artifact.mode.value,
        "status": artifact.status.value,
        "profile": artifact.profile,
        "verdict": None,
        "humanVerdict": None,
        "invalidReason": None,
        "findings": 0,
        "openFindings": 0,
        "durationS": artifact.duration_s,
        "replyP50S": None,
        "overlaps": None,
        "startedAt": artifact.started_at.isoformat() if artifact.started_at else None,
        "timingIsReal": artifact.timing_is_real,
    }


def call_started(artifact: CallArtifact) -> dict[str, Any]:
    return {"type": "callStarted", "call": call_summary(artifact)}


def call_done(artifact: CallArtifact) -> dict[str, Any]:
    return {"type": "callDone", "call": call_summary(artifact)}


def turn(call_id: str, utterance: Utterance) -> dict[str, Any]:
    return {"type": "turn", "callId": call_id, "utterance": _dump(utterance)}


def partial(call_id: str, speaker: Speaker, text: str, at_s: float) -> dict[str, Any]:
    return {
        "type": "partial",
        "callId": call_id,
        "speaker": speaker.value,
        "text": text,
        "atS": at_s,
    }


def event(call_id: str, item: BaseModel) -> dict[str, Any]:
    return {"type": "event", "callId": call_id, "event": _dump(item)}
