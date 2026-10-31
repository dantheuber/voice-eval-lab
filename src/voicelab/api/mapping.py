"""Core models to the JSON shapes in docs/API.md, and back. Pure: no I/O, no app state.

Field names are camelized walking the models, never by rewriting a dump, because dict contents
such as scenario `facts`, tool `arguments`, and `DbChange.key` hold user data and column names
that must keep their spelling.
"""

from __future__ import annotations

import dataclasses
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import PurePath
from typing import Any, get_args

from pydantic import BaseModel, ValidationError
from pydantic.alias_generators import to_camel

from voicelab.core.models import (
    CallArtifact,
    CallMode,
    EvalResult,
    HumanVerdict,
    Scenario,
    Utterance,
)
from voicelab.services.errors import Invalid
from voicelab.store.artifacts import CALL_SAMPLE_RATE
from voicelab.store.calls import timing_metrics

ESTIMATE_PER_CALL_S = {"live_bots": 105.0, "turn_based": 210.0}  # until a run has measured its own


def iso(moment: datetime | str | None) -> str | None:
    if moment is None:
        return None
    text = moment if isinstance(moment, str) else moment.isoformat()
    return text.replace("+00:00", "Z")


def to_api(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {to_camel(name): to_api(getattr(value, name)) for name in type(value).model_fields}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            to_camel(item.name): to_api(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {key: to_api(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_api(item) for item in value]
    if isinstance(value, datetime):
        return iso(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    return value


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        found = _nested_model(argument)
        if found is not None:
            return found
    return None


def snake_fields(model: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    """Rename a camelCase body to `model`'s fields. Unknown keys go: clients echo `usage` back."""
    names = {to_camel(name): name for name in model.model_fields}
    fields: dict[str, Any] = {}
    for key, value in data.items():
        name = names.get(key)
        if name is None:
            continue
        nested = _nested_model(model.model_fields[name].annotation)
        if nested is not None and isinstance(value, dict):
            value = snake_fields(nested, value)
        elif nested is not None and isinstance(value, list):
            value = [snake_fields(nested, v) if isinstance(v, dict) else v for v in value]
        fields[name] = value
    return fields


def from_api[T: BaseModel](model: type[T], data: dict[str, Any], **overrides: Any) -> T:
    try:
        return model.model_validate({**snake_fields(model, data), **overrides})
    except ValidationError as exc:
        problems = [
            {"field": ".".join(str(part) for part in e["loc"]), "problem": e["msg"]}
            for e in exc.errors()
        ]
        raise Invalid(f"invalid {model.__name__}", {"problems": problems}) from exc


# Lists -----------------------------------------------------------------------------------------


def counts(by_verdict: dict[str, int] | None) -> dict[str, int]:
    tally = by_verdict or {}
    return {
        "pass": tally.get("pass", 0),
        "fail": tally.get("fail", 0),
        "needsReview": tally.get("needs_review", 0),
        "invalid": tally.get("invalid", 0),
    }


def usage(entry: dict[str, Any] | None) -> dict[str, Any]:
    entry = entry or {}
    return {"calls": entry.get("calls", 0), "counts": counts(entry.get("counts"))}


def call_summary(row: dict[str, Any]) -> dict[str, Any]:
    real = bool(row["timing_is_real"])
    return {
        "id": row["id"],
        "batchId": row["batch_id"],
        "rep": row["rep"],
        "scenarioId": row["scenario_id"],
        "scenarioTitle": row["scenario_title"],
        "personaId": row["persona_id"],
        "personaName": row["persona_name"],
        "mode": row["mode"],
        "status": row["status"],
        "profile": row["profile"],
        "verdict": row["verdict"],
        "humanVerdict": row["human_verdict"],
        "invalidReason": row["invalid_reason"],
        "findings": row["findings"],
        "openFindings": row["open_findings"],
        "durationS": row["duration_s"],
        "replyP50S": row["reply_p50_s"] if real else None,
        "overlaps": row["overlaps"] if real else None,
        "startedAt": iso(row["started_at"]),
        "timingIsReal": real,
    }


def eta_s(run: dict[str, Any], progress: dict[str, Any], mean_call_s: float | None) -> float | None:
    """Time left at the rate this run has shown, or the mode's usual rate before it has one."""
    if run["status"] not in ("queued", "running"):
        return None
    remaining = progress["calls"] - progress["done"]
    per_call = mean_call_s or ESTIMATE_PER_CALL_S.get(run["mode"])
    if per_call is None:
        return None
    return round(remaining * per_call / max(run["concurrency"], 1), 1)


def run_summary(
    run: dict[str, Any], progress: dict[str, Any], mean_call_s: float | None
) -> dict[str, Any]:
    pairs = run["pairs"]
    return {
        "id": run["id"],
        "label": run["label"],
        "profile": run["profile"],
        "mode": run["mode"],
        "status": run["status"],
        "startedAt": iso(run["started_at"]),
        "endedAt": iso(run["ended_at"]),
        "calls": progress["calls"],
        "done": progress["done"],
        "running": progress["running"],
        "counts": counts(progress["counts"]),
        "etaS": eta_s(run, progress, mean_call_s),
        "scenarioIds": list(dict.fromkeys(p["scenario_id"] for p in pairs)),
        "personaIds": list(dict.fromkeys(p["persona_id"] for p in pairs)),
        "reps": run["reps"],
    }


def queue_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "callId": row["id"],
        "scenarioId": row["scenario_id"],
        "scenarioTitle": row["scenario_title"],
        "personaName": row["persona_name"],
        "openFindings": row["open_findings"],
        "why": row["why"],
        "createdAt": iso(row["created_at"]),
    }


# Review payload --------------------------------------------------------------------------------


def apply_decisions(result: EvalResult, decisions: dict[str, dict[str, Any]]) -> EvalResult:
    """Lay stored human decisions over the machine's findings. eval.json stays machine output."""
    findings = []
    for finding in result.findings:
        decision = decisions.get(finding.id)
        if decision and decision["human_verdict"]:
            finding = finding.model_copy(
                update={
                    "human_verdict": HumanVerdict(decision["human_verdict"]),
                    "human_note": decision["human_note"],
                    "human_at": datetime.fromisoformat(decision["human_at"]),
                }
            )
        findings.append(finding)
    return result.model_copy(update={"findings": findings})


def _transcript_words(result: EvalResult | None) -> dict[str, list[dict[str, Any]]]:
    run = next((r for r in result.evaluators if r.name == "transcribe"), None) if result else None
    stored = run.metrics.get("transcript", {}) if run else {}
    return {speaker: value.get("words", []) for speaker, value in stored.items()}


def timeline_json(artifact: CallArtifact, result: EvalResult | None) -> list[dict[str, Any]]:
    """Add the independent transcript's words to each utterance.

    Uploads have regions but no text, so `heardText` comes from these words. It stays null when
    the call wasn't transcribed, because an empty string would claim silence.
    """
    words = _transcript_words(result)
    events = []
    for event in artifact.timeline:
        item = to_api(event)
        if isinstance(event, Utterance):
            inside = [
                w
                for w in words.get(event.speaker.value, [])
                if event.start_s <= (w["start_s"] + w["end_s"]) / 2 < event.end_s
            ]
            if inside:
                item["words"] = [
                    {
                        "text": w["text"],
                        "startS": w["start_s"],
                        "endS": w["end_s"],
                        "confidence": w.get("confidence"),
                    }
                    for w in inside
                ]
                if item["heardText"] is None:
                    item["heardText"] = " ".join(w["text"] for w in inside)
        events.append(item)
    return events


def timing_block(artifact: CallArtifact, result: EvalResult | None) -> dict[str, Any] | None:
    metrics = timing_metrics(result)
    if metrics is None or not artifact.timing_is_real:
        return None
    latency = metrics.get("response_latency") or {}
    return {
        "replyP50S": latency.get("median_s"),
        "replyP95S": latency.get("p95_s"),
        "overlaps": (metrics.get("overlap") or {}).get("count"),
        "deadAirEvents": (metrics.get("dead_air") or {}).get("count_over_threshold"),
        "bargeStopS": (metrics.get("barge_in") or {}).get("median_reaction_s"),
    }


def evidence_flags(
    artifact: CallArtifact, result: EvalResult | None, audio_available: bool
) -> dict[str, bool]:
    utterances = [e for e in artifact.timeline if isinstance(e, Utterance)]
    spoken = any(u.heard_text or u.spoken_text for u in utterances)
    return {
        "audio": audio_available,
        "transcript": spoken or any(_transcript_words(result).values()),
        "intendedText": artifact.has_intended_text,
        "toolLog": artifact.has_tool_log,
        # Only an outside recording happens without a Fernhill database behind it.
        "dbSnapshots": artifact.mode != CallMode.UPLOAD,
        "scenario": artifact.scenario_id is not None,
    }


def audio_block(call_id: str, available: bool) -> dict[str, Any]:
    url = (lambda name: f"/api/calls/{call_id}/audio/{name}.wav") if available else (lambda _: None)
    return {
        "stereoUrl": url("recording"),
        "callerUrl": url("caller"),
        "botUrl": url("bot"),
        "sampleRate": CALL_SAMPLE_RATE,
        "available": available,
    }


def peaks_block(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    channels = raw["channels"]
    return {
        "perSecond": raw["peaks_per_second"],
        "caller": channels["caller"],
        "bot": channels["bot"],
    }


def review_payload(
    *,
    row: dict[str, Any],
    artifact: CallArtifact,
    result: EvalResult | None,
    decisions: dict[str, dict[str, Any]],
    peaks: dict[str, Any] | None,
    speech: dict[str, list[list[float]]],
    overlaps: list[list[float]],
    audio_available: bool,
    prev_id: str | None,
    next_id: str | None,
) -> dict[str, Any]:
    in_progress = artifact.status.value in ("running", "evaluating")
    shown = None if in_progress or result is None else apply_decisions(result, decisions)
    call = to_api(artifact)
    call["timeline"] = timeline_json(artifact, shown)
    return {
        "call": call,
        "eval": to_api(shown) if shown else None,
        "summary": call_summary(row),
        "audio": audio_block(artifact.id, audio_available),
        "peaks": peaks_block(peaks),
        "speech": speech,
        "overlaps": overlaps,
        "timing": timing_block(artifact, shown),
        "evidence": evidence_flags(artifact, shown, audio_available),
        "prevReviewId": prev_id,
        "nextReviewId": next_id,
    }


# Live streams ----------------------------------------------------------------------------------


def _camel_keys(message: dict[str, Any]) -> dict[str, Any]:
    return {to_camel(key): value for key, value in message.items()}


def stream_message(message: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Normalize an engine `emit`. The engine may pass models or dumps, and `callId` is ours."""
    shaped = _camel_keys(to_api(message))
    shaped.setdefault("callId", call_id)
    for key in ("utterance", "event"):
        if isinstance(shaped.get(key), dict):
            shaped[key] = _camel_keys(shaped[key])
    return shaped


# Talk ------------------------------------------------------------------------------------------


def talk_sheet(scenario: Scenario | None) -> dict[str, Any] | None:
    if scenario is None:
        return None
    you_are = ", ".join(
        f"{key.replace('_', ' ')}: {value}" for key, value in scenario.facts.items()
    )
    return {
        "youAre": you_are,
        "problem": scenario.goal,
        "beats": to_api(scenario.beats),
        # The model has no caller-side rules. This is the one every scripted caller is held to.
        "mustNotSay": ["any detail that is not on this sheet"],
    }
