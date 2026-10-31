"""The overview, the calls list, and the review payload."""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response
from fastapi.responses import FileResponse, JSONResponse

from voicelab.api import mapping
from voicelab.api.schemas import CallVerdict, FindingDecision, Reevaluate, Svc
from voicelab.core.models import EvalResult
from voicelab.services.errors import Conflict, Invalid, NotFound
from voicelab.services.evaluation import check_evaluators
from voicelab.services.review import speech_data
from voicelab.store.calls import CallFilters
from voicelab.store.sqlite import timestamp

router = APIRouter(prefix="/api")

_WINDOW = re.compile(r"^(\d{1,4})([hd])$")
_QUEUE_SHOWN = 8
_RECENT_RUNS = 6
_AUDIO_FILES = {"recording.wav": "recording", "caller.wav": "caller", "bot.wav": "bot"}

Multi = Annotated[list[str], Query()]


def since(window: str) -> str:
    match = _WINDOW.match(window)
    if match is None:
        raise Invalid(f"bad window {window!r}, expected something like 24h or 7d")
    amount, unit = int(match[1]), match[2]
    span = timedelta(hours=amount) if unit == "h" else timedelta(days=amount)
    return timestamp(datetime.now(UTC) - span)


def require_call(svc: Svc, call_id: str) -> dict[str, Any]:
    # Ids only reach the filesystem after matching an index row, so a crafted id goes no further.
    row = svc.calls.get(call_id)
    if row is None:
        raise NotFound(f"no call {call_id}")
    return row


@router.get("/overview")
def overview(svc: Svc, profile: str | None = None, window: str = "24h") -> dict[str, Any]:
    profile = profile or svc.profiles.active_id
    start = since(window)
    filters = CallFilters(profile=[profile] if profile else [], since=start)
    queue = svc.calls.review_queue(profile)
    recent, _ = svc.runs.page(None, _RECENT_RUNS)
    return {
        "window": window,
        "profile": profile,
        "counts": mapping.counts(svc.calls.verdict_counts(filters)),
        "queue": [mapping.queue_item(row) for row in queue[:_QUEUE_SHOWN]],
        "queueTotal": len(queue),
        "openFindingsTotal": sum(row["open_findings"] for row in queue),
        "reviewPaceSPerFinding": svc.calls.review_pace(),
        "callsPerHour": svc.calls.calls_per_hour(profile),
        "recentRuns": [svc.worker.summary(run["id"]) for run in recent],
        "runningRuns": len(svc.runs.with_status("running")),
        "scenarioRates": [
            {
                "scenarioId": group["scenario_id"],
                "scenarioTitle": group["scenario_title"],
                "counts": mapping.counts(group["counts"]),
            }
            for group in svc.calls.counts_by("scenario_id", "scenario_title", filters=filters)
            if group["scenario_id"] is not None
        ],
        "health": mapping.to_api(svc.profiles.health()),
        "topOpenFindings": svc.calls.top_open_findings(profile, start),
    }


@router.get("/calls")
def list_calls(
    svc: Svc,
    verdict: Multi = [],  # noqa: B006  (FastAPI copies query defaults)
    scenario: Multi = [],  # noqa: B006
    persona: Multi = [],  # noqa: B006
    mode: Multi = [],  # noqa: B006
    profile: Multi = [],  # noqa: B006
    run: Multi = [],  # noqa: B006
    status: Multi = [],  # noqa: B006
    openOnly: bool = False,  # noqa: N803  (the contract's query name)
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict[str, Any]:
    filters = CallFilters(
        verdict=verdict,
        scenario=scenario,
        persona=persona,
        mode=mode,
        profile=profile,
        run=run,
        status=status,
        open_only=openOnly,
    )
    page = svc.calls.page(filters, cursor, limit)
    return {
        "items": [mapping.call_summary(row) for row in page.items],
        "nextCursor": page.next_cursor,
        "total": page.total,
        "facets": {"verdict": page.facets},
    }


@router.get("/calls/next-review")
def next_review(svc: Svc, after: str | None = None) -> dict[str, str | None]:
    return {"callId": svc.calls.next_review(after)}


@router.get("/calls/{call_id}")
def call_detail(call_id: str, svc: Svc) -> JSONResponse:
    row = require_call(svc, call_id)
    artifact = svc.store.read_artifact(call_id)
    paths = svc.store.paths(call_id)
    available = paths.recording.exists()
    derived = speech_data(svc.store, artifact)
    payload = mapping.review_payload(
        row=row,
        artifact=artifact,
        result=svc.store.read_eval(call_id),
        decisions=svc.calls.finding_decisions(call_id),
        peaks=svc.store.read_peaks(call_id) if paths.peaks.exists() else None,
        speech=derived["speech"],
        overlaps=derived["overlaps"],
        audio_available=available,
        prev_id=svc.calls.previous_review(call_id),
        next_id=svc.calls.next_review(call_id),
    )
    return JSONResponse(payload)  # built by hand, so FastAPI needn't re-encode a long waveform


@router.get("/calls/{call_id}/audio/{name}")
def call_audio(call_id: str, name: str, svc: Svc) -> FileResponse:
    require_call(svc, call_id)
    attribute = _AUDIO_FILES.get(name)
    if attribute is None:
        raise NotFound(f"no audio file {name!r}")
    path = getattr(svc.store.paths(call_id), attribute)
    if not path.exists():
        raise NotFound(f"call {call_id} has no {name}")
    return FileResponse(path, media_type="audio/wav", headers={"Accept-Ranges": "bytes"})


def _decided_eval(svc: Svc, call_id: str) -> EvalResult:
    result = svc.store.read_eval(call_id)
    if result is None:
        raise NotFound(f"call {call_id} has no evaluation")
    return mapping.apply_decisions(result, svc.calls.finding_decisions(call_id))


@router.post("/calls/{call_id}/findings/{finding_id}")
def decide_finding(
    call_id: str, finding_id: str, body: FindingDecision, svc: Svc
) -> dict[str, Any]:
    require_call(svc, call_id)
    verdict = body.human_verdict.value if body.human_verdict else None
    recorded = svc.calls.decide_finding(
        call_id, finding_id, verdict, body.human_note, svc.settings.reviewer
    )
    if recorded is None:
        raise NotFound(f"no finding {finding_id} on call {call_id}")
    finding = next(f for f in _decided_eval(svc, call_id).findings if f.id == finding_id)
    return mapping.to_api(finding)


@router.post("/calls/{call_id}/verdict")
def decide_call(call_id: str, body: CallVerdict, svc: Svc) -> dict[str, Any]:
    require_call(svc, call_id)
    svc.calls.decide_call(call_id, body.verdict.value, body.note, svc.settings.reviewer)
    return mapping.call_summary(require_call(svc, call_id))


@router.post("/calls/{call_id}/reevaluate")
async def reevaluate(call_id: str, body: Reevaluate, svc: Svc) -> dict[str, bool]:
    row = require_call(svc, call_id)
    names = check_evaluators(body.evaluators)
    if row["status"] not in ("done", "failed"):
        raise Conflict(f"call is {row['status']}")
    if not svc.store.paths(call_id).recording.exists():
        raise Conflict("this call has no recording to evaluate")
    svc.evals.submit(call_id, names)
    return {"accepted": True}


@router.delete("/calls/{call_id}", status_code=204)
def delete_call(call_id: str, svc: Svc) -> Response:
    row = require_call(svc, call_id)
    if row["status"] in ("running", "evaluating"):
        raise Conflict(f"call is {row['status']}")
    batch = svc.runs.get(row["batch_id"]) if row["batch_id"] else None
    if row["status"] == "queued" and batch and batch["status"] in ("queued", "running", "paused"):
        raise Conflict("call is queued in a run that hasn't finished")
    svc.calls.delete(call_id)
    shutil.rmtree(svc.store.paths(call_id).dir, ignore_errors=True)
    return Response(status_code=204)
