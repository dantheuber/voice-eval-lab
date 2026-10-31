"""Mic calls, uploads, and the WebSocket streams."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Response, UploadFile, WebSocket, WebSocketDisconnect

from voicelab.api import mapping
from voicelab.api.routes.calls import require_call
from voicelab.api.schemas import ChannelMapBody, IceBody, OfferBody, Svc, TalkBody
from voicelab.core.models import LineConditions
from voicelab.services.errors import Conflict, Invalid, NotFound
from voicelab.services.events import Subscription

router = APIRouter()

_ICE_SERVERS = [{"urls": "stun:stun.l.google.com:19302"}]
_READ_CHUNK = 1024 * 1024
_RUN_DETAIL_MESSAGES = ("partial", "turn", "event")


# Talk --------------------------------------------------------------------------------------------


@router.post("/api/talk/sessions", status_code=201)
async def create_talk(body: TalkBody, svc: Svc) -> dict[str, Any]:
    line = mapping.from_api(LineConditions, body.line) if body.line else None
    talk, artifact = await svc.talks.create(
        scenario_id=body.scenario_id,
        persona_id=body.persona_id,
        line=line,
        record=body.record,
        profile_id=body.profile,
    )
    return {
        "sessionId": talk.session_id,
        "callId": talk.call_id,
        "offerUrl": f"/api/talk/sessions/{talk.session_id}/offer",
        "iceServers": _ICE_SERVERS,
        "sheet": mapping.talk_sheet(artifact.scenario_snapshot),
    }


@router.post("/api/talk/sessions/{session_id}/offer")
async def talk_offer(session_id: str, body: OfferBody, svc: Svc) -> dict[str, str]:
    return await svc.talks.offer(session_id, body.sdp, body.type, body.pc_id)


@router.patch("/api/talk/sessions/{session_id}/offer")
async def talk_ice(session_id: str, body: IceBody, svc: Svc) -> dict[str, str]:
    """Pipecat's client trickles ICE candidates to the offer URL with PATCH."""
    await svc.talks.add_ice_candidates(session_id, body.candidates)
    return {"status": "ok"}


@router.post("/api/talk/sessions/{session_id}/end")
async def talk_end(session_id: str, svc: Svc) -> dict[str, str]:
    return {"callId": await svc.talks.end(session_id)}


@router.get("/api/talk/sessions/{session_id}")
def talk_state(session_id: str, svc: Svc) -> dict[str, Any]:
    talk = svc.talks.get(session_id)
    return {
        "sessionId": talk.session_id,
        "callId": talk.call_id,
        "state": talk.state,
        "startedAt": mapping.iso(talk.started_at),
        "durationS": svc.talks.duration_s(talk),
    }


# Uploads -----------------------------------------------------------------------------------------


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    chunks = []
    size = 0
    while chunk := await file.read(_READ_CHUNK):
        size += len(chunk)
        if size > limit:
            raise Invalid(f"file is larger than {limit // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/api/uploads", status_code=201)
async def create_upload(
    svc: Svc,
    file: Annotated[UploadFile, File()],
    scenarioId: Annotated[str | None, Form()] = None,  # noqa: N803  (the contract's field names)
    botVersion: Annotated[str | None, Form()] = None,  # noqa: N803
    intendedTranscript: Annotated[str, Form()] = "",  # noqa: N803
    toolLog: Annotated[str, Form()] = "",  # noqa: N803
) -> dict[str, Any]:
    data = await _read_limited(file, svc.settings.max_upload_bytes)
    # Decoding, resampling, and VAD are CPU work, and a mic call may be sharing this loop.
    artifact, probe = await asyncio.to_thread(
        svc.uploads.create,
        data,
        file.filename or "",
        profile=svc.profiles.active_id,
        scenario_id=scenarioId,
        bot_version=botVersion,
        intended=intendedTranscript,
        tool_log=toolLog,
    )
    audio = svc.store.paths(artifact.id).recording.exists()
    return {
        "callId": artifact.id,
        "probe": mapping.to_api(probe),
        "evidence": mapping.evidence_flags(artifact, None, audio),
    }


@router.post("/api/uploads/{call_id}/channel-map")
async def set_channel_map(call_id: str, body: ChannelMapBody, svc: Svc) -> dict[str, bool]:
    await asyncio.to_thread(svc.uploads.set_channel_map, call_id, body.channel_map)
    return {"ok": True}


@router.post("/api/uploads/{call_id}/evaluate")
async def evaluate_upload(call_id: str, svc: Svc) -> dict[str, Any]:
    row = svc.uploads.require(call_id)
    if row["status"] in ("running", "evaluating"):
        raise Conflict("this upload is being evaluated")
    svc.evals.submit(call_id)
    return {"accepted": True, "estimateS": svc.uploads.estimate_s(row["duration_s"])}


@router.get("/api/uploads/{call_id}/preview")
def preview_upload(
    call_id: str, svc: Svc, startS: float = 0.0, durationS: float = 10.0
) -> Response:  # noqa: N803
    return Response(svc.uploads.preview(call_id, startS, durationS), media_type="audio/wav")


# Sockets -----------------------------------------------------------------------------------------


async def _pump(
    websocket: WebSocket,
    subscription: Subscription,
    *,
    keep: Callable[[dict[str, Any]], bool] = lambda _: True,
    final: Callable[[dict[str, Any]], bool] = lambda _: False,
    on_client: Callable[[dict[str, Any]], None] = lambda _: None,
) -> None:
    """Forward messages until `final` says stop or the client leaves."""

    async def send() -> None:
        while True:
            message = await subscription.get()
            if keep(message):
                await websocket.send_json(message)
            if final(message):
                return

    async def receive() -> None:
        try:
            while True:
                on_client(await websocket.receive_json())
        except (WebSocketDisconnect, ValueError):
            return

    tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _close(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except RuntimeError:  # the client already left
        pass


@router.websocket("/ws/calls/{call_id}")
async def call_stream(websocket: WebSocket, call_id: str, svc: Svc) -> None:
    await websocket.accept()
    # Subscribe before reading state, so nothing falls between the read and the first message.
    with svc.hub.subscribe(f"call:{call_id}") as subscription:
        try:
            row = require_call(svc, call_id)
        except NotFound as exc:
            await websocket.send_json({"type": "error", "message": exc.message})
            await websocket.close()
            return
        await websocket.send_json({"type": "status", "status": row["status"]})
        if row["status"] == "done":
            await websocket.send_json(_finished_eval(svc, call_id))
        elif row["status"] == "failed":
            await websocket.send_json({"type": "error", "message": row["error"] or "call failed"})
        else:
            await _pump(websocket, subscription, final=lambda m: m.get("type") in ("eval", "error"))
    await _close(websocket)


def _finished_eval(svc: Svc, call_id: str) -> dict[str, Any]:
    result = svc.store.read_eval(call_id)
    if result is None:
        return {"type": "error", "message": "this call has no evaluation"}
    decided = mapping.apply_decisions(result, svc.calls.finding_decisions(call_id))
    return {"type": "eval", "eval": mapping.to_api(decided)}


class _CallWatch:
    """Which call of a run this socket streams in detail: the client's choice, or the earliest
    one still running."""

    def __init__(self, running: list[str]) -> None:
        self.running = running
        self.chosen: str | None = None

    def wanted(self, message: dict[str, Any]) -> bool:
        kind = message.get("type")
        if kind == "callStarted":
            self.running.append(message["call"]["id"])
        elif kind == "callDone" and message["call"]["id"] in self.running:
            self.running.remove(message["call"]["id"])
        if kind not in _RUN_DETAIL_MESSAGES:
            return True
        watched = self.chosen or (self.running[0] if self.running else None)
        return message.get("callId") == watched

    def on_client(self, message: dict[str, Any]) -> None:
        if message.get("type") == "watch":
            self.chosen = message.get("callId")


@router.websocket("/ws/runs/{run_id}")
async def run_stream(websocket: WebSocket, run_id: str, svc: Svc) -> None:
    await websocket.accept()
    with svc.hub.subscribe(f"run:{run_id}") as subscription:
        run = svc.runs.get(run_id)
        if run is None:
            await websocket.send_json({"type": "error", "message": f"no run {run_id}"})
            await websocket.close()
            return
        summary = svc.worker.summary(run_id)
        await websocket.send_json({"type": "run", "run": summary})
        rows = svc.calls.in_batch(run_id)
        if run["status"] in ("done", "stopped") and not any(
            row["status"] in ("running", "evaluating") for row in rows
        ):
            await websocket.send_json({"type": "runDone", "run": summary})
        else:
            watch = _CallWatch([row["id"] for row in rows if row["status"] == "running"])
            await _pump(
                websocket,
                subscription,
                keep=watch.wanted,
                final=lambda m: m.get("type") == "runDone",
                on_client=watch.on_client,
            )
    await _close(websocket)


@router.websocket("/ws/health")
async def health_stream(websocket: WebSocket, svc: Svc) -> None:
    await websocket.accept()
    with svc.hub.subscribe("health") as subscription:
        items = mapping.to_api(svc.profiles.health())
        await websocket.send_json({"type": "health", "items": items})
        await _pump(websocket, subscription)
    await _close(websocket)
