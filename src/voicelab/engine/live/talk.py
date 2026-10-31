"""A person calling the bot from the browser, over Pipecat's small WebRTC transport.

The API talks to this through `TalkSession`. What the dashboard shows comes over the data channel:
standard RTVI events, plus `voicelab` payloads as server messages, as docs/API.md describes.
"""

from __future__ import annotations

import asyncio
import logging
from statistics import median
from typing import Any

from aiortc.sdp import candidate_from_sdp
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from voicelab.company.db import CompanyDb, create_call_db, diff_against_seed
from voicelab.core.audio import AudioClip
from voicelab.core.models import CallArtifact, CallMode, CallStatus, Speaker, Utterance
from voicelab.core.profiles import ProviderStack
from voicelab.engine.contract import Emit, TalkRequest
from voicelab.engine.live.agent import guard_setting, support_side
from voicelab.engine.live.log import LiveLog
from voicelab.engine.live.parts import (
    EchoGate,
    LineStage,
    LiveSettings,
    MicRecorder,
    PlayoutTap,
    input_rate,
    voice_pipeline,
)
from voicelab.engine.speech_ledger import PlayoutLedger
from voicelab.store.artifacts import ArtifactStore

logger = logging.getLogger(__name__)

_OUT_RATE = 24000
_STOP_TIMEOUT_S = 5.0
_ECHO_WARNING_FRAMES = 100  # about two seconds blanked, at the mic's 20 ms frames


async def create_live_talk_session(
    request: TalkRequest,
    *,
    stack: ProviderStack,
    store: ArtifactStore,
    emit: Emit,
    settings: LiveSettings | None = None,
) -> LiveTalkSession:
    return LiveTalkSession(request, stack, store, emit, settings or LiveSettings())


class LiveTalkSession:
    def __init__(
        self,
        request: TalkRequest,
        stack: ProviderStack,
        store: ArtifactStore,
        emit: Emit,
        settings: LiveSettings,
    ) -> None:
        self.call_id = request.call_id
        self.request = request
        self.stack = stack
        self.settings = settings
        self._notify_api = emit
        providers = {
            "stt_live": stack.stt_live.name,
            "tts_bot": stack.tts_bot.name,
        }
        if stack.profile.support_engine == "realtime":
            providers["realtime"] = stack.realtime.name
        else:
            providers["llm_support"] = stack.llm_support.name
        self.artifact = CallArtifact(
            id=request.call_id,
            mode=CallMode.HUMAN,
            status=CallStatus.RUNNING,
            timing_is_real=True,
            persona_id=request.persona.id if request.persona else None,
            scenario_id=request.scenario.id if request.scenario else None,
            persona_snapshot=request.persona,
            scenario_snapshot=request.scenario,
            profile=stack.profile.id,
            providers=providers,
            metadata=guard_setting(stack),
        )
        self.log = LiveLog(self.artifact, store, self._to_browser)
        self.connection: SmallWebRTCConnection | None = None
        self.pipeline: Any = None
        self.worker: PipelineWorker | None = None
        self.run_task: asyncio.Task[None] | None = None
        self.brain: Any = None
        self.db: CompanyDb | None = None
        self.echo_frames = 0
        self.reply_s: list[float] = []
        self._finished = False

    async def handle_offer(self, sdp: str, sdp_type: str) -> dict[str, str]:
        if self.connection is not None:
            await self.connection.renegotiate(sdp=sdp, type=sdp_type, restart_pc=False)
            return self.connection.get_answer()
        # No STUN here: the browser is on this machine or its LAN, so host candidates reach it,
        # and a STUN lookup would hold up the answer when offline.
        connection = SmallWebRTCConnection()
        await connection.initialize(sdp=sdp, type=sdp_type)
        self.connection = connection
        self.log.start()
        self.log.system("call_started")
        self._start_pipeline(connection)
        answer = connection.get_answer()
        if answer is None:
            raise RuntimeError("the WebRTC connection produced no answer")
        return answer

    async def add_ice_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Trickled ICE from the browser. An empty candidate marks the end of a media line."""
        if self.connection is None:
            return
        for entry in candidates:
            text = entry.get("candidate")
            candidate = None
            if text:
                candidate = candidate_from_sdp(text)
                candidate.sdpMid = entry.get("sdp_mid", entry.get("sdpMid"))
                candidate.sdpMLineIndex = entry.get("sdp_mline_index", entry.get("sdpMLineIndex"))
            await self.connection.add_ice_candidate(candidate)

    def _start_pipeline(self, connection: SmallWebRTCConnection) -> None:
        line = self.request.line
        rate = input_rate(line)
        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_sample_rate=rate,
                audio_out_enabled=True,
                audio_out_sample_rate=_OUT_RATE,
            ),
        )
        self.db = create_call_db()
        ledger = PlayoutLedger(in_transit_s=self.settings.client_buffer_s)
        brain, tts = support_side(
            self.stack,
            db=self.db,
            log=self.log,
            ledger=ledger,
            settings=self.settings,
            fail=self.fail,
            far_side_listens=False,
            on_turn_done=self._turn_done,
            on_call_ended=self._bot_ended,
        )
        self.brain = brain
        echo = EchoGate(self._note_echo)

        def audio_out(clip: AudioClip) -> None:
            brain.on_audio_out(clip)
            echo.add_outgoing(clip)
            self.log.recorder.write_frame(Speaker.BOT, clip)

        inbound = (LineStage(line, seed=0),) if line is not None else ()
        outbound = (LineStage(line, seed=1),) if line is not None else ()
        pipeline, _ = voice_pipeline(
            transport,
            settings=self.settings,
            stt=self.stack.stt_live,
            fail=self.fail,
            brain=brain,
            tts_service=tts,
            tap=PlayoutTap(audio_out),
            wait_for_transcript=tts is not None,
            before_vad=(*inbound, MicRecorder(self.log), echo),
            after_brain=outbound,
        )
        self.pipeline = pipeline
        self.worker = PipelineWorker(
            pipeline,
            params=PipelineParams(audio_in_sample_rate=rate, audio_out_sample_rate=_OUT_RATE),
            enable_rtvi=True,
            cancel_on_idle_timeout=False,
        )

        @transport.event_handler("on_client_connected")
        async def connected(_transport: Any, _client: Any) -> None:
            await brain.greet()

        @transport.event_handler("on_client_disconnected")
        async def disconnected(_transport: Any, _client: Any) -> None:
            await self._notify_api({"type": "disconnected"})

        self.run_task = asyncio.create_task(self._run_worker())

    async def _run_worker(self) -> None:
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        await runner.run()

    async def _to_browser(self, message: dict[str, Any]) -> None:
        if message["type"] == "event":
            await self._server_message({"type": "event", "event": message["event"]})

    async def _server_message(self, payload: dict[str, Any]) -> None:
        if self.worker is not None:
            await self.worker.queue_frame(RTVIServerMessageFrame(data={"voicelab": payload}))

    def _turn_done(self, item: Utterance) -> None:
        if item.ttfb_ms is None:
            return
        self.reply_s.append(item.ttfb_ms / 1000)
        metrics = {
            "type": "metrics",
            "replyP50S": round(median(self.reply_s), 3),
            "lastReplyS": round(self.reply_s[-1], 3),
        }
        asyncio.ensure_future(self._server_message(metrics))

    def _note_echo(self) -> None:
        self.echo_frames += 1
        if self.echo_frames == _ECHO_WARNING_FRAMES:
            warning = {
                "type": "warning",
                "code": "echo_suspected",
                "message": "The bot can hear itself. Headphones will work better.",
            }
            asyncio.ensure_future(self._server_message(warning))

    async def fail(self, message: str) -> None:
        if self.artifact.status != CallStatus.FAILED:
            self.artifact.status = CallStatus.FAILED
            self.artifact.error = message
            self.log.system("call_failed", message)
        await self._notify_api({"type": "disconnected"})

    async def _bot_ended(self, detail: str) -> None:
        self.log.system("bot_ended", detail)
        await self._notify_api({"type": "disconnected"})

    async def end(self) -> CallArtifact:
        """Hang up, finish the recording, and return the artifact."""
        if self._finished:
            return self.artifact
        self._finished = True
        if self.artifact.status != CallStatus.FAILED:
            self.artifact.status = CallStatus.EVALUATING
        try:
            if self.brain is not None:
                await self.brain.stop_for_hang_up()
        finally:
            # Nothing here waits, so a cancel during the settling above can't lose the call.
            if self.brain is not None:
                self.brain.close_cut_off_turn()
            self._close_out()
        await self._stop()
        await self.log.close()
        return self.artifact

    def _close_out(self) -> None:
        self.artifact.metadata["echo_frames_blanked"] = self.echo_frames
        if self.db is not None:
            try:
                self.artifact.db_changes = diff_against_seed(self.db)
            except Exception as failure:
                logger.exception("database diff failed on call %s", self.artifact.id)
                self.artifact.error = self.artifact.error or f"database diff failed: {failure!r}"
            finally:
                self.db.close()
        self.log.system("call_ended")
        self.log.recorder.finish()

    async def _stop(self) -> None:
        close = getattr(self.brain, "close", None)
        if close is not None:
            await close()
        if self.worker is not None:
            await self.worker.cancel()
        if self.run_task is not None:
            await asyncio.wait([self.run_task], timeout=_STOP_TIMEOUT_S)
            self.run_task.cancel()
        if self.connection is not None:
            await self.connection.disconnect()
