"""Two bots on a live call: two pipelines joined by a simulated phone line, in real time.

Each side runs its own VAD, STT, and turn detection on what comes out of the line, so a barge-in
here is the same event as a person's barge-in on a mic call. The channels record what arrived, as
turn-based calls do.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

from voicelab.agents.caller import CallerBot
from voicelab.company.db import CompanyDb, create_call_db, diff_against_seed
from voicelab.core.audio import AudioClip
from voicelab.core.models import CallArtifact, CallMode, CallStatus, Speaker, Utterance
from voicelab.core.profiles import ProviderStack
from voicelab.engine import progress
from voicelab.engine.contract import BotCallRequest, Emit
from voicelab.engine.live.agent import guard_setting, support_side
from voicelab.engine.live.brains import BotBrain, Brain, CallerBrain
from voicelab.engine.live.link import LinkEnd, Wire
from voicelab.engine.live.log import LiveLog
from voicelab.engine.live.parts import (
    LiveSettings,
    PlayoutTap,
    WrappedTTS,
    input_rate,
    voice_pipeline,
)
from voicelab.engine.speech_ledger import PlayoutLedger
from voicelab.store.artifacts import ArtifactStore
from voicelab.telephony import PhoneLine

logger = logging.getLogger(__name__)

_OUT_RATE = 16000
_SECONDS_PER_TURN = 40.0  # generous, since this only catches a call that has stalled
_STOP_TIMEOUT_S = 5.0


async def _run(worker: PipelineWorker) -> None:
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def run_live_bot_call(
    request: BotCallRequest,
    *,
    stack: ProviderStack,
    store: ArtifactStore,
    emit: Emit,
    settings: LiveSettings | None = None,
) -> CallArtifact:
    return await _LiveBotCall(request, stack, store, emit, settings or LiveSettings()).run()


class _LiveBotCall:
    def __init__(
        self,
        request: BotCallRequest,
        stack: ProviderStack,
        store: ArtifactStore,
        emit: Emit,
        settings: LiveSettings,
    ) -> None:
        self.request = request
        self.stack = stack
        self.settings = settings
        persona, scenario = request.persona, request.scenario
        providers = {
            "stt_live": stack.stt_live.name,
            "tts_bot": stack.tts_bot.name,
            "tts_caller": stack.tts_caller.name,
            "llm_caller": stack.llm_caller.name,
        }
        if stack.profile.support_engine == "realtime":
            providers["realtime"] = stack.realtime.name
        else:
            providers["llm_support"] = stack.llm_support.name
        self.artifact = CallArtifact(
            id=request.call_id,
            mode=CallMode.LIVE_BOTS,
            status=CallStatus.RUNNING,
            timing_is_real=True,
            batch_id=request.batch_id,
            persona_id=persona.id,
            scenario_id=scenario.id,
            persona_snapshot=persona,
            scenario_snapshot=scenario,
            profile=stack.profile.id,
            providers=providers,
            metadata={**request.metadata, **guard_setting(stack)},
        )
        self.log = LiveLog(self.artifact, store, emit)
        self.done = asyncio.Event()
        self.conditions = request.line or persona.line
        self.db: CompanyDb | None = None
        self.wires: list[Wire] = []
        self.workers: list[PipelineWorker] = []
        self.runs: list[asyncio.Task[None]] = []
        self.bot_brain: BotBrain | None = None
        self.brains: list[Brain] = []

    async def fail(self, message: str) -> None:
        if self.artifact.status != CallStatus.FAILED:
            self.artifact.status = CallStatus.FAILED
            self.artifact.error = message
            self.log.system("call_failed", message)
        self.done.set()

    async def _ended(self, kind: str, detail: str) -> None:
        self.log.system(kind, detail)
        for brain in self.brains:
            brain.call_over = True
        asyncio.get_running_loop().call_later(self._last_word_s, self.done.set)

    async def _bot_ended(self, detail: str) -> None:
        await self._ended("bot_ended", detail)

    @property
    def _last_word_s(self) -> float:
        """Let the last words cross the line and be heard before the call stops."""
        return self.wires[0].line.latency_s + self.settings.settle_after_end_s

    async def run(self) -> CallArtifact:
        self.log.start()
        self.log.post(progress.call_started(self.artifact))
        self.log.system("call_started")
        try:
            await self._converse()
            if self.artifact.status != CallStatus.FAILED:
                self.artifact.status = CallStatus.EVALUATING
        except asyncio.CancelledError:
            await self.fail("stopped")
            raise
        except Exception as failure:
            logger.exception("call %s failed", self.artifact.id)
            await self.fail(f"{type(failure).__name__}: {failure}")
        finally:
            try:
                await asyncio.gather(*(brain.stop_for_hang_up() for brain in self.brains))
            finally:
                # Nothing here waits, so a second cancel can't lose the recording.
                for brain in self.brains:
                    brain.close_cut_off_turn()
                self._close_out()
            await self._stop_pipelines()
            self.log.send_waiting_turns()
            self.log.post(progress.call_done(self.artifact))
            await self.log.close()
        return self.artifact

    def _build_wires(self) -> tuple[Wire, Wire]:
        seed = self.request.seed
        record = self.log.recorder.write_frame

        def heard_by_bot(clip: AudioClip) -> None:
            record(Speaker.CALLER, clip)

        def heard_by_caller(clip: AudioClip) -> None:
            record(Speaker.BOT, clip)

        to_bot = Wire(
            PhoneLine(self.conditions, seed=seed * 2), rate=_OUT_RATE, on_arrival=heard_by_bot
        )
        to_caller = Wire(
            PhoneLine(self.conditions, seed=seed * 2 + 1),
            rate=_OUT_RATE,
            on_arrival=heard_by_caller,
        )
        self.wires = [to_bot, to_caller]
        return to_bot, to_caller

    async def _converse(self) -> None:
        self.db = create_call_db()
        rate = input_rate(self.conditions)
        to_bot, to_caller = self._build_wires()
        bot_end = LinkEnd(outgoing=to_caller, in_rate=rate, out_rate=_OUT_RATE)
        caller_end = LinkEnd(outgoing=to_bot, in_rate=rate, out_rate=_OUT_RATE)
        to_bot.receiver = bot_end.receiver
        to_caller.receiver = caller_end.receiver

        bot_brain, bot_pipeline = self._support_pipeline(bot_end)
        caller_pipeline = self._caller_pipeline(caller_end)
        self.bot_brain = bot_brain
        ready = [asyncio.Event(), asyncio.Event()]
        params = PipelineParams(audio_in_sample_rate=rate, audio_out_sample_rate=_OUT_RATE)
        for pipeline, event in zip((bot_pipeline, caller_pipeline), ready, strict=True):
            self._start_pipeline(pipeline, params, event)
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 60)
        for wire in self.wires:
            wire.start()
        await bot_brain.greet()
        limit = self.request.scenario.max_turns * _SECONDS_PER_TURN + 60
        try:
            await asyncio.wait_for(self.done.wait(), limit)
        except TimeoutError:
            await self.fail(f"call still going after {limit:.0f} s")

    def _start_pipeline(self, pipeline: Any, params: PipelineParams, ready: asyncio.Event) -> None:
        worker = PipelineWorker(
            pipeline, params=params, enable_rtvi=False, cancel_on_idle_timeout=False
        )

        @worker.event_handler("on_pipeline_started")
        async def started(_worker: Any, _frame: Any) -> None:
            ready.set()

        self.workers.append(worker)
        self.runs.append(asyncio.create_task(_run(worker)))

    def _support_pipeline(self, end: LinkEnd) -> tuple[BotBrain, Any]:
        ledger = PlayoutLedger(in_transit_s=self.wires[0].line.latency_s)

        def turn_done(_item: Utterance) -> None:
            pass

        brain, tts = support_side(
            self.stack,
            db=self.db,
            log=self.log,
            ledger=ledger,
            settings=self.settings,
            fail=self.fail,
            far_side_listens=True,
            on_turn_done=turn_done,
            on_call_ended=self._bot_ended,
            audio_delay_s=self.wires[0].line.latency_s,
        )
        self.brains.append(brain)
        pipeline, _ = voice_pipeline(
            end,
            settings=self.settings,
            stt=self.stack.stt_live,
            fail=self.fail,
            brain=brain,
            tts_service=tts,
            tap=PlayoutTap(brain.on_audio_out),
            wait_for_transcript=tts is not None,
        )
        return brain, pipeline

    def _caller_pipeline(self, end: LinkEnd) -> Any:
        ledger = PlayoutLedger(in_transit_s=self.wires[0].line.latency_s)
        persona = self.request.persona
        caller = CallerBot(
            self.stack.llm_caller, persona, self.request.scenario, seed=self.request.seed
        )
        brain = CallerBrain(
            caller,
            stt=self.stack.stt_live,
            on_hang_up=self._ended,
            log=self.log,
            ledger=ledger,
            settings=self.settings,
            fail=self.fail,
            audio_delay_s=self.wires[0].line.latency_s,
        )
        self.brains.append(brain)
        tts = WrappedTTS(
            self.stack.tts_caller,
            voice_id=persona.voice.voice_id,
            style=persona.voice.style,
            ledger=ledger,
            fail=self.fail,
            role="tts_caller",
        )
        pipeline, turns = voice_pipeline(
            end,
            settings=self.settings,
            stt=self.stack.stt_live,
            fail=self.fail,
            brain=brain,
            tts_service=tts,
            tap=PlayoutTap(brain.on_audio_out),
            # The caller only cuts in when `decide_interruption` says so, not on every sound.
            can_be_interrupted=False,
            idle_s=self.settings.caller_patience_s,
        )

        @turns.event_handler("on_user_turn_idle")
        async def idle(_processor: Any) -> None:
            await brain.on_idle()

        return pipeline

    def _close_out(self) -> None:
        self.log.system("call_ended")
        self.artifact.metadata["line"] = {
            "late_ticks": sum(wire.late_ticks for wire in self.wires),
            "packets_lost": sum(wire.line.packets_lost for wire in self.wires),
        }
        if self.db is not None:
            try:
                self.artifact.db_changes = diff_against_seed(self.db)
            except Exception as failure:
                logger.exception("database diff failed on call %s", self.artifact.id)
                self.artifact.error = self.artifact.error or f"database diff failed: {failure!r}"
            finally:
                self.db.close()
        self.log.recorder.finish()

    async def _stop_pipelines(self) -> None:
        for wire in self.wires:
            await wire.stop()
        close = getattr(self.bot_brain, "close", None)
        if close is not None:
            await close()
        for worker in self.workers:
            await worker.cancel()
        if self.runs:
            await asyncio.wait(self.runs, timeout=_STOP_TIMEOUT_S)
            for run in self.runs:
                run.cancel()
