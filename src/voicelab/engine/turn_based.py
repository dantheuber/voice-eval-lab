"""Bot-to-bot calls that run one turn at a time, slower than real time.

Real audio still crosses the line. A line of speech goes through TTS, through a `PhoneLine`, and
through live STT, and the text that comes out is all the other side gets. Time is virtual: audio
lands on the timeline where it would have landed, the pauses are plausible but invented, and what
each step really took is kept apart in `artifact.metadata["compute"]`.

One clock convention holds throughout. A channel records audio as it comes out of the line, so
every position is an arrival time, and a sender's decision shows up one line delay later.

The caller and support models may share a machine or swap in and out, so nothing here runs them
concurrently, and a slow model only makes the call take longer.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from voicelab.agents.caller import MAX_TURNS_REASON, TYPICAL_TURN_WORDS, CallerBot, CallerLine
from voicelab.agents.speech import SpeechText
from voicelab.agents.support import SupportBot
from voicelab.company.db import CompanyDb, create_call_db, diff_against_seed
from voicelab.company.tools import ToolDispatcher
from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    BargeInEvent,
    CallArtifact,
    CallMode,
    CallStatus,
    Speaker,
    SystemEvent,
    ToolCallEvent,
    Utterance,
)
from voicelab.core.profiles import ProviderStack
from voicelab.engine import progress
from voicelab.engine.contract import BotCallRequest, Emit
from voicelab.engine.heard import prepare_heard_text
from voicelab.engine.recorder import CallRecorder
from voicelab.engine.speech_ledger import words_played
from voicelab.providers.base import LLM, TTS, ChatMessage, LLMDelta, LLMResponse, ToolDef
from voicelab.store.artifacts import ArtifactStore
from voicelab.telephony import PhoneLine

logger = logging.getLogger(__name__)

T = TypeVar("T")

_SECONDS_PER_WORD = 0.4  # to read `Interruption.after_fraction` as a duration
TYPICAL_TURN_S = TYPICAL_TURN_WORDS * _SECONDS_PER_WORD


@dataclass(frozen=True)
class TurnConfig:
    """Invented pauses, in seconds, and how long a model may take. Spans are (low, high)."""

    # A local model server may have to load a model first, so this is generous.
    step_timeout_s: float = 300.0
    answer_s: float = 0.6  # call connects to the greeting's first audio
    think_s: tuple[float, float] = (0.35, 0.75)  # per LLM round
    react_s: tuple[float, float] = (0.4, 1.1)  # caller hears the end, then starts talking
    stop_s: tuple[float, float] = (0.2, 0.45)  # bot notices it was cut in on, then goes quiet
    wait_s: tuple[float, float] = (2.0, 3.5)  # someone waits, having heard nothing


@dataclass(eq=False)
class _VirtualClock:
    now: float = 0.0

    def read(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def advance_to(self, position: float) -> None:
        self.now = max(self.now, position)


@dataclass
class _ComputeLog:
    """Real time spent, per step. The timeline never sees these numbers."""

    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, turn: int, milliseconds: float) -> None:
        self.steps.append({"turn": turn, "step": step, "ms": round(milliseconds, 1)})

    def summary(self) -> dict[str, Any]:
        totals: dict[str, float] = {}
        for entry in self.steps:
            totals[entry["step"]] = totals.get(entry["step"], 0.0) + entry["ms"]
        return {
            "total_ms": round(sum(totals.values()), 1),
            "by_step_ms": {step: round(ms, 1) for step, ms in totals.items()},
            "steps": self.steps,
        }


class _MeteredLLM:
    """Measures real time on an LLM, and charges the virtual clock for each round of thinking."""

    def __init__(self, inner: LLM, step: str, on_round: Any, log: _ComputeLog, turn: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self._step = step
        self._on_round = on_round
        self._log = log
        self._turn = turn

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._on_round()
        started = time.perf_counter()
        try:
            return await self.inner.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )
        finally:
            self._log.add(self._step, self._turn(), (time.perf_counter() - started) * 1000)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]:
        self._on_round()
        stream = self.inner.stream_chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        spent = 0.0
        try:
            while True:
                # Only time inside the model counts. The gaps between deltas are this engine's.
                started = time.perf_counter()
                try:
                    delta = await anext(stream)
                except StopAsyncIteration:
                    return
                finally:
                    spent += time.perf_counter() - started
                yield delta
        finally:
            self._log.add(self._step, self._turn(), spent * 1000)
            close = getattr(stream, "aclose", None)
            if close:
                await close()


class _CallFailed(Exception):
    """A step failed, and this message is what the artifact should say about it."""


@dataclass(frozen=True)
class _BotTurn:
    """What came of a bot turn: what the caller's STT heard, and when the caller cut in."""

    heard: str = ""
    barge_at_s: float | None = None


async def _one_segment(segment: SpeechText) -> AsyncIterator[SpeechText]:
    yield segment


async def run_turn_based_call(
    request: BotCallRequest,
    *,
    stack: ProviderStack,
    store: ArtifactStore,
    emit: Emit,
    config: TurnConfig | None = None,
) -> CallArtifact:
    """Run one call to completion. See `BotCallRunner` for what the caller of this can expect."""
    return await _Call(request, stack, store, emit, config or TurnConfig()).run()


class _Call:
    def __init__(
        self,
        request: BotCallRequest,
        stack: ProviderStack,
        store: ArtifactStore,
        emit: Emit,
        config: TurnConfig,
    ) -> None:
        self.request = request
        self.stack = stack
        self.emit = emit
        self.config = config
        persona, scenario = request.persona, request.scenario
        self.artifact = CallArtifact(
            id=request.call_id,
            mode=CallMode.TURN_BASED,
            status=CallStatus.RUNNING,
            timing_is_real=False,
            batch_id=request.batch_id,
            persona_id=persona.id,
            scenario_id=scenario.id,
            persona_snapshot=persona,
            scenario_snapshot=scenario,
            profile=stack.profile.id,
            providers={
                "stt_live": stack.stt_live.name,
                "tts_bot": stack.tts_bot.name,
                "tts_caller": stack.tts_caller.name,
                "llm_support": stack.llm_support.name,
                "llm_caller": stack.llm_caller.name,
            },
            metadata=dict(request.metadata),
        )
        self.clock = _VirtualClock()
        self.recorder = CallRecorder(self.artifact, store, clock=self.clock.read)
        # The call id stays out of the seed, so a repeat of a seed is a repeat of the call.
        self.rng = random.Random(f"{request.seed}:{scenario.id}")
        conditions = request.line or persona.line
        self.to_bot = PhoneLine(conditions, seed=request.seed * 2)
        self.to_caller = PhoneLine(conditions, seed=request.seed * 2 + 1)
        self.latency_s = self.to_bot.latency_s
        self.compute = _ComputeLog()
        self.turn = 0
        self._unsent: list[ToolCallEvent | SystemEvent | BargeInEvent] = []
        self.support: SupportBot
        self.caller: CallerBot

    async def run(self) -> CallArtifact:
        db: CompanyDb | None = None
        self.recorder.start()
        try:
            db = create_call_db()
            self._seat_bots(db)
            await self._converse()
            self.artifact.status = CallStatus.EVALUATING
        except asyncio.CancelledError:
            self._fail("stopped")
            raise
        except _CallFailed as failure:
            self._fail(str(failure))
        except Exception as failure:
            logger.exception("call %s failed", self.artifact.id)
            self._fail(f"{type(failure).__name__}: {failure}")
        finally:
            # Nothing in here awaits until the artifact is on disk, so a second cancel can't
            # lose the recording.
            self._close_out(db)
            await self._flush()
            await self._emit(progress.call_done(self.artifact))
        return self.artifact

    def _seat_bots(self, db: CompanyDb) -> None:
        def think() -> None:
            self.clock.advance(self.rng.uniform(*self.config.think_s))

        self.support = SupportBot(
            _MeteredLLM(
                self.stack.llm_support, "llm_support", think, self.compute, lambda: self.turn
            ),
            ToolDispatcher(db),
            on_event=self._queue,
            clock=self.clock.read,
        )
        # The caller's thinking is already inside `react_s`.
        self.caller = CallerBot(
            _MeteredLLM(
                self.stack.llm_caller, "llm_caller", lambda: None, self.compute, lambda: self.turn
            ),
            self.request.persona,
            self.request.scenario,
            seed=self.request.seed,
        )

    async def _converse(self) -> None:
        await self._emit(progress.call_started(self.artifact))
        self._system("call_started")
        await self._flush()
        self.clock.advance(self.config.answer_s)
        played = await self._bot_turn(_one_segment(self.support.greet()))
        while True:
            line = await self._step("llm_caller", self.caller.next_line(played.heard))
            bot_heard = await self._caller_speaks(line, played.barge_at_s)
            if line.completed_beat:
                self._system("beat_completed", line.completed_beat)
            await self._flush()
            if line.hang_up:
                at_limit = line.hang_up_reason == MAX_TURNS_REASON
                self._system(
                    MAX_TURNS_REASON if at_limit else "caller_hung_up", line.hang_up_reason
                )
                return
            self.turn += 1
            if not bot_heard:
                # A real pipeline gets no turn out of an empty transcript, so the bot says nothing
                # and the caller is left waiting.
                self._system("nothing_heard", Speaker.BOT.value)
                self.clock.advance(self.rng.uniform(*self.config.wait_s))
                played = _BotTurn()
                continue
            played = await self._bot_turn(self.support.respond(prepare_heard_text(bot_heard)))
            if self.support.ended:
                self._system("bot_ended", self.support.end_reason)
                return

    async def _bot_turn(self, reply: AsyncIterator[SpeechText]) -> _BotTurn:
        """Pull the reply only as far as it gets played, then send what played down the line.

        A caller cutting in closes the reply, so later sentences are never generated and later
        tool rounds never run. Audio for the sentences before the cut is real, and it's cut
        exactly where the caller's interruption stopped it.
        """
        decision = self.caller.decide_interruption(self.turn)
        wants_cut = decision.interrupt
        turnaround = 2 * self.latency_s + self.rng.uniform(*self.config.stop_s)
        cut_in_s = decision.after_fraction * TYPICAL_TURN_S
        clips: list[AudioClip] = []
        spoken_parts: list[tuple[str, float]] = []
        total_s = 0.0
        ready_at: float | None = None
        try:
            while True:
                try:
                    segment = await self._step("llm_support", anext(reply))
                except StopAsyncIteration:
                    break
                if ready_at is None:
                    ready_at = self.clock.now
                clip = await self._speak(
                    self.stack.tts_bot, segment.spoken, self.bot_voice, None, "tts_bot"
                )
                clips.append(clip)
                spoken_parts.append((segment.spoken, clip.duration_s))
                total_s += clip.duration_s
                self._partial(Speaker.BOT, " ".join(text for text, _ in spoken_parts))
                await self._flush()
                if wants_cut and total_s > cut_in_s + turnaround and not self.support.ended:
                    break
        finally:
            await reply.aclose()
        if not clips or ready_at is None:
            return _BotTurn()

        audio = clips[0]
        for clip in clips[1:]:
            audio = audio + clip
        played_s = audio.duration_s
        barge_after_s: float | None = None
        if wants_cut and not self.support.ended:
            # A reply shorter than the cut-in point is still cut, at that fraction of itself.
            heard_s = cut_in_s if total_s > cut_in_s else decision.after_fraction * total_s
            barge_after_s = heard_s + self.latency_s
            played_s = min(heard_s + turnaround, heard_s + (total_s - heard_s) / 2)
            barge_after_s = min(barge_after_s, played_s)
            audio = audio.slice(0, played_s)

        start_s = ready_at + self.latency_s
        arrived, heard = await self._cross(self.to_caller, audio)
        end_s = start_s + arrived.duration_s
        self.recorder.place(Speaker.BOT, arrived, start_s)
        intended = " ".join(text for text, _ in spoken_parts)
        interrupted = barge_after_s is not None
        spoken = words_played(spoken_parts, played_s) if interrupted else intended
        utterance = Utterance(
            speaker=Speaker.BOT,
            start_s=start_s,
            end_s=end_s,
            intended_text=intended,
            spoken_text=spoken,
            heard_text=heard,
            interrupted=interrupted,
        )
        self.recorder.add_event(utterance)
        await self._emit(progress.turn(self.artifact.id, utterance))
        self.clock.advance_to(end_s)
        barge_at_s = None
        if barge_after_s is not None:
            barge_at_s = start_s + barge_after_s
            self._queue(
                BargeInEvent(
                    at_s=barge_at_s, bot_stopped_at_s=end_s, interrupted_utterance_id=utterance.id
                )
            )
            self.support.on_interrupted(spoken)
        if not heard:
            self._system("nothing_heard", Speaker.CALLER.value)
        await self._flush()
        return _BotTurn(heard, barge_at_s)

    async def _caller_speaks(self, line: CallerLine, barge_at_s: float | None) -> str:
        """Play the caller's line, over the bot's speech if it cut in, and return what was heard."""
        if not line.text:
            return ""
        self._partial(Speaker.CALLER, line.text)
        await self._flush()
        voice = self.request.persona.voice
        clip = await self._speak(
            self.stack.tts_caller, line.text, voice.voice_id, voice.style, "tts_caller"
        )
        arrived, heard = await self._cross(self.to_bot, clip)
        if barge_at_s is None:
            start_s = self.clock.now + self.rng.uniform(*self.config.react_s) + self.latency_s
        else:
            start_s = barge_at_s
        end_s = start_s + arrived.duration_s
        self.recorder.place(Speaker.CALLER, arrived, start_s)
        utterance = Utterance(
            speaker=Speaker.CALLER,
            start_s=start_s,
            end_s=end_s,
            intended_text=line.text,
            spoken_text=line.text,
            heard_text=heard,
        )
        self.recorder.add_event(utterance)
        await self._emit(progress.turn(self.artifact.id, utterance))
        self.clock.advance_to(end_s)
        return heard

    async def _speak(
        self, tts: TTS, text: str, voice_id: str, style: str | None, step: str
    ) -> AudioClip:
        clip = await self._step(step, tts.synthesize(text, voice_id=voice_id, style=style))
        if clip.n_samples == 0:
            raise _CallFailed(f"{step} returned no audio for {text!r}")
        return clip

    async def _cross(self, line: PhoneLine, clip: AudioClip) -> tuple[AudioClip, str]:
        """One direction of the line: returns the audio that came out and what STT made of it."""
        started = time.perf_counter()
        # The line is a Python sample loop, and other calls share this event loop.
        arrived = await asyncio.to_thread(line.process, clip)
        self.compute.add("phone_line", self.turn, (time.perf_counter() - started) * 1000)
        transcript = await self._step(
            "stt_live", self.stack.stt_live.transcribe(arrived, language="en")
        )
        return arrived, transcript.text.strip()

    @property
    def bot_voice(self) -> str:
        return self.stack.profile.roles["tts_bot"].get("voice_id", "default")

    async def _step(self, step: str, work: Awaitable[T]) -> T:
        """Run one model call. A model that hangs or errors ends the call, not the process."""
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.config.step_timeout_s):
                return await work
        except TimeoutError as failure:
            limit = self.config.step_timeout_s
            raise _CallFailed(f"{step} timed out after {limit:g} s") from failure
        except StopAsyncIteration:
            raise
        except Exception as failure:
            logger.warning("%s failed on call %s: %r", step, self.artifact.id, failure)
            raise _CallFailed(f"{step} failed: {failure!r}") from failure
        finally:
            if not step.startswith("llm_"):  # `_MeteredLLM` measures those from inside
                self.compute.add(step, self.turn, (time.perf_counter() - started) * 1000)

    def _queue(self, event: ToolCallEvent | SystemEvent | BargeInEvent) -> None:
        """Add to the timeline at once. Runs inside `SupportBot`, which can't await."""
        self.recorder.add_event(event)
        self._unsent.append(event)

    def _system(self, kind: str, detail: str = "") -> None:
        self._queue(SystemEvent(at_s=self.clock.now, kind=kind, detail=detail))

    def _partial(self, speaker: Speaker, text: str) -> None:
        self._unsent.append(progress.partial(self.artifact.id, speaker, text, self.clock.now))

    async def _flush(self) -> None:
        unsent, self._unsent = self._unsent, []
        for item in unsent:
            is_event = not isinstance(item, dict)
            await self._emit(progress.event(self.artifact.id, item) if is_event else item)

    async def _emit(self, message: dict[str, Any]) -> None:
        # A dashboard that went away must not take the call down with it.
        try:
            await self.emit(message)
        except Exception:
            logger.exception("progress emit failed on call %s", self.artifact.id)

    def _fail(self, message: str) -> None:
        self.artifact.status = CallStatus.FAILED
        self.artifact.error = message
        self._system("call_failed", message)

    def _close_out(self, db: CompanyDb | None) -> None:
        """Write everything known about the call. Runs however the call ended."""
        last_audio = max(
            self.recorder.channel_end_s(Speaker.CALLER), self.recorder.channel_end_s(Speaker.BOT)
        )
        self.clock.advance_to(last_audio)
        self._system("call_ended")
        self.artifact.metadata["compute"] = self.compute.summary()
        if db is not None:
            try:
                self.artifact.db_changes = diff_against_seed(db)
            except Exception as failure:
                logger.exception("database diff failed on call %s", self.artifact.id)
                self.artifact.error = self.artifact.error or f"database diff failed: {failure!r}"
            finally:
                db.close()
        self.recorder.finish()
