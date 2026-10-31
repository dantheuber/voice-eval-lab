"""The thinking side of a live agent: what to do when the far side talks or stops talking.

Model calls run as background tasks, because `process_frame` has to stay quick or audio stalls. A
brain only pushes text. TTS, the output transport, and the tap handle the rest, and they tell the
brain how far the speech got through its `PlayoutLedger`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voicelab.agents.caller import MAX_TURNS_REASON, CallerBot, CallerLine
from voicelab.agents.speech import SpeechText
from voicelab.agents.support import SupportBot
from voicelab.core.audio import AudioClip
from voicelab.core.models import BargeInEvent, Speaker, Utterance
from voicelab.engine.heard import prepare_heard_text
from voicelab.engine.live.log import LiveLog
from voicelab.engine.live.parts import Fail, LiveSettings
from voicelab.engine.speech_ledger import PlayoutLedger
from voicelab.engine.turn_based import TYPICAL_TURN_S
from voicelab.providers.base import STT

logger = logging.getLogger(__name__)

_END_SLACK_S = 0.1  # the output drops a partial last chunk


async def _one(segment: SpeechText) -> AsyncIterator[SpeechText]:
    yield segment


class SpokenTurn:
    """Audio timing of the turn in progress, as seen at the tap."""

    def __init__(self) -> None:
        self.started_s: float | None = None
        self.ended_s: float | None = None

    def add_audio(self, now_s: float, duration_s: float) -> None:
        if self.started_s is None:
            self.started_s = now_s - duration_s
        self.ended_s = now_s

    def clear(self) -> None:
        self.started_s = self.ended_s = None


class Brain(FrameProcessor):
    """Shared by both bots: collect what was heard, and hold one turn's speech at a time."""

    speaker: Speaker

    def __init__(
        self,
        *,
        log: LiveLog,
        ledger: PlayoutLedger,
        settings: LiveSettings,
        fail: Fail,
        audio_delay_s: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.audio_delay_s = audio_delay_s  # channels record arrivals, and this is the trip there
        self.call_over = False
        self.log = log
        self.ledger = ledger
        self.settings = settings
        self.fail = fail
        self.spoken = SpokenTurn()
        self.heard_parts: list[str] = []
        self.far_side_started_s: float | None = None
        self.far_side_ended_s: float | None = None
        self.task: asyncio.Task | None = None  # the model work for the turn in progress
        self.turn_open = False
        self.sentences_pushed = 0
        self.bot_speaking = False

    def on_audio_out(self, clip: AudioClip) -> None:
        self.ledger.add_played(clip.duration_s)
        self.spoken.add_audio(self.log.now() + self.audio_delay_s, clip.duration_s)

    def take_heard(self) -> str:
        text = " ".join(self.heard_parts).strip()
        self.heard_parts = []
        return text

    def run(self, work: Awaitable[None]) -> None:
        """Do model work beside the pipeline. A failure in it fails the call."""

        async def guarded() -> None:
            try:
                await work
            except asyncio.CancelledError:
                raise
            except Exception as failure:
                logger.warning("%s failed: %r", type(self).__name__, failure)
                await self.fail(f"{self.role} failed: {failure!r}")

        self.task = self.create_task(guarded())

    role = "llm"

    async def stop_work(self) -> None:
        if self.task is not None and not self.task.done():
            await self.cancel_task(self.task)
        self.task = None

    @property
    def working(self) -> bool:
        return self.task is not None and not self.task.done()

    async def speak(self, segments: AsyncIterator[SpeechText]) -> None:
        """Push sentences as they come. Closing `segments` stops the model at the next word."""
        self.turn_open = True
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for segment in segments:
                self.log.partial(self.speaker, segment.spoken)
                self.sentences_pushed += 1
                await self.push_frame(LLMTextFrame(segment.spoken))
        finally:
            await segments.aclose()
        await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self.on_start()
            return
        if isinstance(frame, TranscriptionFrame):
            self.on_transcript(frame.text)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self.far_side_started_s = self.log.now() - frame.start_secs
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.far_side_ended_s = self.log.now() - frame.stop_secs
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self.on_far_side_started()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.on_far_side_finished()
        elif isinstance(frame, InterruptionFrame):
            await self.on_interrupted()
        elif isinstance(frame, BotStartedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
            self.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
            self.bot_speaking = False
            await self.on_speech_paused()
        elif isinstance(frame, InputAudioRawFrame):
            self.on_audio_in(frame)
        await self.push_frame(frame, direction)

    @property
    def all_speech_out(self) -> bool:
        """Every sentence pushed has gone through TTS and out. Either can finish last."""
        segments = self.ledger.segments
        if self.working or self.bot_speaking or len(segments) != self.sentences_pushed:
            return False
        made = sum(segment.produced_s for segment in segments)
        return all(s.finished for s in segments) and self.ledger.played_s >= made - _END_SLACK_S

    def begin_turn(self) -> None:
        self.ledger.reset()
        self.spoken.clear()
        self.sentences_pushed = 0

    async def on_start(self) -> None: ...

    async def on_far_side_started(self) -> None: ...

    async def on_far_side_finished(self) -> None: ...

    async def on_interrupted(self) -> None: ...

    async def on_speech_paused(self) -> None: ...

    def on_audio_in(self, frame: InputAudioRawFrame) -> None: ...

    def on_transcript(self, text: str) -> None:
        self.heard_parts.append(text)


class BotBrain(Brain):
    """The support side, however it's built. Barge-in works the same for a cascade and realtime."""

    speaker = Speaker.BOT

    def __init__(
        self,
        *,
        far_side_listens: bool,
        on_turn_done: Callable[[Utterance], None],
        on_call_ended: Callable[[str], Awaitable[None]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.far_side_listens = far_side_listens
        self.on_turn_done = on_turn_done
        self.on_call_ended = on_call_ended
        self.cut_pending = False
        self.barged_at_s: float | None = None
        self.caller_started_s: float | None = None

    @property
    def intended_text(self) -> str:
        return self.ledger.intended_text

    @property
    def played_text(self) -> str:
        return self.ledger.spoken_text

    async def on_far_side_started(self) -> None:
        self.caller_started_s = self.far_side_started_s

    async def on_far_side_finished(self) -> None:
        heard = self.take_heard()
        interrupted_by_nothing = self.barged_at_s is not None and not heard
        self.barged_at_s = None
        self._log_caller_turn(heard)
        if interrupted_by_nothing:
            # The bot stopped for noise or leftover echo. Count it, since this is the number to
            # watch when tuning barge-in.
            self.log.system("false_barge_in")
            count = self.log.artifact.metadata.get("false_barge_ins", 0)
            self.log.artifact.metadata["false_barge_ins"] = count + 1
        if heard and not self.call_over:
            await self.answer(heard)

    def _log_caller_turn(self, heard: str) -> None:
        if self.far_side_listens:
            self.log.hear(Speaker.CALLER, heard)
        elif heard:
            start = self.caller_started_s if self.caller_started_s is not None else self.log.now()
            end = self.far_side_ended_s if self.far_side_ended_s is not None else self.log.now()
            self.log.utterance(
                Utterance(
                    speaker=Speaker.CALLER, start_s=start, end_s=max(start, end), heard_text=heard
                ),
                far_side_listens=False,
            )

    async def answer(self, heard: str) -> None:
        raise NotImplementedError

    async def stop_model(self, played_ms: int) -> None:
        """Stop generating. Runs at the cut, before the output has settled."""
        await self.stop_work()

    async def forget_unheard(self, played_text: str) -> None:
        """Make the model's memory match what was heard."""
        raise NotImplementedError

    def close_turn(self, *, interrupted: bool) -> Utterance | None:
        self.turn_open = False
        started, ended = self.spoken.started_s, self.spoken.ended_s
        if started is None or ended is None:
            return None  # nothing got out, so as far as the caller knows the turn never happened
        ttfb = None
        if self.far_side_ended_s is not None:
            ttfb = max(0.0, (started - self.far_side_ended_s) * 1000)
        item = Utterance(
            speaker=Speaker.BOT,
            start_s=started,
            end_s=ended,
            intended_text=self.intended_text,
            spoken_text=self.played_text if interrupted else self.intended_text,
            interrupted=interrupted,
            ttfb_ms=ttfb,
        )
        self.log.utterance(item, far_side_listens=self.far_side_listens)
        self.on_turn_done(item)
        self.spoken.clear()
        return item

    async def on_interrupted(self) -> None:
        if not self.turn_open or self.cut_pending:
            return
        self.cut_pending = True
        self.barged_at_s = self.log.now()
        await self.stop_model(round(self.ledger.played_s * 1000))
        self.create_task(self._settle_cut())

    async def _settle_cut(self) -> None:
        """Wait for the chunk already being written, then count what really went out."""
        await asyncio.sleep(self.settings.settle_s)
        await self.forget_unheard(self.played_text)
        self.cut_pending = False
        stopped_s = self.spoken.ended_s
        item = self.close_turn(interrupted=True)
        if item is None:
            return
        started = self.caller_started_s if self.caller_started_s is not None else self.barged_at_s
        self.log.event(
            BargeInEvent(at_s=started, bot_stopped_at_s=stopped_s, interrupted_utterance_id=item.id)
        )


class SupportBrain(BotBrain):
    """Delegates to `SupportBot`, so tools and `on_interrupted` work as they do turn-based."""

    role = "llm_support"

    def __init__(self, support: SupportBot, **kwargs) -> None:
        super().__init__(**kwargs)
        self.support = support

    async def greet(self) -> None:
        self.begin_turn()
        self.run(self._say(_one(self.support.greet())))

    async def answer(self, heard: str) -> None:
        self.begin_turn()
        self.run(self._say(self.support.respond(prepare_heard_text(heard))))

    async def _say(self, segments: AsyncIterator[SpeechText]) -> None:
        await self.speak(segments)
        self.task = None  # this is the last line of the work, so `all_speech_out` can pass
        await self.on_speech_paused()

    async def forget_unheard(self, played_text: str) -> None:
        self.support.on_interrupted(played_text)

    async def on_speech_paused(self) -> None:
        if not self.turn_open or self.cut_pending or not self.all_speech_out:
            return
        self.close_turn(interrupted=False)
        if self.support.ended:
            await self.on_call_ended(self.support.end_reason)


class CallerBrain(Brain):
    """Delegates to `CallerBot`. It may cut in, which makes the far side's barge-in fire for real.

    It hears through its own STT and speaks through its own TTS, so it knows only what it heard.
    """

    speaker = Speaker.CALLER
    role = "llm_caller"

    def __init__(
        self,
        caller: CallerBot,
        *,
        stt: STT,
        on_hang_up: Callable[[str, str], Awaitable[None]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.caller = caller
        self.stt = stt
        self.on_hang_up = on_hang_up
        self.bot_turn = -1  # the greeting is turn 0, as in turn-based calls
        self.cut_in_this_turn = False
        self.cut_timer: asyncio.Task | None = None
        self.leaving: CallerLine | None = None
        self.intended = ""
        self.audio_this_turn: list[AudioClip] = []
        self.hearing = False

    def on_audio_in(self, frame: InputAudioRawFrame) -> None:
        if self.hearing:
            self.audio_this_turn.append(AudioClip(frame.audio, frame.sample_rate))

    async def on_far_side_started(self) -> None:
        self.bot_turn += 1
        self.cut_in_this_turn = False
        self.hearing = True
        self.audio_this_turn = []
        decision = self.caller.decide_interruption(self.bot_turn)
        if decision.interrupt:
            wait_s = decision.after_fraction * TYPICAL_TURN_S
            self.cut_timer = self.create_task(self._cut_in(wait_s))

    async def _cut_in(self, wait_s: float) -> None:
        await asyncio.sleep(wait_s)
        self.cut_in_this_turn = True
        if not self.audio_this_turn:
            return
        audio = self.audio_this_turn[0]
        for clip in self.audio_this_turn[1:]:
            audio = audio + clip
        self.run(self._heard_so_far(audio))

    async def _heard_so_far(self, audio: AudioClip) -> None:
        transcript = await self.stt.transcribe(audio)
        await self._say_next(transcript.text.strip())

    async def on_far_side_finished(self) -> None:
        self.hearing = False
        if self.cut_timer is not None and not self.cut_timer.done():
            await self.cancel_task(self.cut_timer)
        self.cut_timer = None
        heard = self.take_heard()
        self.log.hear(Speaker.BOT, heard)
        if self.cut_in_this_turn or self.call_over:
            return  # already answered, over the top of the bot
        self.run(self._say_next(heard))

    async def _say_next(self, heard: str) -> None:
        line = await self.caller.next_line(heard)
        if line.completed_beat:
            self.log.system("beat_completed", line.completed_beat)
        if line.hang_up:
            self.leaving = line
        if not line.text:
            if line.hang_up:
                await self._hang_up(line)
            return
        self.begin_turn()
        self.intended = line.text
        await self.speak(_one(SpeechText(display=line.text, spoken=line.text)))
        self.task = None
        await self.on_speech_paused()

    async def on_speech_paused(self) -> None:
        if not self.turn_open or not self.all_speech_out or self.spoken.started_s is None:
            return
        self.turn_open = False
        self.log.utterance(
            Utterance(
                speaker=Speaker.CALLER,
                start_s=self.spoken.started_s,
                end_s=self.spoken.ended_s,
                intended_text=self.intended,
                spoken_text=self.intended,
            ),
            far_side_listens=True,
        )
        if self.leaving is not None:
            await self._hang_up(self.leaving)

    async def _hang_up(self, line: CallerLine) -> None:
        self.leaving = None
        self.call_over = True
        at_limit = line.hang_up_reason == MAX_TURNS_REASON
        kind = MAX_TURNS_REASON if at_limit else "caller_hung_up"
        await self.on_hang_up(kind, line.hang_up_reason)

    async def on_idle(self) -> None:
        """Nobody has said anything for a while, so the caller speaks up rather than wait."""
        if not self.working and not self.turn_open and not self.call_over:
            self.run(self._say_next(""))
