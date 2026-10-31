"""Pipecat processors around our provider interfaces, so one adapter works in every call mode."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601

from voicelab.core.audio import AudioClip
from voicelab.core.models import LineConditions, Speaker
from voicelab.engine.echo import EchoDetector
from voicelab.engine.live.log import LiveLog
from voicelab.engine.speech_ledger import PlayoutLedger
from voicelab.providers.base import STT, TTS
from voicelab.telephony import LINE_RATE, PhoneLine

Fail = Callable[[str], Awaitable[None]]

_VAD_HANG_S = 0.3  # measured: longer adds latency, and Smart Turn covers pauses inside a turn
_VAD_START_S = 0.2
_RESUME_WINDOW_S = 0.5  # on top of the VAD hang, so about p90 of the gaps inside a turn
_STT_P99_S = 0.4  # Parakeet runs at about 150x real time, so even a long turn is quick


def _silero() -> VADAnalyzer:
    return SileroVADAnalyzer(params=VADParams(start_secs=_VAD_START_S, stop_secs=_VAD_HANG_S))


def _smart_turn(*, wait_for_transcript: bool = True) -> list[BaseUserTurnStopStrategy]:
    """Smart Turn, plus a window to carry on, and both have to agree the turn is over.

    A sentence that ended looks finished to Smart Turn even with more to come. Gaps inside one
    turn measured 0.32 s at the median and 0.5 to 0.76 s at p90 across Pocket and Phonon voices.
    Without the window, turns ended at those gaps and the bot talked over its caller.
    """
    return [
        TurnAnalyzerUserTurnStopStrategy(
            turn_analyzer=LocalSmartTurnAnalyzerV3(), wait_for_transcript=wait_for_transcript
        ),
        SpeechTimeoutUserTurnStopStrategy(
            user_speech_timeout=_RESUME_WINDOW_S, wait_for_transcript=wait_for_transcript
        ),
    ]


class EnergyVAD(VADAnalyzer):
    """Plain level detection, for audio Silero wouldn't call speech, such as the fake voice."""

    def __init__(self, threshold: float = 0.01) -> None:
        super().__init__(
            params=VADParams(confidence=0.5, start_secs=0.06, stop_secs=0.25, min_volume=0.0)
        )
        self._threshold = threshold

    def num_frames_required(self) -> int:
        return round(self.sample_rate * 0.02)

    def voice_confidence(self, buffer: bytes) -> float:
        return (
            1.0 if level(AudioClip(buffer, self.sample_rate).samples()) > self._threshold else 0.0
        )


def _timed_stop(*, wait_for_transcript: bool = True) -> list[BaseUserTurnStopStrategy]:
    return [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.05)]


@dataclass(frozen=True)
class LiveSettings:
    """What a live pipeline is built from. Tests swap the models for simple stand-ins."""

    vad: Callable[[], VADAnalyzer] = _silero
    turn_stop: Callable[..., list[BaseUserTurnStopStrategy]] = field(default=_smart_turn)
    stop_timeout_s: float = 5.0  # an open turn with no transcript ends after this long
    caller_patience_s: float = 12.0  # a caller left in silence this long speaks up
    settle_s: float = 0.06  # the output's in-flight chunk finishes within this after a cut
    settle_after_end_s: float = 1.5  # time for the far side to hear the last words of a call
    filler_after_s: float | None = 0.6  # say "one moment" if the tool call took this long to come


def simple_settings() -> LiveSettings:
    """For stacks with no real speech in them: the fake profile, and tests."""
    return LiveSettings(vad=EnergyVAD, turn_stop=_timed_stop, stop_timeout_s=1.5)


class WrappedSTT(SegmentedSTTService):
    """Our `STT` on one VAD segment at a time. A failure ends the call instead of going quiet."""

    def __init__(self, stt: STT, *, fail: Fail, role: str = "stt_live", **kwargs) -> None:
        super().__init__(
            settings=STTSettings(model=stt.name, language=None),
            ttfs_p99_latency=_STT_P99_S,
            **kwargs,
        )
        self._stt = stt
        self._fail = fail
        self._role = role

    @property
    def wants_wav_segments(self) -> bool:
        return False

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        try:
            transcript = await self._stt.transcribe(AudioClip(audio, self.sample_rate))
        except Exception as failure:
            await self._fail(f"{self._role} failed: {failure!r}")
            return
        text = transcript.text.strip()
        if text:
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601())


class WrappedTTS(TTSService):
    """Our `TTS`, streaming. Text arrives a sentence at a time, so nothing waits to aggregate."""

    def __init__(
        self,
        tts: TTS,
        *,
        voice_id: str,
        ledger: PlayoutLedger,
        fail: Fail,
        role: str,
        style: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            text_aggregation_mode=TextAggregationMode.TOKEN,
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=tts.sample_rate,
            settings=TTSSettings(model=tts.name, voice=voice_id, language=None),
            **kwargs,
        )
        self._tts = tts
        self._voice_id = voice_id
        self._style = style
        self._ledger = ledger
        self._fail = fail
        self._role = role

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        segment = self._ledger.start_segment(text)
        try:
            async for clip in self._tts.stream(text, voice_id=self._voice_id, style=self._style):
                await self.stop_ttfb_metrics()
                segment.produced_s += clip.duration_s
                yield TTSAudioRawFrame(
                    audio=clip.pcm,
                    sample_rate=clip.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as failure:
            await self._fail(f"{self._role} failed: {failure!r}")
        finally:
            segment.finished = True


class PlayoutTap(FrameProcessor):
    """Goes after `transport.output()`, which pushes a frame on only once it has been written.

    So what passes here is what really went out, and a cut can't count audio that was queued and
    thrown away. The browser's jitter buffer holds a little more back, which this can't see.
    """

    def __init__(self, on_audio: Callable[[AudioClip], None], **kwargs) -> None:
        super().__init__(**kwargs)
        self._on_audio = on_audio

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            self._on_audio(AudioClip(frame.audio, frame.sample_rate))
        await self.push_frame(frame, direction)


class EchoGate(FrameProcessor):
    """Blanks mic audio that is only the bot coming back, so VAD never sees it as a person.

    Browser echo cancellation does the first pass, and this catches what's left when someone is on
    speakers. The bot stays interruptible, since a person talking doesn't match the bot's contour.
    """

    def __init__(self, on_echo: Callable[[], None], **kwargs) -> None:
        super().__init__(**kwargs)
        self._detector = EchoDetector()
        self._on_echo = on_echo

    def add_outgoing(self, clip: AudioClip) -> None:
        self._detector.add_outgoing(clip.samples(), clip.sample_rate)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            samples = AudioClip(frame.audio, frame.sample_rate).samples()
            if self._detector.is_echo(samples, frame.sample_rate):
                self._on_echo()
                frame.audio = bytes(len(frame.audio))
        await self.push_frame(frame, direction)


class MicRecorder(FrameProcessor):
    """Keeps the caller's channel, at the point in the pipeline it's placed."""

    def __init__(self, log: LiveLog, **kwargs) -> None:
        super().__init__(**kwargs)
        self._log = log

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            clip = AudioClip(frame.audio, frame.sample_rate)
            self._log.recorder.write_frame(Speaker.CALLER, clip)
        await self.push_frame(frame, direction)


class LineStage(FrameProcessor):
    """A phone line in a mic call's path: mic audio inbound, bot audio outbound.

    Mic calls run the pipeline's input at the line's rate when there's a codec, since a person on
    a real phone line would reach the bot at 8 kHz.
    """

    def __init__(self, conditions: LineConditions, seed: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._to_bot = PhoneLine(conditions, seed=seed * 2)
        self._to_caller = PhoneLine(conditions, seed=seed * 2 + 1)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            clip = self._to_bot.process_frame(AudioClip(frame.audio, frame.sample_rate))
            if clip.n_samples:
                frame.audio, frame.sample_rate = clip.pcm, clip.sample_rate
                await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSAudioRawFrame):
            clip = self._to_caller.process_frame(AudioClip(frame.audio, frame.sample_rate))
            if clip.n_samples:
                frame.audio, frame.sample_rate = clip.pcm, clip.sample_rate
                await self.push_frame(frame, direction)
            return
        await self.push_frame(frame, direction)


def input_rate(line: LineConditions | None) -> int:
    return LINE_RATE if line is not None and line.codec == "mulaw_8k" else 16000


def voice_pipeline(
    transport: BaseTransport,
    *,
    settings: LiveSettings,
    stt: STT,
    fail: Fail,
    brain: FrameProcessor,
    tts_service: FrameProcessor | None,
    tap: FrameProcessor,
    can_be_interrupted: bool = True,
    wait_for_transcript: bool = True,
    idle_s: float = 0.0,
    before_vad: tuple[FrameProcessor, ...] = (),
    after_brain: tuple[FrameProcessor, ...] = (),
) -> tuple[Pipeline, UserTurnProcessor]:
    """The stages every live agent shares. Order matters.

    STT goes ahead of the turn processor because the stop strategy waits on its transcripts, and
    the tap goes after the output because that's where audio is known to have left.
    """
    turns = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(enable_interruptions=can_be_interrupted)],
            stop=settings.turn_stop(wait_for_transcript=wait_for_transcript),
        ),
        user_turn_stop_timeout=settings.stop_timeout_s,
        user_idle_timeout=idle_s,
    )
    stages: list[FrameProcessor] = [
        transport.input(),
        *before_vad,
        VADProcessor(vad_analyzer=settings.vad()),
        WrappedSTT(stt, fail=fail),
        turns,
        brain,
    ]
    if tts_service is not None:
        stages.append(tts_service)
    stages.extend([*after_brain, transport.output(), tap])
    return Pipeline(stages), turns


def level(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
