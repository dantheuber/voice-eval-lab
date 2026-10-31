"""An in-memory phone line between two pipelines, with audio moving at wall-clock pace.

Each direction ticks every 20 ms like RTP. It takes the next 20 ms of queued speech, or silence,
sends it through a `PhoneLine`, and delivers the result to the far end one line delay later. The
silence matters: a line never goes quiet, VAD needs a steady stream to notice speech has ended,
and noise has to keep playing between turns.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from voicelab.core.audio import AudioClip
from voicelab.telephony import PhoneLine

_TICK_S = 0.02


class Wire:
    """One direction of the call."""

    def __init__(
        self,
        line: PhoneLine,
        *,
        rate: int,
        on_arrival: Callable[[AudioClip], None] = lambda clip: None,
    ) -> None:
        self.line = line
        self.rate = rate
        self.on_arrival = on_arrival
        self.receiver: LinkInput | None = None
        self._queue: deque[tuple[bytes, asyncio.Future[None] | None]] = deque()
        self._task: asyncio.Task[None] | None = None
        self._silence = bytes(round(_TICK_S * rate) * 2)
        self.late_ticks = 0

    def write(self, clip: AudioClip) -> asyncio.Future[None]:
        """Queue audio. The future resolves as its last tick goes out, which paces the sender."""
        audio = clip.resampled(self.rate).pcm
        size = len(self._silence)
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        chunks = [audio[i : i + size] for i in range(0, len(audio), size)] or [b""]
        for index, chunk in enumerate(chunks):
            padded = chunk.ljust(size, b"\x00")
            self._queue.append((padded, done if index == len(chunks) - 1 else None))
        return done

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for _, done in self._queue:
            if done is not None and not done.done():
                done.cancel()
        self._queue.clear()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        delay = self.line.latency_s
        while True:
            pcm, done = self._queue.popleft() if self._queue else (self._silence, None)
            arrived = self.line.process_frame(AudioClip(pcm, self.rate))
            if done is not None and not done.done():
                done.set_result(None)
            if arrived.n_samples:
                loop.call_at(loop.time() + delay, self._deliver, arrived)
            next_tick += _TICK_S
            wait = next_tick - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            else:
                self.late_ticks += 1
                await asyncio.sleep(0)

    def _deliver(self, clip: AudioClip) -> None:
        self.on_arrival(clip)
        if self.receiver is not None:
            asyncio.ensure_future(self.receiver.hear(clip))


class LinkInput(BaseInputTransport):
    def __init__(self, params: TransportParams, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._ready = False

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._ready = True

    async def hear(self, clip: AudioClip) -> None:
        if self._ready:
            await self.push_audio_frame(
                InputAudioRawFrame(audio=clip.pcm, sample_rate=clip.sample_rate, num_channels=1)
            )


class LinkOutput(BaseOutputTransport):
    def __init__(self, params: TransportParams, wire: Wire, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._wire = wire

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        await self._wire.write(AudioClip(frame.audio, frame.sample_rate))
        return True


class LinkEnd(BaseTransport):
    """One party's end of the call."""

    def __init__(self, *, outgoing: Wire, in_rate: int, out_rate: int) -> None:
        super().__init__()
        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=in_rate,
            audio_out_enabled=True,
            audio_out_sample_rate=out_rate,
        )
        self._input = LinkInput(params, name=f"{self.name}#input")
        self._output = LinkOutput(params, outgoing, name=f"{self.name}#output")

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    @property
    def receiver(self) -> LinkInput:
        return self._input
