"""Accumulates one call on a shared timeline and writes its artifact.

Audio is kept as it arrived and only rendered in `finish`. Resampling each 20 ms frame on its own
would click at every boundary, and resampling them as one run at the end doesn't.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from voicelab.core.audio import AudioClip
from voicelab.core.models import CallArtifact, Speaker, TimelineEvent, Utterance, utcnow
from voicelab.store.artifacts import CALL_SAMPLE_RATE, ArtifactStore

# A live frame this close to the end of the previous one continues it. Frame timestamps jitter by
# a few milliseconds, and without this each frame would leave a tiny gap or overlap.
_SNAP_S = 0.04


@dataclass(eq=False)
class Placement:
    """A run of audio placed on the timeline. `cut_s` is where it was truncated, if it was."""

    speaker: Speaker
    start_s: float
    sample_rate: int
    pcm: list[bytes] = field(default_factory=list)
    cut_s: float | None = None
    _samples: int = 0

    def append(self, clip: AudioClip) -> None:
        self.pcm.append(clip.pcm)
        self._samples += clip.n_samples

    @property
    def end_s(self) -> float:
        full = self.start_s + self._samples / self.sample_rate
        return full if self.cut_s is None else min(full, self.cut_s)

    def render(self) -> np.ndarray:
        clip = AudioClip(b"".join(self.pcm), self.sample_rate).resampled(CALL_SAMPLE_RATE)
        samples = clip.samples()
        if self.cut_s is not None:
            keep = max(0, round((self.cut_s - self.start_s) * CALL_SAMPLE_RATE))
            samples = samples[:keep]
        return samples


class CallRecorder:
    """Builds a call from placed audio and timeline events.

    Turn-based calls use `place` with computed virtual times. Live calls use `write_frame` as
    frames arrive. Channels are independent, so caller and bot may overlap freely. Two placements
    on one channel that overlap are mixed, as they would be on a real line.
    """

    def __init__(
        self,
        artifact: CallArtifact,
        store: ArtifactStore | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.artifact = artifact
        self.store = store or ArtifactStore()
        self._clock = clock
        self._started: float | None = None
        self._placements: list[Placement] = []
        self._live: dict[Speaker, Placement] = {}

    def start(self) -> None:
        """Set time zero for `now` and live frames. The first of those calls it if you don't."""
        if self._started is None:
            self._started = self._clock()
            if self.artifact.started_at is None:
                self.artifact.started_at = utcnow()

    def now(self) -> float:
        """Seconds since the call started."""
        self.start()
        return self._clock() - self._started

    def place(self, speaker: Speaker, clip: AudioClip, at_s: float) -> Placement:
        """Put a whole clip at a position. Keep the result to truncate this clip alone later."""
        placement = Placement(speaker, at_s, clip.sample_rate)
        placement.append(clip)
        self._placements.append(placement)
        return placement

    def write_frame(self, speaker: Speaker, frame: AudioClip, at_s: float | None = None) -> float:
        """Add a live frame and return where it landed.

        With no `at_s`, the frame goes at the current call time, or right after the channel's
        previous frame if that one reaches past the current time. A frame that lands near the
        end of the previous one joins it, and a later one starts a new run after a silent gap.
        """
        current = self._live.get(speaker)
        position = self._position(current, at_s)
        joins = (
            current is not None
            and current.sample_rate == frame.sample_rate
            and abs(position - current.end_s) <= _SNAP_S
        )
        if joins:
            position = current.end_s
        else:
            current = Placement(speaker, position, frame.sample_rate)
            self._placements.append(current)
            self._live[speaker] = current
        current.append(frame)
        return position

    def _position(self, current: Placement | None, at_s: float | None) -> float:
        if at_s is not None:
            return at_s
        now = self.now()
        return now if current is None else max(now, current.end_s)

    def truncate(
        self, speaker: Speaker, at_s: float, *, placement: Placement | None = None
    ) -> None:
        """Remove `speaker`'s audio from `at_s` on, as when an utterance is cut off.

        Pass `placement` to cut only that clip. Otherwise everything on the channel past `at_s`
        goes, which covers live audio queued ahead of the clock.
        """
        targets = [placement] if placement else self._channel(speaker)
        for target in targets:
            target.cut_s = at_s if target.cut_s is None else min(target.cut_s, at_s)
        # A later frame must start a new run, or it would land inside the removed part.
        live = self._live.get(speaker)
        if live is not None and live in targets:
            del self._live[speaker]

    def add_event(self, event: TimelineEvent) -> None:
        self.artifact.timeline.append(event)

    def channel_end_s(self, speaker: Speaker) -> float:
        """Where this channel's audio currently ends."""
        return max((p.end_s for p in self._channel(speaker)), default=0.0)

    def finish(self) -> CallArtifact:
        """Render both channels and write the artifact. Sets timeline order, duration and end."""
        artifact = self.artifact
        artifact.timeline.sort(key=_event_time)
        last_event = max((_event_end(e) for e in artifact.timeline), default=0.0)
        duration = max(
            last_event, self.channel_end_s(Speaker.CALLER), self.channel_end_s(Speaker.BOT)
        )
        caller = self._render(Speaker.CALLER, duration)
        bot = self._render(Speaker.BOT, duration)
        artifact.duration_s = caller.duration_s
        artifact.ended_at = artifact.ended_at or utcnow()
        self.store.write_call(artifact, caller, bot)
        return artifact

    def _channel(self, speaker: Speaker) -> list[Placement]:
        return [p for p in self._placements if p.speaker == speaker]

    def _render(self, speaker: Speaker, duration_s: float) -> AudioClip:
        channel = np.zeros(round(duration_s * CALL_SAMPLE_RATE), dtype=np.float32)
        for placement in self._channel(speaker):
            samples = placement.render()
            start = round(placement.start_s * CALL_SAMPLE_RATE)
            end = min(start + len(samples), len(channel))
            if end > start:
                channel[start:end] += samples[: end - start]
        return AudioClip.from_samples(channel, CALL_SAMPLE_RATE)


def _event_time(event: TimelineEvent) -> float:
    return event.start_s if isinstance(event, Utterance) else event.at_s


def _event_end(event: TimelineEvent) -> float:
    return event.end_s if isinstance(event, Utterance) else event.at_s
