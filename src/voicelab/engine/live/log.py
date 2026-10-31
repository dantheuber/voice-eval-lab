"""One live call's record as it happens: the timeline, messages to watchers, and the recorder."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from voicelab.core.models import (
    BargeInEvent,
    CallArtifact,
    Speaker,
    SystemEvent,
    ToolCallEvent,
    Utterance,
)
from voicelab.engine import progress
from voicelab.engine.contract import Emit
from voicelab.engine.recorder import CallRecorder
from voicelab.store.artifacts import ArtifactStore

logger = logging.getLogger(__name__)

_HEARD_WAIT_S = 2.0  # the far side has VAD hang and STT to get through
_SEGMENT_SLACK_S = 0.6  # a segment ends where VAD saw it end, a little past the audio's end


class LiveLog:
    """Everything here is synchronous, so a brain or a tool callback can record without waiting."""

    def __init__(self, artifact: CallArtifact, store: ArtifactStore, emit: Emit) -> None:
        self.artifact = artifact
        self.recorder = CallRecorder(artifact, store)
        self._emit = emit
        self._outbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None
        self._open: dict[Speaker, list[Utterance]] = {s: [] for s in Speaker}
        self._heard: dict[Speaker, deque[tuple[float, str]]] = {s: deque() for s in Speaker}

    def start(self) -> None:
        self.recorder.start()
        self._pump = asyncio.create_task(self._run_pump())

    def now(self) -> float:
        return self.recorder.now()

    def post(self, message: dict[str, Any]) -> None:
        self._outbox.put_nowait(message)

    async def _run_pump(self) -> None:
        while (message := await self._outbox.get()) is not None:
            # A dashboard that went away must not take the call down with it.
            try:
                await self._emit(message)
            except Exception:
                logger.exception("progress emit failed on call %s", self.artifact.id)

    def event(self, item: ToolCallEvent | SystemEvent | BargeInEvent) -> None:
        self.recorder.add_event(item)
        self.post(progress.event(self.artifact.id, item))

    def system(self, kind: str, detail: str = "") -> None:
        self.event(SystemEvent(at_s=self.now(), kind=kind, detail=detail))

    def partial(self, speaker: Speaker, text: str) -> None:
        self.post(progress.partial(self.artifact.id, speaker, text, self.now()))

    def utterance(self, item: Utterance, *, far_side_listens: bool) -> Utterance:
        """Add a finished turn. With a bot on the far end, its STT gets a moment to report."""
        self.recorder.add_event(item)
        if not far_side_listens:
            self.post(progress.turn(self.artifact.id, item))
            return item
        self._open[item.speaker].append(item)
        self._attach(item.speaker)
        asyncio.get_running_loop().call_later(_HEARD_WAIT_S, self._send_turn, item)
        return item

    def hear(self, speaker: Speaker, text: str, at_s: float) -> None:
        """Record what the far side's STT made of `speaker`, in a segment that ended at `at_s`.

        One turn can come through as several segments, and they can land either side of the turn
        being closed, so they are matched on time and not on arrival order.
        """
        self._heard[speaker].append((at_s, text))
        self._attach(speaker)

    def _attach(self, speaker: Speaker) -> None:
        heard = self._heard[speaker]
        for item in self._open[speaker]:
            while heard and heard[0][0] <= item.end_s + _SEGMENT_SLACK_S:
                _, text = heard.popleft()
                item.heard_text = f"{item.heard_text} {text}" if item.heard_text else text

    def _send_turn(self, item: Utterance) -> None:
        if item in self._open[item.speaker]:
            self._open[item.speaker].remove(item)
            item.heard_text = item.heard_text or ""
            self.post(progress.turn(self.artifact.id, item))

    def send_waiting_turns(self) -> None:
        for items in self._open.values():
            for item in list(items):
                self._send_turn(item)

    async def close(self) -> None:
        """Send what's left. Call only after the artifact is written."""
        self.send_waiting_turns()
        self._outbox.put_nowait(None)
        if self._pump is not None:
            await self._pump
