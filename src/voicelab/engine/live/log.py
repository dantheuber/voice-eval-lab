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


class LiveLog:
    """Everything here is synchronous, so a brain or a tool callback can record without waiting."""

    def __init__(self, artifact: CallArtifact, store: ArtifactStore, emit: Emit) -> None:
        self.artifact = artifact
        self.recorder = CallRecorder(artifact, store)
        self._emit = emit
        self._outbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None
        # A turn waits here until the far side's STT reports what it heard.
        self._waiting: dict[Speaker, deque[Utterance]] = {s: deque() for s in Speaker}
        self._early: dict[Speaker, deque[str]] = {s: deque() for s in Speaker}

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
        """Add a finished turn. A bot on the far end must hear it before it goes out."""
        self.recorder.add_event(item)
        if far_side_listens:
            early = self._early[item.speaker]
            if early:
                item.heard_text = early.popleft()
            else:
                self._waiting[item.speaker].append(item)
                return item
        self.post(progress.turn(self.artifact.id, item))
        return item

    def hear(self, speaker: Speaker, text: str) -> None:
        """Record what the far side's STT made of `speaker`'s latest turn."""
        waiting = self._waiting[speaker]
        if waiting:
            item = waiting.popleft()
            item.heard_text = text
            self.post(progress.turn(self.artifact.id, item))
        else:
            self._early[speaker].append(text)

    async def close(self) -> None:
        """Send what's left. Call only after the artifact is written."""
        for waiting in self._waiting.values():
            for item in waiting:
                item.heard_text = item.heard_text or ""
                self.post(progress.turn(self.artifact.id, item))
            waiting.clear()
        self._outbox.put_nowait(None)
        if self._pump is not None:
            await self._pump
