"""The seam between the API layer and call orchestration.

`api/` depends on these signatures and the engine implements them, so neither needs the other's
internals. `emit` messages use the shapes in docs/API.md for run and talk streams.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from voicelab.core.models import CallArtifact, CallMode, LineConditions, Persona, Scenario
from voicelab.core.profiles import ProviderStack
from voicelab.store.artifacts import ArtifactStore

Emit = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class BotCallRequest:
    """One bot-to-bot call. `mode` is TURN_BASED or LIVE_BOTS, already resolved from "auto"."""

    call_id: str
    mode: CallMode
    persona: Persona
    scenario: Scenario
    batch_id: str | None = None
    seed: int = 0
    line: LineConditions | None = None  # overrides the persona's line when set
    metadata: dict[str, Any] = field(default_factory=dict)


class BotCallRunner(Protocol):
    async def __call__(
        self, request: BotCallRequest, *, stack: ProviderStack, store: ArtifactStore, emit: Emit
    ) -> CallArtifact:
        """Run the call to completion and return the finished artifact, ready for evaluation.

        Writes audio and `call.json` through `store`. Raises nothing for call-level failures:
        those end as an artifact with `status=FAILED` and `error` set. `asyncio.CancelledError`
        propagates, and a stopped run still writes whatever was recorded.
        """
        ...


@dataclass
class TalkRequest:
    """A person calling the bot from the browser. Persona and scenario are optional context."""

    call_id: str
    persona: Persona | None = None
    scenario: Scenario | None = None
    line: LineConditions | None = None


class TalkSession(Protocol):
    """A live microphone call. One session per call."""

    call_id: str

    async def handle_offer(self, sdp: str, sdp_type: str) -> dict[str, str]:
        """Answer a WebRTC offer. Returns `{"sdp": ..., "type": ...}`."""
        ...

    async def end(self) -> CallArtifact:
        """Hang up, finish the recording, and return the artifact, ready for evaluation."""
        ...


class TalkFactory(Protocol):
    async def __call__(
        self, request: TalkRequest, *, stack: ProviderStack, store: ArtifactStore, emit: Emit
    ) -> TalkSession: ...
