"""The seam between the call engine and whatever carries audio between the two ends."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from voicelab.core.audio import AudioClip


@runtime_checkable
class Transport(Protocol):
    """One end of a call. A SIP transport implements this with RTP and a real phone number.

    Frames are 20 ms of mono PCM. The far end's line conditions and delay belong to the transport:
    a real network supplies them, and a simulated one applies a `PhoneLine`.
    """

    sample_rate: int

    async def send(self, frame: AudioClip) -> None:
        """Queue one frame toward the far end. Never waits for the far end to hear it."""
        ...

    def receive(self) -> AsyncIterator[AudioClip]:
        """Yield frames from the far end until the call ends."""
        ...

    async def hang_up(self) -> None:
        """End the call. `receive` stops after any frames still in flight."""
        ...
