"""The engine's public entry points, which `api/` calls through the seams in `contract.py`."""

from __future__ import annotations

from voicelab.core.models import CallArtifact, CallMode
from voicelab.core.profiles import ProviderStack
from voicelab.engine.contract import BotCallRequest, Emit
from voicelab.engine.turn_based import run_turn_based_call
from voicelab.store.artifacts import ArtifactStore


async def run_bot_call(
    request: BotCallRequest, *, stack: ProviderStack, store: ArtifactStore, emit: Emit
) -> CallArtifact:
    """Run one bot-to-bot call in the mode already chosen for it. See `BotCallRunner`."""
    if request.mode == CallMode.TURN_BASED:
        return await run_turn_based_call(request, stack=stack, store=store, emit=emit)
    raise ValueError(f"not a bot-to-bot mode: {request.mode}")
