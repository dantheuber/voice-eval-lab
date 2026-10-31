"""The engine's public entry points, which `api/` calls through the seams in `contract.py`.

Live calls import Pipecat inside these functions, so the API starts without loading it.
"""

from __future__ import annotations

from voicelab.core.models import CallArtifact, CallMode
from voicelab.core.profiles import ProviderStack
from voicelab.engine.contract import BotCallRequest, Emit, TalkRequest, TalkSession
from voicelab.engine.turn_based import run_turn_based_call
from voicelab.store.artifacts import ArtifactStore


def _live_settings(stack: ProviderStack):
    """Real speech gets the real detectors. A stack with no real speech can't use them."""
    from voicelab.engine.live.parts import LiveSettings, simple_settings

    return LiveSettings() if stack.profile.live_capable else simple_settings()


async def run_bot_call(
    request: BotCallRequest, *, stack: ProviderStack, store: ArtifactStore, emit: Emit
) -> CallArtifact:
    """Run one bot-to-bot call in the mode already chosen for it. See `BotCallRunner`."""
    if request.mode == CallMode.TURN_BASED:
        return await run_turn_based_call(request, stack=stack, store=store, emit=emit)
    if request.mode == CallMode.LIVE_BOTS:
        from voicelab.engine.live.bots import run_live_bot_call

        settings = _live_settings(stack)
        return await run_live_bot_call(
            request, stack=stack, store=store, emit=emit, settings=settings
        )
    raise ValueError(f"not a bot-to-bot mode: {request.mode}")


async def create_talk_session(
    request: TalkRequest, *, stack: ProviderStack, store: ArtifactStore, emit: Emit
) -> TalkSession:
    """Open a live microphone call. See `TalkFactory`."""
    from voicelab.engine.live.talk import create_live_talk_session

    return await create_live_talk_session(
        request, stack=stack, store=store, emit=emit, settings=_live_settings(stack)
    )
