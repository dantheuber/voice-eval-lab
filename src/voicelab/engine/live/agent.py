"""Builds the support side of a live call, which both call types share."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from voicelab.agents.support import SupportBot, support_instructions
from voicelab.company.db import CompanyDb
from voicelab.company.tools import ToolDispatcher
from voicelab.core.models import Utterance
from voicelab.core.profiles import ProviderStack
from voicelab.engine.live.brains import BotBrain, SupportBrain
from voicelab.engine.live.log import LiveLog
from voicelab.engine.live.parts import Fail, LiveSettings, WrappedTTS
from voicelab.engine.speech_ledger import PlayoutLedger


def bot_voice(stack: ProviderStack) -> str:
    return stack.profile.roles["tts_bot"].get("voice_id", "default")


def support_side(
    stack: ProviderStack,
    *,
    db: CompanyDb,
    log: LiveLog,
    ledger: PlayoutLedger,
    settings: LiveSettings,
    fail: Fail,
    far_side_listens: bool,
    on_turn_done: Callable[[Utterance], None],
    on_call_ended: Callable[[str], Awaitable[None]],
    audio_delay_s: float = 0.0,
) -> tuple[BotBrain, WrappedTTS | None]:
    """The bot's brain, and its TTS service if it has one. A realtime model is its own TTS."""
    shared = {
        "log": log,
        "ledger": ledger,
        "settings": settings,
        "fail": fail,
        "far_side_listens": far_side_listens,
        "on_turn_done": on_turn_done,
        "on_call_ended": on_call_ended,
        "audio_delay_s": audio_delay_s,
    }
    dispatcher = ToolDispatcher(db)
    if stack.profile.support_engine == "realtime":
        if stack.realtime is None:
            raise ValueError(f"profile {stack.profile.id} has no realtime role")
        from voicelab.engine.live.realtime_brain import RealtimeBrain

        return (
            RealtimeBrain(
                stack.realtime,
                dispatcher=dispatcher,
                instructions=support_instructions(db.today),
                greeter=stack.tts_bot,
                greeter_voice=bot_voice(stack),
                **shared,
            ),
            None,
        )
    support = SupportBot(
        stack.llm_support,
        dispatcher,
        on_event=log.event,
        clock=log.now,
        filler_after_s=settings.filler_after_s,
    )
    tts = WrappedTTS(
        stack.tts_bot, voice_id=bot_voice(stack), ledger=ledger, fail=fail, role="tts_bot"
    )
    return SupportBrain(support, **shared), tts
