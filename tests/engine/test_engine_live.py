"""Live calls through the real Pipecat pipeline at wall-clock pace, on the fake voice stack.

Each call takes seconds of real time, so the scripts stay short. Energy VAD stands in for Silero,
which rightly doesn't call tones speech. Real models are in test_engine_live_models.py.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest
from enginefakes import (
    CLEAN,
    DeafSTT,
    FailingTTS,
    HangingLLM,
    Messages,
    Rewriting,
    Scripted,
    persona,
    request,
    says,
    scenario,
    stack,
    uses,
)

from voicelab.agents import CallerBot, Interruption
from voicelab.agents.support import FILLER_LINE
from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    BargeInEvent,
    CallArtifact,
    CallMode,
    CallStatus,
    Speaker,
    SystemEvent,
    ToolCallEvent,
    Utterance,
)
from voicelab.engine.live.bots import run_live_bot_call
from voicelab.engine.live.parts import simple_settings
from voicelab.engine.service import run_bot_call
from voicelab.providers.adapters.fake_speech import decode_speech
from voicelab.store.artifacts import ArtifactStore

LIVE = CLEAN.model_copy(update={"one_way_latency_ms": 60})
LONG_REPLY = "I can look at that for you right now while we stay on the line together today."


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path)


def utterances(artifact: CallArtifact, speaker: Speaker | None = None) -> list[Utterance]:
    found = [e for e in artifact.timeline if isinstance(e, Utterance)]
    return [u for u in found if speaker is None or u.speaker == speaker]


def kinds(artifact: CallArtifact) -> list[str]:
    return [e.kind for e in artifact.timeline if isinstance(e, SystemEvent)]


async def live_call(store: ArtifactStore, *, emit=None, req=None, **adapters) -> CallArtifact:
    call = run_live_bot_call(
        req or request(mode=CallMode.LIVE_BOTS, persona=persona(line=LIVE)),
        stack=stack(**adapters),
        store=store,
        emit=emit or Messages(),
        settings=simple_settings(),
    )
    return await asyncio.wait_for(call, 45)


def quick_caller(*lines: str) -> Scripted:
    return Scripted(*lines, says("Bye.", hang_up=True, reason="finished"))


async def test_a_live_call_runs_in_real_time_and_writes_the_artifact(store):
    messages = Messages()
    plan = uses("list_plans", text="Let me check.")

    artifact = await live_call(
        store,
        emit=messages,
        llm_caller=quick_caller(says("Hello.", beat="only")),
        llm_support=Scripted(plan, default="We have plans."),
    )

    assert artifact.status == CallStatus.EVALUATING and artifact.error is None
    assert artifact.mode == CallMode.LIVE_BOTS and artifact.timing_is_real is True
    assert "caller_hung_up" in kinds(artifact) and kinds(artifact)[-1] == "call_ended"
    assert "beat_completed" in kinds(artifact)
    assert artifact.metadata["line"]["late_ticks"] < 5
    paths = store.paths(artifact.id)
    assert paths.recording.exists() and store.read_artifact(artifact.id).timing_is_real is True
    assert messages.items[0]["type"] == "callStarted"
    assert messages.items[-1]["type"] == "callDone"
    assert [m["utterance"]["heardText"] for m in messages.of("turn")][:2] == [
        utterances(artifact)[0].intended_text,
        "Hello.",
    ]


async def test_each_side_hears_only_what_came_out_of_its_stt(store):
    caller = quick_caller(says("My name is Dana."))
    support = Scripted(default="Your balance is fine.")
    hearing = Rewriting({"Dana": "Diana", "fine": "nine"})

    artifact = await live_call(store, llm_caller=caller, llm_support=support, stt_live=hearing)

    assert support.requests[0][-1].content == "My name is Diana."
    assert "Your balance is nine." in caller.requests[1][-1].content
    assert "is fine" not in caller.requests[1][-1].content
    reply = utterances(artifact, Speaker.BOT)[1]
    assert (reply.intended_text, reply.heard_text) == (
        "Your balance is fine.",
        "Your balance is nine.",
    )


async def test_timing_on_the_timeline_matches_the_audio_on_disk(store):
    artifact = await live_call(
        store,
        llm_caller=quick_caller(says("Hello there.")),
        llm_support=Scripted(default="Noted."),
    )

    bot = AudioClip.from_file(store.paths(artifact.id).bot)
    for turn in utterances(artifact, Speaker.BOT):
        heard = decode_speech(bot.slice(turn.start_s - 0.1, turn.end_s + 0.1))
        assert " ".join(word.text for word in heard) == turn.intended_text
    reply = utterances(artifact, Speaker.BOT)[1]
    assert reply.ttfb_ms is not None and 0 < reply.ttfb_ms < 2000


async def test_a_tool_lands_on_the_timeline_between_the_turns_around_it(store):
    tool_first = uses("list_plans")
    artifact = await live_call(
        store,
        llm_caller=quick_caller(says("What plans?")),
        llm_support=Scripted(tool_first, default="Several."),
    )

    tool = next(e for e in artifact.timeline if isinstance(e, ToolCallEvent))
    question = utterances(artifact, Speaker.CALLER)[0]
    reply = utterances(artifact, Speaker.BOT)[1]
    assert question.start_s < tool.at_s < reply.start_s


async def test_a_live_barge_in_cuts_the_bot_and_memory_keeps_only_what_went_out(store):
    support = Scripted(default="Okay.")
    # The greeting runs 5 s, and rate 1 makes the caller cut in part way through it.
    artifact = await live_call(
        store,
        req=request(mode=CallMode.LIVE_BOTS, persona=persona(interruption_rate=1.0, line=LIVE)),
        llm_caller=quick_caller(says("Hold on.")),
        llm_support=support,
    )

    barge = next(e for e in artifact.timeline if isinstance(e, BargeInEvent))
    greeting = next(
        u for u in utterances(artifact, Speaker.BOT) if u.id == barge.interrupted_utterance_id
    )
    cut_in = utterances(artifact, Speaker.CALLER)[0]
    assert greeting.interrupted
    assert greeting.intended_text.startswith(greeting.spoken_text) and greeting.spoken_text
    assert greeting.spoken_text != greeting.intended_text
    assert greeting.start_s < cut_in.start_s < greeting.end_s
    assert barge.at_s == pytest.approx(cut_in.start_s, abs=0.2)
    assert barge.bot_stopped_at_s == pytest.approx(greeting.end_s)
    assert 0 < barge.bot_stopped_at_s - barge.at_s < 1.0

    bot = AudioClip.from_file(store.paths(artifact.id).bot)
    on_disk = decode_speech(bot.slice(greeting.start_s - 0.1, greeting.end_s + 0.1))
    played = " ".join(word.text for word in on_disk)
    assert greeting.intended_text.startswith(played)
    assert abs(len(played.split()) - len(greeting.spoken_text.split())) <= 1

    # Finished sentences stay in the bot's turn, and the cut one reaches it in the caller's next.
    last_two = " ".join(m.content for m in support.requests[0][-2:])
    assert all(word in last_two for word in greeting.spoken_text.split())


async def test_the_bot_stops_for_a_caller_who_talks_over_a_long_reply(store, monkeypatch):
    support = Scripted(LONG_REPLY, default="Okay.")
    caller = quick_caller(says("Hello."), says("Wait a second."))
    decisions = iter([False, True])

    def decide(self, turn):
        return Interruption(interrupt=next(decisions, False), after_fraction=0.2)

    monkeypatch.setattr(CallerBot, "decide_interruption", decide)

    artifact = await live_call(store, llm_caller=caller, llm_support=support)

    reply = next(u for u in utterances(artifact) if u.intended_text == LONG_REPLY)
    assert reply.interrupted and reply.spoken_text != LONG_REPLY
    assert any(isinstance(e, BargeInEvent) for e in artifact.timeline)


async def test_the_caller_waits_when_told_to_hold_on(store):
    slow_tool = uses("list_plans", text=FILLER_LINE)

    class PausesAfterTheFiller(Scripted):
        async def stream_chat(self, messages, **options):
            if messages[-1].role == "tool":
                await asyncio.sleep(1.5)  # a slow second LLM round, in silence
            async for delta in super().stream_chat(messages, **options):
                yield delta

    caller = quick_caller(says("Hello."))
    artifact = await live_call(
        store, llm_caller=caller, llm_support=PausesAfterTheFiller(slow_tool, default="Done now.")
    )

    assert not any(isinstance(e, BargeInEvent) for e in artifact.timeline)
    assert "Done now." in caller.requests[1][-1].content
    assert FILLER_LINE in caller.requests[1][-1].content


async def test_the_bot_ending_the_call_ends_it(store):
    goodbye = uses("end_call", text="Goodbye.", reason="resolved")

    artifact = await live_call(
        store, llm_caller=Scripted(default=says("Thanks.")), llm_support=Scripted(goodbye)
    )

    assert "bot_ended" in kinds(artifact)
    assert artifact.status == CallStatus.EVALUATING
    assert utterances(artifact, Speaker.BOT)[-1].intended_text == "Goodbye."


async def test_changes_land_in_db_changes(store):
    ticket = uses("create_ticket", subject="s", summary="t")

    artifact = await live_call(
        store,
        llm_caller=quick_caller(says("Open a ticket.")),
        llm_support=Scripted(ticket, default="Done."),
    )

    assert [change.table for change in artifact.db_changes] == ["tickets"]


async def test_nothing_is_asked_of_the_bot_when_stt_hears_nothing(store):
    support = Scripted(default="Should never run.")

    settings = simple_settings()
    artifact = await asyncio.wait_for(
        run_live_bot_call(
            request(
                mode=CallMode.LIVE_BOTS,
                persona=persona(line=LIVE),
                scenario=scenario(max_turns=2),
            ),
            stack=stack(
                llm_caller=Scripted(default=says("Hello?")),
                llm_support=support,
                stt_live=DeafSTT(deaf_from=0),
            ),
            store=store,
            emit=Messages(),
            settings=type(settings)(
                vad=settings.vad,
                turn_stop=settings.turn_stop,
                stop_timeout_s=0.5,
                caller_patience_s=1.0,
            ),
        ),
        45,
    )

    assert support.requests == []
    assert artifact.status == CallStatus.EVALUATING


# Failure paths -----------------------------------------------------------------------------------


async def test_a_tts_failure_fails_the_call_and_keeps_what_was_recorded(store):
    artifact = await live_call(
        store,
        llm_caller=Scripted(default=says("Hello.")),
        llm_support=Scripted(default="Fine."),
        tts_bot=FailingTTS(ok_for=1),
    )

    assert artifact.status == CallStatus.FAILED and "tts_bot" in artifact.error
    assert "call_failed" in kinds(artifact)
    saved = store.read_artifact(artifact.id)
    assert saved.status == CallStatus.FAILED
    assert AudioClip.from_file(store.paths(artifact.id).bot).duration_s > 1


async def test_an_llm_failure_fails_the_call(store):
    class Broken(Scripted):
        async def chat(self, messages, **options):
            raise ConnectionError("no route to model server")

    artifact = await live_call(store, llm_caller=Broken(), llm_support=Scripted(default="Fine."))

    assert artifact.status == CallStatus.FAILED and "no route" in artifact.error


async def test_cancelling_a_live_call_writes_the_recording_and_propagates(store):
    messages = Messages()
    run = asyncio.create_task(
        run_live_bot_call(
            request(mode=CallMode.LIVE_BOTS, persona=persona(line=LIVE)),
            stack=stack(llm_caller=HangingLLM(), llm_support=Scripted(default="Fine.")),
            store=store,
            emit=messages,
            settings=simple_settings(),
        )
    )
    while not messages.of("partial"):  # noqa: ASYNC110  (a plain list, filled by another task)
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.0)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    saved = store.read_artifact("call_test")
    assert saved.status == CallStatus.FAILED and saved.error == "stopped"
    assert np.abs(AudioClip.from_file(store.paths("call_test").bot).samples()).max() > 0.02
    assert messages.of("callDone")[0]["call"]["status"] == "failed"


async def test_cancelling_a_live_call_during_a_reply_keeps_the_words_that_went_out(store):
    messages = Messages()
    long_reply = " ".join(["This reply takes a long time to say out loud."] * 4)
    run = asyncio.create_task(
        run_live_bot_call(
            request(mode=CallMode.LIVE_BOTS, persona=persona(line=LIVE)),
            stack=stack(
                llm_caller=quick_caller(says("I need some help.")),
                llm_support=Scripted(default=long_reply),
            ),
            store=store,
            emit=messages,
            settings=simple_settings(),
        )
    )
    async with asyncio.timeout(20):
        while "long time" not in str(messages.of("partial")):  # noqa: ASYNC110
            await asyncio.sleep(0.05)
    await asyncio.sleep(1.5)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    saved = store.read_artifact("call_test")
    replies = [e for e in saved.timeline if isinstance(e, Utterance)]
    cut = next(r for r in replies if "long time" in r.intended_text)
    assert cut.speaker == Speaker.BOT and cut.interrupted
    assert cut.spoken_text and cut.intended_text.startswith(cut.spoken_text)


async def test_run_bot_call_routes_live_mode_to_the_live_engine(store):
    artifact = await asyncio.wait_for(
        run_bot_call(
            request(mode=CallMode.LIVE_BOTS, persona=persona(line=LIVE)),
            stack=stack(llm_caller=quick_caller(), llm_support=Scripted(default="Fine.")),
            store=store,
            emit=Messages(),
        ),
        30,
    )

    assert artifact.mode == CallMode.LIVE_BOTS and artifact.status == CallStatus.EVALUATING


def test_the_engine_service_does_not_load_pipecat():
    import subprocess
    import sys

    script = (
        "import sys, voicelab.engine.service, voicelab.providers.adapters\n"
        "assert not [m for m in sys.modules if m.startswith('pipecat')]"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
