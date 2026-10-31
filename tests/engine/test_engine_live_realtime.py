"""The realtime path through the live pipeline, with a scripted model behind `RealtimeVoice`."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from enginefakes import Messages, Scripted, persona, request, says, stack

from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    BargeInEvent,
    CallMode,
    CallStatus,
    LineConditions,
    Speaker,
    ToolCallEvent,
    Utterance,
)
from voicelab.engine.live.bots import run_live_bot_call
from voicelab.engine.live.parts import simple_settings
from voicelab.providers.adapters.fake_speech import encode_speech
from voicelab.providers.base import RealtimeEvent, ToolCall
from voicelab.store.artifacts import ArtifactStore

LIVE = LineConditions(codec="none", one_way_latency_ms=60)


class ScriptedSession:
    def __init__(self, turns: list[list[RealtimeEvent]]) -> None:
        self.turns = turns
        self.events_out: asyncio.Queue[RealtimeEvent | None] = asyncio.Queue()
        self.audio_bytes = 0
        self.commits = 0
        self.user_texts: list[str] = []
        self.tool_results: list[tuple[str, str]] = []
        self.interrupts: list[int] = []
        self.closed = False

    async def send_audio(self, pcm16: bytes) -> None:
        self.audio_bytes += len(pcm16)

    def _play_next(self) -> None:
        for event in self.turns.pop(0) if self.turns else []:
            self.events_out.put_nowait(event)

    async def commit_turn(self) -> None:
        self.commits += 1
        self._play_next()

    def record_user_text(self, text: str) -> None:
        self.user_texts.append(text)

    async def send_tool_result(self, call_id: str, result: str) -> None:
        self.tool_results.append((call_id, result))

    async def continue_after_tools(self) -> None:
        self._play_next()

    async def interrupt(self, *, response_id: str | None, played_ms: int) -> None:
        self.interrupts.append(played_ms)

    async def events(self) -> Any:
        while (event := await self.events_out.get()) is not None:
            yield event

    async def close(self) -> None:
        self.closed = True
        self.events_out.put_nowait(None)


class ScriptedVoice:
    name = "scripted-realtime"
    input_sample_rate = 16000
    output_sample_rate = 24000

    def __init__(self, *turns: list[RealtimeEvent]) -> None:
        self.session = ScriptedSession(list(turns))
        self.instructions = ""

    async def connect(self, *, instructions, tools, voice_id=None) -> ScriptedSession:
        self.instructions = instructions
        self.tools = tools
        return self.session


def says_aloud(text: str) -> list[RealtimeEvent]:
    audio = encode_speech(text, 24000).pcm
    return [
        RealtimeEvent(type="bot_transcript", text=text, response_id="r"),
        RealtimeEvent(type="audio", audio_pcm16=audio, response_id="r"),
        RealtimeEvent(type="response_done", response_id="r"),
    ]


def calls(*items: tuple[str, dict, str]) -> list[RealtimeEvent]:
    events = [
        RealtimeEvent(
            type="tool_call",
            tool_call=ToolCall(id=f"c{index}", name=name, arguments=arguments),
            text=problem,
            response_id="r",
        )
        for index, (name, arguments, problem) in enumerate(items)
    ]
    return [*events, RealtimeEvent(type="response_done", response_id="r")]


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path)


def realtime_stack(voice: ScriptedVoice, **adapters: Any) -> Any:
    built = stack(realtime=voice, **adapters)
    built.profile = built.profile.model_copy(update={"support_engine": "realtime"})
    return built


async def run(store: ArtifactStore, voice: ScriptedVoice, **adapters: Any):
    caller = adapters.pop(
        "llm_caller",
        Scripted(says("Hello."), says("Thanks."), says("Bye.", hang_up=True, reason="finished")),
    )
    call = run_live_bot_call(
        request(mode=CallMode.LIVE_BOTS, persona=adapters.pop("who", persona(line=LIVE))),
        stack=realtime_stack(voice, llm_caller=caller, **adapters),
        store=store,
        emit=Messages(),
        settings=simple_settings(),
    )
    return await asyncio.wait_for(call, 40)


async def test_the_model_gets_audio_and_a_commit_and_the_transcript_comes_from_our_stt(store):
    voice = ScriptedVoice(says_aloud("Sure thing."), says_aloud("Goodbye then."))

    artifact = await run(store, voice)

    session = voice.session
    assert session.audio_bytes > 16000 and session.commits >= 2
    assert session.user_texts[:2] == ["Hello.", "Thanks."]
    assert session.closed
    assert artifact.status == CallStatus.EVALUATING
    assert artifact.providers["realtime"] == "scripted-realtime"
    assert "llm_support" not in artifact.providers
    bot = [e for e in artifact.timeline if isinstance(e, Utterance) and e.speaker == Speaker.BOT]
    assert bot[1].intended_text == "Sure thing." and bot[1].heard_text == "Sure thing."
    assert bot[0].intended_text.startswith("Thanks for calling")
    assert "already answered the call" in voice.instructions


async def test_tool_calls_run_in_order_and_bad_ones_get_an_error_not_a_crash(store):
    voice = ScriptedVoice(
        calls(
            ("find_customer", {"domain": "bellhardware.example"}, ""),
            ("get_invoice", {}, ""),
            ("issue_refund", {}, "arguments were not valid JSON: {"),
            ("list_invoices", {"customer_id": "[customer_id]\n</function>"}, ""),
        ),
        says_aloud("Found you."),
        says_aloud("Goodbye."),
    )

    artifact = await run(store, voice)

    results = voice.session.tool_results
    assert [call_id for call_id, _ in results] == ["c0", "c1", "c2", "c3"]
    assert "found" in results[0][1]
    assert "missing invoice_id" in results[1][1]
    assert "not valid JSON" in results[2][1]
    assert "template syntax" in results[3][1]
    events = [e for e in artifact.timeline if isinstance(e, ToolCallEvent)]
    assert [e.name for e in events][:2] == ["find_customer", "get_invoice"]
    assert events[1].error and not events[0].error
    assert not artifact.db_changes


async def test_a_barge_in_tells_the_session_how_much_was_played(store):
    long_reply = "I can certainly look at that for you right now while we are on the line."
    voice = ScriptedVoice(says_aloud(long_reply), says_aloud("Goodbye."))
    who = persona(interruption_rate=1.0, line=LIVE)

    artifact = await run(store, voice, who=who)

    assert voice.session.interrupts
    barges = [e for e in artifact.timeline if isinstance(e, BargeInEvent)]
    assert barges


async def test_a_realtime_server_that_refuses_the_connection_fails_the_call(store):
    class Busy(ScriptedVoice):
        async def connect(self, *, instructions, tools, voice_id=None):
            raise RuntimeError("every slot is busy")

    artifact = await run(store, Busy())

    assert artifact.status == CallStatus.FAILED and "every slot is busy" in artifact.error


async def test_an_error_from_the_model_fails_the_call(store):
    voice = ScriptedVoice([RealtimeEvent(type="error", text="context full")])

    artifact = await run(store, voice)

    assert artifact.status == CallStatus.FAILED and "context full" in artifact.error
    assert AudioClip.from_file(store.paths(artifact.id).bot).duration_s > 1
