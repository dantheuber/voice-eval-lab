"""The realtime adapter against a small local server that follows the protocol notes."""

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
import websockets

from voicelab.providers.adapters.lfm3_realtime import (
    Lfm3RealtimeVoice,
    RealtimeError,
    heard_prefix,
)
from voicelab.providers.base import RealtimeVoice, ToolDef

TOOL = ToolDef(
    name="find_customer",
    description="Find a customer",
    parameters={"type": "object", "properties": {"phone": {"type": "string"}}},
)


class Server:
    """Holds a transcript of each connection, in order. `reply` scripts `response.create`."""

    def __init__(self, reply: Callable[[Any, "Server"], Awaitable[None]] | None = None) -> None:
        self.reply = reply
        self.connections: list[list[dict]] = []
        self.closed = 0
        self.greet = True

    async def __call__(self, socket: Any) -> None:
        log: list[dict] = []
        self.connections.append(log)
        if self.greet:
            await socket.send(json.dumps({"type": "session.created", "session": {}}))
        try:
            async for raw in socket:
                event = json.loads(raw)
                log.append(event)
                if event["type"] == "session.update":
                    await socket.send(json.dumps({"type": "session.updated", "session": {}}))
                elif event["type"] == "response.create" and self.reply:
                    await self.reply(socket, self)
        except websockets.ConnectionClosed:
            pass
        self.closed += 1


async def serve(server: Server):
    running = await websockets.serve(server, "127.0.0.1", 0)
    port = running.sockets[0].getsockname()[1]
    return running, Lfm3RealtimeVoice(f"http://127.0.0.1:{port}", connect_timeout_s=0.3)


async def send(socket: Any, **event: Any) -> None:
    await socket.send(json.dumps(event))


async def speak(socket: Any, text: str, audio_s: float, *, response_id: str = "r1") -> None:
    audio = base64.b64encode(b"\x00\x00" * round(24000 * audio_s)).decode()
    await send(socket, type="response.created", response={"id": response_id})
    await send(socket, type="response.output_text.delta", delta=text)
    await send(socket, type="response.output_audio.delta", delta=audio)


async def finish(socket: Any, *output: dict, status: str = "completed") -> None:
    await send(socket, type="response.done", response={"status": status, "output": list(output)})


async def collect(session: Any, count: int) -> list[Any]:
    events = []
    stream = session.events()
    for _ in range(count):
        events.append(await asyncio.wait_for(anext(stream), 2))
    return events


async def test_it_configures_a_session_in_the_order_that_works():
    server = Server()
    running, voice = await serve(server)

    session = await voice.connect(instructions="You are Fern.", tools=[TOOL])
    await session.close()
    running.close()

    assert isinstance(voice, RealtimeVoice)
    update = server.connections[0][0]
    assert update["type"] == "session.update"
    settings = update["session"]
    assert settings["output_modalities"] == ["text", "audio"]  # or replies go text-only
    assert settings["instructions"].startswith("You are Fern.")
    assert settings["instructions"].endswith("interleaved text and audio.\n\nList of tools:")
    assert settings["tools"] == [
        {
            "type": "function",
            "name": "find_customer",
            "description": "Find a customer",
            "parameters": TOOL.parameters,
        }
    ]


async def test_audio_is_base64_and_committing_starts_a_response():
    server = Server()
    running, voice = await serve(server)
    session = await voice.connect(instructions="", tools=[])

    await session.send_audio(b"\x01\x02")
    await session.commit_turn()
    await session.close()
    running.close()
    await asyncio.sleep(0.01)

    log = server.connections[0]
    assert log[1] == {"type": "input_audio_buffer.append", "audio": "AQI="}
    assert log[2] == {"type": "response.create"}


async def test_text_and_audio_events_are_normalized_with_their_response_id():
    async def reply(socket, _):
        await speak(socket, "Hello", 0.5)
        await finish(socket)

    running, voice = await serve(Server(reply))
    session = await voice.connect(instructions="", tools=[])

    await session.commit_turn()
    events = await collect(session, 3)
    await session.close()
    running.close()

    assert [e.type for e in events] == ["bot_transcript", "audio", "response_done"]
    assert events[0].text == "Hello" and events[0].response_id == "r1"
    assert len(events[1].audio_pcm16) == 24000


async def test_a_good_tool_call_comes_out_parsed():
    async def reply(socket, _):
        await send(socket, type="response.created", response={"id": "r1"})
        await finish(
            socket,
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "find_customer",
                "arguments": '{"phone": "555"}',
            },
        )

    running, voice = await serve(Server(reply))
    session = await voice.connect(instructions="", tools=[TOOL])

    await session.commit_turn()
    tool, done = await collect(session, 2)
    await session.close()
    running.close()

    assert tool.type == "tool_call" and done.type == "response_done"
    assert tool.tool_call.arguments == {"phone": "555"} and tool.text == ""


@pytest.mark.parametrize("arguments", ['{"phone": "5', '["list"]'])
async def test_a_bad_tool_call_still_comes_out_so_it_can_be_answered(arguments):
    async def reply(socket, _):
        await finish(
            socket,
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "find_customer",
                "arguments": arguments,
            },
        )

    running, voice = await serve(Server(reply))
    session = await voice.connect(instructions="", tools=[TOOL])

    await session.commit_turn()
    tool, _ = await collect(session, 2)
    await session.close()
    running.close()

    assert tool.tool_call.id == "c1" and tool.tool_call.arguments == {}
    assert "arguments were not" in tool.text


async def test_tool_results_go_back_and_the_reply_is_requested():
    server = Server()
    running, voice = await serve(server)
    session = await voice.connect(instructions="", tools=[TOOL])

    await session.send_tool_result("c1", '{"found": true}')
    await session.continue_after_tools()
    await session.close()
    running.close()
    await asyncio.sleep(0.01)

    assert server.connections[0][1] == {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": "c1", "output": '{"found": true}'},
    }
    assert server.connections[0][2] == {"type": "response.create"}


async def test_an_interrupt_replaces_the_session_with_heard_only_history():
    async def reply(socket, _):
        await speak(socket, "one two three four five six seven eight", 4.0)

    server = Server(reply)
    running, voice = await serve(server)
    session = await voice.connect(instructions="You are Fern.", tools=[TOOL])
    session.record_user_text("My email is down.")

    await session.commit_turn()
    await collect(session, 2)
    await session.interrupt(response_id="r1", played_ms=1000)
    await session.send_audio(b"\x01\x02")
    await session.close()
    running.close()
    await asyncio.sleep(0.01)

    assert len(server.connections) == 2
    reopened = server.connections[1][0]["session"]
    prompt = reopened["instructions"]
    assert "Caller: My email is down." in prompt
    assert "You: one two" in prompt and "seven" not in prompt
    assert "did not hear anything else" in prompt
    assert reopened["tools"] and reopened["output_modalities"] == ["text", "audio"]
    assert server.connections[1][1]["type"] == "input_audio_buffer.append"


async def test_events_from_the_old_socket_never_reach_the_new_session():
    async def reply(socket, server):
        await speak(socket, "stale words", 1.0)
        await asyncio.sleep(0.05)
        await send(socket, type="response.output_text.delta", delta="late")

    running, voice = await serve(Server(reply))
    session = await voice.connect(instructions="", tools=[])

    await session.commit_turn()
    await collect(session, 2)
    await session.interrupt(response_id=None, played_ms=0)
    await asyncio.sleep(0.1)
    leftover = session._events.qsize()
    await session.close()
    running.close()

    assert leftover == 0


async def test_an_interrupt_for_a_finished_response_does_nothing():
    server = Server()
    running, voice = await serve(server)
    session = await voice.connect(instructions="", tools=[])

    await session.interrupt(response_id="an-old-one", played_ms=100)
    await session.close()
    running.close()

    assert len(server.connections) == 1


async def test_a_finished_reply_joins_the_history_and_a_cancelled_one_does_not():
    turns = iter(["Finished words.", "Cancelled words."])

    async def reply(socket, _):
        text = next(turns)
        await speak(socket, text, 0.1)
        await finish(socket, status="completed" if text.startswith("Fin") else "cancelled")

    server = Server(reply)
    running, voice = await serve(server)
    session = await voice.connect(instructions="Base.", tools=[])

    await session.commit_turn()
    await collect(session, 3)
    await session.commit_turn()
    await collect(session, 3)
    await session.interrupt(response_id=None, played_ms=0)
    await session.close()
    running.close()

    prompt = server.connections[1][0]["session"]["instructions"]
    assert "You: Finished words." in prompt and "Cancelled" not in prompt


async def test_a_queued_connection_is_closed_when_no_slot_comes_free():
    server = Server()
    server.greet = False  # no session.created is how a busy -np 1 server behaves
    running, voice = await serve(server)

    with pytest.raises(RealtimeError, match="every slot may be busy"):
        await voice.connect(instructions="", tools=[])
    await asyncio.sleep(0.05)
    running.close()

    assert server.closed == 1


async def test_a_setup_error_is_reported_and_the_socket_closed():
    class Refuses(Server):
        async def __call__(self, socket):
            await send(socket, type="session.created", session={})
            await send(socket, type="error", error={"message": "tools are locked"})
            await socket.wait_closed()
            self.closed += 1

    server = Refuses()
    running, voice = await serve(server)

    with pytest.raises(RealtimeError, match="tools are locked"):
        await voice.connect(instructions="", tools=[])
    await asyncio.sleep(0.05)
    running.close()

    assert server.closed == 1


async def test_a_dropped_connection_is_an_error_event():
    async def reply(socket, _):
        await socket.close()

    running, voice = await serve(Server(reply))
    session = await voice.connect(instructions="", tools=[])

    await session.commit_turn()
    events = await collect(session, 1)
    await session.close()
    running.close()

    assert events[0].type == "error" and "closed" in events[0].text


async def test_server_errors_are_passed_on():
    async def reply(socket, _):
        await send(socket, type="error", error={"message": "context full"})

    running, voice = await serve(Server(reply))
    session = await voice.connect(instructions="", tools=[])

    await session.commit_turn()
    events = await collect(session, 1)
    await session.close()
    running.close()

    assert events[0].type == "error" and events[0].text == "context full"


async def test_closing_ends_the_event_stream():
    running, voice = await serve(Server())
    session = await voice.connect(instructions="", tools=[])

    await session.close()
    await session.close()
    running.close()

    assert [event async for event in session.events()] == []


def test_heard_prefix_keeps_only_finished_words():
    text = "aaaa bbbb cccc dddd"

    assert heard_prefix(text, 0, 2000) == ""
    assert heard_prefix(text, 500, 2000) == "aaaa"
    assert heard_prefix(text, 1000, 2000) == "aaaa bbbb"
    assert heard_prefix(text, 2000, 2000) == text
    assert heard_prefix(text, 500, 0) == text
