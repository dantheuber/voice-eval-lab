"""LFM3-Audio speech-to-speech over llama.cpp's `/v1/realtime`.

Run one server process per session, with `-np 1`. The protocol notes in
`docs/research/benchmarks-2026-10.md` are the source for everything here, and they differ from
OpenAI's Realtime API in ways that matter:

  - There is no server VAD, so the engine calls `commit_turn` when its own VAD says the caller is
    done. There are no user transcripts either, so the engine runs STT beside this and reports
    what it heard through `record_user_text`.
  - Text the model generated stays in its context even when the caller never heard it, and there
    is no truncate. A text note didn't correct it (0 of 2). What worked, 4 of 4, was closing the
    session and opening a new one with heard-only history in `instructions`, ready in 4 to 6 ms.
    So `interrupt` does exactly that. Earlier caller audio leaves the context, so from then on the
    model knows the caller only through the STT transcript, and `tools` are sent again.
  - `conversation.item.create` refuses assistant and system items, which is why the history has
    to go in `instructions`.
  - A connection with no free slot opens and then waits with no event, and it keeps its place in
    line. A client that gives up must close it, or it takes the slot later.
  - With a persona in `instructions`, replies silently go text-only unless `output_modalities` is
    forced to `["text", "audio"]`. Forcing `["audio"]` alone is broken upstream.
  - There is one voice. `audio.output.voice` and prompt instructions both left pitch unchanged.

Use this for the support bot only. As a caller, LFM3 slipped into the agent's role.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import websockets

from voicelab.providers.base import RealtimeEvent, ToolCall, ToolDef, register

logger = logging.getLogger(__name__)

_INPUT_RATE = 16000
_OUTPUT_RATE = 24000
_BYTES_PER_SECOND = _OUTPUT_RATE * 2
_CHARACTERS_PER_S = 14.0  # speaking rate, for text whose audio hasn't been generated yet
_MODALITIES = ["text", "audio"]
_VOICE_CARD = "Respond with function calls, and interleaved text and audio.\n\nList of tools:"
_HISTORY_HEADER = "Conversation so far:"
_CUT_OFF_NOTE = "The caller cut you off there. They did not hear anything else you meant to say."


class RealtimeError(RuntimeError):
    """The server refused something the session can't go on without."""


def _wire_tool(tool: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def heard_prefix(text: str, played_ms: float, generated_ms: float) -> str:
    """The words that had finished playing, assuming speech runs evenly through the text."""
    if generated_ms <= 0:
        return ""
    if played_ms >= generated_ms:
        return text.strip()
    words = text.split()
    weights = [len(word) + 1 for word in words]
    limit = sum(weights) * max(0, played_ms) / generated_ms
    kept: list[str] = []
    position = 0.0
    for word, weight in zip(words, weights, strict=True):
        position += weight
        if position > limit:
            break
        kept.append(word)
    return " ".join(kept)


def _history_line(entry: str | _Reply) -> str:
    if isinstance(entry, str):
        return entry
    return f"You: {entry.text.strip()}" if entry.text.strip() else ""


@dataclass(eq=False)
class _Reply:
    """One generated response. It sits in the history where it happened, and a cut trims it."""

    id: str | None
    text: str = ""
    audio_bytes: int = 0
    done: bool = False

    @property
    def length_ms(self) -> float:
        """Text outruns audio while generating, so until it's done the audio can't be the length."""
        audio_ms = self.audio_bytes * 1000 / _BYTES_PER_SECOND
        return audio_ms if self.done else max(audio_ms, len(self.text) / _CHARACTERS_PER_S * 1000)


@register("realtime", "lfm3_realtime")
class Lfm3RealtimeVoice:
    input_sample_rate = _INPUT_RATE
    output_sample_rate = _OUTPUT_RATE

    def __init__(
        self,
        url: str,
        *,
        name: str = "lfm3-audio-realtime",
        connect_timeout_s: float = 10.0,
        max_output_tokens: int = 512,
    ) -> None:
        self.url = url.rstrip("/").replace("http://", "ws://")
        if not self.url.endswith("/v1/realtime"):
            self.url = f"{self.url}/v1/realtime"
        self.name = name
        self.connect_timeout_s = connect_timeout_s
        self.max_output_tokens = max_output_tokens

    async def connect(
        self, *, instructions: str, tools: list[ToolDef], voice_id: str | None = None
    ) -> Lfm3RealtimeSession:
        session = Lfm3RealtimeSession(self, instructions, tools)
        await session.open()
        return session


class Lfm3RealtimeSession:
    """One call. It may span several server sessions, since each cut-in replaces the socket."""

    def __init__(self, voice: Lfm3RealtimeVoice, instructions: str, tools: list[ToolDef]) -> None:
        self._voice = voice
        self._instructions = instructions
        self._tools = [_wire_tool(tool) for tool in tools]
        self._history: list[str | _Reply] = []
        self._turn: list[_Reply] = []  # replies since the last commit, not yet known to be heard
        self._tool_names: dict[str, str] = {}
        self._events: asyncio.Queue[tuple[int, RealtimeEvent] | None] = asyncio.Queue()
        self._socket: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._closed = False
        self._epoch = 0  # events from a replaced socket are stale

    def _prompt(self) -> str:
        lines = [_history_line(entry) for entry in self._history]
        history = "\n".join(line for line in lines if line)
        if not history:
            return f"{self._instructions}\n\n{_VOICE_CARD}"
        return f"{self._instructions}\n\n{_HISTORY_HEADER}\n{history}\n\n{_VOICE_CARD}"

    async def open(self) -> None:
        """Open a socket and configure it. `session.created` means a slot was free."""
        self._epoch += 1
        epoch = self._epoch
        socket = await asyncio.wait_for(
            websockets.connect(self._voice.url, max_size=None, ping_interval=None),
            self._voice.connect_timeout_s,
        )
        try:
            await self._await_type(socket, "session.created")
            await socket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "instructions": self._prompt(),
                            "tools": self._tools,
                            "tool_choice": "auto",
                            "output_modalities": _MODALITIES,
                            "max_output_tokens": self._voice.max_output_tokens,
                        },
                    }
                )
            )
            await self._await_type(socket, "session.updated")
        except BaseException:
            # A queued connection takes the slot later unless it's closed.
            await socket.close()
            raise
        self._socket = socket
        self._reader = asyncio.create_task(self._read(socket, epoch))
        self._ready.set()

    async def _await_type(self, socket: Any, kind: str) -> None:
        async def wait() -> None:
            async for raw in socket:
                event = json.loads(raw)
                if event.get("type") == kind:
                    return
                if event.get("type") == "error":
                    raise RealtimeError(f"server error during setup: {event.get('error')}")
            raise RealtimeError("the server closed the connection during setup")

        try:
            await asyncio.wait_for(wait(), self._voice.connect_timeout_s)
        except TimeoutError as failure:
            raise RealtimeError(
                f"no {kind} from {self._voice.url}: every slot may be busy"
            ) from failure

    async def _read(self, socket: Any, epoch: int) -> None:
        try:
            async for raw in socket:
                if epoch != self._epoch:
                    return
                for event in self._normalize(json.loads(raw)):
                    self._events.put_nowait((epoch, event))
        except websockets.ConnectionClosed:
            pass
        finally:
            if epoch == self._epoch and not self._closed and self._ready.is_set():
                closed = RealtimeEvent(type="error", text="the realtime connection closed")
                self._events.put_nowait((epoch, closed))

    def _normalize(self, event: dict[str, Any]) -> list[RealtimeEvent]:
        kind = event.get("type")
        if kind == "response.created":
            reply = _Reply((event.get("response") or {}).get("id"))
            self._turn.append(reply)
            self._history.append(reply)
        elif kind == "response.output_text.delta" and self._turn:
            self._turn[-1].text += event.get("delta", "")
            return [self._event("bot_transcript", text=event.get("delta", ""))]
        elif kind == "response.output_audio.delta" and self._turn:
            audio = base64.b64decode(event.get("delta", ""))
            self._turn[-1].audio_bytes += len(audio)
            return [self._event("audio", audio_pcm16=audio)]
        elif kind == "response.done":
            return self._finish_response(event.get("response") or {})
        elif kind == "error":
            detail = event.get("error") or {}
            return [self._event("error", text=str(detail.get("message", detail)))]
        return []

    def _event(self, kind: str, **fields: Any) -> RealtimeEvent:
        response_id = self._turn[-1].id if self._turn else None
        return RealtimeEvent(type=kind, response_id=response_id, **fields)

    def _finish_response(self, response: dict[str, Any]) -> list[RealtimeEvent]:
        events: list[RealtimeEvent] = []
        if self._turn:
            self._turn[-1].done = True
            if response.get("status") == "cancelled":
                self._turn[-1].text = ""
        for item in response.get("output") or []:
            if item.get("type") == "function_call":
                events.append(self._tool_event(item))
        events.append(self._event("response_done"))
        return events

    def _tool_event(self, item: dict[str, Any]) -> RealtimeEvent:
        """Turn a function call item into an event, keeping the bad ones.

        Arguments can arrive as truncated JSON. Every call needs an answer, so a bad one still
        goes out, with the problem in `text`, for the engine to answer with a tool error.
        """
        raw = item.get("arguments") or "{}"
        arguments: dict[str, Any] = {}
        problem = ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                arguments = parsed
            else:
                problem = "arguments were not a JSON object"
        except ValueError:
            problem = f"arguments were not valid JSON: {raw[:200]}"
        call = ToolCall(id=item.get("call_id", ""), name=item.get("name", ""), arguments=arguments)
        self._tool_names[call.id] = call.name
        return self._event("tool_call", tool_call=call, text=problem)

    async def _send(self, **event: Any) -> None:
        await self._ready.wait()
        if not self._closed:
            await self._socket.send(json.dumps(event))

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send(
            type="input_audio_buffer.append", audio=base64.b64encode(pcm16).decode("ascii")
        )

    async def commit_turn(self) -> None:
        """The caller finished. `response.create` commits pending audio by itself.

        Nothing cut the replies before this, so they played to the end.
        """
        self._turn = []
        await self._send(type="response.create")

    def record_user_text(self, text: str) -> None:
        if text.strip():
            self._history.append(f"Caller: {text.strip()}")

    async def send_tool_result(self, call_id: str, result: str) -> None:
        # A finished tool stays finished, so it has to survive a session being replaced.
        self._history.append(f"Tool {self._tool_names.get(call_id, 'call')} returned: {result}")
        await self._send(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": call_id, "output": result},
        )

    async def continue_after_tools(self) -> None:
        """Ask for the reply to the tool results just sent."""
        await self._send(type="response.create")

    async def interrupt(self, *, response_id: str | None, played_ms: int) -> None:
        """Forget what the caller didn't hear, by replacing the session with one that never did.

        Cancelling alone leaves the unheard text in context, where the model recites it later.
        """
        if response_id is not None and all(reply.id != response_id for reply in self._turn):
            return
        self._keep_only_heard(played_ms)
        self._history.append(_CUT_OFF_NOTE)
        self._turn = []
        self._ready.clear()
        await self._close_socket()
        await self.open()

    def _keep_only_heard(self, played_ms: int) -> None:
        """`played_ms` counts from the start of the turn, so earlier replies use it up first."""
        remaining_ms = float(played_ms)
        for reply in self._turn:
            reply.text = heard_prefix(reply.text, remaining_ms, reply.length_ms)
            remaining_ms = max(0.0, remaining_ms - reply.length_ms)

    async def _close_socket(self) -> None:
        reader, socket = self._reader, self._socket
        self._reader = self._socket = None
        self._epoch += 1
        if socket is not None:
            await socket.close()
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            queued = await self._events.get()
            if queued is None:
                return
            epoch, event = queued
            if epoch == self._epoch:  # it may have waited here while its session was replaced
                yield event

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready.set()  # anything waiting on a reopen must not wait forever
        await self._close_socket()
        self._events.put_nowait(None)
