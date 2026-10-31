"""A fake LLM that plays back a fixed script and records every request."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from voicelab.providers.base import ChatMessage, LLMDelta, LLMResponse, ToolCall, ToolDef


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


def uses(*calls: ToolCall, text: str = "") -> LLMResponse:
    return LLMResponse(text=text, tool_calls=list(calls))


def caller_says(
    say: str, beat: str | None = None, *, hang_up: bool = False, reason: str = ""
) -> str:
    return json.dumps(
        {"say": say, "completed_beat": beat, "hang_up": hang_up, "hang_up_reason": reason}
    )


class ScriptedLLM:
    """Each request takes the next script entry, a string or an LLMResponse.

    When the script runs out it repeats `default`, or fails the test if there is none.
    """

    name = "scripted"

    def __init__(self, *script: str | LLMResponse, default: str | LLMResponse | None = None):
        self.script = list(script)
        self.default = default
        self.requests: list[dict[str, Any]] = []
        self.chunks_sent = 0
        self.closed_streams = 0

    def _next(self) -> LLMResponse:
        if self.script:
            entry = self.script.pop(0)
        elif self.default is not None:
            entry = self.default
        else:
            raise AssertionError("the script ran out")
        return LLMResponse(text=entry) if isinstance(entry, str) else entry

    def _record(self, messages: list[ChatMessage], **options: Any) -> None:
        self.requests.append({"messages": [m.model_copy() for m in messages], **options})

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._record(messages, tools=tools, json_schema=json_schema, temperature=temperature)
        return self._next()

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]:
        self._record(messages, tools=tools, temperature=temperature)
        response = self._next()
        try:
            words = response.text.split(" ")
            for index, word in enumerate(words):
                self.chunks_sent += 1
                yield LLMDelta(text=word if index == len(words) - 1 else f"{word} ")
            yield LLMDelta(final=response)
        finally:
            self.closed_streams += 1
