"""Chat against any OpenAI-compatible server: llama.cpp, Ollama, vLLM, Bedrock's OpenAI endpoint.

`base_url` ends where `/chat/completions` starts, so for example `http://127.0.0.1:8092/v1` for
llama-server or `https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1` for Bedrock, whose
bearer token goes in `api_key_env`.

These servers all claim the same API and differ in the details, and small models add quirks of
their own. Tool calls may arrive split across deltas, whole in one delta, with no index, or with
arguments already parsed. Reasoning arrives under several field names and is never speech. A
reply can have empty content. A tool call with arguments that won't parse is dropped, leaving the
caller's retry logic to ask again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import httpx

from voicelab.providers.base import (
    ChatMessage,
    LLMDelta,
    LLMResponse,
    ToolCall,
    ToolDef,
    register,
)

logger = logging.getLogger(__name__)

# A server that is starting answers 503 while its model loads. 502 and 504 come from proxies.
_RETRY_STATUSES = {429, 502, 503, 504}
_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LLMError(RuntimeError):
    """The server failed or refused. The message carries its status and body."""


class LLMTimeoutError(LLMError):
    """The server stopped answering."""


def _wire_message(message: ChatMessage) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        wire["tool_call_id"] = message.tool_call_id
        if message.name:
            wire["name"] = message.name
    return wire


def _wire_tool(tool: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _text_of(content: Any) -> str:
    """Content is a string, null, or on some servers a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _reasoning_of(fields: dict[str, Any]) -> str:
    return "".join(str(fields.get(name) or "") for name in _REASONING_FIELDS)


def _clean_text(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def _parse_arguments(raw: Any) -> dict[str, Any] | None:
    """Arguments as a dict, or None if they can't be used."""
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


class _ToolCallParts:
    """Assembles tool calls from however a server chose to split them."""

    def __init__(self) -> None:
        self._slots: list[dict[str, Any]] = []
        self._by_index: dict[int, dict[str, Any]] = {}

    def add(self, delta: dict[str, Any]) -> None:
        function = delta.get("function") or {}
        slot = self._slot_for(delta.get("index"), delta.get("id"))
        if delta.get("id"):
            slot["id"] = delta["id"]
        name = function.get("name")
        if name and name != slot["name"]:  # most send it once, a few repeat it on every delta
            slot["name"] += name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            slot["arguments"] += arguments
        elif arguments is not None:
            slot["arguments"] = arguments

    def _slot_for(self, index: int | None, call_id: str | None) -> dict[str, Any]:
        slot = self._by_index.get(index) if index is not None else None
        if index is None and self._slots and not call_id:
            slot = self._slots[-1]
        # Some servers number every call 0 and tell them apart only by id.
        if slot is not None and not (call_id and slot["id"] and call_id != slot["id"]):
            return slot
        slot = {"id": None, "name": "", "arguments": ""}
        self._slots.append(slot)
        if index is not None:
            self._by_index[index] = slot
        return slot

    def finish(self) -> tuple[list[ToolCall], bool]:
        """The usable calls, and whether any had to be dropped."""
        calls: list[ToolCall] = []
        dropped = False
        for position, slot in enumerate(self._slots):
            arguments = _parse_arguments(slot["arguments"])
            if not slot["name"] or arguments is None:
                logger.warning("dropping malformed tool call: %.300r", slot)
                dropped = True
                continue
            call_id = slot["id"] or f"call_{position}_{time.monotonic_ns()}"
            calls.append(ToolCall(id=call_id, name=slot["name"], arguments=arguments))
        return calls, dropped


def _schema_note(schema: dict[str, Any]) -> str:
    return f"Reply with JSON only, matching this JSON schema: {json.dumps(schema)}"


def _with_note(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    if messages and messages[0]["role"] == "system":
        first = {**messages[0], "content": f"{messages[0]['content']}\n\n{note}"}
        return [first, *messages[1:]]
    return [{"role": "system", "content": note}, *messages]


@register("llm", "openai_compat")
class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        model: str = "",
        *,
        name: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 5.0,
        retries: int = 2,
        retry_backoff_s: float = 0.5,
        structured_output: str = "json_schema",  # or "json_object", or "prompt" for neither
        include_usage: bool = True,
        send_temperature: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or f"openai_compat/{model or urlsplit(base_url).netloc}"
        self.timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self.structured_output = structured_output
        self.include_usage = include_usage
        self.send_temperature = send_temperature
        self.extra_body = extra_body or {}
        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None

    def _client(self) -> httpx.AsyncClient:
        # A client's connections belong to one event loop, and tests run one loop per test.
        loop = asyncio.get_running_loop()
        if self._http is None or self._http_loop is not loop:
            self._http = httpx.AsyncClient(timeout=self.timeout, headers=self._headers)
            self._http_loop = loop
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _body(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDef] | None,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        wire = [_wire_message(message) for message in messages]
        body: dict[str, Any] = {"max_tokens": max_tokens}
        if self.model:
            body["model"] = self.model
        if self.send_temperature:
            body["temperature"] = temperature
        if tools:
            body["tools"] = [_wire_tool(tool) for tool in tools]
        if json_schema is not None:
            if self.structured_output == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "schema": json_schema},
                }
            else:
                wire = _with_note(wire, _schema_note(json_schema))
                if self.structured_output == "json_object":
                    body["response_format"] = {"type": "json_object"}
        return {"messages": wire, **body, **self.extra_body}

    async def _send(self, body: dict[str, Any]) -> httpx.Response:
        """Open a response, retrying while nothing has been received yet."""
        client = self._client()
        url = f"{self.base_url}/chat/completions"
        attempt = 0
        while True:
            try:
                request = client.build_request("POST", url, json=body)
                response = await client.send(request, stream=True)
            except httpx.ReadTimeout as failure:
                # The model is slow, not the network. Asking again would only queue behind it.
                raise LLMTimeoutError(f"{self.name} timed out waiting for a reply") from failure
            except httpx.TransportError as failure:
                if attempt >= self.retries:
                    kind = (
                        LLMTimeoutError if isinstance(failure, httpx.TimeoutException) else LLMError
                    )
                    raise kind(f"{self.name} unreachable: {failure!r}") from failure
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in _RETRY_STATUSES or attempt >= self.retries:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    await response.aclose()
                    raise LLMError(f"{self.name} answered {response.status_code}: {detail[:500]}")
                await response.aclose()
            attempt += 1
            await asyncio.sleep(self.retry_backoff_s * 2 ** (attempt - 1))

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        body = self._body(messages, tools, temperature, max_tokens, json_schema)
        response = await self._send(body)
        try:
            payload = json.loads(await response.aread())
        except httpx.TimeoutException as failure:
            raise LLMTimeoutError(f"{self.name} timed out mid-reply") from failure
        except ValueError as failure:
            raise LLMError(f"{self.name} sent a reply that isn't JSON") from failure
        finally:
            await response.aclose()
        return self._response_from(payload, wants_json=json_schema is not None)

    def _response_from(self, payload: dict[str, Any], *, wants_json: bool) -> LLMResponse:
        if payload.get("error"):
            raise LLMError(f"{self.name} reported an error: {payload['error']}")
        choices = payload.get("choices") or [{}]
        message = choices[0].get("message") or {}
        parts = _ToolCallParts()
        for call in message.get("tool_calls") or []:
            parts.add(call)
        calls, dropped = parts.finish()
        text = _clean_text(_text_of(message.get("content")))
        if not text and wants_json:
            # Some templates leave structured output in the reasoning channel.
            text = _clean_text(_reasoning_of(message))
        timings = payload.get("timings") or {}
        return self._assemble(
            text,
            calls,
            dropped,
            finish=choices[0].get("finish_reason"),
            usage=payload.get("usage") or {},
            timings=timings,
            first_token_ms=timings.get("prompt_ms"),
            model=payload.get("model"),
        )

    def _assemble(
        self,
        text: str,
        calls: list[ToolCall],
        dropped: bool,
        *,
        finish: str | None,
        usage: dict[str, Any],
        timings: dict[str, Any],
        first_token_ms: float | None,
        model: str | None,
    ) -> LLMResponse:
        return LLMResponse(
            text=text,
            tool_calls=calls,
            finish_reason="malformed_tool_call" if dropped and not calls else finish,
            input_tokens=usage.get("prompt_tokens") or timings.get("prompt_n"),
            output_tokens=usage.get("completion_tokens") or timings.get("predicted_n"),
            first_token_ms=first_token_ms,
            model=model or self.model or self.name,
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]:
        """Yield text as it arrives, then a final delta with the assembled response.

        `first_token_ms` times the first content or tool call, since that is what a voice call
        waits on, and reasoning tokens before it don't count. Closing this generator closes the
        connection, which stops generation on a single-slot server.
        """
        body = self._body(messages, tools, temperature, max_tokens)
        body["stream"] = True
        if self.include_usage:
            body["stream_options"] = {"include_usage": True}
        started = time.perf_counter()
        response = await self._send(body)
        raw = ""
        parts = _ToolCallParts()
        first_token_ms: float | None = None
        finish: str | None = None
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        model: str | None = None
        try:
            async for event in self._events(response):
                if event.get("error"):
                    raise LLMError(f"{self.name} reported an error: {event['error']}")
                usage = event.get("usage") or usage
                timings = event.get("timings") or timings
                model = event.get("model") or model
                for choice in event.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    text = _text_of(delta.get("content"))
                    call_deltas = delta.get("tool_calls") or []
                    if (text or call_deltas) and first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    for call in call_deltas:
                        parts.add(call)
                    if text:
                        raw += text
                        yield LLMDelta(text=text)
        except httpx.TimeoutException as failure:
            raise LLMTimeoutError(f"{self.name} stalled mid-reply") from failure
        except httpx.TransportError as failure:
            # Text may already be out, so asking again would repeat it.
            raise LLMError(f"{self.name} dropped the connection: {failure!r}") from failure
        finally:
            await response.aclose()
        calls, dropped = parts.finish()
        final = self._assemble(
            _clean_text(raw),
            calls,
            dropped,
            finish=finish,
            usage=usage,
            timings=timings,
            first_token_ms=first_token_ms,
            model=model,
        )
        yield LLMDelta(final=final)

    @staticmethod
    async def _events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):  # blanks and ": keep-alive" comments
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except ValueError:
                logger.warning("skipping unreadable stream event: %.200r", data)
