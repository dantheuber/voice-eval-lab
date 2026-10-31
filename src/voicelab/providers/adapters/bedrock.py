"""AWS adapters: Bedrock Converse, Transcribe streaming, Polly, and Nova Sonic.

NONE OF THIS HAS EVER RUN AGAINST LIVE AWS. There is no AWS account on the machine this was
written on. Requests and parsing follow the AWS documentation and were checked against stubbed
clients only, so expect to fix things on first contact. Each adapter says so with `untested = True`.

AWS packages are optional, install them with the `aws` extra, and nothing here imports them until
an adapter is used. boto3 covers Converse and Polly. It can't do HTTP/2 bidirectional streams, so
Transcribe and Nova Sonic use AWS's experimental `aws-sdk-*` Python clients over the CRT transport.
Check those first if a stream fails: they are preview software, and AWS says not to run them in
production.

Claude on Bedrock has rules a local model doesn't, which the adapter options cover:
  - Claude Opus 5.5 rejects `temperature`, and Haiku 5.5 rejects non-default values, so profiles
    set `send_temperature: false`.
  - They always think, and thinking counts against `maxTokens`, so a caller asking for 256 tokens
    can get back nothing. `min_max_tokens` sets a floor.
  - Opus 5.5 returns a 400 on forced `toolChoice`, so JSON goes through `outputConfig`.
  - An assistant turn still waiting on its tool results must go back with its reasoning intact.
    The history type has nowhere to keep reasoning, so the adapter holds it, keyed by tool use id.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import aclosing
from typing import Any

from voicelab.core.audio import AudioClip
from voicelab.core.models import Word
from voicelab.providers.base import (
    ChatMessage,
    LLMDelta,
    LLMResponse,
    RealtimeEvent,
    ToolCall,
    ToolDef,
    Transcript,
    VoiceInfo,
    register,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = "AWS packages are missing. Install them with: uv sync --extra aws"
_END = object()


def _boto_client(service: str, region: str | None) -> Any:
    try:
        import boto3
    except ImportError as failure:
        raise RuntimeError(_INSTALL_HINT) from failure
    return boto3.client(service, region_name=region)


async def _pump_in_thread[T](
    produce: Callable[[], Iterable[T]], on_close: Callable[[], None] = lambda: None
) -> AsyncIterator[T]:
    """Consume a blocking iterator on a worker thread, so it can't stall the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    stop = threading.Event()

    def pump() -> None:
        try:
            for item in produce():
                if stop.is_set():
                    return
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as failure:
            loop.call_soon_threadsafe(queue.put_nowait, failure)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _END)

    worker = loop.run_in_executor(None, pump)
    try:
        while True:
            item = await queue.get()
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        on_close()
        await asyncio.shield(worker)


# Converse --------------------------------------------------------------------------------------


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Claude's structured output requires every object to close its properties."""
    closed = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(closed)
    return closed


def _converse_messages(
    messages: list[ChatMessage], reasoning: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Convert history. Converse needs turns to alternate, so neighbours of one role merge."""
    last_user_text = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
    wire: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.role == "system":
            continue
        if message.role == "tool":
            role = "user"
            content = [
                {
                    "toolResult": {
                        "toolUseId": message.tool_call_id,
                        "content": [{"text": message.content}],
                    }
                }
            ]
        else:
            role = message.role
            content = []
            # Only the round in progress needs its reasoning. Older turns can be rewritten by a
            # barge-in, and an edited turn sent with its reasoning is rejected.
            if index > last_user_text:
                for call in message.tool_calls[:1]:
                    content.extend(reasoning.get(call.id, []))
            if message.content.strip():  # Converse rejects blank text blocks
                content.append({"text": message.content})
            for call in message.tool_calls:
                content.append(
                    {"toolUse": {"toolUseId": call.id, "name": call.name, "input": call.arguments}}
                )
        if wire and wire[-1]["role"] == role:
            wire[-1]["content"].extend(content)
        elif content:
            wire.append({"role": role, "content": content})
    return wire


@register("llm", "bedrock_converse")
class BedrockConverseLLM:
    """Any Bedrock model through Converse and ConverseStream. Never run against live AWS."""

    untested = True

    def __init__(
        self,
        model: str,
        region: str | None = None,
        *,
        name: str | None = None,
        send_temperature: bool = True,
        min_max_tokens: int = 0,
        effort: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.region = region
        self.name = name or f"bedrock/{model}"
        self.send_temperature = send_temperature
        self.min_max_tokens = min_max_tokens
        self.effort = effort
        self._boto = client
        self._reasoning: dict[str, list[dict[str, Any]]] = {}

    @property
    def _client(self) -> Any:
        if self._boto is None:
            self._boto = _boto_client("bedrock-runtime", self.region)
        return self._boto

    def _request(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDef] | None,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        inference: dict[str, Any] = {"maxTokens": max(max_tokens, self.min_max_tokens)}
        if self.send_temperature:
            inference["temperature"] = temperature
        request: dict[str, Any] = {
            "modelId": self.model,
            "messages": _converse_messages(messages, self._reasoning),
            "inferenceConfig": inference,
        }
        system = [{"text": m.content} for m in messages if m.role == "system"]
        if system:
            request["system"] = system
        if tools:
            request["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": {"json": tool.parameters},
                        }
                    }
                    for tool in tools
                ]
            }
        output: dict[str, Any] = {}
        if self.effort:
            output["effort"] = self.effort
        if json_schema is not None:
            output["textFormat"] = {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": json.dumps(_strict_schema(json_schema)),
                        "name": "reply",
                    }
                },
            }
        if output:
            request["outputConfig"] = output
        return request

    def _keep_reasoning(self, blocks: list[dict[str, Any]], calls: list[ToolCall]) -> None:
        if blocks and calls:
            self._reasoning[calls[0].id] = blocks

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        request = self._request(messages, tools, temperature, max_tokens, json_schema)
        reply = await asyncio.to_thread(self._client.converse, **request)
        text = ""
        calls: list[ToolCall] = []
        reasoning: list[dict[str, Any]] = []
        for block in reply.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                text += block["text"]
            elif "toolUse" in block:
                use = block["toolUse"]
                calls.append(
                    ToolCall(
                        id=use["toolUseId"], name=use["name"], arguments=use.get("input") or {}
                    )
                )
            elif "reasoningContent" in block:
                reasoning.append(block)
        self._keep_reasoning(reasoning, calls)
        usage = reply.get("usage", {})
        return LLMResponse(
            text=text,
            tool_calls=calls,
            finish_reason=reply.get("stopReason"),
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            model=self.model,
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]:
        request = self._request(messages, tools, temperature, max_tokens)
        started = time.perf_counter()
        opened: dict[str, Any] = {}

        def produce() -> Iterable[dict[str, Any]]:
            opened["stream"] = self._client.converse_stream(**request)["stream"]
            return opened["stream"]

        def close() -> None:
            stream = opened.get("stream")
            if stream is not None and hasattr(stream, "close"):
                stream.close()

        text = ""
        first_token_ms: float | None = None
        finish: str | None = None
        usage: dict[str, Any] = {}
        tools_by_block: dict[int, dict[str, Any]] = {}
        reasoning_by_block: dict[int, dict[str, Any]] = {}
        async with aclosing(_pump_in_thread(produce, close)) as events:
            async for event in events:
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"]
                    use = start.get("start", {}).get("toolUse")
                    if use:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        tools_by_block[start["contentBlockIndex"]] = {**use, "input": ""}
                elif "contentBlockDelta" in event:
                    block = event["contentBlockDelta"]
                    delta = block.get("delta", {})
                    if delta.get("text"):
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        text += delta["text"]
                        yield LLMDelta(text=delta["text"])
                    elif "toolUse" in delta:
                        tools_by_block[block["contentBlockIndex"]]["input"] += delta["toolUse"].get(
                            "input", ""
                        )
                    elif "reasoningContent" in delta:
                        self._add_reasoning(reasoning_by_block, block)
                elif "messageStop" in event:
                    finish = event["messageStop"].get("stopReason")
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
        calls = self._finish_tools(tools_by_block)
        blocks = [{"reasoningContent": reasoning_by_block[i]} for i in sorted(reasoning_by_block)]
        self._keep_reasoning(blocks, calls)
        final = LLMResponse(
            text=text,
            tool_calls=calls,
            finish_reason=finish,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            first_token_ms=first_token_ms,
            model=self.model,
        )
        yield LLMDelta(final=final)

    @staticmethod
    def _add_reasoning(blocks: dict[int, dict[str, Any]], event: dict[str, Any]) -> None:
        delta = event["delta"]["reasoningContent"]
        block = blocks.setdefault(event["contentBlockIndex"], {})
        if "redactedContent" in delta:
            block["redactedContent"] = delta["redactedContent"]
            return
        reasoning = block.setdefault("reasoningText", {"text": ""})
        reasoning["text"] += delta.get("text", "")
        if delta.get("signature"):
            reasoning["signature"] = delta["signature"]

    @staticmethod
    def _finish_tools(by_block: dict[int, dict[str, Any]]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(by_block):
            use = by_block[index]
            try:
                arguments = json.loads(use["input"] or "{}")
            except ValueError:
                logger.warning("dropping tool call with bad input: %.300r", use)
                continue
            if isinstance(arguments, dict):
                calls.append(ToolCall(id=use["toolUseId"], name=use["name"], arguments=arguments))
        return calls


# Transcribe ------------------------------------------------------------------------------------

_TRANSCRIBE_RATE = 16000
_CHUNK_S = 0.1  # AWS recommends uniform chunks of 50 to 200 ms


async def _open_transcribe(region: str, request: Any) -> Any:
    try:
        from aws_sdk_transcribe_streaming.client import AsyncTranscribeStreamingClient
        from aws_sdk_transcribe_streaming.config import AsyncTranscribeStreamingConfig
        from smithy_http.aio.crt import AWSCRTHTTPClient
    except ImportError as failure:
        raise RuntimeError(_INSTALL_HINT) from failure
    config = await AsyncTranscribeStreamingConfig.resolve(
        region=region, transport=AWSCRTHTTPClient()
    )
    return await AsyncTranscribeStreamingClient(config=config).start_stream_transcription(request)


@register("stt", "transcribe")
class TranscribeSTT:
    """Amazon Transcribe streaming. Never run against live AWS.

    Transcribe biases recognition through a vocabulary created ahead of time, not a list passed
    with the request. `vocabulary_name` is used only when a caller passes `hints`, and eval code
    passes None, so an eval transcript can't be biased however the profile is set.

    A finished clip is still streamed, because streaming is what gives word confidence without an
    S3 bucket. AWS asks for streams close to real time, so `speed` defaults to 1 and a long clip
    takes as long as it plays. Raising it is unverified.
    """

    untested = True

    def __init__(
        self,
        region: str = "us-east-1",
        *,
        name: str = "amazon-transcribe",
        language_code: str = "en-US",
        vocabulary_name: str | None = None,
        speed: float = 1.0,
        open_stream: Callable[[str, Any], Awaitable[Any]] = _open_transcribe,
    ) -> None:
        self.region = region
        self.name = name
        self.language_code = language_code
        self.vocabulary_name = vocabulary_name
        self.speed = speed
        self._open = open_stream

    def _start_request(self, hints: list[str] | None) -> Any:
        try:
            from aws_sdk_transcribe_streaming.models import StartStreamTranscriptionInput
        except ImportError as failure:
            raise RuntimeError(_INSTALL_HINT) from failure
        return StartStreamTranscriptionInput(
            language_code=self.language_code,
            media_sample_rate_hertz=_TRANSCRIBE_RATE,
            media_encoding="pcm",
            vocabulary_name=self.vocabulary_name if hints else None,
        )

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        stream = await self._open(self.region, self._start_request(hints))
        pcm = audio.resampled(_TRANSCRIBE_RATE).pcm
        sender = asyncio.create_task(self._send_audio(stream, pcm))
        try:
            _, output = await stream.await_output()
            transcript = await self._read_results(output)
            await sender
        finally:
            sender.cancel()
        return transcript.model_copy(update={"language": language})

    async def _send_audio(self, stream: Any, pcm: bytes) -> None:
        from aws_sdk_transcribe_streaming.models import AudioEvent, AudioStreamAudioEvent

        size = round(_CHUNK_S * _TRANSCRIBE_RATE) * 2
        try:
            for position in range(0, len(pcm), size):
                chunk = pcm[position : position + size]
                await stream.input_stream.send(
                    AudioStreamAudioEvent(value=AudioEvent(audio_chunk=chunk))
                )
                await asyncio.sleep(_CHUNK_S / self.speed)
        finally:
            await stream.input_stream.close()

    async def _read_results(self, output: Any) -> Transcript:
        texts: list[str] = []
        words: list[Word] = []
        async for event in output:
            value = getattr(event, "value", None)
            if isinstance(value, Exception):
                raise value
            for result in getattr(getattr(value, "transcript", None), "results", None) or []:
                # Partial results are revised later, and only final ones are safe to keep.
                if result.is_partial or not result.alternatives:
                    continue
                best = result.alternatives[0]
                texts.append((best.transcript or "").strip())
                words.extend(self._timed_words(best.items or []))
        return Transcript(text=" ".join(t for t in texts if t), words=words, model=self.name)

    @staticmethod
    def _timed_words(items: list[Any]) -> list[Word]:
        words: list[Word] = []
        for item in items:
            if item.type == "punctuation":
                if words:
                    words[-1] = words[-1].model_copy(
                        update={"text": f"{words[-1].text}{item.content}"}
                    )
                continue
            words.append(
                Word(
                    text=item.content,
                    start_s=item.start_time,
                    end_s=item.end_time,
                    confidence=item.confidence,
                )
            )
        return words


# Polly -----------------------------------------------------------------------------------------

_POLLY_CHUNK_BYTES = 4096


@register("tts", "polly")
class PollyTTS:
    """Amazon Polly over `SynthesizeSpeech` in raw PCM. Never run against live AWS.

    Persona voice ids are provider neutral, so `voice_map` turns them into Polly names. An unmapped
    id is passed through, which also lets a persona name a Polly voice directly. `style` is
    ignored, since Polly has no free-text delivery hint.
    """

    untested = True

    def __init__(
        self,
        region: str | None = None,
        *,
        name: str = "amazon-polly",
        engine: str = "neural",
        voice_id: str = "Joanna",
        voice_map: dict[str, str] | None = None,
        language_code: str = "en-US",
        sample_rate: int = 16000,  # PCM is 8000 or 16000
        client: Any = None,
    ) -> None:
        self.region = region
        self.name = name
        self.engine = engine
        self.voice_id = voice_id
        self.voice_map = voice_map or {}
        self.language_code = language_code
        self.sample_rate = sample_rate
        self._boto = client

    @property
    def _client(self) -> Any:
        if self._boto is None:
            self._boto = _boto_client("polly", self.region)
        return self._boto

    async def stream(
        self, text: str, *, voice_id: str, style: str | None = None
    ) -> AsyncIterator[AudioClip]:
        request = {
            "Text": text,
            "TextType": "text",
            "VoiceId": self.voice_map.get(voice_id, voice_id),
            "Engine": self.engine,
            "LanguageCode": self.language_code,
            "OutputFormat": "pcm",
            "SampleRate": str(self.sample_rate),
        }

        def produce() -> Iterable[bytes]:
            body = self._client.synthesize_speech(**request)["AudioStream"]
            with body:
                yield from body.iter_chunks(_POLLY_CHUNK_BYTES)

        leftover = b""
        async with aclosing(_pump_in_thread(produce)) as chunks:
            async for chunk in chunks:
                data = leftover + chunk
                usable = len(data) - len(data) % 2  # a sample can straddle two chunks
                leftover = data[usable:]
                if usable:
                    yield AudioClip(data[:usable], self.sample_rate)

    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        chunks = [clip.pcm async for clip in self.stream(text, voice_id=voice_id, style=style)]
        return AudioClip(b"".join(chunks), self.sample_rate)

    async def voices(self) -> list[VoiceInfo]:
        reply = await asyncio.to_thread(
            self._client.describe_voices, Engine=self.engine, LanguageCode=self.language_code
        )
        return [
            VoiceInfo(
                id=voice["Id"],
                label=voice.get("Name", voice["Id"]),
                gender=voice.get("Gender"),
                language=voice.get("LanguageCode", self.language_code),
            )
            for voice in reply.get("Voices", [])
        ]


# Nova Sonic ------------------------------------------------------------------------------------

_SONIC_INPUT_RATE = 16000
_SONIC_OUTPUT_RATE = 24000
_INTERRUPTED_MARK = '"interrupted"'


async def _open_sonic(region: str, model: str) -> Any:
    try:
        from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
        from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
        from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamOperationInput
        from smithy_http.aio.crt import AWSCRTHTTPClient
    except ImportError as failure:
        raise RuntimeError(_INSTALL_HINT) from failure
    config = await AsyncBedrockRuntimeConfig.resolve(region=region, transport=AWSCRTHTTPClient())
    return await AsyncBedrockRuntimeClient(config=config).invoke_model_with_bidirectional_stream(
        InvokeModelWithBidirectionalStreamOperationInput(model_id=model)
    )


@register("realtime", "nova_sonic")
class NovaSonicVoice:
    """Nova Sonic over `InvokeModelWithBidirectionalStream`. Never run against live AWS.

    The Bedrock docs found describe `amazon.nova-sonic-v1:0` and `amazon.nova-2-sonic-v1:0`. The
    event protocol here comes from those, and `amazon.nova-2-5-sonic` is assumed to match.
    """

    untested = True
    input_sample_rate = _SONIC_INPUT_RATE
    output_sample_rate = _SONIC_OUTPUT_RATE

    def __init__(
        self,
        model: str = "amazon.nova-2-5-sonic",
        region: str = "us-east-1",
        *,
        name: str | None = None,
        voice_id: str = "matthew",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        endpointing_sensitivity: str = "HIGH",
        open_stream: Callable[[str, str], Awaitable[Any]] = _open_sonic,
    ) -> None:
        self.model = model
        self.region = region
        self.name = name or f"bedrock/{model}"
        self.voice_id = voice_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.endpointing_sensitivity = endpointing_sensitivity
        self._open = open_stream

    async def connect(
        self, *, instructions: str, tools: list[ToolDef], voice_id: str | None = None
    ) -> NovaSonicSession:
        stream = await self._open(self.region, self.model)
        session = NovaSonicSession(stream)
        await session.start(self, instructions, tools, voice_id or self.voice_id)
        return session


class NovaSonicSession:
    """One open stream. AWS closes these after 8 minutes, and this does not renew them.

    Limitation on interruption: Nova detects barge-in by itself and stops speaking. There is no
    event to tell it how much was played, so `interrupt` can't drop unheard text from its context,
    and it only stops forwarding audio already on its way.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._prompt = str(uuid.uuid4())
        self._audio_content = str(uuid.uuid4())
        self._role = ""
        self._stage = ""
        self._dropping_audio = False
        self._closed = False

    async def _send(self, event: dict[str, Any]) -> None:
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )

        payload = json.dumps({"event": event}).encode("utf-8")
        await self._stream.input_stream.send(
            InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=payload)
            )
        )

    async def start(
        self, voice: NovaSonicVoice, instructions: str, tools: list[ToolDef], voice_id: str
    ) -> None:
        await self._send(
            {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": voice.max_tokens,
                        "topP": voice.top_p,
                        "temperature": voice.temperature,
                    },
                    "turnDetectionConfiguration": {
                        "endpointingSensitivity": voice.endpointing_sensitivity
                    },
                }
            }
        )
        prompt: dict[str, Any] = {
            "promptName": self._prompt,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": _SONIC_OUTPUT_RATE,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": voice_id,
                "encoding": "base64",
                "audioType": "SPEECH",
            },
        }
        if tools:
            prompt["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            prompt["toolConfiguration"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool.name,
                            "description": tool.description,
                            # A string here, unlike Converse, where it is a JSON object.
                            "inputSchema": {"json": json.dumps(tool.parameters)},
                        }
                    }
                    for tool in tools
                ]
            }
        await self._send({"promptStart": prompt})
        system = str(uuid.uuid4())
        await self._send(
            {
                "contentStart": {
                    "promptName": self._prompt,
                    "contentName": system,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        )
        await self._send(
            {
                "textInput": {
                    "promptName": self._prompt,
                    "contentName": system,
                    "content": instructions,
                }
            }
        )
        await self._send({"contentEnd": {"promptName": self._prompt, "contentName": system}})
        await self._send(
            {
                "contentStart": {
                    "promptName": self._prompt,
                    "contentName": self._audio_content,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": _SONIC_INPUT_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        )

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send(
            {
                "audioInput": {
                    "promptName": self._prompt,
                    "contentName": self._audio_content,
                    "content": base64.b64encode(pcm16).decode("ascii"),
                }
            }
        )

    async def send_tool_result(self, call_id: str, result: str) -> None:
        content = str(uuid.uuid4())
        await self._send(
            {
                "contentStart": {
                    "promptName": self._prompt,
                    "contentName": content,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": call_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            }
        )
        await self._send(
            {"toolResult": {"promptName": self._prompt, "contentName": content, "content": result}}
        )
        await self._send({"contentEnd": {"promptName": self._prompt, "contentName": content}})

    async def interrupt(self, *, response_id: str | None, played_ms: int) -> None:
        self._dropping_audio = True

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        _, output = await self._stream.await_output()
        async for item in output:
            value = getattr(item, "value", None)
            if isinstance(value, Exception):
                yield RealtimeEvent(type="error", text=str(value))
                return
            payload = getattr(value, "bytes_", None)
            if not payload:
                continue
            try:
                event = json.loads(payload.decode("utf-8")).get("event", {})
            except ValueError:
                logger.warning("skipping unreadable Nova event: %.200r", payload)
                continue
            for normalized in self._normalize(event):
                yield normalized

    def _normalize(self, event: dict[str, Any]) -> list[RealtimeEvent]:
        if "contentStart" in event:
            start = event["contentStart"]
            self._role = start.get("role", "")
            self._stage = self._generation_stage(start)
            if start.get("type") == "AUDIO":
                self._dropping_audio = False
            return []
        if "textOutput" in event:
            return self._text_event(event["textOutput"])
        if "audioOutput" in event:
            if self._dropping_audio:
                return []
            audio = base64.b64decode(event["audioOutput"].get("content", ""))
            return [RealtimeEvent(type="audio", audio_pcm16=audio)]
        if "toolUse" in event:
            return [self._tool_event(event["toolUse"])]
        if "contentEnd" in event and event["contentEnd"].get("stopReason") == "INTERRUPTED":
            return [RealtimeEvent(type="speech_started")]
        if "completionEnd" in event:
            return [RealtimeEvent(type="response_done")]
        return []

    @staticmethod
    def _generation_stage(start: dict[str, Any]) -> str:
        fields = start.get("additionalModelFields")
        if not fields:
            return ""
        try:
            return json.loads(fields).get("generationStage", "")
        except ValueError:
            return ""

    def _text_event(self, output: dict[str, Any]) -> list[RealtimeEvent]:
        content = output.get("content", "")
        if _INTERRUPTED_MARK in content:
            return [RealtimeEvent(type="speech_started")]
        role = output.get("role") or self._role
        if role == "USER":
            return [RealtimeEvent(type="user_transcript", text=content)]
        # SPECULATIVE text is what the model plans to say. FINAL is what it did say, and a
        # barge-in leaves the FINAL text short, which is the only record of what was played.
        if role == "ASSISTANT" and self._stage != "SPECULATIVE":
            return [RealtimeEvent(type="bot_transcript", text=content)]
        return []

    @staticmethod
    def _tool_event(use: dict[str, Any]) -> RealtimeEvent:
        try:
            arguments = json.loads(use.get("content") or "{}")
        except ValueError:
            arguments = None
        if not isinstance(arguments, dict):
            return RealtimeEvent(type="error", text=f"bad tool input: {use.get('content')!r:.300}")
        call = ToolCall(id=use["toolUseId"], name=use["toolName"], arguments=arguments)
        return RealtimeEvent(type="tool_call", tool_call=call)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._send(
                {"contentEnd": {"promptName": self._prompt, "contentName": self._audio_content}}
            )
            await self._send({"promptEnd": {"promptName": self._prompt}})
            await self._send({"sessionEnd": {}})
        finally:
            await self._stream.input_stream.close()
