"""Provider seams. Engine, agent, and eval code depends only on these interfaces.

An adapter is a class implementing one interface, registered under a type name with `@register`,
and constructed from a config block. Swapping a model means editing a profile YAML. Local models
run as separate server processes, so adapters here stay thin HTTP or WebSocket clients and the
main app never imports torch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from voicelab.core.audio import AudioClip
from voicelab.core.models import Word

# STT -------------------------------------------------------------------------------------------


class Transcript(BaseModel):
    text: str
    words: list[Word] = Field(default_factory=list)  # empty when the provider can't time words
    language: str | None = None
    model: str = ""


@runtime_checkable
class STT(Protocol):
    name: str

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        """Transcribe one finished clip. `hints` biases recognition, so eval code passes None."""
        ...


# TTS -------------------------------------------------------------------------------------------


class VoiceInfo(BaseModel):
    id: str
    label: str
    gender: str | None = None
    age: str | None = None
    accent: str | None = None
    language: str = "en"
    notes: str = ""


@runtime_checkable
class TTS(Protocol):
    name: str
    sample_rate: int

    def stream(
        self, text: str, *, voice_id: str, style: str | None = None
    ) -> AsyncIterator[AudioClip]:
        """Yield audio chunks at `sample_rate` as they are generated."""
        ...

    async def synthesize(
        self, text: str, *, voice_id: str, style: str | None = None
    ) -> AudioClip: ...

    async def voices(self) -> list[VoiceInfo]: ...


# LLM -------------------------------------------------------------------------------------------


class ToolDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # role=assistant
    tool_call_id: str | None = None  # role=tool
    name: str | None = None  # role=tool


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    first_token_ms: float | None = None
    model: str = ""


class LLMDelta(BaseModel):
    """One streaming event. Text arrives incrementally; `final` carries the assembled response."""

    text: str = ""
    final: LLMResponse | None = None


@runtime_checkable
class LLM(Protocol):
    name: str

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """One completion. With `json_schema`, `text` is JSON matching that schema."""
        ...

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]: ...


# Speech-to-speech ------------------------------------------------------------------------------


class RealtimeEvent(BaseModel):
    """Normalized event from a speech-to-speech session."""

    type: Literal[
        "speech_started",  # the model detected user speech
        "speech_stopped",
        "user_transcript",
        "audio",  # bot audio chunk in `audio_pcm16`
        "bot_transcript",
        "tool_call",
        "response_done",
        "error",
    ]
    text: str = ""
    audio_pcm16: bytes = b""
    tool_call: ToolCall | None = None
    response_id: str | None = None  # the response this event belongs to, when the model has ids


class RealtimeSession(Protocol):
    async def send_audio(self, pcm16: bytes) -> None: ...

    async def send_tool_result(self, call_id: str, result: str) -> None: ...

    async def interrupt(self, *, response_id: str | None, played_ms: int) -> None:
        """Stop a response after `played_ms` of its audio actually reached the listener.

        Audio arrives faster than it plays, so the session alone can't know the cut point. The
        engine tracks playback and passes it here so the adapter can drop unheard text from the
        model's context. An adapter whose model can't do that must document the limitation.
        """
        ...

    def events(self) -> AsyncIterator[RealtimeEvent]: ...

    async def close(self) -> None: ...


@runtime_checkable
class RealtimeVoice(Protocol):
    name: str
    input_sample_rate: int
    output_sample_rate: int

    async def connect(
        self, *, instructions: str, tools: list[ToolDef], voice_id: str | None = None
    ) -> RealtimeSession: ...


# Audio listening -------------------------------------------------------------------------------


@runtime_checkable
class AudioAnalyst(Protocol):
    """An audio-capable model that hears a clip and answers about it. Advisory evidence only."""

    name: str

    async def analyze(
        self, audio: AudioClip, prompt: str, *, json_schema: dict[str, Any] | None = None
    ) -> str: ...


# Registry --------------------------------------------------------------------------------------

Role = Literal["stt", "tts", "llm", "realtime", "audio_analyst"]
T = TypeVar("T")
_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}


def register(role: Role, type_name: str) -> Callable[[T], T]:
    """Class decorator: make an adapter constructible from `{type: type_name, ...}` config."""

    def wrap(cls: T) -> T:
        _REGISTRY.setdefault(role, {})[type_name] = cls  # type: ignore[assignment]
        return cls

    return wrap


def build(role: Role, config: dict[str, Any]) -> Any:
    """Construct an adapter from a profile config block. Remaining keys go to its constructor."""
    settings = dict(config)
    type_name = settings.pop("type")
    try:
        factory = _REGISTRY[role][type_name]
    except KeyError as exc:
        known = sorted(_REGISTRY.get(role, {}))
        raise ValueError(f"unknown {role} provider {type_name!r}, known: {known}") from exc
    return factory(**settings)
