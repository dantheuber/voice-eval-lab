"""Shared pieces for engine tests: scripted models around the fake speech adapters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    Beat,
    CallMode,
    LineConditions,
    Persona,
    Scenario,
    VoiceRef,
)
from voicelab.core.profiles import Profile, ProviderStack, build_stack
from voicelab.engine.contract import BotCallRequest
from voicelab.providers.adapters.fake import FakeLLM, FakeSTT, FakeTTS
from voicelab.providers.base import (
    ChatMessage,
    LLMDelta,
    LLMResponse,
    ToolCall,
    ToolDef,
    Transcript,
)

# No codec, noise, or loss, so what the fake STT reads back is exactly what was played.
CLEAN = LineConditions(codec="none", one_way_latency_ms=80)


def persona(*, interruption_rate: float = 0.0, line: LineConditions = CLEAN) -> Persona:
    return Persona(
        id="per_test",
        name="Test caller",
        interruption_rate=interruption_rate,
        voice=VoiceRef(voice_id="default_female_2"),
        line=line,
    )


def scenario(*, max_turns: int = 8, facts: dict[str, str] | None = None) -> Scenario:
    return Scenario(
        id="scn_test",
        title="Test",
        goal="get help",
        facts=facts or {"name": "Dana"},
        beats=[Beat(id="only", instruction="Say hello.")],
        max_turns=max_turns,
    )


def request(call_id: str = "call_test", **overrides: Any) -> BotCallRequest:
    values: dict[str, Any] = {
        "call_id": call_id,
        "mode": CallMode.TURN_BASED,
        "persona": persona(),
        "scenario": scenario(),
        "seed": 1,
    }
    return BotCallRequest(**(values | overrides))


def fake_profile() -> Profile:
    fake = {"type": "fake"}
    return Profile(
        id="fake",
        label="Fake",
        roles={
            "stt_live": fake,
            "stt_eval": fake,
            "tts_bot": fake | {"voice_id": "default_female_1"},
            "tts_caller": fake,
            "llm_support": fake,
            "llm_caller": fake,
            "llm_judge": fake,
        },
    )


def stack(**replacements: Any) -> ProviderStack:
    built = build_stack(fake_profile())
    for role, adapter in replacements.items():
        setattr(built, role, adapter)
    return built


def says(text: str, *, beat: str | None = None, hang_up: bool = False, reason: str = "") -> str:
    return json.dumps(
        {"say": text, "completed_beat": beat, "hang_up": hang_up, "hang_up_reason": reason}
    )


def uses(name: str, *, text: str = "", **arguments: Any) -> LLMResponse:
    call = ToolCall(id=f"call_{name}", name=name, arguments=arguments)
    return LLMResponse(text=text, tool_calls=[call])


class Scripted:
    """Plays back entries, then repeats `default`. Records every request as it arrives."""

    def __init__(self, *script: str | LLMResponse, default: str | LLMResponse | None = None):
        self.name = "scripted"
        self.script = list(script)
        self.default = default
        self.requests: list[list[ChatMessage]] = []

    def _next(self) -> LLMResponse:
        entry = self.script.pop(0) if self.script else self.default
        if entry is None:
            raise AssertionError("the script ran out")
        return LLMResponse(text=entry) if isinstance(entry, str) else entry

    async def chat(self, messages: list[ChatMessage], **options: Any) -> LLMResponse:
        self.requests.append([m.model_copy() for m in messages])
        return self._next()

    async def stream_chat(
        self, messages: list[ChatMessage], **options: Any
    ) -> AsyncIterator[LLMDelta]:
        self.requests.append([m.model_copy() for m in messages])
        response = self._next()
        if response.text:
            yield LLMDelta(text=response.text)
        yield LLMDelta(final=response)


class HangingLLM:
    """Never answers, as a stuck model server wouldn't."""

    name = "hanging"

    def __init__(self) -> None:
        self.closed = False

    async def chat(self, messages: list[ChatMessage], **options: Any) -> LLMResponse:
        await asyncio.Event().wait()
        raise AssertionError

    async def stream_chat(
        self, messages: list[ChatMessage], **options: Any
    ) -> AsyncIterator[LLMDelta]:
        try:
            await asyncio.Event().wait()
            yield LLMDelta()
        finally:
            self.closed = True


class Rewriting(FakeSTT):
    """Hears through a fixed mapping, so a test can tell heard text from intended text."""

    def __init__(self, replace: dict[str, str]) -> None:
        super().__init__()
        self.replace = replace
        self.clips: list[AudioClip] = []

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        self.clips.append(audio)
        heard = await super().transcribe(audio, language=language, hints=hints)
        text = heard.text
        for old, new in self.replace.items():
            text = text.replace(old, new)
        return heard.model_copy(update={"text": text})


class DeafSTT(FakeSTT):
    """Hears nothing from the `deaf_from`-th clip on."""

    def __init__(self, deaf_from: int = 0) -> None:
        super().__init__()
        self.deaf_from = deaf_from
        self.count = 0

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        self.count += 1
        if self.count > self.deaf_from:
            return Transcript(text="")
        return await super().transcribe(audio, language=language, hints=hints)


class SlowSTT(FakeSTT):
    """Hears correctly, but takes longer than a turn is allowed to stay open."""

    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        await asyncio.sleep(self.delay_s)
        return await super().transcribe(audio, language=language, hints=hints)


class FailingTTS(FakeTTS):
    """Works for `ok_for` lines and then raises."""

    def __init__(self, ok_for: int = 0) -> None:
        super().__init__()
        self.remaining = ok_for

    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        if self.remaining <= 0:
            raise RuntimeError("tts server went away")
        self.remaining -= 1
        return await super().synthesize(text, voice_id=voice_id, style=style)


class SilentTTS(FakeTTS):
    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        return AudioClip(b"", self.sample_rate)


class Messages:
    """Collects what the engine emits."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.items.append(message)

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [item for item in self.items if item["type"] == kind]


__all__ = ["CLEAN", "FakeLLM", "ToolDef"]
