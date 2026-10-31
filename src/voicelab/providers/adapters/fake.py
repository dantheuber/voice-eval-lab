"""Deterministic adapters with no models, so a whole call can run on any machine.

Audio is real and crosses the phone line: see `fake_speech`. LLM replies come from the rules in
`fake_brains`. The same inputs give the same call, byte for byte.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any

from voicelab.core.audio import AudioClip
from voicelab.core.models import Word
from voicelab.providers.adapters.fake_brains import (
    caller_reply,
    draft_reply,
    judge_reply,
    support_reply,
)
from voicelab.providers.adapters.fake_speech import decode_speech, encode_speech
from voicelab.providers.base import (
    ChatMessage,
    LLMDelta,
    LLMResponse,
    ToolDef,
    Transcript,
    VoiceInfo,
    register,
)

_STREAM_CHUNK_S = 0.1
_VOICES = [
    VoiceInfo(id=f"default_{gender}_{number}", label=f"Fake {gender} {number}", gender=gender)
    for gender, count in (("female", 5), ("male", 4))
    for number in range(1, count + 1)
]


@register("stt", "fake")
class FakeSTT:
    """Reads the text `FakeTTS` put in the audio, from whatever survived the trip."""

    def __init__(self, name: str = "fake-stt", delay_s: float = 0.0) -> None:
        self.name = name
        self.delay_s = delay_s

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        await asyncio.sleep(self.delay_s)
        heard = decode_speech(audio)
        words = [Word(text=w.text, start_s=w.start_s, end_s=w.end_s, confidence=w.confidence)
                 for w in heard]  # fmt: skip
        return Transcript(
            text=" ".join(w.text for w in heard), words=words, language=language, model=self.name
        )


@register("tts", "fake")
class FakeTTS:
    """Speech-shaped audio whose length follows the text. Voices differ only in level."""

    def __init__(
        self,
        name: str = "fake-tts",
        sample_rate: int = 16000,
        voice_id: str = "default_female_1",
        delay_s: float = 0.0,
    ) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.voice_id = voice_id
        self.delay_s = delay_s

    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        await asyncio.sleep(self.delay_s)
        return encode_speech(text, self.sample_rate, level=_voice_level(voice_id))

    async def stream(
        self, text: str, *, voice_id: str, style: str | None = None
    ) -> AsyncIterator[AudioClip]:
        clip = await self.synthesize(text, voice_id=voice_id, style=style)
        position = 0.0
        while position < clip.duration_s:
            yield clip.slice(position, position + _STREAM_CHUNK_S)
            position += _STREAM_CHUNK_S

    async def voices(self) -> list[VoiceInfo]:
        return list(_VOICES)


def _voice_level(voice_id: str) -> float:
    digest = hashlib.sha256(voice_id.encode()).digest()
    return 0.18 + 0.1 * digest[0] / 255


@register("llm", "fake")
class FakeLLM:
    """Picks its job from the request shape, so one instance can fill every LLM role."""

    def __init__(self, name: str = "fake-llm", delay_s: float = 0.0) -> None:
        self.name = name
        self.delay_s = delay_s

    def _answer(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDef] | None,
        json_schema: dict[str, Any] | None,
    ) -> LLMResponse:
        properties = (json_schema or {}).get("properties", {})
        system = messages[0].content if messages[0].role == "system" else ""
        last = messages[-1].content
        if "say" in properties:
            reply = caller_reply(system, last)
        elif "status" in properties or "results" in properties:
            reply = judge_reply(last)
        elif "beats" in properties:
            reply = draft_reply(system, last.split("this:", 1)[-1].strip())
        elif tools:
            return support_reply(messages).model_copy(update={"model": self.name})
        else:
            return LLMResponse(text="Okay.", finish_reason="stop", model=self.name)
        return LLMResponse(text=json.dumps(reply), finish_reason="stop", model=self.name)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        await asyncio.sleep(self.delay_s)
        return self._answer(messages, tools, json_schema).model_copy(
            update={"first_token_ms": self.delay_s * 1000}
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]:
        await asyncio.sleep(self.delay_s)
        response = self._answer(messages, tools, None)
        words = response.text.split(" ") if response.text else []
        for index, word in enumerate(words):
            yield LLMDelta(text=word if index == len(words) - 1 else f"{word} ")
        yield LLMDelta(final=response.model_copy(update={"first_token_ms": self.delay_s * 1000}))


@register("audio_analyst", "fake")
class FakeAnalyst:
    name = "fake-analyst"

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s

    async def analyze(
        self, audio: AudioClip, prompt: str, *, json_schema: dict[str, Any] | None = None
    ) -> str:
        await asyncio.sleep(self.delay_s)
        return json.dumps(
            {"tone": "even", "delivery": "Steady pace.", "odd_pronunciations": [], "issues": []}
        )
