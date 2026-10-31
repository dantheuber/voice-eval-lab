"""Fakes and audio builders for eval tests. No model servers, no real speech."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from voicelab.core.audio import AudioClip
from voicelab.core.models import CallArtifact, CallMode, Speaker, Word
from voicelab.providers.base import LLMResponse, Transcript
from voicelab.store.artifacts import ArtifactStore, CallPaths

RATE = 16000


def timed_words(
    text: str, start_s: float = 0.0, *, per_word_s: float = 0.4, confidence: float | None = 0.95
) -> list[Word]:
    return [
        Word(
            text=word,
            start_s=start_s + i * per_word_s,
            end_s=start_s + (i + 1) * per_word_s,
            confidence=confidence,
        )
        for i, word in enumerate(text.split())
    ]


def transcript(*segments: tuple[str, float] | str) -> Transcript:
    """Build a transcript from (text, start_s) pairs."""
    words: list[Word] = []
    for segment in segments:
        text, start = (segment, 0.0) if isinstance(segment, str) else segment
        words.extend(timed_words(text, start))
    return Transcript(text=" ".join(w.text for w in words), words=words, model="fake-stt")


class FakeSTT:
    """Answers per channel. Channels are told apart by call order: caller, then bot."""

    name = "fake-stt"

    def __init__(self, caller: Any = None, bot: Any = None) -> None:
        self._replies = [caller or Transcript(text=""), bot or Transcript(text="")]
        self.calls: list[dict[str, Any]] = []

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        self.calls.append({"duration_s": audio.duration_s, "language": language, "hints": hints})
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeLLM:
    """Answers from `reply(prompt)`, which returns a dict, a raw string, or raises."""

    name = "fake-judge"

    def __init__(self, reply: Callable[[str], Any] | None = None) -> None:
        self._reply = reply or (lambda prompt: verdict_json("pass", quotes=[]))
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    async def chat(self, messages, **kwargs) -> LLMResponse:
        prompt = messages[-1].content
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        reply = self._reply(prompt)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, model="fake-judge-v1")

    def stream_chat(self, messages, **kwargs):
        raise NotImplementedError


class FakeAnalyst:
    name = "fake-analyst"

    def __init__(self, reply: Callable[[AudioClip, str], Any] | None = None) -> None:
        self._reply = reply or (
            lambda audio, prompt: {
                "tone": "warm",
                "delivery": "Even pace.",
                "odd_pronunciations": [],
                "issues": [],
            }
        )
        self.clips: list[AudioClip] = []
        self.prompts: list[str] = []

    async def analyze(self, audio, prompt, *, json_schema=None) -> str:
        self.clips.append(audio)
        self.prompts.append(prompt)
        reply = self._reply(audio, prompt)
        return reply if isinstance(reply, str) else json.dumps(reply)


def verdict_json(status: str, *, reason: str = "because", quotes: Iterable[str] = ()) -> dict:
    return {"status": status, "reason": reason, "quotes": list(quotes)}


def speech(
    spans: Iterable[tuple[float, float]], duration_s: float, *, amp: float = 0.3
) -> AudioClip:
    """A channel with a tone during each (start, end) span and digital silence elsewhere."""
    samples = np.zeros(round(duration_s * RATE), dtype=np.float32)
    for start, end in spans:
        first, last = round(start * RATE), round(end * RATE)
        t = np.arange(last - first) / RATE
        samples[first:last] = (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return AudioClip.from_samples(samples, RATE)


def write_call(
    store: ArtifactStore,
    artifact: CallArtifact | None = None,
    *,
    caller: Iterable[tuple[float, float]] = (),
    bot: Iterable[tuple[float, float]] = (),
    duration_s: float = 10.0,
) -> tuple[CallArtifact, CallPaths]:
    artifact = artifact or CallArtifact(mode=CallMode.LIVE_BOTS)
    artifact.duration_s = duration_s
    paths = store.write_call(artifact, speech(caller, duration_s), speech(bot, duration_s))
    return artifact, paths


def speakers() -> tuple[Speaker, Speaker]:
    return Speaker.CALLER, Speaker.BOT
