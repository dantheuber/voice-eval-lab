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


def synth_voice(
    words: int = 16,
    *,
    speed: float = 1.0,
    f0: float = 120.0,
    flat: bool = False,
    seed: int = 0,
    wpm: float = 165.0,
) -> np.ndarray:
    """Speech-like audio: a pitched buzz through vowel resonances, with rhythm and intonation.

    `speed` time-compresses or stretches it the way a bad TTS does: rhythm and intonation play
    faster while the voice stays at its pitch. `flat` removes the intonation.
    """
    rng = np.random.default_rng(seed)
    seconds = words / wpm * 60 / speed
    n = round(seconds * RATE)
    t = np.arange(n) / RATE
    story = t * speed  # where in the sentence each moment is
    phases = rng.uniform(0, 2 * np.pi, 3)
    contour = 3.5 * np.sin(2 * np.pi * story / 1.7 + phases[0])
    contour = contour + 2.0 * np.sin(2 * np.pi * story / 0.63 + phases[1])
    if flat:
        contour = np.zeros(n)
    phase = 2 * np.pi * np.cumsum(f0 * 2 ** (contour / 12)) / RATE
    buzz = sum(np.sin(k * phase) / k for k in range(1, int(3600 / f0)))
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    formants = sum(
        np.exp(-(((freqs - hz) / width) ** 2)) * gain
        for hz, width, gain in ((500.0, 250.0, 1.0), (1500.0, 350.0, 0.7), (2500.0, 400.0, 0.4))
    )
    voice = np.fft.irfft(np.fft.rfft(buzz) * formants, n)
    # Real speech keeps closing between sounds, and those quiet moments are what a noise estimate
    # leans on. Here that's about 60 ms in every 250 ms, down 26 dB. They ease in and out over
    # 40 ms, and voicing carries through them, as on real speech, rather than stopping dead.
    openness = np.abs(np.sin(np.pi * story * 4.0 + phases[2]))
    syllables = np.clip((openness - 0.35) / 0.5, 0.0, 1.0)
    syllables = 0.05 + 0.95 * syllables * syllables * (3 - 2 * syllables)
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.05)
    voice = voice * syllables * (0.5 - 0.5 * np.cos(np.pi * fade))
    voice = 0.1 * voice / np.sqrt(np.mean(np.square(voice)))
    breath = rng.standard_normal(n) * 0.1 * 10 ** (-45 / 20)  # microphones and TTS are never silent
    return (voice + breath).astype(np.float32)


def loud_moment(samples: np.ndarray, near_s: float) -> int:
    """A sample at full level near `near_s`, so a cut lands on sound and not on a closure."""
    start = round(near_s * RATE)
    window = samples[start : start + RATE // 4]
    frames = len(window) // 80
    level = np.square(window[: frames * 80]).reshape(frames, 80).mean(axis=1)
    return start + int(np.argmax(level)) * 80


def with_noise(samples: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    gain = np.sqrt(np.mean(np.square(samples))) * 10 ** (-snr_db / 20)
    noise = np.random.default_rng(seed).standard_normal(len(samples)) * gain
    return (samples + noise).astype(np.float32)


def gated(samples: np.ndarray, *, every_s: float = 0.25, gap_s: float = 0.04) -> np.ndarray:
    """Hard dropouts through the speech, as a broken stream or a bad noise gate would leave."""
    out = samples.copy()
    every, gap = round(every_s * RATE), round(gap_s * RATE)
    for start in range(every, len(out) - every, every):
        out[start : start + gap] = 0
    return out


def with_pauses(samples: np.ndarray, *, every_s: float = 0.7, pause_s: float = 0.6) -> np.ndarray:
    parts = []
    every = round(every_s * RATE)
    for start in range(0, len(samples), every):
        parts += [samples[start : start + every], np.zeros(round(pause_s * RATE), np.float32)]
    return np.concatenate(parts[:-1])


def robotized(samples: np.ndarray, *, size: int = 512, hop: int = 128) -> np.ndarray:
    """Phase zeroing, the benchmark's robot filter. Pitch locks to the frame rate."""
    out = np.zeros(len(samples) + size)
    window = np.hanning(size)
    for start in range(0, len(samples) - size, hop):
        spectrum = np.abs(np.fft.rfft(samples[start : start + size] * window))
        out[start : start + size] += np.fft.fftshift(np.fft.irfft(spectrum, size)) * window
    out = out[: len(samples)]
    scale = np.sqrt(np.mean(np.square(samples)) / np.mean(np.square(out)))
    return (out * scale).astype(np.float32)
