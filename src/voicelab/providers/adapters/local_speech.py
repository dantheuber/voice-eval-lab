"""Clients for the speech servers this project runs on its own ports.

Each server's launch command is in `config/profiles/local-*.yaml`, and `docs/research/
benchmarks-2026-10.md` has what each was measured to do. These stay thin HTTP clients, so the app
never loads a model or imports torch.

  http_transcriptions   Qwen3-ASR, Parakeet, or any OpenAI-style /v1/audio/transcriptions
  whisper_cpp           whisper.cpp whisper-server, the one that returns real word probabilities
  phonon_tts            Phonon through scripts/phonon_server.py
  llama_tts             llama-server POST /tts, for LFM3-Audio
  pocket_tts            Pocket TTS preset voices
  llama_audio_analyst   an audio-capable llama-server model listening to a clip
"""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
from collections.abc import AsyncIterator
from typing import Any

import httpx
import numpy as np

from voicelab.core.audio import AudioClip
from voicelab.core.models import Word
from voicelab.providers.adapters.http_client import PerLoopClient
from voicelab.providers.base import Transcript, VoiceInfo, register

logger = logging.getLogger(__name__)

_STT_RATE = 16000
_TTS_RATE = 24000
_ASR_TEXT_MARK = "<asr_text>"


class SpeechServerError(RuntimeError):
    """A speech server refused a request or couldn't be reached."""


def _timeout(timeout_s: float) -> httpx.Timeout:
    return httpx.Timeout(timeout_s, connect=5.0)


def _wav_upload(audio: AudioClip) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("audio.wav", audio.resampled(_STT_RATE).to_wav_bytes(), "audio/wav")}


def _checked(response: httpx.Response, name: str) -> dict[str, Any]:
    if response.status_code >= 400:
        raise SpeechServerError(f"{name} answered {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as failure:
        raise SpeechServerError(f"{name} sent a reply that isn't JSON") from failure


@register("stt", "http_transcriptions")
class HttpTranscriptionSTT:
    """OpenAI-style transcription. `hints` are ignored, since these servers take none.

    Parakeet reports every word's confidence as exactly 1.0, on all 265 words checked, so
    `trust_confidence: false` reports them as unknown rather than as certain. Set
    `word_timing: false` for llama-server, which has no word output on this route.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "",
        *,
        name: str | None = None,
        trust_confidence: bool = True,
        word_timing: bool = True,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or f"transcriptions/{model or base_url}"
        self.trust_confidence = trust_confidence
        self.word_timing = word_timing
        self._http = PerLoopClient(timeout=_timeout(timeout_s))

    async def _model_id(self) -> str:
        if not self.model:
            listing = _checked(await self._http.get().get(f"{self.base_url}/v1/models"), self.name)
            self.model = listing["data"][0]["id"]
        return self.model

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        if audio.n_samples == 0:
            return Transcript(text="", model=self.name)
        data = {"model": await self._model_id(), "language": language, "response_format": "json"}
        if self.word_timing:  # llama-server refuses anything but plain json
            data |= {"response_format": "verbose_json", "timestamp_granularities[]": "word"}
        try:
            response = await self._http.get().post(
                f"{self.base_url}/v1/audio/transcriptions", files=_wav_upload(audio), data=data
            )
        except httpx.HTTPError as failure:
            raise SpeechServerError(f"{self.name} unreachable: {failure!r}") from failure
        reply = _checked(response, self.name)
        words = [
            Word(
                text=word.get("word", "").strip(),
                start_s=word.get("start", 0.0),
                end_s=word.get("end", 0.0),
                confidence=word.get("confidence") if self.trust_confidence else None,
            )
            for word in reply.get("words") or []
        ]
        # Qwen3-ASR writes "language English<asr_text>..." ahead of what it heard.
        text = (reply.get("text") or "").rsplit(_ASR_TEXT_MARK, 1)[-1].strip()
        return Transcript(text=text, words=words, language=language, model=self.name)


@register("stt", "whisper_cpp")
class WhisperCppSTT:
    """whisper.cpp's `whisper-server`. Word probabilities are real, so eval depends on this one.

    Pass no prompt, ever: a prompt biases what Whisper hears, and eval has to catch mispronounced
    words, not have them corrected away.
    """

    def __init__(
        self, base_url: str, *, name: str = "whisper.cpp", timeout_s: float = 900.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self._http = PerLoopClient(timeout=_timeout(timeout_s))

    async def transcribe(
        self, audio: AudioClip, *, language: str = "en", hints: list[str] | None = None
    ) -> Transcript:
        if audio.n_samples == 0:
            return Transcript(text="", model=self.name)
        data = {"response_format": "verbose_json", "temperature": "0", "language": language}
        try:
            response = await self._http.get().post(
                f"{self.base_url}/inference", files=_wav_upload(audio), data=data
            )
        except httpx.HTTPError as failure:
            raise SpeechServerError(f"{self.name} unreachable: {failure!r}") from failure
        reply = _checked(response, self.name)
        return Transcript(
            text=(reply.get("text") or "").strip(),
            words=_whisper_words(reply),
            language=language,
            model=self.name,
        )


def _whisper_words(reply: dict[str, Any]) -> list[Word]:
    """Words with probabilities from verbose JSON, dropping `[_BEG_]`-style special tokens.

    What whisper.cpp calls words are tokens: "I" and "'m", "60", "6" and "14". A leading space is
    the only mark of a new word, so tokens without one join the word before them. Everything
    downstream compares and quotes whole words.
    """
    words: list[Word] = []
    for segment in reply.get("segments") or []:
        for token in segment.get("words") or []:
            raw = token.get("word") or ""
            text = raw.strip()
            if not text or text.startswith("[_"):
                continue
            start = token.get("start", segment.get("start", 0.0))
            end = token.get("end", segment.get("end", 0.0))
            confidence = token.get("probability")
            if words and not raw[:1].isspace():
                words[-1] = _extended(words[-1], text, end, confidence)
            else:
                words.append(Word(text=text, start_s=start, end_s=end, confidence=confidence))
    return words


def _extended(word: Word, text: str, end_s: float, confidence: float | None) -> Word:
    """A word with one more token on it. Its weakest letters speak for it, its commas don't."""
    if any(c.isalnum() for c in text) and confidence is not None:
        confidence = min(confidence, word.confidence if word.confidence is not None else 1.0)
    else:
        confidence = word.confidence
    return word.model_copy(
        update={"text": word.text + text, "end_s": end_s, "confidence": confidence}
    )


@register("tts", "llama_tts")
class LlamaTTS:
    """llama-server `POST /tts`, streaming 32-bit float PCM. Used for LFM3-Audio.

    Send it the normalized spoken form only. Given "AC9077" it says "AC nineteen seventy seven".
    """

    sample_rate = _TTS_RATE

    def __init__(
        self,
        base_url: str,
        *,
        name: str = "llama-tts",
        voice_id: str = "female",
        voice_map: dict[str, str] | None = None,
        language: str = "en",
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.voice_id = voice_id
        self.voice_map = voice_map or {}
        self.language = language
        self._http = PerLoopClient(timeout=_timeout(timeout_s))

    async def stream(
        self, text: str, *, voice_id: str, style: str | None = None
    ) -> AsyncIterator[AudioClip]:
        body = {
            "input": text,
            "voice": self.voice_map.get(voice_id, voice_id),
            "lang": self.language,
            "stream": True,
            "response_format": "pcm",
        }
        try:
            async with self._http.get().stream(
                "POST", f"{self.base_url}/tts", json=body
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise SpeechServerError(
                        f"{self.name} answered {response.status_code}: {detail}"
                    )
                leftover = b""
                async for chunk in response.aiter_bytes():
                    data = leftover + chunk
                    usable = len(data) - len(data) % 4
                    leftover = data[usable:]
                    if usable:
                        samples = np.frombuffer(data[:usable], dtype="<f4")
                        yield AudioClip.from_samples(samples, self.sample_rate)
        except httpx.HTTPError as failure:
            raise SpeechServerError(f"{self.name} unreachable: {failure!r}") from failure

    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        return await _gather_clip(self.stream(text, voice_id=voice_id), self.sample_rate)

    async def voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id="female", label="LFM3 female", gender="female"),
            VoiceInfo(id="male", label="LFM3 male", gender="male"),
        ]


PHONON_VOICES: dict[str, tuple[str, str]] = {
    "Easton": ("male", "American"),
    "Delia": ("female", "American"),
    "Harper": ("female", "American"),
    "Hadley": ("female", "American"),
    "Everly": ("female", "American"),
    "Juniper": ("female", "American"),
    "Preston": ("male", "American"),
    "Miles": ("male", "American"),
    "Trevor": ("male", "American"),
    "Russell": ("male", "American"),
    "Micah": ("male", "American"),
    "Maeve": ("female", "British"),
    "Hugh": ("male", "British"),
    "Edwin": ("male", "British"),
}

PHONON_PERSONA_VOICES = {
    "default_female_1": "Juniper",  # chatty, polite, 62
    "default_female_2": "Everly",  # confused retiree, 78
    "default_female_3": "Hadley",  # soft spoken, 34
    "default_female_4": "Harper",  # rushed, 29
    "default_female_5": "Delia",  # skeptical bookkeeper, 58
    "default_male_1": "Preston",  # impatient owner, 47
    "default_male_2": "Russell",  # angry customer, 53
    "default_male_3": "Miles",  # sysadmin, 38
    "default_male_4": "Micah",  # student, 22
}


@register("tts", "phonon_tts")
class PhononTTS:
    """Phonon 31m on CPU. Genders above are read from names and catalog notes, not verified."""

    sample_rate = _TTS_RATE

    def __init__(
        self,
        base_url: str,
        *,
        name: str = "phonon-31m",
        voice_id: str = "Easton",
        voice_map: dict[str, str] | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.voice_id = voice_id
        self.voice_map = PHONON_PERSONA_VOICES | (voice_map or {})
        self._http = PerLoopClient(timeout=_timeout(timeout_s))

    async def stream(
        self, text: str, *, voice_id: str, style: str | None = None
    ) -> AsyncIterator[AudioClip]:
        body = {"text": text, "voice": self.voice_map.get(voice_id, voice_id)}
        try:
            async with self._http.get().stream(
                "POST", f"{self.base_url}/tts", json=body
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise SpeechServerError(
                        f"{self.name} answered {response.status_code}: {detail[:300]}"
                    )
                leftover = b""
                async for chunk in response.aiter_bytes():
                    data = leftover + chunk
                    usable = len(data) - len(data) % 2
                    leftover = data[usable:]
                    if usable:
                        yield AudioClip(data[:usable], self.sample_rate)
        except httpx.HTTPError as failure:
            raise SpeechServerError(f"{self.name} unreachable: {failure!r}") from failure

    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        return await _gather_clip(self.stream(text, voice_id=voice_id), self.sample_rate)

    async def voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id=voice, label=voice, gender=gender, accent=accent)
            for voice, (gender, accent) in PHONON_VOICES.items()
        ]


async def _gather_clip(chunks: AsyncIterator[AudioClip], rate: int) -> AudioClip:
    return AudioClip(b"".join([clip.pcm async for clip in chunks]), rate)


# License-safe presets only. marius and javert are CC0, and the rest are VCTK voices under
# CC-BY-4.0. cosette and jean are non-commercial and must stay out, and the alba, voice-zero, and
# unmute voices have unclear per-file terms.
POCKET_VOICES: dict[str, tuple[str, str]] = {
    "marius": ("male", "American"),
    "javert": ("male", "American"),
    "anna": ("female", "English"),
    "vera": ("female", "English"),
    "fantine": ("female", "English"),
    "charles": ("male", "English"),
    "paul": ("male", "English"),
    "eponine": ("female", "Scottish"),
    "azelma": ("female", "Canadian"),
    "george": ("male", "American"),
    "mary": ("female", "American"),
    "jane": ("female", "American"),
    "michael": ("male", "American"),
    "eve": ("female", "American"),
}

# The seeded personas name abstract voices. Each gets its own preset, picked to suit who they are.
POCKET_PERSONA_VOICES = {
    "default_female_1": "mary",  # chatty, polite, 62
    "default_female_2": "fantine",  # confused retiree, 78
    "default_female_3": "jane",  # soft spoken, 34
    "default_female_4": "eve",  # rushed, 29
    "default_female_5": "vera",  # skeptical bookkeeper, 58
    "default_male_1": "marius",  # impatient owner, 47
    "default_male_2": "george",  # angry customer, 53
    "default_male_3": "javert",  # sysadmin, 38
    "default_male_4": "michael",  # student, 22
}


@register("tts", "pocket_tts")
class PocketTTS:
    """`pocket-tts serve`, preset voices only. Cloning from a WAV needs gated weights.

    The model isn't thread-safe, and one server fills this role, so requests go one at a time.
    A voice outside the license-safe set fails the request rather than playing it.
    """

    sample_rate = _TTS_RATE

    def __init__(
        self,
        base_url: str,
        *,
        name: str = "pocket-tts",
        voice_id: str = "marius",
        voice_map: dict[str, str] | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.voice_id = voice_id
        self.voice_map = POCKET_PERSONA_VOICES | (voice_map or {})
        self._http = PerLoopClient(timeout=_timeout(timeout_s))
        self._one_at_a_time: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _preset(self, voice_id: str) -> str:
        preset = self.voice_map.get(voice_id, voice_id)
        if preset not in POCKET_VOICES:
            raise ValueError(f"{preset!r} is not a license-safe Pocket TTS voice")
        return preset

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._one_at_a_time is None or self._lock_loop is not loop:
            self._one_at_a_time = asyncio.Lock()
            self._lock_loop = loop
        return self._one_at_a_time

    async def stream(
        self, text: str, *, voice_id: str, style: str | None = None
    ) -> AsyncIterator[AudioClip]:
        form = {"text": text, "voice_url": self._preset(voice_id)}
        async with self._lock():
            try:
                async with self._http.get().stream(
                    "POST", f"{self.base_url}/tts", data=form
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", "replace")
                        raise SpeechServerError(
                            f"{self.name} answered {response.status_code}: {detail[:300]}"
                        )
                    reader = _WavStream()
                    async for chunk in response.aiter_bytes():
                        pcm = reader.feed(chunk)
                        if pcm:
                            yield AudioClip(pcm, reader.sample_rate)
            except httpx.HTTPError as failure:
                raise SpeechServerError(f"{self.name} unreachable: {failure!r}") from failure

    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        return await _gather_clip(self.stream(text, voice_id=voice_id), self.sample_rate)

    async def voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id=preset, label=preset.capitalize(), gender=gender, accent=accent)
            for preset, (gender, accent) in POCKET_VOICES.items()
        ]


class _WavStream:
    """Peels the header off a WAV arriving in pieces. Its lengths are placeholders."""

    def __init__(self) -> None:
        self.sample_rate = _TTS_RATE
        self._header = b""
        self._in_data = False
        self._odd = b""

    def feed(self, chunk: bytes) -> bytes:
        if not self._in_data:
            self._header += chunk
            start = self._data_start()
            if start is None:
                return b""
            self._in_data = True
            chunk = self._header[start:]
        data = self._odd + chunk
        usable = len(data) - len(data) % 2
        self._odd = data[usable:]
        return data[:usable]

    def _data_start(self) -> int | None:
        position = 12  # past RIFF, size, WAVE
        header = self._header
        while position + 8 <= len(header):
            kind = header[position : position + 4]
            size = struct.unpack("<I", header[position + 4 : position + 8])[0]
            if kind == b"data":
                return position + 8
            if kind == b"fmt " and position + 16 <= len(header):
                self.sample_rate = struct.unpack("<I", header[position + 12 : position + 16])[0]
            position += 8 + size + size % 2
        return None


_LISTEN_SYSTEM = (
    "Answer questions about the provided audio clip. Base every answer only on what can be heard."
)


@register("audio_analyst", "llama_audio_analyst")
class LlamaAudioAnalyst:
    """An audio model answering about a clip, over llama-server's chat endpoint. Advisory only."""

    def __init__(
        self, base_url: str, *, name: str = "llama-audio", timeout_s: float = 120.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self._http = PerLoopClient(timeout=_timeout(timeout_s))

    async def analyze(
        self, audio: AudioClip, prompt: str, *, json_schema: dict[str, Any] | None = None
    ) -> str:
        clip = base64.b64encode(audio.resampled(_STT_RATE).to_wav_bytes()).decode("ascii")
        body: dict[str, Any] = {
            "temperature": 0,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": _LISTEN_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": clip, "format": "wav"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": json_schema},
            }
        try:
            response = await self._http.get().post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as failure:
            raise SpeechServerError(f"{self.name} unreachable: {failure!r}") from failure
        reply = _checked(response, self.name)
        return reply["choices"][0]["message"].get("content") or ""
