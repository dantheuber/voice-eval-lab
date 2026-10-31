"""STT and TTS backends for the end-to-end benchmarks. Each has the same small surface."""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import httpx
import numpy as np

from common import MODELS, RT, free_port, gpu_mem_gb, rss_gb

NEMO_DIR = RT / "nemo" / "nemo-speech-0.3.0-linux-x86_64-vulkan"
PARAKEET = MODELS / "nvidia__parakeet" / "parakeet-tdt-0.6b-v3.q8_0.gguf"
NEMOTRON = MODELS / "nvidia__nemotron-streaming" / "nemotron-speech-streaming-en-0.6b.q8_0.gguf"


def wav_bytes(pcm: np.ndarray, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


class NemoServer:
    """nemo-speech serve on an 18xxx port, Vulkan or CPU."""

    def __init__(self, model: Path = PARAKEET, device: str = "vulkan:0", extra: list[str] | None = None):
        self.model, self.device, self.extra = model, device, extra or []
        self.port = free_port(18300)
        self.url = f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{NEMO_DIR / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
        gpu0 = gpu_mem_gb()
        log = open(RT / "logs" / f"nemo-{self.port}.log", "w")
        self.proc = subprocess.Popen(
            [str(NEMO_DIR / "bin" / "nemo-speech"), "serve", "--no-ui", "--port", str(self.port), "--asr-model", str(self.model),
             "--device", self.device, *self.extra], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 120:
            if self.proc.poll() is not None:
                raise RuntimeError(f"nemo-speech exited {self.proc.returncode}, see {log.name}")
            for path in ("/healthz", "/health", "/v1/health"):
                try:
                    if httpx.get(self.url + path, timeout=1).status_code == 200:
                        self.load_s = time.perf_counter() - t0
                        self.gpu_delta_gb = gpu_mem_gb() - gpu0
                        return self
                except httpx.HTTPError:
                    pass
            time.sleep(0.2)
        raise TimeoutError("nemo-speech not healthy")

    def rss(self) -> float:
        return rss_gb(self.proc.pid)

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class HttpSTT:
    """Any OpenAI-compatible /v1/audio/transcriptions server."""

    def __init__(self, url: str, model: str = "x", extra: dict | None = None):
        self.url, self.model, self.extra = url, model, extra or {}
        self.client = httpx.Client(timeout=120)

    def transcribe(self, pcm: np.ndarray) -> str:
        r = self.client.post(self.url + "/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(pcm), "audio/wav")},
                             data={"model": self.model, "language": "en", **self.extra})
        r.raise_for_status()
        return r.json()["text"].strip()


class KokoroTTS:
    """Kokoro ONNX on CPU. No sub-sentence streaming: first audio is the full sentence."""
    name = "kokoro"
    rate = 24000

    def __init__(self, voice: str = "af_heart", threads: int | None = None):
        if threads:
            os.environ["ONNX_NUM_THREADS"] = str(threads)
        from kokoro_onnx import Kokoro
        self.k = Kokoro(str(MODELS / "kokoro" / "kokoro-v1.0.onnx"), str(MODELS / "kokoro" / "voices-v1.0.bin"))
        self.voice = voice
        self.lock = threading.Lock()

    def synth(self, text: str, on_first=None) -> np.ndarray:
        with self.lock:
            pcm, _ = self.k.create(text, voice=self.voice, speed=1.0, lang="en-us")
        if on_first:
            on_first(time.perf_counter())
        return (np.clip(pcm, -1, 1) * 32767).astype("<i2")

    def stream(self, text: str):
        yield self.synth(text)


class LfmTTS:
    """LFM3-Audio POST /tts, streamed PCM float32."""
    name = "lfm3"
    rate = 24000

    def stream(self, text: str):
        with self.client.stream("POST", self.url + "/tts", json={"input": text, "voice": self.voice, "lang": "en", "stream": True,
                                                                  "response_format": "pcm"}) as r:
            r.raise_for_status()
            left = b""
            for chunk in r.iter_bytes():
                data = left + chunk
                n = len(data) // 4 * 4
                left = data[n:]
                if n:
                    yield (np.clip(np.frombuffer(data[:n], dtype="<f4"), -1, 1) * 32767).astype("<i2")

    def __init__(self, url: str, voice: str = "female"):
        self.url, self.voice = url, voice
        self.client = httpx.Client(timeout=120)

    def synth(self, text: str, on_first=None) -> np.ndarray:
        buf = bytearray()
        with self.client.stream("POST", self.url + "/tts", json={"input": text, "voice": self.voice, "lang": "en", "stream": True,
                                                                  "response_format": "pcm"}) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                if chunk and not buf and on_first:
                    on_first(time.perf_counter())
                buf += chunk
        pcm = np.frombuffer(bytes(buf[: len(buf) // 4 * 4]), dtype="<f4")
        return (np.clip(pcm, -1, 1) * 32767).astype("<i2")


class _StreamTTS:
    """Shared synth() for engines that stream natively."""

    def synth(self, text: str, on_first=None) -> np.ndarray:
        parts = []
        for chunk in self.stream(text):
            if not parts and on_first:
                on_first(time.perf_counter())
            parts.append(chunk)
        return np.concatenate(parts) if parts else np.zeros(0, dtype="<i2")


class PhononTTS(_StreamTTS):
    """Phonon 31m through the ptts Rust runtime, on CPU."""
    name = "phonon"
    rate = 24000

    def __init__(self, voice: str = "Easton", quant: str = "q8", threads: int | None = None):
        import ptts
        if threads:
            ptts.set_num_threads(threads)
        self.tts = ptts.TTS("hf://gradium/phonon-31m-en", lang="en", quant=quant)
        self.voice = voice

    def stream(self, text: str):
        with self.tts.stream(text, voice=self.voice) as audio:
            self.rate = audio.sample_rate
            for pcm in audio:
                yield (np.clip(np.asarray(pcm), -1, 1) * 32767).astype("<i2")


class PocketTTS(_StreamTTS):
    """Pocket TTS on CPU through its Python package. Not thread-safe, so each side needs an instance."""
    name = "pocket"

    def __init__(self, language: str | None = "english_7m_2026-10", voice: str = "alba", threads: int | None = None):
        import torch
        from pocket_tts import TTSModel
        if threads:
            torch.set_num_threads(threads)
        self.model = TTSModel.load_model(language=language) if language else TTSModel.load_model()
        self.rate = self.model.sample_rate
        self.state = self.model.get_state_for_audio_prompt(voice)

    def stream(self, text: str):
        for chunk in self.model.generate_audio_stream(self.state, text):
            yield (np.clip(chunk.numpy(), -1, 1) * 32767).astype("<i2")


def make_tts(name: str, lfm_url: str | None = None, side: int = 0):
    """One TTS per call side. side picks a contrasting voice."""
    if name == "kokoro":
        return KokoroTTS(("af_heart", "am_michael")[side])
    if name == "lfm3":
        return LfmTTS(lfm_url, ("female", "male")[side])
    if name == "phonon":
        return PhononTTS(("Easton", "Ivy")[side] if False else "Easton")
    if name.startswith("pocket"):
        lang = {"pocket": "english_7m_2026-10", "pocket-2m": "english_2m_2026-10", "pocket-18m": "english_18m_2026-10"}[name]
        return PocketTTS(lang, ("alba", "marius")[side])
    raise SystemExit(f"unknown tts {name}")


def tmp_wav() -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return path
