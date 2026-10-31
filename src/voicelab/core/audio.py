"""Audio value type and conversions. All internal audio is mono signed 16-bit PCM."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


@dataclass(frozen=True)
class AudioClip:
    """Mono 16-bit little-endian PCM at a known sample rate."""

    pcm: bytes
    sample_rate: int

    @property
    def n_samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate

    def samples(self) -> np.ndarray:
        """Float32 samples in [-1, 1]."""
        return np.frombuffer(self.pcm, dtype=np.int16).astype(np.float32) / 32768.0

    @classmethod
    def from_samples(cls, samples: np.ndarray, sample_rate: int) -> AudioClip:
        clipped = np.clip(samples, -1.0, 1.0)
        return cls((clipped * 32767.0).astype(np.int16).tobytes(), sample_rate)

    @classmethod
    def silence(cls, duration_s: float, sample_rate: int) -> AudioClip:
        return cls(b"\x00\x00" * round(duration_s * sample_rate), sample_rate)

    @classmethod
    def from_file(cls, path: str | Path) -> AudioClip:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return cls.from_samples(data.mean(axis=1), rate)

    @classmethod
    def from_wav_bytes(cls, wav: bytes) -> AudioClip:
        data, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        return cls.from_samples(data.mean(axis=1), rate)

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, self.samples(), self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def write_wav(self, path: str | Path) -> None:
        sf.write(str(path), self.samples(), self.sample_rate, format="WAV", subtype="PCM_16")

    def resampled(self, sample_rate: int) -> AudioClip:
        if sample_rate == self.sample_rate:
            return self
        out = soxr.resample(self.samples(), self.sample_rate, sample_rate, quality="HQ")
        return AudioClip.from_samples(out, sample_rate)

    def slice(self, start_s: float, end_s: float) -> AudioClip:
        start = min(self.n_samples, max(0, round(start_s * self.sample_rate)))
        end = min(self.n_samples, max(start, round(end_s * self.sample_rate)))
        return AudioClip(self.pcm[start * 2 : end * 2], self.sample_rate)

    def __add__(self, other: AudioClip) -> AudioClip:
        if other.sample_rate != self.sample_rate:
            other = other.resampled(self.sample_rate)
        return AudioClip(self.pcm + other.pcm, self.sample_rate)


def write_stereo_wav(path: str | Path, left: AudioClip, right: AudioClip) -> None:
    """Write two mono clips as one stereo file, padding the shorter channel with silence."""
    if right.sample_rate != left.sample_rate:
        right = right.resampled(left.sample_rate)
    a, b = left.samples(), right.samples()
    n = max(len(a), len(b))
    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[: len(a), 0] = a
    stereo[: len(b), 1] = b
    sf.write(str(path), stereo, left.sample_rate, format="WAV", subtype="PCM_16")


def read_stereo_wav(path: str | Path) -> tuple[AudioClip, AudioClip]:
    """Read a stereo file as (left, right). A mono file comes back on both channels."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    left = AudioClip.from_samples(data[:, 0], rate)
    right = AudioClip.from_samples(data[:, 1] if data.shape[1] > 1 else data[:, 0], rate)
    return left, right
