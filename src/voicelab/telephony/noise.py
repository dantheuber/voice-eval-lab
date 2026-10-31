"""Procedural background noise, so the line needs no noise dataset.

Each kind is rendered once into a long loop with unit RMS. The line then reads from the loop, which
makes streaming cheap and the target SNR exact: scaling a unit-RMS signal needs no level estimate.
Everything is built in the frequency domain, so the loop is circular and has no seam.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

LOOP_SECONDS = 30.0
HUM_HZ = 60.0


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def _unit(signal: np.ndarray) -> np.ndarray:
    rms = _rms(signal)
    return signal / rms if rms > 0 else signal


def _shaped(
    rng: np.random.Generator, n: int, rate: int, gains: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """White noise with its spectrum multiplied by `gains(frequencies)`."""
    freqs = np.fft.rfftfreq(n, 1 / rate)
    return _unit(np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * gains(freqs), n))


def _colored(rng: np.random.Generator, n: int, rate: int, exponent: float) -> np.ndarray:
    """Noise with power falling as 1/f^exponent: 1 is pink, 2 is brown."""
    return _shaped(rng, n, rate, lambda f: np.maximum(f, 20.0) ** (-exponent / 2))


def _band(rng: np.random.Generator, n: int, rate: int, low: float, high: float) -> np.ndarray:
    return _shaped(rng, n, rate, lambda f: ((f >= low) & (f <= high)).astype(float))


def _envelope(rng: np.random.Generator, n: int, rate: int, hz: float, floor: float) -> np.ndarray:
    """A slow random gain in [floor, 1], for traffic swells and syllable rhythm."""
    slow = _shaped(rng, n, rate, lambda f: ((f > 0) & (f <= hz)).astype(float))
    span = slow.max() - slow.min()
    return floor + (1 - floor) * (slow - slow.min()) / span


def _clicks(
    rng: np.random.Generator, n: int, rate: int, per_second: float, decay_ms: float
) -> np.ndarray:
    """Short decaying transients: keystrokes, cups on saucers."""
    impulses = np.zeros(n)
    count = max(1, round(per_second * n / rate))
    impulses[rng.integers(0, n, count)] = rng.uniform(0.3, 1.0, count) * rng.choice([-1, 1], count)
    tail = np.exp(-np.arange(n) / (decay_ms * rate / 1000))
    return _unit(np.fft.irfft(np.fft.rfft(impulses) * np.fft.rfft(tail), n))


def _babble(rng: np.random.Generator, n: int, rate: int, voices: int) -> np.ndarray:
    """Overlapping unintelligible talkers: speech-band noise with syllable-rate gating."""
    top = min(3000.0, rate * 0.45)
    mix = np.zeros(n)
    for _ in range(voices):
        mix += _band(rng, n, rate, 200.0, top) * _envelope(rng, n, rate, 4.0, 0.02) ** 2
    return _unit(mix)


def _office(rng: np.random.Generator, n: int, rate: int) -> np.ndarray:
    return (
        _colored(rng, n, rate, 1.0)
        + 0.5 * _babble(rng, n, rate, 3)
        + 0.4 * _clicks(rng, n, rate, 2.0, 4.0)
    )


def _street(rng: np.random.Generator, n: int, rate: int) -> np.ndarray:
    traffic = _colored(rng, n, rate, 2.0) * _envelope(rng, n, rate, 0.2, 0.3)
    return traffic + 0.4 * _colored(rng, n, rate, 1.0)


def _cafe(rng: np.random.Generator, n: int, rate: int) -> np.ndarray:
    return _babble(rng, n, rate, 8) + 0.3 * _clicks(rng, n, rate, 1.0, 12.0)


def _car(rng: np.random.Generator, n: int, rate: int) -> np.ndarray:
    road = _colored(rng, n, rate, 2.0)
    wind = _colored(rng, n, rate, 1.0) * _envelope(rng, n, rate, 0.5, 0.7)
    return road + 0.5 * wind


def _tv(rng: np.random.Generator, n: int, rate: int) -> np.ndarray:
    """One voice in another room: a pitched buzz through vowel-like resonances."""
    f0 = 120.0 + 30.0 * (_envelope(rng, n, rate, 1.0, 0.0) - 0.5)
    phase = np.cumsum(f0) / rate
    # A whole number of cycles over the loop keeps the wrap free of a click.
    phase *= round(phase[-1]) / phase[-1]
    harmonics = np.arange(1, int(min(3200.0, rate * 0.45) / 150.0) + 1)
    buzz = np.sum(np.sin(2 * np.pi * np.outer(harmonics, phase)) / harmonics[:, None], axis=0)
    freqs = np.fft.rfftfreq(n, 1 / rate)
    formants = sum(np.exp(-(((freqs - hz) / 200.0) ** 2)) for hz in (500.0, 1500.0, 2500.0))
    voice = np.fft.irfft(np.fft.rfft(buzz) * formants, n)
    return _unit(voice) * _envelope(rng, n, rate, 4.0, 0.0) ** 2 + 0.1 * _colored(rng, n, rate, 1.0)


def _hum(rng: np.random.Generator, n: int, rate: int) -> np.ndarray:
    """Mains hum. The upper harmonics matter because the phone band removes 60 Hz itself."""
    t = np.arange(n) / rate
    top = int(min(1500.0, rate * 0.45) // HUM_HZ)
    return sum(np.sin(2 * np.pi * HUM_HZ * k * t) / np.sqrt(k) for k in range(1, top + 1))


_KINDS: dict[str, Callable[[np.random.Generator, int, int], np.ndarray]] = {
    "office": _office,
    "street": _street,
    "cafe": _cafe,
    "car": _car,
    "tv": _tv,
    "hum": _hum,
}


def make_noise(
    kind: str,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    band: tuple[float, float] | None = None,
    seconds: float = LOOP_SECONDS,
) -> np.ndarray:
    """Render a loop of `kind` with RMS 1.0 after limiting it to `band`."""
    try:
        generate = _KINDS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown noise kind {kind!r}, known: {sorted(_KINDS)}") from exc
    n = round(seconds * sample_rate)
    signal = generate(rng, n, sample_rate)
    if band is not None:
        freqs = np.fft.rfftfreq(n, 1 / sample_rate)
        in_band = (freqs >= band[0]) & (freqs <= band[1])
        signal = np.fft.irfft(np.fft.rfft(signal) * in_band, n)
    return _unit(signal).astype(np.float32)
