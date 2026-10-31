"""Signal measurements of how speech was delivered. Pure numpy, so they run without a model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voicelab.eval.vad import Region

_BAND_HZ = (300.0, 3400.0)  # measured here on every call, so wideband and phone audio compare
_DIGITAL_SILENCE = 1e-10
_SNR_LIMITS_DB = (-10.0, 60.0)
# The 10th percentile of a noise bin's power sits at -ln(0.9) of its mean.
_NOISE_PERCENTILE = 10
_NOISE_BIAS = 1 / 0.10536
_MIN_NOISE_FRAMES = 8

_ENVELOPE_HOP_S = 0.005
_EDGE_HOP_S = 0.001
_EDGE_WINDOW_S = 0.012  # longer than a pitch period, so one glottal pulse can't look like an edge
_EDGE_REACH_S = 0.15
_EDGE_HOLD_S = 0.04


def rms_envelope(samples: np.ndarray, rate: int, window_s: float, hop_s: float) -> np.ndarray:
    """RMS level per hop, through a cumulative sum, so long windows cost the same as short."""
    window = max(1, round(window_s * rate))
    hop = max(1, round(hop_s * rate))
    if len(samples) < window:
        return np.zeros(0)
    energy = np.concatenate([[0.0], np.cumsum(np.square(samples.astype(np.float64)))])
    starts = np.arange(0, len(samples) - window + 1, hop)
    return np.sqrt(np.maximum(energy[starts + window] - energy[starts], 0.0) / window)


def estimate_snr_db(samples: np.ndarray, rate: int) -> float | None:
    """Speech to noise in the phone band, or None for too little audio.

    Noise is read per frequency bin from its quietest frames, the minimum statistics idea, because
    the bins between a voice's harmonics stay quiet even when no pause does. Digital silence is left
    out: it's the gap around an utterance, not the noise floor on it.
    """
    size = round(0.032 * rate)
    if len(samples) < size:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(samples.astype(np.float64), size)[
        :: size // 2
    ]
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(size), axis=1)) ** 2
    freqs = np.fft.rfftfreq(size, 1 / rate)
    in_band = spectrum[:, (freqs >= _BAND_HZ[0]) & (freqs <= _BAND_HZ[1])]
    totals = in_band.sum(axis=1)
    alive = totals > _DIGITAL_SILENCE
    if alive.sum() < _MIN_NOISE_FRAMES:
        return None
    noise = float(np.sum(np.percentile(in_band[alive], _NOISE_PERCENTILE, axis=0))) * _NOISE_BIAS
    loud = totals[alive]
    active = loud[loud >= np.median(loud)]
    signal = float(np.mean(active)) - noise
    if signal <= 0:
        return _SNR_LIMITS_DB[0]
    if noise <= 0:
        return _SNR_LIMITS_DB[1]
    return float(np.clip(10 * np.log10(signal / noise), *_SNR_LIMITS_DB))


@dataclass(frozen=True)
class Chop:
    start_s: float
    end_s: float


def find_chops(
    samples: np.ndarray, rate: int, *, min_s: float, max_s: float, cliff_db: float
) -> list[Chop]:
    """Short dropouts with a cliff on both sides: sound, nothing, and sound again.

    Natural closures can be just as short and deep, but level eases in or out on at least one side.
    Requiring `cliff_db` within 10 ms on both sides separated them cleanly on real speech: 0 of 660
    natural dips against about 200 gate edges. Noise fills a gap and hides its cliffs, so a chop
    under noise goes unseen here, and the noise is what gets reported instead.
    """
    level = 20 * np.log10(rms_envelope(samples, rate, _ENVELOPE_HOP_S, _ENVELOPE_HOP_S) + 1e-7)
    audible = level[level > -80]
    if len(audible) < 4:
        return []
    low = level < float(np.percentile(audible, 70)) - cliff_db / 2
    edges = np.flatnonzero(np.diff(np.concatenate([[0], low.astype(np.int8), [0]])))
    chops = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        length = (end - start) * _ENVELOPE_HOP_S
        if not min_s <= length <= max_s or start < 2 or end + 2 > len(level):
            continue
        fall = level[start - 2 : start].max() - level[start : start + 2].min()
        rise = level[end : end + 2].max() - level[end - 2 : end].min()
        if fall >= cliff_db and rise >= cliff_db:
            chops.append(Chop(float(start * _ENVELOPE_HOP_S), float(end * _ENVELOPE_HOP_S)))
    return chops


def edge_ms(samples: np.ndarray, rate: int, *, at_end: bool = False) -> float | None:
    """How fast speech arrives at this edge: milliseconds from 10% to 70% of its typical level.

    `samples` starts just before the speech, or for `at_end` finishes just after it. None means
    there's nothing to time: speech already running at the start of the audio has no edge, and
    speech that never reaches that level near the edge is a soft one.
    """
    signal = samples[::-1] if at_end else samples
    level = rms_envelope(signal, rate, _EDGE_WINDOW_S, _EDGE_HOP_S)
    if len(level) == 0:
        return None
    typical = float(np.percentile(level, 75))
    if typical <= 0:
        return None
    reach = min(len(level), round(_EDGE_REACH_S / _EDGE_HOP_S))
    arrived = np.flatnonzero(level[:reach] >= 0.7 * typical)
    if len(arrived) == 0:
        return None
    quiet = np.flatnonzero(level[: arrived[0]] <= 0.1 * typical)
    if len(quiet) == 0:
        return None
    # A plosive arrives this fast too, but its burst is over at once. A cut lands on held sound.
    held = level[arrived[0] : arrived[0] + round(_EDGE_HOLD_S / _EDGE_HOP_S)]
    if float(np.median(held)) < 0.5 * typical:
        return None
    return float((arrived[0] - quiet[-1]) * _EDGE_HOP_S * 1000)


def speech_time(regions: list[Region], pause_s: float) -> tuple[float, list[Region]]:
    """Time spent speaking and the pauses in it. Gaps under `pause_s` count as speaking."""
    if not regions:
        return 0.0, []
    pauses = [
        Region(a.end_s, b.start_s)
        for a, b in zip(regions, regions[1:], strict=False)
        if b.start_s - a.end_s >= pause_s
    ]
    total = regions[-1].end_s - regions[0].start_s - sum(p.duration_s for p in pauses)
    return total, pauses
