"""Pitch tracking with YIN. Pure numpy, batched through the FFT, so it needs no model.

YIN looks for the lag at which a waveform repeats, not for energy at the fundamental. That matters
on a phone line, where the band starts at 300 Hz and removes the fundamental of most voices: the
harmonics that remain still repeat at the same period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOW_HZ = 70.0
HIGH_HZ = 400.0
HOP_S = 0.01
_THRESHOLD = 0.3  # highest normalized difference that still counts as periodic
_MARGIN = 0.08  # how far over the deepest dip a shorter lag can sit and still be the period
_MIN_RMS = 1e-4  # quieter frames have no pitch worth reading
_BATCH_FRAMES = 2048


def track_pitch(
    samples: np.ndarray, rate: int, *, low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ
) -> np.ndarray:
    """Fundamental frequency per `HOP_S`, NaN where the audio isn't periodic."""
    tau_min = max(2, int(rate / high_hz))
    tau_max = int(np.ceil(rate / low_hz))
    width = tau_max + tau_min  # the comparison window has to hold a whole longest period
    frame = width + tau_max
    hop = round(HOP_S * rate)
    if len(samples) < frame:
        return np.zeros(0)
    starts = np.arange(0, len(samples) - frame + 1, hop)
    pitch = np.full(len(starts), np.nan)
    view = np.lib.stride_tricks.sliding_window_view(samples.astype(np.float64), frame)
    for first in range(0, len(starts), _BATCH_FRAMES):
        batch = view[starts[first : first + _BATCH_FRAMES]]
        cmnd = _normalized_difference(batch, width, tau_max)
        loud = np.sqrt(np.mean(np.square(batch), axis=1)) >= _MIN_RMS
        for i in np.flatnonzero(loud):
            tau = _best_lag(cmnd[i], tau_min, tau_max)
            if tau is not None:
                pitch[first + i] = rate / tau
    return pitch


def _normalized_difference(frames: np.ndarray, width: int, tau_max: int) -> np.ndarray:
    """YIN's cumulative mean normalized difference, one row per frame, one column per lag."""
    size = 2 ** int(np.ceil(np.log2(frames.shape[1] + width)))
    whole = np.fft.rfft(frames, size, axis=1)
    head = np.fft.rfft(frames[:, :width], size, axis=1)
    correlation = np.fft.irfft(whole * np.conj(head), size, axis=1)[:, : tau_max + 1]
    energy = np.concatenate(
        [np.zeros((len(frames), 1)), np.cumsum(np.square(frames), axis=1)], axis=1
    )
    lags = np.arange(tau_max + 1)
    difference = energy[:, [width]] + energy[:, width + lags] - energy[:, lags] - 2 * correlation
    difference = np.maximum(difference, 0.0)
    running = np.cumsum(difference[:, 1:], axis=1)
    cmnd = np.ones_like(difference)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmnd[:, 1:] = difference[:, 1:] * lags[1:] / running
    return np.nan_to_num(cmnd, nan=1.0, posinf=1.0)


def _best_lag(cmnd: np.ndarray, tau_min: int, tau_max: int) -> float | None:
    """The period: the shortest lag about as deep as the deepest dip, refined to a fraction.

    Plain YIN takes the first dip under a threshold. On a phone line that goes wrong, because with
    the fundamental gone a strong third harmonic dips at a third of the period. Those false dips
    are shallower than the true one, while multiples of the period are just as deep, so "shortest
    lag near the deepest" lands on the period itself.
    """
    span = cmnd[tau_min : tau_max + 1]
    deepest = float(span.min())
    if deepest >= _THRESHOLD:
        return None
    tau = int(np.flatnonzero(span <= deepest + _MARGIN)[0]) + tau_min
    while tau < tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1
    if tau >= tau_max:
        return float(tau)
    left, middle, right = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
    curve = left - 2 * middle + right
    # Only a true minimum refines. A shoulder would throw the parabola's vertex out of range.
    offset = (left - right) / (2 * curve) if curve > 0 else 0.0
    return tau + offset if abs(offset) <= 1 else float(tau)


@dataclass(frozen=True)
class PitchStats:
    median_hz: float
    range_st: float  # p10 to p90
    std_st: float
    iqr_st: float  # p25 to p75, what flatness is judged on


def pitch_stats(pitch: np.ndarray) -> PitchStats | None:
    """How much pitch moves, in semitones around its median.

    Trackers sometimes lock onto a multiple of the period, more so without the fundamental. Whole
    octave errors fold back, but a stray frame at 1.5 times the period can't, and on a phone line
    about one frame in ten does that. IQR ignores them, and it's the measure that keeps its meaning
    through the line: over 21 clips, robotic speech read at most 0.25 st and natural speech at
    least 2.5 st, wideband or not, while the standard deviation of the two nearly met.
    """
    voiced = pitch[np.isfinite(pitch)]
    if len(voiced) == 0:
        return None
    middle = float(np.median(voiced))
    semitones = 12 * np.log2(voiced / middle)
    semitones = np.where(semitones > 9, semitones - 12, semitones)
    semitones = np.where(semitones < -9, semitones + 12, semitones)
    p10, p25, p75, p90 = np.percentile(semitones, [10, 25, 75, 90])
    return PitchStats(middle, float(p90 - p10), float(np.std(semitones)), float(p75 - p25))
