"""Energy-based speech detection on one channel. Pure numpy, no model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voicelab.core.audio import AudioClip

FRAME_S = 0.01
_ABSOLUTE_FLOOR_DB = -50.0
_ABOVE_NOISE_DB = 10.0
_BELOW_PEAK_DB = 30.0


@dataclass(frozen=True)
class Region:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def frame_levels_db(samples: np.ndarray, frame: int) -> np.ndarray:
    """RMS level in dBFS of consecutive frames. A partial last frame is dropped."""
    count = len(samples) // frame
    if count == 0:
        return np.zeros(0)
    frames = samples[: count * frame].reshape(count, frame)
    return 10 * np.log10(np.mean(np.square(frames), axis=1) + 1e-12)


def speech_regions(
    clip: AudioClip, *, min_speech_s: float = 0.1, min_gap_s: float = 0.2
) -> list[Region]:
    """Find speech as stretches above a threshold set from this channel's own levels.

    The threshold sits above the noise floor and within reach of the peak, so it follows a noisy
    line instead of assuming studio silence. A channel with no modulation, such as steady noise
    with no speech on it, has its threshold above its peak and returns nothing.
    """
    frame = round(FRAME_S * clip.sample_rate)
    levels = frame_levels_db(clip.samples(), frame)
    if len(levels) == 0:
        return []
    floor = float(np.percentile(levels, 10))
    peak = float(np.percentile(levels, 99.5))
    threshold = max(_ABSOLUTE_FLOOR_DB, floor + _ABOVE_NOISE_DB, peak - _BELOW_PEAK_DB)
    if threshold >= peak:
        return []
    return _runs(levels > threshold, min_speech_s, min_gap_s)


def _runs(active: np.ndarray, min_speech_s: float, min_gap_s: float) -> list[Region]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(np.int8), [0]])))
    merged: list[list[float]] = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        start_s, end_s = float(start) * FRAME_S, float(end) * FRAME_S
        if merged and start_s - merged[-1][1] < min_gap_s:
            merged[-1][1] = end_s
        else:
            merged.append([start_s, end_s])
    return [Region(s, e) for s, e in merged if e - s >= min_speech_s]
