"""Tells the bot's own voice coming back through a speaker from a person talking over it.

Browser echo cancellation removes most of the echo, and what's left is enough to set off VAD and
make the bot interrupt itself. Echo is the outgoing audio again, a little later and quieter, so
its level follows the outgoing level's rise and fall. A person's speech doesn't. Comparing level
contours instead of waveforms survives the codec and the speaker, which change the waveform.

A contour takes 400 ms to fill, and VAD fires at 200 ms, so the first moments of each bot turn go
by level alone: echo is much quieter than the speech causing it, and a person cutting in isn't.
"""

from __future__ import annotations

from collections import deque

import numpy as np

STEP_S = 0.01
_WINDOW_STEPS = 40  # 400 ms of contour to compare
_MAX_DELAY_STEPS = 40  # speaker, room, and network delay up to 400 ms
_HOLD_STEPS = 30  # echo keeps arriving for a moment after the bot stops
_MIN_CORRELATION = 0.7
_SILENT_LEVEL = 1e-4
_ONSET_STEPS = 10  # the last 100 ms of each side, while there's too little contour to compare
_ONSET_RATIO = 0.6


class LevelContour:
    """Turns audio of any frame size into levels at a fixed step."""

    def __init__(self, keep_steps: int) -> None:
        self.levels: deque[float] = deque(maxlen=keep_steps)
        self._pending = np.zeros(0, dtype=np.float32)

    def add(self, samples: np.ndarray, rate: int) -> int:
        """Add audio and return how many new steps it completed."""
        step = max(1, round(STEP_S * rate))
        pending = np.concatenate([self._pending, samples])
        count = len(pending) // step
        for index in range(count):
            chunk = pending[index * step : (index + 1) * step]
            self.levels.append(float(np.sqrt(np.mean(np.square(chunk)))))
        self._pending = pending[count * step :]
        return count


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    scale = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / scale) if scale > 1e-12 else 0.0


class EchoDetector:
    """Feed it both directions. `is_echo` is true while the mic only repeats the bot."""

    def __init__(self, *, min_correlation: float = _MIN_CORRELATION) -> None:
        keep = _WINDOW_STEPS + _MAX_DELAY_STEPS
        self._mic = LevelContour(_WINDOW_STEPS)
        self._bot = LevelContour(keep)
        self._min_correlation = min_correlation
        self._quiet_steps = _HOLD_STEPS + 1  # the bot hasn't spoken yet
        self._speaking_steps = 0

    def add_outgoing(self, samples: np.ndarray, rate: int) -> None:
        new = self._bot.add(samples, rate)
        recent = list(self._bot.levels)[-new:] if new else []
        if any(level > _SILENT_LEVEL for level in recent):
            if self._quiet_steps > _HOLD_STEPS:
                self._speaking_steps = 0
            self._quiet_steps = 0
            self._speaking_steps += new
        else:
            self._quiet_steps += new

    def is_echo(self, samples: np.ndarray, rate: int) -> bool:
        """Check one mic frame. Quiet mic audio is never echo, since there's nothing to remove."""
        self._mic.add(samples, rate)
        if self._quiet_steps > _HOLD_STEPS or not self._mic.levels:
            return False
        mic = np.array(self._mic.levels)
        bot = np.array(self._bot.levels)
        if mic.max() < _SILENT_LEVEL or len(bot) == 0:
            return False
        if self._speaking_steps < _WINDOW_STEPS or len(mic) < _WINDOW_STEPS:
            loudest_bot = bot[-_ONSET_STEPS:].max()
            return bool(mic[-_ONSET_STEPS:].max() < loudest_bot * _ONSET_RATIO)
        latest_start = len(bot) - _WINDOW_STEPS
        for start in range(latest_start, max(-1, latest_start - _MAX_DELAY_STEPS), -1):
            window = bot[start : start + _WINDOW_STEPS]
            if _correlation(mic, window) >= self._min_correlation:
                return True
        return False
