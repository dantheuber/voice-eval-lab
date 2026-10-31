"""Tells the bot's own voice coming back through a speaker from a person talking over it.

Browser echo cancellation removes most of the echo, and what's left is enough to set off VAD and
make the bot interrupt itself. Echo is the outgoing audio again, a little later and quieter, so
its energy moves through the frequency bands as the outgoing audio's did. A person's speech
doesn't. Comparing band contours instead of waveforms survives the codec and the speaker, which
change the waveform. Eight bands separate far better than one overall level: on real speech with
room reverb, a person's correlation reached 0.78 at p95 against 0.90 for level alone, and echo
stayed above that on 98 to 100% of steps against 52 to 74%.

VAD fires 200 ms into anything, sooner than a 400 ms contour of speech can fill. So both contours
run on one timeline with silence in them: silence then a rise is already a shape to compare, and
a person who was talking before the bot started doesn't fit it. Nothing here compares how loud
the two sides are, since a person quieter than the bot isn't echo.

A person who starts within 400 ms of a bot rise can match it for a moment. That costs them the
first moments of speech, and the contours part as soon as there is more to compare.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

STEP_S = 0.01
_BANDS = 8
_LOW_HZ = 250.0
_HIGH_HZ = 3800.0  # inside what an 8 kHz phone line carries
_WINDOW_STEPS = 40  # 400 ms of contour to compare
_MAX_DELAY_STEPS = 40  # speaker, room, and network delay up to 400 ms
_HOLD_STEPS = 30  # echo keeps arriving for a moment after the bot stops
_MIN_CORRELATION = 0.8
_SILENT_LEVEL = 1e-4
_OUTPUT_JITTER_STEPS = 6  # outgoing audio can run this far behind the mic without being a gap


@lru_cache
def _analysis(size: int, rate: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """A window scaled so noise of level x reads as x in every band, and each band's FFT bins."""
    window = np.hanning(size)
    freqs = np.fft.rfftfreq(size, 1 / rate)
    edges = np.geomspace(_LOW_HZ, _HIGH_HZ, _BANDS + 1)
    bands = [
        np.flatnonzero((freqs >= low) & (freqs < high))
        for low, high in zip(edges, edges[1:], strict=False)
    ]
    return window / np.sqrt(np.sum(window**2)), bands


class BandContour:
    """Turns audio of any frame size into band levels at a fixed step, on overlapping frames."""

    def __init__(self, keep_steps: int) -> None:
        silence = np.zeros(_BANDS)
        self.levels: deque[np.ndarray] = deque([silence] * keep_steps, maxlen=keep_steps)
        self._pending = np.zeros(0, dtype=np.float32)

    def add(self, samples: np.ndarray, rate: int) -> int:
        """Add audio and return how many new steps it completed."""
        step = max(1, round(STEP_S * rate))
        pending = np.concatenate([self._pending, samples])
        count = max(0, (len(pending) - step) // step)
        window, bands = _analysis(2 * step, rate)
        for index in range(count):
            frame = pending[index * step : (index + 2) * step]
            power = np.abs(np.fft.rfft(frame * window)) ** 2
            self.levels.append(np.array([np.sqrt(power[band].mean()) for band in bands]))
        self._pending = pending[count * step :]
        return count


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    scale = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / scale) if scale > 1e-12 else 0.0


class EchoDetector:
    """Feed it both directions. `is_echo` is true while the mic only repeats the bot.

    Mic audio arrives without gaps and outgoing audio doesn't, since nothing is sent between
    replies. So the mic is the clock, and the time the output said nothing counts as silence.
    """

    def __init__(self, *, min_correlation: float = _MIN_CORRELATION) -> None:
        self._mic = BandContour(_WINDOW_STEPS)
        self._bot = BandContour(_WINDOW_STEPS + _MAX_DELAY_STEPS)
        self._min_correlation = min_correlation
        self._mic_steps = 0
        self._bot_steps = 0
        self._bot_last_spoke = -_HOLD_STEPS - 1  # on the mic's clock

    def add_outgoing(self, samples: np.ndarray, rate: int) -> None:
        new = self._bot.add(samples, rate)
        self._bot_steps += new
        if new and np.max(list(self._bot.levels)[-new:]) > _SILENT_LEVEL:
            self._bot_last_spoke = self._mic_steps

    def _fill_output_gap(self) -> None:
        gap = self._mic_steps - self._bot_steps
        if gap > _OUTPUT_JITTER_STEPS:
            self._bot.levels.extend([np.zeros(_BANDS)] * gap)
            self._bot_steps += gap

    def is_echo(self, samples: np.ndarray, rate: int) -> bool:
        """Check one mic frame. Quiet mic audio is never echo, since there's nothing to remove."""
        self._mic_steps += self._mic.add(samples, rate)
        self._fill_output_gap()
        if self._mic_steps - self._bot_last_spoke > _HOLD_STEPS:
            return False
        mic = np.array(self._mic.levels)
        bot = np.array(self._bot.levels)
        if mic.max() < _SILENT_LEVEL:
            return False
        latest_start = len(bot) - _WINDOW_STEPS
        for start in range(latest_start, latest_start - _MAX_DELAY_STEPS, -1):
            window = bot[start : start + _WINDOW_STEPS]
            if _correlation(mic, window) >= self._min_correlation:
                return True
        return False
