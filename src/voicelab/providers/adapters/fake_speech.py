"""Text carried inside audio, so the fake STT really has to listen.

The fake TTS writes each character as one tone in the telephone band, with a syllable envelope and
a short gap between words. The fake STT reads tones back from whatever audio reaches it. Nothing
is shared between the two, so a cut, a dropped packet, or a noisy line changes what is heard the
way it would for real speech: a word cut in half is lost, and a word hit by packet loss can come
back garbled with lower confidence. Timing on the shared timeline works for free, since a whole
recorded channel decodes to words at their true positions.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np

from voicelab.core.audio import AudioClip

# One character takes 72 ms, about 14 characters a second, which is ordinary speech. It is that
# long because a lost 20 ms packet must leave most of a symbol intact.
SYMBOL_S = 0.072
GAP_S = SYMBOL_S / 2

_ALPHABET = string.ascii_letters + string.digits + string.punctuation
_FIRST_HZ = 430.0
_SPACING_HZ = 30.0  # wider than the 14 Hz a 72 ms window resolves, and inside the 3.4 kHz band
_TONES_HZ = _FIRST_HZ + _SPACING_HZ * np.arange(len(_ALPHABET))
_END_HZ = float(_TONES_HZ[-1] + 3 * _SPACING_HZ)  # closes a word, so a cut-off word is detectable
_ALL_HZ = np.append(_TONES_HZ, _END_HZ)
_END_TONE = len(_ALPHABET)
_INDEX = {character: index for index, character in enumerate(_ALPHABET)}
_UNKNOWN = _INDEX["?"]

_EDGE_S = 0.003
_SYLLABLE_HZ = 3.5
_ENVELOPE_FLOOR = 0.75  # dips stay well above the detector threshold, even on a noisy line
_FRAME_S = 0.004
_SMOOTH_FRAMES = 4
_TRAILING_S = 0.012
_FADE_TO = 0.3  # how far the closing tone falls, because real speech tails off


@dataclass(frozen=True)
class HeardWord:
    text: str
    start_s: float
    end_s: float
    confidence: float


def speech_duration_s(text: str) -> float:
    """How long `text` lasts when encoded. A word is its characters plus one for the gap."""
    return sum((len(word) + 1) * SYMBOL_S for word in text.split())


def encode_speech(text: str, sample_rate: int, *, level: float = 0.25) -> AudioClip:
    gap = round(GAP_S * sample_rate)
    pad = round(_TRAILING_S * sample_rate)
    pieces = []
    for word in text.split():
        # The gap leads, so a word's closing tone ends where its share of the time ends.
        pieces.append(np.zeros(gap, dtype=np.float32))
        pieces.append(_word_samples(word, sample_rate, level))
    if not pieces:
        return AudioClip(b"", sample_rate)
    # Line filters delay audio a few milliseconds and would otherwise clip the last word's end.
    pieces[0] = pieces[0][pad:]
    pieces.append(np.zeros(pad, dtype=np.float32))
    return AudioClip.from_samples(np.concatenate(pieces), sample_rate)


def _word_samples(word: str, rate: int, level: float) -> np.ndarray:
    symbol = round(SYMBOL_S * rate)
    tones = [float(_TONES_HZ[_INDEX.get(c, _UNKNOWN)]) for c in word]
    frequency = np.concatenate([np.repeat(tones, symbol), np.full(round(GAP_S * rate), _END_HZ)])
    # Phase runs on across symbols, since a jump at each boundary would splatter energy.
    wave = np.sin(2 * np.pi * np.cumsum(frequency) / rate)
    t = np.arange(len(wave)) / rate
    syllable = np.sin(np.pi * _SYLLABLE_HZ * t) ** 2
    envelope = _ENVELOPE_FLOOR + (1 - _ENVELOPE_FLOOR) * syllable
    closing = round(GAP_S * rate)
    envelope[-closing:] *= np.linspace(1.0, _FADE_TO, closing)
    edge = min(round(_EDGE_S * rate), len(wave) // 2)
    ramp = np.linspace(0.0, 1.0, edge, endpoint=False)
    envelope[:edge] *= ramp
    envelope[len(wave) - edge :] *= ramp[::-1]
    return (level * envelope * wave).astype(np.float32)


def decode_speech(clip: AudioClip) -> list[HeardWord]:
    """Read every complete word in a clip, with its position and how clean its tones were."""
    samples = clip.samples().astype(np.float64)
    rate = clip.sample_rate
    levels, threshold = _levels(samples, rate)
    words: list[HeardWord] = []
    if threshold is None:
        return words
    frame = max(1, round(_FRAME_S * rate))
    index = 0
    while True:
        rising = np.flatnonzero(levels[index:] > threshold)
        if len(rising) == 0:
            return words
        start = (index + int(rising[0])) * frame
        word, end = _read_word(samples, start, rate, threshold)
        if word is not None:
            words.append(HeardWord(word[0], start / rate, end / rate, word[1]))
        # The next word starts one gap later, or the level falls and this finds the next rise.
        index = max(index + 1, (end + round(GAP_S * rate)) // frame)


def _levels(samples: np.ndarray, rate: int) -> tuple[np.ndarray, float | None]:
    """Frame levels, and the level that means someone is speaking, or None for silence."""
    frame = max(1, round(_FRAME_S * rate))
    count = len(samples) // frame
    if count == 0:
        return np.zeros(0), None
    power = np.mean(np.square(samples[: count * frame].reshape(count, frame)), axis=1)
    # Smoothed over about one 60 Hz period, or mains hum would pulse in and out of the gaps.
    levels = np.sqrt(np.convolve(power, np.ones(_SMOOTH_FRAMES) / _SMOOTH_FRAMES, mode="same"))
    peak = float(np.percentile(levels, 99))
    if peak < 1e-3:
        return levels, None
    # A low percentile, because word gaps are under a tenth of a short clip and they set the floor.
    noise = float(np.percentile(levels, 2))
    # Halfway between floor and peak on a dB scale, and never so low that clean tails count.
    threshold = max(peak * 0.2, np.sqrt(max(noise, 1e-9) * peak))
    return levels, threshold if threshold < peak else None


def _read_word(
    samples: np.ndarray, start: int, rate: int, threshold: float
) -> tuple[tuple[str, float] | None, int]:
    """Read symbols at their fixed spacing until the closing tone. Returns the word and its end.

    Reading by position rather than by energy means a dip inside a word costs at most a letter.
    Only a closing tone that played to its end proves the word finished.
    """
    symbol = round(SYMBOL_S * rate)
    closing = round(GAP_S * rate)
    characters: list[str] = []
    purity: list[float] = []
    position = start
    while position + symbol / 2 <= len(samples):
        window = samples[position : position + symbol]
        if np.sqrt(np.mean(np.square(window))) < threshold / 2:
            break
        tone, clean = _strongest_tone(window, rate)
        if tone == _END_TONE:
            end = position + closing
            if characters and _closing_finished(samples[position:end], closing):
                return ("".join(characters), round(float(min(purity)), 3)), end
            return None, end
        characters.append(_ALPHABET[tone])
        purity.append(clean)
        position += symbol
    return None, position


def _closing_finished(window: np.ndarray, closing: int) -> bool:
    if len(window) < closing * 0.95:
        return False
    fifth = max(1, closing // 5)
    first = np.mean(np.square(window[:fifth]))
    last = np.mean(np.square(window[-fifth:]))
    return bool(last >= first * 0.04)  # the tone fades on purpose, but it doesn't vanish


def _strongest_tone(window: np.ndarray, rate: int) -> tuple[int, float]:
    size = round(SYMBOL_S * rate)
    magnitudes = np.abs(np.fft.rfft(window * np.hanning(len(window)), n=size))
    frequencies = np.fft.rfftfreq(size, 1 / rate)
    bins = np.round(np.interp(_ALL_HZ, frequencies, np.arange(len(frequencies)))).astype(int)
    power = np.square(magnitudes[bins])
    best = int(np.argmax(power))
    # Clean is the share of the energy that sits on the winning tone and the bins beside it.
    near = np.square(magnitudes[max(0, bins[best] - 2) : bins[best] + 3]).sum()
    return best, min(1.0, float(near / (np.square(magnitudes).sum() + 1e-12)))
