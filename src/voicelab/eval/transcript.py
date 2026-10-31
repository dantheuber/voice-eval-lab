"""The independent transcript: rendering it for judges and finding quotes in it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from voicelab.core.models import Evidence, Speaker, Word
from voicelab.eval.vad import Region
from voicelab.providers.base import Transcript

_NOT_WORD = re.compile(r"[^\w]+")

# Longer quotes may differ by a word or two when a judge fixes up a stumble, and short ones can't:
# one changed word in four is a different statement.
_NEAR_MATCH_MIN_WORDS = 5
_NEAR_MATCH_RATIO = 0.9


def clean_word(text: str) -> str:
    """Lowercase with punctuation gone, so "Refund," and "refund" compare equal."""
    return _NOT_WORD.sub("", text.lower())


def estimate_words(text: str, regions: list[Region], fallback_end_s: float) -> list[Word]:
    """Spread text over a channel's speech when the STT gave no word timing.

    Providers are allowed to omit words, and everything downstream works in timed words. These
    times are only good enough to find the right part of the call.
    """
    tokens = text.split()
    spans = regions or [Region(0.0, fallback_end_s)]
    total = sum(r.duration_s for r in spans)
    if not tokens or total <= 0:
        return []
    weights = [len(t) + 1 for t in tokens]
    per_weight = total / sum(weights)
    words: list[Word] = []
    span_index, cursor = 0, spans[0].start_s
    for token, weight in zip(tokens, weights, strict=True):
        length = weight * per_weight
        while cursor + length > spans[span_index].end_s + 1e-9 and span_index + 1 < len(spans):
            span_index += 1
            cursor = spans[span_index].start_s
        words.append(Word(text=token, start_s=cursor, end_s=cursor + length))
        cursor += length
    return words


def render_transcript(transcripts: dict[Speaker, Transcript], *, gap_s: float = 1.0) -> str:
    """Time-ordered speaker turns as text, for judge prompts."""
    timed = [(word, speaker) for speaker, t in transcripts.items() for word in t.words]
    timed.sort(key=lambda item: item[0].start_s)
    lines: list[str] = []
    current: Speaker | None = None
    words: list[str] = []
    started, last_end = 0.0, 0.0
    for word, speaker in timed:
        if speaker != current or word.start_s - last_end > gap_s:
            if words:
                lines.append(_turn_line(current, started, words))
            current, words, started = speaker, [], word.start_s
        words.append(word.text)
        last_end = word.end_s
    if words:
        lines.append(_turn_line(current, started, words))
    return "\n".join(lines)


def _turn_line(speaker: Speaker, start_s: float, words: list[str]) -> str:
    return f"[{start_s:.1f}s] {speaker.value.upper()}: {' '.join(words)}"


@dataclass
class _Channel:
    speaker: Speaker
    words: list[Word]
    tokens: list[str]


class TranscriptIndex:
    """Looks up quotes in the transcript and returns where they were spoken."""

    def __init__(self, transcripts: dict[Speaker, Transcript]) -> None:
        self._channels = [self._build(s, t) for s, t in transcripts.items()]

    @staticmethod
    def _build(speaker: Speaker, transcript: Transcript) -> _Channel:
        kept = [w for w in transcript.words if clean_word(w.text)]
        return _Channel(speaker, kept, [clean_word(w.text) for w in kept])

    def locate(self, quote: str, prefer: Speaker | None = None) -> Evidence | None:
        """Find a quote, or None if it isn't in the transcript."""
        needle = [t for t in map(clean_word, quote.split()) if t]
        if not needle:
            return None
        channels = sorted(self._channels, key=lambda c: c.speaker != prefer)
        for channel in channels:
            span = _find_exact(channel.tokens, needle)
            if span is not None:
                return _evidence(channel, *span)
        if len(needle) >= _NEAR_MATCH_MIN_WORDS:
            for channel in channels:
                span = _find_near(channel.tokens, needle)
                if span is not None:
                    return _evidence(channel, *span)
        return None


def _find_exact(tokens: list[str], needle: list[str]) -> tuple[int, int] | None:
    size = len(needle)
    for start in range(len(tokens) - size + 1):
        if tokens[start : start + size] == needle:
            return start, start + size
    return None


def _find_near(tokens: list[str], needle: list[str]) -> tuple[int, int] | None:
    best, best_span = _NEAR_MATCH_RATIO, None
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)
    for size in (len(needle) - 1, len(needle), len(needle) + 1):
        for start in range(len(tokens) - size + 1):
            matcher.set_seq1(tokens[start : start + size])
            ratio = matcher.ratio()
            if ratio >= best:
                best, best_span = ratio, (start, start + size)
    return best_span


def _evidence(channel: _Channel, start: int, end: int) -> Evidence:
    # Evidence carries the transcript's own words, never the judge's copy of them.
    words = channel.words[start:end]
    return Evidence(
        quote=" ".join(w.text for w in words),
        speaker=channel.speaker,
        start_s=words[0].start_s,
        end_s=words[-1].end_s,
    )
