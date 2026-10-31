"""The independent transcript: rendering it for judges and finding quotes in it."""

from __future__ import annotations

import re
from dataclasses import dataclass

from voicelab.core.models import Evidence, Speaker, Word
from voicelab.eval.vad import Region
from voicelab.providers.base import Transcript

_NOT_WORD = re.compile(r"[^\w]+")


def clean_word(text: str) -> str:
    """Lowercase with punctuation gone, so "Refund," and "refund" compare equal."""
    return _NOT_WORD.sub("", text.lower())


def estimate_words(text: str, regions: list[Region], fallback_end_s: float) -> list[Word]:
    """Spread text over a channel's speech when the STT gave no word timing.

    Providers are allowed to omit words, and everything downstream works in timed words. Words are
    laid out over the channel's total speech time and then mapped back onto its regions, so no
    estimate lands in silence or past the recording. The times are only good enough to find the
    right part of the call.
    """
    tokens = text.split()
    spans = regions or [Region(0.0, fallback_end_s)]
    total = sum(r.duration_s for r in spans)
    if not tokens or total <= 0:
        return []
    weights = [len(t) + 1 for t in tokens]
    per_weight = total / sum(weights)
    words: list[Word] = []
    position = 0.0
    for token, weight in zip(tokens, weights, strict=True):
        end = min(total, position + weight * per_weight)
        words.append(
            Word(
                text=token,
                start_s=_time_at(position, spans, ending=False),
                end_s=_time_at(end, spans, ending=True),
            )
        )
        position = end
    return words


def _time_at(position: float, spans: list[Region], *, ending: bool) -> float:
    """Recording time at `position` seconds into the speech, counting only time inside spans.

    A position on the seam between two spans is the end of one and the start of the next, so a
    word that finishes there stays in its own span.
    """
    passed = 0.0
    for span in spans:
        into_span = position - passed
        if into_span < span.duration_s or (ending and into_span <= span.duration_s):
            return span.start_s + max(0.0, into_span)
        passed += span.duration_s
    return spans[-1].end_s


def speech_chunks(
    regions: list[Region], *, gap_s: float, pad_s: float, end_s: float
) -> list[Region]:
    """Clips of a channel that hold its speech, padded, with the silence between them left out.

    Whisper fed a whole channel loops on silence, repeating a sentence dozens of times with
    invented timestamps. On 5 real calls that made 62% of caller tokens spurious, and heard 80% of
    what was said. One clip per turn gave 2% spurious and 99% heard, in about half the time.
    """
    chunks: list[Region] = []
    for region in regions:
        start, end = max(0.0, region.start_s - pad_s), min(end_s, region.end_s + pad_s)
        if chunks and start - chunks[-1].end_s < gap_s:
            chunks[-1] = Region(chunks[-1].start_s, end)
        else:
            chunks.append(Region(start, end))
    return chunks


def render_transcript(transcripts: dict[Speaker, Transcript], *, gap_s: float = 1.0) -> str:
    """Time-ordered speaker turns as text, for judge prompts.

    Turns are found on each channel alone and then put in order. Sorting words across channels
    would cut every overlap into one-word lines, and a judge can't read or quote those.
    """
    turns: list[tuple[float, Speaker, list[str]]] = []
    for speaker, transcript in transcripts.items():
        words: list[str] = []
        started, last_end = 0.0, 0.0
        for word in transcript.words:
            if words and word.start_s - last_end > gap_s:
                turns.append((started, speaker, words))
                words = []
            if not words:
                started = word.start_s
            words.append(word.text)
            last_end = word.end_s
        if words:
            turns.append((started, speaker, words))
    turns.sort(key=lambda turn: turn[0])
    return "\n".join(_turn_line(speaker, start, words) for start, speaker, words in turns)


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
        """Find a quote word for word, or None if it isn't in the transcript.

        Only case and punctuation are forgiven. A near match won't do, because the word a judge
        gets wrong can be the "not" that reverses the meaning. When both sides said it, the
        `prefer` speaker's copy is returned.
        """
        needle = [t for t in map(clean_word, quote.split()) if t]
        if not needle:
            return None
        for channel in sorted(self._channels, key=lambda c: c.speaker != prefer):
            span = _find_exact(channel.tokens, needle)
            if span is not None:
                return _evidence(channel, *span)
        return None


def _find_exact(tokens: list[str], needle: list[str]) -> tuple[int, int] | None:
    size = len(needle)
    for start in range(len(tokens) - size + 1):
        if tokens[start : start + size] == needle:
            return start, start + size
    return None


def _evidence(channel: _Channel, start: int, end: int) -> Evidence:
    # Evidence carries the transcript's own words, never the judge's copy of them.
    words = channel.words[start:end]
    return Evidence(
        quote=" ".join(w.text for w in words),
        speaker=channel.speaker,
        start_s=words[0].start_s,
        end_s=words[-1].end_s,
    )
