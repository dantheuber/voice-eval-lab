"""The independent transcript: rendering it for judges and finding quotes in it."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain

from voicelab.core.models import Evidence, Speaker, Word
from voicelab.eval.normalize import Lexicon, Token, phrase_spans, tokenize
from voicelab.eval.vad import Region
from voicelab.providers.base import Transcript

_ELLIPSIS = re.compile(r"\.{2,}|\u2026")


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
    """Time-ordered speaker turns as text, for judge prompts."""
    return "\n".join(line for _, line in timed_turns(transcripts, gap_s=gap_s))


def timed_turns(
    transcripts: dict[Speaker, Transcript],
    *,
    gap_s: float = 1.0,
    tool_times: Sequence[float] = (),
) -> list[tuple[float, str]]:
    """Each turn's start and its line.

    Turns are found on each channel alone and then put in order. Sorting words across channels
    would cut every overlap into one-word lines, and a judge can't read or quote those.

    A bot turn also breaks at each tool call in it, or words said after the tool would be stamped
    with the turn's start and read as coming before it. "One moment" running straight into the
    answer made real results look like claims made ahead of the tool.
    """
    turns: list[tuple[float, Speaker, list[str]]] = []
    for speaker, transcript in transcripts.items():
        words: list[str] = []
        started, last_start, last_end = 0.0, 0.0, 0.0
        breaks = tool_times if speaker == Speaker.BOT else ()
        for word in transcript.words:
            crossed = any(last_start < at <= word.start_s for at in breaks)
            if words and (word.start_s - last_end > gap_s or crossed):
                turns.append((started, speaker, words))
                words = []
            if not words:
                started = word.start_s
            words.append(word.text)
            last_start, last_end = word.start_s, word.end_s
        if words:
            turns.append((started, speaker, words))
    turns.sort(key=lambda turn: turn[0])
    return [(start, _turn_line(speaker, start, words)) for start, speaker, words in turns]


def _turn_line(speaker: Speaker, start_s: float, words: list[str]) -> str:
    return f"[{start_s:.1f}s] {speaker.value.upper()}: {' '.join(words)}"


@dataclass
class _Channel:
    speaker: Speaker
    words: list[Word]
    tokens: list[Token]


class TranscriptIndex:
    """Looks up quotes in the transcript and returns where they were spoken."""

    def __init__(
        self, transcripts: dict[Speaker, Transcript], lexicon: Lexicon | None = None
    ) -> None:
        self._lexicon = lexicon
        self._channels = [self._build(s, t) for s, t in transcripts.items()]

    def _build(self, speaker: Speaker, transcript: Transcript) -> _Channel:
        tokens = tokenize([w.text for w in transcript.words], lexicon=self._lexicon)
        return _Channel(speaker, transcript.words, tokens)

    def locate(self, quote: str, prefer: Speaker | None = None) -> Evidence | None:
        """Find a quote, or None if it isn't in the transcript.

        Both sides are reduced to the tokens pronunciation compares, so forms that say the same
        thing are the same: case, punctuation, "I'm" and "I am", "$49" and "forty nine dollars".
        What's left has to match token for token, so "not" and every digit count, though word
        breaks can differ, as in "O'Car for" and "O'Carfor". A judge's near miss is still a miss,
        because the word it gets wrong can be the one that reverses the meaning. An ellipsis joins
        parts, and each part has to be there. When both sides said it, the `prefer` speaker's copy
        is returned.
        """
        parts = [self.norms(part) for part in _ELLIPSIS.split(quote)]
        parts = [part for part in parts if part]
        if not parts:
            return None
        for channel in sorted(self._channels, key=lambda c: c.speaker != prefer):
            spans = [self._find(channel, part) for part in parts]
            if all(span is not None for span in spans):
                return _evidence(channel, *spans[0])
        return None

    def norms(self, text: str) -> list[str]:
        """The tokens a text comes down to, as a quote is compared."""
        return [token.norm for token in tokenize(text.split(), lexicon=self._lexicon)]

    def _find(self, channel: _Channel, needle: list[str]) -> tuple[int, int] | None:
        """The first place this quote stands whole.

        One word can become several tokens, as "can't" becomes "can not" and "$500" becomes three
        digits, so tokens alone would let "I can" match "I can't" and "$50" match "$500". A match
        has to start and end on word edges, and the words it covers have to read back as the quote.
        """
        tokens = channel.tokens
        size = len(needle)
        norms = [token.norm for token in tokens]
        token_runs = (
            (start, start + size)
            for start in range(len(norms) - size + 1)
            if norms[start : start + size] == needle
        )
        # An ASR and a judge can split a name differently, and every letter still has to be there.
        for start, end in chain(token_runs, phrase_spans(tokens, "".join(needle))):
            if _on_word_edges(tokens, start, end) and self._reads_as(channel, start, end, needle):
                return start, end
        return None

    def _reads_as(self, channel: _Channel, start: int, end: int, needle: list[str]) -> bool:
        words = channel.words[channel.tokens[start].start : channel.tokens[end - 1].end]
        tokens = tokenize([word.text for word in words], lexicon=self._lexicon)
        return "".join(token.norm for token in tokens) == "".join(needle)


def _on_word_edges(tokens: list[Token], start: int, end: int) -> bool:
    starts_inside = start > 0 and tokens[start - 1].end > tokens[start].start
    ends_inside = end < len(tokens) and tokens[end].start < tokens[end - 1].end
    return not (starts_inside or ends_inside)


def _evidence(channel: _Channel, start: int, end: int) -> Evidence:
    # Evidence carries the transcript's own words, never the judge's copy of them.
    words = channel.words[channel.tokens[start].start : channel.tokens[end - 1].end]
    return Evidence(
        quote=" ".join(w.text for w in words),
        speaker=channel.speaker,
        start_s=words[0].start_s,
        end_s=words[-1].end_s,
    )
