"""What a bot meant to say against how much of it actually went out.

Used on interruption: `SupportBot.on_interrupted` must get the words the listener heard, and those
come from counting audio sent, because TTS produces speech faster than it plays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_CHARACTERS_PER_S = 14.0  # for a sentence whose audio is still being made


def words_played(segments: list[tuple[str, float]], played_s: float) -> str:
    """The words that finished before a cut, from each segment's text and length.

    TTS gives no word timing, so a word's share of a segment follows its characters. A word still
    playing at the cut is left out, since the listener didn't get all of it.
    """
    kept: list[str] = []
    offset = 0.0
    for text, duration_s in segments:
        words = text.split()
        weights = [len(word) + 1 for word in words]
        per_weight = duration_s / sum(weights) if weights else 0.0
        position = offset
        for word, weight in zip(words, weights, strict=True):
            position += weight * per_weight
            if position > played_s:
                return " ".join(kept)
            kept.append(word)
        offset += duration_s
    return " ".join(kept)


@dataclass
class _Segment:
    text: str
    produced_s: float = 0.0
    finished: bool = False

    @property
    def duration_s(self) -> float:
        if self.finished:
            return self.produced_s
        return max(self.produced_s, len(self.text) / _CHARACTERS_PER_S)


@dataclass
class PlayoutLedger:
    """One bot turn: sentences given to TTS, the audio made for them, and the audio sent out."""

    segments: list[_Segment] = field(default_factory=list)
    played_s: float = 0.0

    def reset(self) -> None:
        self.segments = []
        self.played_s = 0.0

    def start_segment(self, text: str) -> _Segment:
        """Add a sentence. TTS keeps the result, since a cut can start a new turn mid-sentence."""
        segment = _Segment(text)
        self.segments.append(segment)
        return segment

    def add_played(self, seconds: float) -> None:
        self.played_s += seconds

    @property
    def intended_text(self) -> str:
        return " ".join(segment.text for segment in self.segments)

    @property
    def spoken_text(self) -> str:
        return words_played(
            [(segment.text, segment.duration_s) for segment in self.segments], self.played_s
        )
