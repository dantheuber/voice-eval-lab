"""Turns written and transcribed text into tokens that can be compared word by word.

TTS input says "$49.99" and an ASR may return "forty nine dollars and ninety nine cents". Both
become the digit tokens 4 9 9 9. Digits compare one at a time so a single wrong digit in a long
number is one substitution, and money words are dropped so "forty nine ninety nine" still matches.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

_EDGE_PUNCTUATION = "\"'`.,!?;:()[]{}<>"
_SENTENCE_END = (".", "?", "!", ":")

_NUMERIC = re.compile(r"^\$?\d[\d,.\-:/]*(st|nd|rd|th|%)?$")
_ALPHANUMERIC_ID = re.compile(r"^[A-Za-z]{1,6}-?\d[\d-]*$")
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
_DOMAIN = re.compile(r"^(https?://)?([\w-]+\.)+[a-z]{2,}(/\S*)?$", re.IGNORECASE)
_ACRONYM = re.compile(r"^(?:[A-Z]{2,6}s?|(?:[A-Z]\.){2,})$")
_URL_PIECE = re.compile(r"[^\W_]+|[./@_-]")

_URL_SPOKEN = {".": "dot", "@": "at", "/": "slash", "-": "dash", "_": "underscore"}

_CONTRACTIONS = {
    "can't": ["can", "not"],
    "cannot": ["can", "not"],
    "won't": ["will", "not"],
    "n't": ["not"],
    "i'm": ["i", "am"],
    "it's": ["it", "is"],
    "that's": ["that", "is"],
    "there's": ["there", "is"],
    "let's": ["let", "us"],
}
_CONTRACTION_SUFFIXES = {"'re": "are", "'ll": "will", "'ve": "have", "'d": "would"}

_ONES = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9,
}  # fmt: skip
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "tenth": 10, "eleventh": 11, "twelfth": 12,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "twentieth": 20, "thirtieth": 30,
}  # fmt: skip
_SCALES = {"hundred", "thousand"}
_NUMBER_WORDS = _ONES.keys() | _TEENS.keys() | _TENS.keys() | _SCALES
_MONEY_WORDS = {"dollar", "dollars", "buck", "bucks", "cent", "cents"}
_NUMBER_JOINERS = {"and", "point"}

# Ordinary speech as often as quantities, so alone they aren't critical: "one moment", "just a
# second", "oh, I see", "first, let me check".
_INCIDENTAL_NUMBERS = {"one", "oh", "first", "second"}


@dataclass(frozen=True)
class Token:
    norm: str  # the form that gets compared
    start: int  # first input word this came from
    end: int  # one past the last input word
    critical: str | None = None  # number, money, url, id, acronym, name, watch, lexicon


# Numbers -----------------------------------------------------------------------------------------


class _SpokenNumber:
    """Reads number words as digits, in the groups people say them.

    "twenty twenty four" is 2024 and "two thousand twenty four" is the same number. A word that
    can't extend the current group starts a new one, and groups are written side by side, which
    also covers digit strings like "five five five oh one".
    """

    def __init__(self) -> None:
        self.digits = ""
        self._thousands = 0
        self._value: int | None = None

    def add_word(self, word: str) -> None:
        if word == "thousand":
            self._thousands = (self._value or 1) * 1000
            self._value = None
        elif word == "hundred":
            self._value = (self._value or 1) * 100
        elif word in _TENS or word in _TEENS:
            value = _TENS.get(word, _TEENS.get(word))
            if self._value is not None and self._value >= 100 and self._value % 100 == 0:
                self._value += value
            else:
                self._close_group()
                self._value = value
        else:
            value = _ONES[word]
            if self._value is not None and self._value >= 20 and self._value % 10 == 0 and value:
                self._value += value
            else:
                self._close_group()
                self._value = value

    def add_digits(self, digits: str) -> None:
        self._close_group()
        self.digits += digits

    def finish(self) -> str:
        self._close_group()
        return self.digits

    def _close_group(self) -> None:
        if self._thousands or self._value is not None:
            self.digits += str(self._thousands + (self._value or 0))
        self._thousands, self._value = 0, None


def _numeric_digits(word: str) -> str:
    if word.startswith("$") and word.endswith(".00"):
        word = word[:-3]
    return re.sub(r"\D", "", word)


def _consume_number(words: Sequence[str], start: int) -> tuple[int, str, bool]:
    """Read a number starting at `start`. Returns its end, its digits, and whether it's money."""
    number = _SpokenNumber()
    money = False
    i = start
    while i < len(words):
        word = words[i]
        if _NUMERIC.match(word):
            number.add_digits(_numeric_digits(word))
            money = money or word.startswith("$")
        elif word in _NUMBER_WORDS:
            number.add_word(word)
        elif word in _MONEY_WORDS and i > start:
            money = True
        elif word in _NUMBER_JOINERS and i + 1 < len(words) and _starts_number(words[i + 1]):
            pass
        else:
            break
        i += 1
    return i, number.finish(), money


def _starts_number(word: str) -> bool:
    return bool(_NUMERIC.match(word)) or word in _NUMBER_WORDS


# Tokenizing --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Piece:
    text: str  # lowercase
    original: str
    source: int  # index of the input word


def _pieces(words: Sequence[str]) -> list[_Piece]:
    pieces = []
    for source, word in enumerate(words):
        stripped = word.strip(_EDGE_PUNCTUATION)
        if not stripped:
            continue
        keep_whole = (
            _NUMERIC.match(stripped) or _EMAIL.match(stripped) or _DOMAIN.match(stripped)
            or _ALPHANUMERIC_ID.match(stripped)
        )  # fmt: skip
        parts = [stripped] if keep_whole else [p for p in stripped.split("-") if p]
        pieces.extend(_Piece(part.lower(), part, source) for part in parts)
    return pieces


def _expand_word(text: str) -> list[str]:
    """Plain word to its compared form, undoing contractions that ASRs and TTS text disagree on."""
    text = text.replace("’", "'")
    if text in _CONTRACTIONS:
        return _CONTRACTIONS[text]
    if text.endswith("n't"):
        return [text[:-3], "not"]
    for suffix, word in _CONTRACTION_SUFFIXES.items():
        if text.endswith(suffix):
            return [text[: -len(suffix)], word]
    cleaned = re.sub(r"[^\w]", "", text)
    return [cleaned] if cleaned else []


def _url_tokens(text: str) -> list[str]:
    text = re.sub(r"^https?://", "", text)
    return [_URL_SPOKEN.get(piece, piece) for piece in _URL_PIECE.findall(text)]


def _id_tokens(text: str) -> list[str]:
    letters = re.match(r"^[a-z]+", text).group()
    return [letters, *re.sub(r"\D", "", text)]


def _is_sentence_start(words: Sequence[str], source: int) -> bool:
    return source == 0 or words[source - 1].rstrip("\"')").endswith(_SENTENCE_END)


def _name_kind(piece: _Piece, words: Sequence[str], known_names: set[str]) -> str | None:
    if piece.text in known_names:
        return "name"
    original = piece.original
    capitalized = len(original) > 1 and original[0].isupper() and original[1:].islower()
    return "name" if capitalized and not _is_sentence_start(words, piece.source) else None


def tokenize(
    words: Sequence[str],
    *,
    lexicon: Lexicon | None = None,
    known_names: Iterable[str] = (),
    watch: Iterable[str] = (),
) -> list[Token]:
    """Normalize words and mark the critical ones. `start`/`end` index back into `words`."""
    names = {n.lower() for n in known_names}
    pieces = _pieces(words)
    lowered = [p.text for p in pieces]
    tokens: list[Token] = []
    i = 0
    while i < len(pieces):
        piece = pieces[i]
        if _starts_number(piece.text):
            end, digits, money = _consume_number(lowered, i)
            incidental = end == i + 1 and piece.text in _INCIDENTAL_NUMBERS
            kind = "money" if money else None if incidental else "number"
            last = pieces[end - 1].source + 1
            tokens.extend(Token(d, piece.source, last, kind) for d in digits)
            i = end
            continue
        text = piece.text
        if _EMAIL.match(text) or _DOMAIN.match(text):
            norms, kind = _url_tokens(text), "url"
        elif _ALPHANUMERIC_ID.match(text):
            norms, kind = _id_tokens(text), "id"
        elif _ACRONYM.match(piece.original):
            norms, kind = [text.replace(".", "").removesuffix("s")], "acronym"
        else:
            norms, kind = _expand_word(text), _name_kind(piece, words, names)
        tokens.extend(Token(n, piece.source, piece.source + 1, kind) for n in norms)
        i += 1
    if lexicon is not None:
        tokens = lexicon.canonicalize(tokens)
    for phrase in watch:
        tokens = _mark_phrase(tokens, _plain_norms(phrase.split()), "watch")
    return tokens


def _plain_norms(words: Sequence[str]) -> list[str]:
    return [t.norm for t in tokenize(words)]


def _phrase_positions(tokens: Sequence[Token], phrase: Sequence[str]) -> Iterable[int]:
    size = len(phrase)
    for i in range(len(tokens) - size + 1):
        if size and all(tokens[i + k].norm == phrase[k] for k in range(size)):
            yield i


def _mark_phrase(tokens: list[Token], phrase: list[str], kind: str) -> list[Token]:
    marked = list(tokens)
    for i in _phrase_positions(tokens, phrase):
        for k in range(i, i + len(phrase)):
            marked[k] = replace(marked[k], critical=kind)
    return marked


# Lexicon -----------------------------------------------------------------------------------------


class Lexicon:
    """Domain terms and the ways they are said out loud, from `seeds/lexicon.yaml`.

    Spoken forms become the written form on both sides of a comparison, and listed terms count as
    critical. A missing file gives an empty lexicon, because a comparison still works without it.
    """

    def __init__(self, spoken_forms: dict[str, list[str]] | None = None) -> None:
        self._terms: list[tuple[list[str], list[list[str]]]] = []
        for written, forms in (spoken_forms or {}).items():
            canonical = _plain_norms(written.split())
            variants = [_plain_norms(form.split()) for form in forms]
            variants = [v for v in variants if v and v != canonical]
            self._terms.append((canonical, sorted(variants, key=len, reverse=True)))

    @classmethod
    def load(cls, path: Path | None = None) -> Lexicon:
        target = path or Path("seeds/lexicon.yaml")
        if not target.exists():
            return cls()
        return cls(_spoken_forms(yaml.safe_load(target.read_text())))

    def canonicalize(self, tokens: list[Token]) -> list[Token]:
        for canonical, variants in self._terms:
            for variant in variants:
                tokens = _replace_phrase(tokens, variant, canonical)
            tokens = _mark_phrase(tokens, canonical, "lexicon")
        return tokens


def _replace_phrase(tokens: list[Token], phrase: list[str], with_norms: list[str]) -> list[Token]:
    out: list[Token] = []
    i = 0
    while i < len(tokens):
        if tokens[i].norm == phrase[0] and [t.norm for t in tokens[i : i + len(phrase)]] == phrase:
            start, end = tokens[i].start, tokens[i + len(phrase) - 1].end
            out.extend(Token(norm, start, end, "lexicon") for norm in with_norms)
            i += len(phrase)
        else:
            out.append(tokens[i])
            i += 1
    return out


def _spoken_forms(data: Any) -> dict[str, list[str]]:
    """Accept the plausible lexicon shapes: a term map, or a list of term records, maybe nested."""
    if isinstance(data, dict):
        for key in ("terms", "entries", "lexicon", "words"):
            if key in data:
                return _spoken_forms(data[key])
        return {str(term): _forms(value) for term, value in data.items()}
    if isinstance(data, list):
        forms: dict[str, list[str]] = {}
        for entry in data:
            if isinstance(entry, str):
                forms[entry] = []
            elif isinstance(entry, dict):
                term = next(
                    (entry[k] for k in ("term", "written", "word", "text") if k in entry), None
                )
                if term is not None:
                    forms[str(term)] = _forms(entry)
        return forms
    return {}


def _forms(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict):
        for key in ("spoken", "spoken_forms", "say", "sounds_like", "forms", "aliases"):
            if key in value:
                return _forms(value[key])
    return []


# Alignment ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Aligned:
    ref: Token
    hyp: tuple[Token, ...]  # what was heard here, empty if nothing was
    matched: bool


def align(ref: Sequence[Token], hyp: Sequence[Token]) -> list[Aligned]:
    """Line a reference up against a hypothesis. Hypothesis-only words are ignored."""
    matcher = SequenceMatcher(a=[t.norm for t in ref], b=[t.norm for t in hyp], autojunk=False)
    aligned: list[Aligned] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            aligned.extend(
                Aligned(r, (h,), True) for r, h in zip(ref[i1:i2], hyp[j1:j2], strict=True)
            )
        elif tag != "insert":
            aligned.extend(_align_block(ref[i1:i2], hyp[j1:j2]))
    return aligned


def _align_block(ref: Sequence[Token], hyp: Sequence[Token]) -> list[Aligned]:
    """Inside a differing stretch, catch word splits such as "fernhill" and "fern hill"."""
    aligned: list[Aligned] = []
    i, j = 0, 0
    while i < len(ref):
        if j >= len(hyp):
            aligned.append(Aligned(ref[i], (), False))
            i += 1
            continue
        split = _joined_length(ref[i].norm, hyp[j:])
        merged = _joined_length(hyp[j].norm, ref[i:])
        if split:
            aligned.append(Aligned(ref[i], tuple(hyp[j : j + split]), True))
            i, j = i + 1, j + split
        elif merged:
            aligned.extend(Aligned(r, (hyp[j],), True) for r in ref[i : i + merged])
            i, j = i + merged, j + 1
        else:
            aligned.append(Aligned(ref[i], (hyp[j],), False))
            i, j = i + 1, j + 1
    return aligned


def _joined_length(target: str, tokens: Sequence[Token]) -> int:
    """How many tokens join to spell `target`, or 0. One token alone doesn't count."""
    joined = ""
    for count, token in enumerate(tokens, start=1):
        joined += token.norm
        if joined == target:
            return count if count > 1 else 0
        if not target.startswith(joined):
            return 0
    return 0
