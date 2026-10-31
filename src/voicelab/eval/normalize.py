"""Turns written and transcribed text into tokens that can be compared word by word.

A reference can be display text ("$49.99", "FH-4419") or the TTS form that `agents.speech` builds
("forty nine dollars and ninety nine cents", "F H dash 4 4 1 9"), and an ASR can return either.
All of them become the same tokens. Digits compare one at a time, so one wrong digit in a long
number is one substitution.

Amounts keep their decimal point as a token, because $5.50 and $550 are not the same money. A
symbol or unit word only says it's money: $5, 5.00, and "five dollars" all read as 5. A bare
"forty nine ninety nine" has no unit, and can't be told from a year, so it stays a plain number.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from voicelab.company.db import SEEDS_DIR

LEXICON_FILE = SEEDS_DIR / "lexicon.yaml"

_EDGE_PUNCTUATION = "\"'`.,!?;:()[]{}<>"
_SENTENCE_END = (".", "?", "!", ":")

_NUMERIC = re.compile(r"^\$?\d[\d,.\-:/]*(st|nd|rd|th|%)?$")
_NUMBER_WITH_UNIT = re.compile(r"^(\d+)([A-Za-z]{2,})$")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DECIMAL = re.compile(r"^\$?(\d[\d,]*)\.(\d+)%?$")
_ALPHANUMERIC_ID = re.compile(r"^[A-Za-z]{1,6}-?\d[\d-]*$")
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
_DOMAIN = re.compile(r"^(https?://)?([\w-]+\.)+([a-z]{2,})(/\S*)?$", re.IGNORECASE)
_ACRONYM = re.compile(r"^(?:[A-Z]{2,6}s?|(?:[A-Z]\.){2,})$")
_URL_PIECE = re.compile(r"[^\W_]+|[./@_]")

_URL_SPOKEN = {".": "dot", "@": "at", "/": "slash", "_": "underscore"}
_DEFAULT_TLDS = ("com", "net", "org", "example")
_MONTHS = (
    "January February March April May June July August September October November December"
).split()

# Said aloud in hyphenated codes and hostnames, and absent when an ASR writes them joined.
_DROPPED_WORDS = {"dash"}

_CONTRACTIONS = {
    "can't": ["can", "not"],
    "cannot": ["can", "not"],
    "won't": ["will", "not"],
    "i'm": ["i", "am"],
    "it's": ["it", "is"],
    "that's": ["that", "is"],
    "there's": ["there", "is"],
    "what's": ["what", "is"],
    "who's": ["who", "is"],
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
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "twentieth": 20, "thirtieth": 30,
}  # fmt: skip
_SCALES = {"hundred", "thousand"}
_NUMBER_WORDS = _ONES.keys() | _TEENS.keys() | _TENS.keys() | _SCALES
_MONEY_WORDS = {"dollar", "dollars", "buck", "bucks", "cent", "cents"}
_CENT_WORDS = {"cent", "cents"}
_NUMBER_UNITS = {"percent", "o'clock"}  # dropped after a number, since "5%" and "3:00" have none
_NUMBER_JOINERS = {"and", "point"}

# Ordinary speech as often as quantities, so alone they aren't critical: "one moment", "just a
# second", "oh, I see", "first, let me check".
_INCIDENTAL_NUMBERS = {"one", "oh", "first", "second"}

# Spoken forms containing these are everyday English, like "who is" for WHOIS. Folding them is
# harmless, but treating every "who is" as a critical token would fail calls on an ASR slip.
_COMMON_WORDS = {"a", "an", "are", "is", "of", "the", "to", "was"}


@dataclass(frozen=True)
class Token:
    norm: str  # the form that gets compared
    start: int  # first input word this came from
    end: int  # one past the last input word
    critical: str | None = None  # number, money, url, id, acronym, name, watch, lexicon


# Numbers -----------------------------------------------------------------------------------------


class _SpokenNumber:
    """Reads numbers as digits, in the groups people say them, with an optional fraction.

    "twenty twenty four" is 2024 and "two thousand twenty four" is the same number. A word that
    can't extend the current group starts a new one, and groups are written side by side, which
    also covers digit strings like "five five five oh one". After "dollars" or "point", digits go
    to the fraction, so "$5.50" and "five dollars and fifty cents" both come out as 5.50.
    """

    def __init__(self) -> None:
        self._digits = ""
        self._thousands = 0
        self._value: int | None = None
        self._fraction_at: int | None = None
        self._cents = False  # the fraction is cents, so five of them is .05 and not .50
        self.complete = False  # a finished amount: another number after it is a new number

    def add_word(self, word: str) -> None:
        if word == "thousand":
            self._thousands = (self._value or 1) * 1000
            self._value = None
        elif word == "hundred":
            self._value = (self._value or 1) * 100
        else:
            self._add(_TENS.get(word, _TEENS.get(word, _ONES.get(word))))

    def _add(self, value: int) -> None:
        if self._extends_group(value):
            self._value = (self._value or 0) + value
        else:
            self._close_group()
            self._value = value

    def _extends_group(self, value: int) -> bool:
        if self._value is None:
            return self._thousands > 0
        if value >= 10:  # needs both low digits free, as after "two hundred"
            return self._value >= 100 and self._value % 100 == 0
        return value > 0 and self._value >= 20 and self._value % 10 == 0

    def add_digits(self, digits: str) -> None:
        self._close_group()
        self._digits += digits

    def add_decimal(self, whole: str, fraction: str) -> None:
        self.add_digits(whole)
        self._fraction_at = len(self._digits)
        self._digits += fraction
        self.complete = True

    def start_fraction(self, *, cents: bool) -> None:
        """A "dollars" or "point" boundary: whatever comes next is after the decimal point."""
        self._close_group()
        if self._fraction_at is None:
            self._fraction_at = len(self._digits)
            self._cents = cents

    def end_cents(self) -> None:
        """A "cents" word. With no dollars before it, the amount is under a dollar."""
        self._close_group()
        if self._fraction_at is None:
            self._digits, self._fraction_at = f"0{self._digits}", 1
        self._cents = True
        self.complete = True

    def finish(self) -> str:
        self._close_group()
        if self._fraction_at is None:
            return self._digits
        whole, fraction = self._digits[: self._fraction_at], self._digits[self._fraction_at :]
        fraction = fraction.zfill(2) if self._cents else fraction.ljust(2, "0")
        return f"{whole}.{fraction}" if fraction.strip("0") else whole

    def _close_group(self) -> None:
        if self._thousands or self._value is not None:
            self._digits += str(self._thousands + (self._value or 0))
        self._thousands, self._value = 0, None


def _numeric_digits(word: str) -> str:
    if word.endswith(":00"):
        word = word[:-3]
    return re.sub(r"\D", "", word)


def _consume_number(words: Sequence[str], start: int) -> tuple[int, str, bool]:
    """Read a number starting at `start`. Returns its end, its digits, and whether it's money."""
    number = _SpokenNumber()
    money = False
    i = start
    while i < len(words):
        word = words[i]
        if number.complete and _starts_number(word):
            break
        decimal = _DECIMAL.match(word)
        if decimal:
            number.add_decimal(decimal.group(1).replace(",", ""), decimal.group(2))
            money = money or word.startswith("$")
        elif _NUMERIC.match(word):
            number.add_digits(_numeric_digits(word))
            if word.startswith("$"):
                money = True
                number.start_fraction(cents=True)
        elif word in _NUMBER_WORDS:
            number.add_word(word)
        elif word in _MONEY_WORDS and i > start:
            money = True
            if word in _CENT_WORDS:
                number.end_cents()
            else:
                number.start_fraction(cents=True)
        elif word in _NUMBER_UNITS and i > start:
            pass
        elif word in _NUMBER_JOINERS and _joins_number(words, i, number):
            if word == "point":
                number.start_fraction(cents=False)
        else:
            break
        i += 1
    return i, number.finish(), money


def _joins_number(words: Sequence[str], i: int, number: _SpokenNumber) -> bool:
    """Whether "and" or "point" continues this number. "$5 and $20" are two amounts."""
    if number.complete or i + 1 >= len(words):
        return False
    following = words[i + 1]
    return _starts_number(following) and not following.startswith("$")


def _starts_number(word: str) -> bool:
    return bool(_NUMERIC.match(word)) or word in _NUMBER_WORDS


# Tokenizing --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Piece:
    text: str  # lowercase
    original: str
    source: int  # index of the input word


def _is_url(text: str, tlds: Iterable[str]) -> bool:
    if _EMAIL.match(text):
        return True
    domain = _DOMAIN.match(text)
    # A typo like "done.Thanks" looks like a hostname, so the ending has to be a real one.
    return domain is not None and domain.group(3).lower() in tlds


def _split_word(word: str, tlds: Iterable[str]) -> list[str]:
    date = _ISO_DATE.match(word)
    if date:
        # Said as "November 6th, 2027", never in the order it's written.
        year, month, day = date.groups()
        return [_MONTHS[int(month) - 1], str(int(day)), year] if 1 <= int(month) <= 12 else [word]
    if _NUMERIC.match(word) or _is_url(word, tlds) or _ALPHANUMERIC_ID.match(word):
        return [word]
    unit = _NUMBER_WITH_UNIT.match(word)
    if unit:
        return list(unit.groups())
    return [part for part in word.split("-") if part]


def _pieces(words: Sequence[str], tlds: Iterable[str]) -> list[_Piece]:
    pieces = []
    for source, word in enumerate(words):
        stripped = word.strip(_EDGE_PUNCTUATION).replace("’", "'")
        if stripped:
            parts = _split_word(stripped, tlds)
            pieces.extend(_Piece(part.lower(), part, source) for part in parts)
    return pieces


def _expand_word(text: str) -> list[str]:
    """Plain word to its compared form, undoing contractions that ASRs and TTS text disagree on."""
    if text in _DROPPED_WORDS:
        return []
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


def _is_spelled_letter(piece: _Piece) -> bool:
    return len(piece.original) == 1 and piece.original.isalpha() and piece.original.isupper()


def _spelled_run_end(pieces: Sequence[_Piece], start: int) -> int:
    """End of a run of spelled letters, as the TTS form writes "S S L"."""
    end = start
    while end < len(pieces) and _is_spelled_letter(pieces[end]):
        end += 1
    return end


def _classified(
    pieces: Sequence[_Piece], words: Sequence[str], names: set[str], tlds
) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(pieces):
        piece = pieces[i]
        if _starts_number(piece.text):
            end, digits, money = _consume_number([p.text for p in pieces], i)
            incidental = end == i + 1 and piece.text in _INCIDENTAL_NUMBERS
            kind = "money" if money else None if incidental else "number"
            last = pieces[end - 1].source + 1
            tokens.extend(Token(d, piece.source, last, kind) for d in digits)
            i = end
            continue
        spelled_end = _spelled_run_end(pieces, i)
        if spelled_end - i >= 2:
            last = pieces[spelled_end - 1].source + 1
            tokens.extend(
                Token(p.text, piece.source, last, "acronym") for p in pieces[i:spelled_end]
            )
            i = spelled_end
            continue
        text = piece.text
        if _is_url(text, tlds):
            norms, kind = _url_tokens(text), "url"
        elif _ALPHANUMERIC_ID.match(text):
            norms, kind = _id_tokens(text), "id"
        elif _ACRONYM.match(piece.original):
            norms, kind = [text.replace(".", "").removesuffix("s")], "acronym"
        else:
            norms, kind = _expand_word(text), _name_kind(piece, words, names)
        tokens.extend(Token(n, piece.source, piece.source + 1, kind) for n in norms)
        i += 1
    return tokens


def _merge_span(tokens: list[Token], start: int, end: int, kind: str) -> None:
    """Make tokens one critical unit, reported once however many words it covers."""
    first, last = tokens[start].start, tokens[end - 1].end
    for i in range(start, end):
        tokens[i] = replace(tokens[i], start=first, end=last, critical=kind)


def _mark_spelled_ids(tokens: list[Token]) -> None:
    """Spelled letters then digits are one code: "F H 4 4 1 9"."""
    i = 0
    while i < len(tokens):
        if tokens[i].critical != "acronym":
            i += 1
            continue
        start = i
        while i < len(tokens) and tokens[i].critical == "acronym":
            i += 1
        digits_end = i
        while digits_end < len(tokens) and tokens[digits_end].critical == "number":
            digits_end += 1
        if digits_end > i:
            _merge_span(tokens, start, digits_end, "id")
            i = digits_end


def _mark_spoken_urls(tokens: list[Token], tlds: Iterable[str]) -> None:
    """The TTS form of a hostname has no dots to spot, so look for "dot" before a known ending."""
    for i in range(len(tokens) - 1):
        if tokens[i].norm != "dot" or tokens[i + 1].norm not in tlds:
            continue
        start = i - 1
        while start >= 2 and tokens[start - 1].norm == "dot":
            start -= 2
        if start >= 2 and tokens[start - 1].norm == "at":
            start -= 2
        if start >= 0:
            _merge_span(tokens, start, i + 2, "url")


def _phrase_spans(tokens: Sequence[Token], phrase: str) -> Iterator[tuple[int, int]]:
    """Where tokens join to spell `phrase`, so "fh4288" finds both "FH-4288" and "F H 4 2 8 8"."""
    for start in range(len(tokens)):
        joined = ""
        for end in range(start, len(tokens)):
            joined += tokens[end].norm
            if joined == phrase:
                yield start, end + 1
                break
            if not phrase.startswith(joined):
                break


def tokenize(
    words: Sequence[str],
    *,
    lexicon: Lexicon | None = None,
    known_names: Iterable[str] = (),
    watch: Iterable[str] = (),
) -> list[Token]:
    """Normalize words and mark the critical ones. `start`/`end` index back into `words`."""
    tlds = lexicon.tlds if lexicon is not None else _DEFAULT_TLDS
    names = {n.lower() for n in known_names}
    tokens = _classified(_pieces(words, tlds), words, names, tlds)
    if lexicon is not None:
        tokens = lexicon.fold(tokens)
    _mark_spelled_ids(tokens)
    _mark_spoken_urls(tokens, tlds)
    for phrase in watch:
        spelled = "".join(t.norm for t in tokenize(phrase.split()))
        for start, end in list(_phrase_spans(tokens, spelled)):
            _merge_span(tokens, start, end, "watch")
    return tokens


# Lexicon -----------------------------------------------------------------------------------------


class Lexicon:
    """Fernhill terms and how they're said, from `seeds/lexicon.yaml`.

    Spoken forms fold into the written term on both sides of a comparison, so "eye map" matches
    "IMAP". A missing file gives an empty lexicon, because a comparison still works without one.
    """

    def __init__(
        self,
        spoken_forms: dict[str, list[str]] | None = None,
        tlds: Sequence[str] = _DEFAULT_TLDS,
    ) -> None:
        self.tlds = tuple(tlds)
        self._by_first_word: dict[str, list[tuple[list[str], list[str], bool]]] = {}
        for term, forms in (spoken_forms or {}).items():
            canonical = [t.norm for t in tokenize(term.split())]
            for form in forms:
                variant = [t.norm for t in tokenize(form.split())]
                if variant and variant != canonical:
                    entry = (variant, canonical, _is_distinctive(variant))
                    self._by_first_word.setdefault(variant[0], []).append(entry)
        for entries in self._by_first_word.values():
            entries.sort(key=lambda entry: len(entry[0]), reverse=True)

    @classmethod
    def load(cls, path: Path | None = None) -> Lexicon:
        target = path or LEXICON_FILE
        if not target.exists():
            return cls()
        data = yaml.safe_load(target.read_text()) or {}
        forms: dict[str, list[str]] = {
            str(term): [str(spoken)] for term, spoken in (data.get("terms") or {}).items()
        }
        for unit, spoken in (data.get("units") or {}).items():
            forms.setdefault(str(unit), []).extend(str(form) for form in spoken)
        return cls(forms, data.get("tlds") or _DEFAULT_TLDS)

    def fold(self, tokens: list[Token]) -> list[Token]:
        out: list[Token] = []
        i = 0
        while i < len(tokens):
            match = self._match(tokens, i)
            if match is None:
                out.append(tokens[i])
                i += 1
                continue
            variant, canonical, distinctive = match
            last = tokens[i + len(variant) - 1]
            kind = "lexicon" if distinctive else tokens[i].critical
            out.extend(Token(norm, tokens[i].start, last.end, kind) for norm in canonical)
            i += len(variant)
        return out

    def _match(self, tokens: list[Token], i: int) -> tuple[list[str], list[str], bool] | None:
        for entry in self._by_first_word.get(tokens[i].norm, ()):
            variant = entry[0]
            if [t.norm for t in tokens[i : i + len(variant)]] == variant:
                return entry
        return None


def _is_distinctive(spoken: list[str]) -> bool:
    """Whether hearing this form means the term was meant, rather than ordinary English."""
    return len(spoken) >= 2 and not _COMMON_WORDS.intersection(spoken)


# Alignment ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Aligned:
    ref: Token
    hyp: tuple[Token, ...]  # what was heard here, empty if nothing was
    matched: bool


def align(ref: Sequence[Token], hyp: Sequence[Token]) -> list[Aligned]:
    """Line a reference up against a hypothesis, one entry per reference token, in order.

    Extra heard words are ignored, except inside a critical token: "$449.99" against "$49.99"
    matches every reference digit and is still the wrong amount.
    """
    matcher = SequenceMatcher(a=[t.norm for t in ref], b=[t.norm for t in hyp], autojunk=False)
    aligned: list[Aligned] = []
    insertions: list[tuple[int, Sequence[Token]]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            aligned.extend(
                Aligned(r, (h,), True) for r, h in zip(ref[i1:i2], hyp[j1:j2], strict=True)
            )
        elif tag == "insert":
            insertions.append((i1, hyp[j1:j2]))
        else:
            aligned.extend(_align_block(ref[i1:i2], hyp[j1:j2]))
    for position, extras in insertions:
        for neighbor in (position - 1, position):
            if 0 <= neighbor < len(aligned) and _heard_inside(aligned[neighbor], extras):
                item = aligned[neighbor]
                aligned[neighbor] = replace(item, hyp=(*item.hyp, *extras), matched=False)
    return aligned


def _heard_inside(item: Aligned, extras: Sequence[Token]) -> bool:
    """Whether extra heard tokens belong to the same heard word as this critical token."""
    if item.ref.critical is None or not item.hyp:
        return False
    return any(extra.start == item.hyp[0].start for extra in extras)


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
