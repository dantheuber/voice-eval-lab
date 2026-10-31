"""Speech normalization: turn text written for the eye into text a TTS voice says correctly.

A TTS model reading "FH-4419" or "$49.99" raw will guess, and a wrong guess on an invoice number
or an amount is a real support failure. Every token is rewritten in one regex pass, so the output
of one rule is never picked up by a later rule. Digits stay as spaced numerals because TTS reads
those one at a time, which is how a person reads out an invoice number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from voicelab.company.db import SEEDS_DIR

LEXICON_FILE = SEEDS_DIR / "lexicon.yaml"


@dataclass(frozen=True)
class SpeechText:
    """One piece of bot speech. `spoken` goes to TTS, `display` goes in logs and transcripts."""

    display: str
    spoken: str


# Number words ----------------------------------------------------------------------------------

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred"))
_IRREGULAR_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}


def number_to_words(n: int) -> str:
    """Spoken English without hyphens or "and": 1234 is one thousand two hundred thirty four."""
    if n < 0:
        return f"minus {number_to_words(-n)}"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    for size, name in _SCALES:
        if n >= size:
            count, rest = divmod(n, size)
            head = f"{number_to_words(count)} {name}"
            return f"{head} {number_to_words(rest)}" if rest else head
    raise AssertionError(f"unreachable for {n}")


def ordinal_to_words(n: int) -> str:
    *head, last = number_to_words(n).split()
    if last in _IRREGULAR_ORDINALS:
        last = _IRREGULAR_ORDINALS[last]
    elif last.endswith("y"):
        last = f"{last[:-1]}ieth"
    else:
        last = f"{last}th"
    return " ".join([*head, last])


def year_to_words(year: int) -> str:
    century, rest = divmod(year, 100)
    if rest == 0:
        return "two thousand" if year == 2000 else f"{number_to_words(century)} hundred"
    if 2000 < year < 2010:
        return number_to_words(year)
    if rest < 10:
        return f"{number_to_words(century)} oh {number_to_words(rest)}"
    return f"{number_to_words(century)} {number_to_words(rest)}"


def spell_digits(digits: str) -> str:
    return " ".join(digits)


def _decimal_words(text: str) -> str:
    """Read "1,234" as a number and "1.5" as one point five."""
    whole, _, fraction = text.replace(",", "").partition(".")
    words = number_to_words(int(whole))
    if fraction:
        spoken = " ".join(_ONES[int(digit)] for digit in fraction)
        words = f"{words} point {spoken}"
    return words


# Lexicon ---------------------------------------------------------------------------------------

_MONTHS = (
    "January February March April May June July August September October November December"
).split()
_MONTH_NUMBERS = {name[:3]: number for number, name in enumerate(_MONTHS, start=1)}
_MONTH_PATTERN = "|".join([*_MONTHS, "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sept"]) + (
    "|Sep|Oct|Nov|Dec"
)

# After these words a number is a code to read digit by digit, not a quantity.
_DIGIT_CONTEXT = re.compile(
    r"(?:\bports?|\bpin|\bzip(?: code)?|\bpostal code|\bext(?:ension)?\.?|\bending(?: in)?"
    r"|\blast (?:four|4)(?: digits)?|\binvoice|\bticket|\bcode|\bnumber|\baccount|\bcard|#)"
    r"(?: number| is| was|:|#)*\s*(?:\d+(?:\s*,\s*|\s+(?:and|or)\s+))*$",
    re.IGNORECASE,
)


def _token_pattern(terms: list[str], units: list[str], tlds: list[str]) -> re.Pattern[str]:
    """Build the single pattern for every rewritable token. Earlier alternatives win a tie."""
    term_alternatives = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    host = (
        r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+"
        rf"(?i:{'|'.join(tlds)})(?![\w-])"
    )
    alternatives = [
        r"(?P<email>[A-Za-z0-9._+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})",
        r"(?P<url>https?://[^\s,;)\"']*[^\s,;)\"'.?!])",
        r"(?P<ip>(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w.]))",
        rf"(?P<host>(?<![\w@.-]){host}(?:/[\w./-]*\w)?)",
        r"(?P<code>(?<![\w-])(?:[A-Z]{1,5}|fh|tk)-?\d{2,}(?![\w-]))",
        r"(?P<money>-?\$\d[\d,]*(?:\.\d+)?(?:/(?:mo|month|yr|year)\b)?)",
        r"(?P<iso_date>(?<![\d-])\d{4}-\d{2}-\d{2}(?![\d-]))",
        r"(?P<slash_date>(?<![\d/])\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})(?![\d/]))",
        rf"(?P<named_date>\b(?:{_MONTH_PATTERN})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b"
        r"(?:,?\s+\d{4}\b)?)",
        r"(?P<phone>(?<![\w.$-])(?:(?:\+?1[ .-]?)?(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}"
        r"|\d{3}-\d{4})(?![\w-]))",
        rf"(?P<size>(?<![\w.])\d[\d,]*(?:\.\d+)? ?(?i:{'|'.join(units)})\b)",
        r"(?P<clock_ampm>(?<![\w:.])\d{1,2}(?::\d{2})? ?[AaPp]\.?[Mm]\b\.?)",
        r"(?P<clock>(?<![\w:.])\d{1,2}:\d{2}(?![\w:]))",
        r"(?P<always>(?<![\w/])24/7(?![\w/]))",
        r"(?P<percent>(?<![\w.])\d[\d,]*(?:\.\d+)?%)",
        r"(?P<ordinal>(?<![\w.])\d+(?:st|nd|rd|th)\b)",
        r"(?P<number>(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]))",
        rf"(?P<term>(?<![\w-])(?i:{term_alternatives})(?![\w-]))",
        r"(?P<amp>(?<=\s)&(?=\s))",
    ]
    return re.compile("|".join(alternatives))


@dataclass(frozen=True)
class Lexicon:
    """Spoken forms for terms and units. Shared with the pronunciation evaluator."""

    terms: dict[str, str]
    units: dict[str, tuple[str, str]]  # symbol -> (singular, plural)
    tlds: tuple[str, ...]
    _by_lower: dict[str, str] = field(init=False, repr=False)
    _pattern: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lowered = {term.lower(): spoken for term, spoken in self.terms.items()}
        object.__setattr__(self, "_by_lower", lowered)
        pattern = _token_pattern(list(self.terms), list(self.units), list(self.tlds))
        object.__setattr__(self, "_pattern", pattern)

    def spoken_term(self, term: str) -> str | None:
        return self._by_lower.get(term.lower())


@lru_cache
def load_lexicon(path: Path = LEXICON_FILE) -> Lexicon:
    data = yaml.safe_load(Path(path).read_text())
    return Lexicon(
        terms={str(term): str(spoken) for term, spoken in data["terms"].items()},
        units={symbol: (forms[0], forms[1]) for symbol, forms in data["units"].items()},
        tlds=tuple(data["tlds"]),
    )


# Token rules. Each takes a regex match and returns the words to say ----------------------------


def _speak_label(label: str, lexicon: Lexicon) -> str:
    """Say one piece of a hostname or email address."""
    known = lexicon.spoken_term(label)
    if known:
        return known
    pieces = re.findall(r"\d+|[^\d]+", label)
    return " ".join(spell_digits(p) if p.isdigit() else lexicon.spoken_term(p) or p for p in pieces)


def _speak_host(host: str, lexicon: Lexicon) -> str:
    def speak_part(part: str) -> str:
        return " dash ".join(_speak_label(piece, lexicon) for piece in part.split("-"))

    return " dot ".join(speak_part(part) for part in host.split("."))


def _speak_path(path: str, lexicon: Lexicon) -> str:
    parts = [part for part in path.split("/") if part]
    return "".join(f" slash {_speak_host(part, lexicon)}" for part in parts)


def _speak_email(match: re.Match[str], lexicon: Lexicon) -> str:
    local, _, host = match.group("email").partition("@")
    separators = {".": "dot", "_": "underscore", "-": "dash", "+": "plus"}
    pieces = re.split(r"([._+-])", local)
    spoken = [separators.get(piece) or _speak_label(piece, lexicon) for piece in pieces if piece]
    return f"{' '.join(spoken)} at {_speak_host(host, lexicon)}"


def _speak_url(match: re.Match[str], lexicon: Lexicon) -> str:
    # Nobody reads "https colon slash slash" over the phone, and a query string is unsayable.
    address = re.sub(r"^https?://", "", match.group("url")).split("?")[0].split("#")[0]
    host, _, path = address.partition("/")
    return _speak_host(host, lexicon) + _speak_path(path, lexicon)


def _speak_ip(match: re.Match[str], lexicon: Lexicon) -> str:
    return " dot ".join(spell_digits(part) for part in match.group("ip").split("."))


def _speak_hostname(match: re.Match[str], lexicon: Lexicon) -> str:
    host, _, path = match.group("host").partition("/")
    return _speak_host(host, lexicon) + _speak_path(path, lexicon)


def _speak_code(match: re.Match[str], lexicon: Lexicon) -> str:
    letters, digits = re.fullmatch(r"([A-Za-z]+)-?(\d+)", match.group("code")).groups()
    dash = " dash " if "-" in match.group("code") else " "
    return f"{spell_digits(letters.upper())}{dash}{spell_digits(digits)}"


def _speak_money(match: re.Match[str], lexicon: Lexicon) -> str:
    token = match.group("money")
    negative = token.startswith("-")
    amount, _, period = token.lstrip("-$").partition("/")
    value = Decimal(amount.replace(",", "")).quantize(Decimal("0.01"), ROUND_HALF_UP)
    dollars, cents = divmod(int(value * 100), 100)
    parts = []
    if dollars or not cents:
        parts.append(f"{number_to_words(dollars)} dollar{'' if dollars == 1 else 's'}")
    if cents:
        parts.append(f"{number_to_words(cents)} cent{'' if cents == 1 else 's'}")
    spoken = " and ".join(parts)
    if negative:
        spoken = f"minus {spoken}"
    if period:
        spoken += " a month" if period.startswith("mo") else " a year"
    return spoken


def _speak_date(month: int, day: int, year: int | None, fallback: str) -> str:
    try:
        date(year or 2000, month, day)
    except ValueError:
        return spell_digits(re.sub(r"\D", "", fallback))
    spoken = f"{_MONTHS[month - 1]} {ordinal_to_words(day)}"
    return f"{spoken}, {year_to_words(year)}" if year else spoken


def _speak_iso_date(match: re.Match[str], lexicon: Lexicon) -> str:
    token = match.group("iso_date")
    year, month, day = (int(part) for part in token.split("-"))
    return _speak_date(month, day, year, token)


def _speak_slash_date(match: re.Match[str], lexicon: Lexicon) -> str:
    token = match.group("slash_date")
    month, day, year = (int(part) for part in token.split("/"))
    return _speak_date(month, day, year + 2000 if year < 100 else year, token)


def _speak_named_date(match: re.Match[str], lexicon: Lexicon) -> str:
    token = match.group("named_date")
    pattern = r"([A-Za-z]+)\.?\s+(\d+)\w*(?:,?\s+(\d{4}))?"
    month_name, day, year = re.match(pattern, token).groups()
    month = _MONTH_NUMBERS[month_name[:3]]
    return _speak_date(month, int(day), int(year) if year else None, token)


def _speak_phone(match: re.Match[str], lexicon: Lexicon) -> str:
    # Commas make TTS pause between the groups, the way people say a phone number.
    groups = re.findall(r"\d+", match.group("phone"))
    return ", ".join(spell_digits(group) for group in groups)


def _speak_size(match: re.Match[str], lexicon: Lexicon) -> str:
    amount, symbol = re.fullmatch(r"([\d,.]+) ?(\w+)", match.group("size")).groups()
    singular, plural = lexicon.units[symbol.upper()]
    return f"{_decimal_words(amount)} {singular if amount == '1' else plural}"


def _clock_words(hour: str, minute: str | None) -> str:
    spoken = number_to_words(int(hour))
    if minute and minute != "00":
        tail = number_to_words(int(minute))
        spoken += f" oh {tail}" if minute.startswith("0") else f" {tail}"
    return spoken


def _speak_clock_ampm(match: re.Match[str], lexicon: Lexicon) -> str:
    hour, minute, meridiem = re.match(
        r"(\d+)(?::(\d+))? ?([AaPp])", match.group("clock_ampm")
    ).groups()
    return f"{_clock_words(hour, minute)} {meridiem.upper()} M"


def _speak_clock(match: re.Match[str], lexicon: Lexicon) -> str:
    hour, minute = match.group("clock").split(":")
    if minute == "00":
        return f"{number_to_words(int(hour))} o'clock"
    return _clock_words(hour, minute)


def _speak_always(match: re.Match[str], lexicon: Lexicon) -> str:
    return "twenty four seven"


def _speak_percent(match: re.Match[str], lexicon: Lexicon) -> str:
    return f"{_decimal_words(match.group('percent').rstrip('%'))} percent"


def _speak_ordinal(match: re.Match[str], lexicon: Lexicon) -> str:
    return ordinal_to_words(int(re.sub(r"\D", "", match.group("ordinal"))))


def _speak_number(match: re.Match[str], lexicon: Lexicon) -> str:
    token = match.group("number")
    before = match.string[: match.start()]
    if "," not in token and "." not in token:
        if _DIGIT_CONTEXT.search(before) or token.startswith("0") and len(token) > 1:
            return spell_digits(token)
        if len(token) == 10:
            return ", ".join(spell_digits(part) for part in (token[:3], token[3:6], token[6:]))
        if len(token) >= 5:  # zip codes and other codes, never a quantity on a support call
            return spell_digits(token)
        if len(token) == 4 and 1900 <= int(token) < 2100:
            return year_to_words(int(token))
    return _decimal_words(token)


def _speak_term(match: re.Match[str], lexicon: Lexicon) -> str:
    return lexicon.spoken_term(match.group("term"))


def _speak_amp(match: re.Match[str], lexicon: Lexicon) -> str:
    return "and"


_SPEAK = {
    "email": _speak_email,
    "url": _speak_url,
    "ip": _speak_ip,
    "host": _speak_hostname,
    "code": _speak_code,
    "money": _speak_money,
    "iso_date": _speak_iso_date,
    "slash_date": _speak_slash_date,
    "named_date": _speak_named_date,
    "phone": _speak_phone,
    "size": _speak_size,
    "clock_ampm": _speak_clock_ampm,
    "clock": _speak_clock,
    "always": _speak_always,
    "percent": _speak_percent,
    "ordinal": _speak_ordinal,
    "number": _speak_number,
    "term": _speak_term,
    "amp": _speak_amp,
}

# Small models write markdown and emoji despite the prompt, and TTS would read them out.
_MARKUP = (
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"^\s*#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"^\s*(?:[-*•]|\d+\.)\s+", re.MULTILINE), ""),
    (re.compile(r"[*`]+"), ""),
    (re.compile("[\U0001f300-\U0001faff☀-➿]"), ""),
)


def unicode_safe(value: Any) -> Any:
    """Replace lone surrogates in model output. JSON can carry them, but UTF-8 can't encode them.

    Left in, one would fail every later request and the save of the call artifact.
    """
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {unicode_safe(key): unicode_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unicode_safe(item) for item in value]
    return value


def clean_markup(text: str) -> str:
    for pattern, replacement in _MARKUP:
        text = pattern.sub(replacement, text)
    return text.strip()


def normalize(text: str, lexicon: Lexicon | None = None) -> SpeechText:
    """Return the cleaned display text and the form to send to TTS."""
    lexicon = lexicon or load_lexicon()
    display = clean_markup(text)
    spoken = lexicon._pattern.sub(lambda m: _SPEAK[m.lastgroup](m, lexicon), display)
    spoken = re.sub(r"\s*[—–]\s*", ", ", spoken)
    spoken = re.sub(r"\s+", " ", spoken)
    return SpeechText(display=display, spoken=re.sub(r"\s+([,.?!])", r"\1", spoken).strip())


# Sentence splitting for streaming --------------------------------------------------------------

_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s)|\n+")
_ABBREVIATIONS = {
    "approx.",
    "apt.",
    "ave.",
    "blvd.",
    "co.",
    "corp.",
    "dept.",
    "dr.",
    "est.",
    "etc.",
    "ext.",
    "inc.",
    "jr.",
    "llc.",
    "ltd.",
    "mr.",
    "mrs.",
    "ms.",
    "mt.",
    "no.",
    "prof.",
    "sec.",
    "sr.",
    "st.",
    "ste.",
    "tel.",
    "vs.",
}
# Only capitalized, so "the sun." and "he sat." still end their sentences.
_CALENDAR_ABBREVIATIONS = {
    f"{name}."
    for name in (
        "Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec "
        "Mon Tue Tues Wed Thu Thur Thurs Fri Sat Sun"
    ).split()
}


def _is_abbreviation(word: str) -> bool:
    word = word.lstrip("(\"'[")
    if word in _CALENDAR_ABBREVIATIONS or word.lower() in _ABBREVIATIONS:
        return True
    return re.fullmatch(r"(?:[A-Za-z]\.)+", word) is not None  # J. Smith, a.m., U.S., e.g.


def split_sentences(text: str) -> tuple[list[str], str]:
    """Split into finished sentences and the unfinished tail.

    A period only ends a sentence once whitespace has arrived after it. While streaming, that keeps
    "$49." from being cut before ".99" comes in, and leaves "mail.fernhill.example" whole.
    Abbreviations never end one, since "Nov. 6" has to reach `normalize` as a single date. The cost
    is that a sentence really ending in one waits to join the next, which only delays it.
    """
    sentences: list[str] = []
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        candidate = text[start : boundary.end()].strip()
        last_word = candidate.rsplit(None, 1)[-1] if candidate else ""
        is_list_number = re.fullmatch(r"\d+\.", candidate) is not None
        if _is_abbreviation(last_word) or is_list_number:
            continue
        if candidate:
            sentences.append(candidate)
        start = boundary.end()
    return sentences, text[start:].strip()
