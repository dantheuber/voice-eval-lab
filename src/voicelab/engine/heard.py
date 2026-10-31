"""Repair what STT does to spelled ids and addresses, before a bot's brain sees the text.

Speech recognizers hear "AC9077" as "a C nine zero seven seven", and a small LLM then reads an
article and looks up the wrong thing. This puts spelled runs of letters and digits back together,
including phone-style spelling ("F as in Frank"), where the anchor word is more reliable than the
letter before it. It only changes what the brain reads: `heard_text` in the artifact stays the raw
STT output, so evals still see every recognition error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import ascii_lowercase

_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_MIN_DIGITS = 3  # shorter runs are ordinary speech: "one or two", "two zero"
_MAX_LETTERS = 3
# Common words that are also letters, so they only start an id when clearly spelled.
_WORD_LETTERS = {"a", "i"}
_TOKEN = re.compile(r"\S+")
_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
_ANCHORS = {"as", "for"}
_SPOKEN_HOST = re.compile(r"\b([\w-]+) dot (example|com|net|org)\b", re.IGNORECASE)
_SPOKEN_EMAIL = re.compile(r"\b([\w.+-]+) at ([\w-]+\.(?:example|com|net|org))\b", re.IGNORECASE)


@dataclass(frozen=True)
class _Token:
    lead: str
    word: str
    trail: str


def _digit(word: str) -> str | None:
    if word.isdigit():
        return word
    return _DIGITS.get(word.lower())


def _letter(word: str) -> str | None:
    bare = word.rstrip(".")
    return bare.upper() if len(bare) == 1 and bare.lower() in ascii_lowercase else None


def _tokens(text: str) -> list[_Token]:
    return [_Token(*_EDGE.match(m.group()).groups()) for m in _TOKEN.finditer(text)]


def _anchor_word(tokens: list[_Token], position: int) -> tuple[str, int] | None:
    """Read "as in Frank" or "for Frank" at `position`, returning the word and what it used."""
    if position >= len(tokens) or tokens[position].word.lower() not in _ANCHORS:
        return None
    if tokens[position].word.lower() == "for":
        following = position + 1
    elif position + 1 < len(tokens) and tokens[position + 1].word.lower() == "in":
        following = position + 2
    else:
        return None
    if following >= len(tokens) or not tokens[following].word.replace("-", "").isalpha():
        return None
    word = tokens[following].word
    if word.lower() == "x" and following + 1 < len(tokens):
        if tokens[following + 1].word.lower() == "ray":  # STT also writes it as two words
            following += 1
    return word, following - position + 1


def _spelled_id(tokens: list[_Token], start: int) -> tuple[str, int]:
    """Read one id at `start`: spelled letters, then a run of digits."""
    letters: list[str] = []
    position = start
    clearly_spelled = False
    while position < len(tokens) and len(letters) < _MAX_LETTERS:
        letter = _letter(tokens[position].word)
        if letter is None:
            break
        anchor = _anchor_word(tokens, position + 1)
        if anchor is not None:
            letters.append(anchor[0][0].upper())  # "S as in Frank" can only be F
            position += 1 + anchor[1]
            clearly_spelled = True
            continue
        letters.append(letter)
        position += 1
    digits: list[str] = []
    while position < len(tokens):
        digit = _digit(tokens[position].word)
        if digit is None:
            break
        digits.append(digit)
        position += 1
    run = "".join(digits)
    if len(run) < _MIN_DIGITS or (len(digits) == 1 and not letters):
        return "", 0
    if len(letters) == 1 and not clearly_spelled:
        word = tokens[start].word
        if word.lower() in _WORD_LETTERS and word.islower():
            return "", 0  # "a five digit code"
    return "".join(letters) + run, position - start


def prepare_heard_text(text: str) -> str:
    """Join spelled ids and addresses, so "F as in Frank H as in Henry 4419" reads "FH4419"."""
    tokens = _tokens(text)
    output: list[str] = []
    index = 0
    while index < len(tokens):
        joined, consumed = _spelled_id(tokens, index)
        token = tokens[index]
        if consumed:
            output.append(f"{token.lead}{joined}{tokens[index + consumed - 1].trail}")
            index += consumed
        else:
            output.append(f"{token.lead}{token.word}{token.trail}")
            index += 1
    repaired = _SPOKEN_HOST.sub(r"\1.\2", " ".join(output))
    return _SPOKEN_EMAIL.sub(r"\1@\2", repaired)
