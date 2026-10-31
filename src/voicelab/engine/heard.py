"""Repair what STT does to spelled-out ids, before a bot's brain sees the text.

Speech recognizers hear "AC9077" as "a C nine zero seven seven", and a small LLM then looks up the
wrong thing. This puts spelled runs of letters and digits back together. It only changes what the
brain reads: `heard_text` in the artifact stays the raw STT output, so evals still see every
recognition error.
"""

from __future__ import annotations

import re
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
# Two common words that are also single letters, so they can only start an id before a capital.
_WORD_LETTERS = {"a", "i"}
_TOKEN = re.compile(r"\S+")
_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def _digit(word: str) -> str | None:
    if word.isdigit():
        return word
    return _DIGITS.get(word.lower())


def _letter(word: str) -> str | None:
    bare = word.rstrip(".")
    return bare.upper() if len(bare) == 1 and bare.lower() in ascii_lowercase else None


def prepare_heard_text(text: str) -> str:
    """Join spelled ids, so "a C nine zero seven seven" reads "AC9077"."""
    tokens = [(_EDGE.match(m.group()).groups()) for m in _TOKEN.finditer(text)]
    output: list[str] = []
    index = 0
    while index < len(tokens):
        joined, consumed = _spelled_id(tokens, index)
        if consumed:
            lead = tokens[index][0]
            trail = tokens[index + consumed - 1][2]
            output.append(f"{lead}{joined}{trail}")
            index += consumed
        else:
            output.append("".join(tokens[index]))
            index += 1
    return " ".join(output)


def _spelled_id(tokens: list[tuple[str, str, str]], start: int) -> tuple[str, int]:
    """Read one id at `start`: optional spelled letters, then a run of digits."""
    letters: list[str] = []
    position = start
    while position < len(tokens) and len(letters) < 3:
        letter = _letter(tokens[position][1])
        if letter is None or _digit(tokens[position][1]) is not None:
            break
        letters.append(letter)
        position += 1
        if tokens[position - 1][2]:  # a comma or full stop ends the id
            break
    digits: list[str] = []
    while position < len(tokens):
        digit = _digit(tokens[position][1])
        if digit is None:
            break
        digits.append(digit)
        position += 1
        if tokens[position - 1][2] and tokens[position - 1][2] not in {",", "-"}:
            break
    run = "".join(digits)
    if len(run) < _MIN_DIGITS or (len(digits) == 1 and not letters):
        return "", 0
    # "a five digit code": a lone common word before digits needs a capital to count as a letter.
    if len(letters) == 1 and letters[0].lower() in _WORD_LETTERS and tokens[start][1].islower():
        return "", 0
    return "".join(letters) + run, position - start
