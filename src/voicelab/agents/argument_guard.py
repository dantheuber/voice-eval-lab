"""A bot-side check on values only the caller can supply, before they reach a tool.

The guard is part of the bot under test, the way a real product might validate tool input. It reads
what the bot heard and never scenario data, so "not heard" means live STT never gave the bot that
value. It doesn't mean the caller never said it, because STT drops digits on a noisy line.

Stopping a call must hide nothing. The bot records every objection on the call's `ToolCallEvent`,
and the `tool_attempts` evaluator reports them. Run with the guard off, the call goes through and
the objection stays on the record, so the same mistake can be read on both settings.
"""

from __future__ import annotations

import re
from typing import Any

from voicelab.core.models import GuardObjection

NOT_HEARD = "not_heard"
DIGIT_COUNT = "digit_count"

# What a tool can only know from the caller. A small model fills these in when it has none, and
# the same zip and PIN came out on call after call. Each one used up one of the caller's two tries.
FROM_THE_CALLER = {
    "find_customer": ("phone", "email", "domain"),
    "verify_identity": ("billing_zip", "pin"),
}
_DIGITS_NEEDED = {"billing_zip": 5, "pin": 4}
# Speech recognition drops digits on a noisy line. A check on three digits can't pass, and it cost a
# real caller both tries on a PIN they gave correctly.
_TOLD_THE_MODEL = {
    NOT_HEARD: "The caller hasn't said that. Ask them for it.",
    DIGIT_COUNT: (
        "That's not a full zip and PIN, so speech recognition missed some digits. This try doesn't "
        "count. Ask them to say both again, slowly."
    ),
}
_WHY = {
    NOT_HEARD: "not in anything the bot heard from the caller",
    DIGIT_COUNT: "wrong number of digits",
}
_DIGIT_WORDS = {
    word: str(value)
    for value, word in enumerate("zero one two three four five six seven eight nine".split())
} | {"oh": "0"}
_LONG_NUMBER_WORDS = re.compile(r"\b(?:\w+teen|\w+ty|hundred|thousand)\b", re.IGNORECASE)


def _is_number(value: str) -> bool:
    return bool(re.search(r"\d", value)) and "@" not in value


def said_exactly(value: str, text: str) -> bool:
    """Whether a value is in this text, as digits or digit words, or by an address's name.

    An address only has to match by its name, which is what tells them apart.
    """
    if _is_number(value):
        said = "".join(
            word if word.isdigit() else _DIGIT_WORDS.get(word, "")
            for word in re.findall(r"\d+|[a-z]+", text.lower())
        )
        return re.sub(r"\D", "", value) in said
    name = re.split(r"[.@]", value.lower().removeprefix("www."))[0]
    return re.sub(r"[^a-z0-9]", "", name) in re.sub(r"[^a-z0-9]", "", text.lower())


def caller_said(value: str, heard: str) -> bool:
    """Whether a value could have come from the caller. Generous, since speech varies, so a number
    said like "nineteen fifty eight" is taken on trust."""
    if said_exactly(value, heard):
        return True
    return _is_number(value) and bool(_LONG_NUMBER_WORDS.search(heard))


def check(name: str, arguments: dict[str, Any], heard: str, *, stop: bool) -> GuardObjection | None:
    """The guard's objection to this call, or None if it has none. `stop` is the switch.

    This isn't policy, and a stopped call never reaches the tool. Like a reply that won't parse,
    it's bad output from the model, which gets told and tries again. It doesn't cost a try.
    """
    values = {field: str(arguments.get(field) or "") for field in FROM_THE_CALLER.get(name, ())}
    unheard = [field for field, value in values.items() if value and not caller_said(value, heard)]
    if unheard:
        return GuardObjection(rule=NOT_HEARD, fields=unheard, stopped=stop)
    short = [
        field
        for field, needed in _DIGITS_NEEDED.items()
        if field in values and len(re.sub(r"\D", "", values[field])) != needed
    ]
    if short:
        return GuardObjection(rule=DIGIT_COUNT, fields=short, stopped=stop)
    return None


def told_the_model(objection: GuardObjection) -> str:
    return _TOLD_THE_MODEL[objection.rule]


def blocked_error(objection: GuardObjection) -> str:
    """The event's `error`. It says what the rule can know, which is what the bot heard."""
    fields = ", ".join(objection.fields)
    rule = objection.rule
    return f"blocked before reaching the tool by argument_guard rule {rule}: {fields} {_WHY[rule]}"
