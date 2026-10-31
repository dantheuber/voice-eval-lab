"""Checks on tool calls from a speech-to-speech model before they run.

In a cascade the LLM goes through `SupportBot`, which recovers from bad calls. A realtime model
has no such layer, and in testing LFM3 realtime sent truncated JSON, leaked template syntax into
values (`[customer_id]\\n</function>...`), and fired a dependent call before the call it depended
on had returned. A bad call gets an error back as its result, so the model can try again.
"""

from __future__ import annotations

import re
from typing import Any

from voicelab.providers.base import ToolCall, ToolDef

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}
_TEMPLATE_SYNTAX = re.compile(r"</?function|</?tool_call|<\|")


def tool_call_problem(call: ToolCall, tools: list[ToolDef]) -> str | None:
    """Why this call shouldn't run, in words the model can act on, or None if it may."""
    tool = next((t for t in tools if t.name == call.name), None)
    if tool is None:
        return f"there is no tool called {call.name}"
    schema = tool.parameters
    properties: dict[str, Any] = schema.get("properties", {})
    missing = [
        name for name in schema.get("required", []) if call.arguments.get(name) in (None, "")
    ]
    if missing:
        return f"missing {', '.join(missing)}"
    for name, value in call.arguments.items():
        if isinstance(value, str) and _TEMPLATE_SYNTAX.search(value):
            return f"{name} contains template syntax instead of a value"
        expected = _JSON_TYPES.get(properties.get(name, {}).get("type", ""))
        if value is None or expected is None or isinstance(value, expected):
            continue
        # A number sent as text, or the reverse, is a near miss the tools themselves accept.
        if not isinstance(value, str | int | float) or isinstance(value, bool):
            return f"{name} has the wrong type"
    return None
