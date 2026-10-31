"""What to do when the audio model and the delivery measurements disagree: believe the measurements.

The benchmark's listening model read words rather than the voice, so what it says about a clip gets
checked against what was measured on it. A claim the numbers contradict stays on record as `info`,
which no longer counts toward review. Defects the model didn't mention stand as measured. Claims
nothing was measured for are left alone, since there's nothing to overrule them with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from voicelab.core.models import Finding

# What the model might say, in its words, for each thing the measurements can settle.
_CLAIM_WORDS = {
    "rushed": ("rushed", "too fast", "hurried", "rapid"),
    "dragging": ("dragging", "too slow", "sluggish", "drawn out"),
    "flat": ("robotic", "monotone", "flat", "mechanical", "metallic"),
    "noisy": ("noisy", "noise", "static", "hiss"),
    "chopped": ("chopped", "choppy", "cutting out", "cuts out", "dropout", "stutter"),
}
_MEASURED_AS = {
    "rushed": ("rushed",),
    "dragging": ("dragging",),
    "flat": ("robotic", "monotone"),
    "noisy": ("noisy",),
    "chopped": ("chopped",),
}
_METRIC = {
    "rushed": "wpm",
    "dragging": "wpm",
    "flat": "pitch_iqr_st",
    "noisy": "snr_db",
    "chopped": "chops",
}
_UNIT = {
    "wpm": "{:.0f} wpm",
    "pitch_iqr_st": "{:.2f} st of pitch movement",
    "snr_db": "{:.0f} dB SNR",
    "chops": "{:.0f} dropouts",
}
# Emotion needs a moving voice to carry it, so hearing one in a flat clip means reading the words.
_EMOTIONS = (
    "frustrated", "angry", "irritated", "annoyed", "upset", "excited", "enthusiastic", "cheerful",
)  # fmt: skip


def _claims_in(text: str) -> list[str]:
    lowered = text.lower()
    return [kind for kind, words in _CLAIM_WORDS.items() if any(w in lowered for w in words)]


def _measured(kind: str, items: Sequence[Any]) -> str:
    metric = _METRIC[kind]
    values = [getattr(item, metric) for item in items if getattr(item, metric) is not None]
    return ", ".join(_UNIT[metric].format(v) for v in values)


def _contradicted(kind: str, items: Sequence[Any]) -> bool:
    """The clip was measured for this, and came back fine."""
    if not _measured(kind, items):
        return False
    return not any(flag in item.flags for item in items for flag in _MEASURED_AS[kind])


def _record(kind: str, start_s: float, end_s: float, note: str) -> dict[str, Any]:
    return {"kind": kind, "start_s": round(start_s, 3), "end_s": round(end_s, 3), "note": note}


def reconcile(
    start_s: float,
    end_s: float,
    payload: dict[str, Any],
    findings: list[Finding],
    measured: Sequence[Any],
) -> list[dict[str, Any]]:
    """Settle one heard clip against the bot's measurements. Overruled findings are edited here."""
    items = [m for m in measured if m.start_s < end_s and start_s < m.end_s]
    if not items:
        return []
    at = f"at {start_s:.1f} s"
    disagreements: list[dict[str, Any]] = []
    claimed: set[str] = set()

    for finding in findings:
        for kind in _claims_in(f"{finding.title} {finding.detail}"):
            claimed.add(kind)
            if _contradicted(kind, items):
                numbers = _measured(kind, items)
                note = f'The audio model said "{finding.title}" {at}, but measured {numbers}.'
                disagreements.append(_record("overruled", start_s, end_s, note))
                finding.severity = "info"
                finding.title = f"Overruled: {finding.title}"
                finding.detail = f"Overruled by measurements, {numbers}. {finding.detail}".strip()

    spoken = f"{payload.get('tone') or ''} {payload.get('delivery') or ''}"
    for kind in _claims_in(spoken):
        if kind not in claimed and _contradicted(kind, items):
            numbers = _measured(kind, items)
            note = f'The audio model described "{spoken.strip()}" {at}, but measured {numbers}.'
            disagreements.append(_record("overruled", start_s, end_s, note))
        claimed.add(kind)

    tone = str(payload.get("tone") or "").lower()
    flat = any(flag in item.flags for item in items for flag in _MEASURED_AS["flat"])
    if flat and any(emotion in tone for emotion in _EMOTIONS):
        numbers = _measured("flat", items)
        note = f'The audio model heard "{tone}" {at} in a flat voice, {numbers}.'
        disagreements.append(_record("emotion", start_s, end_s, note))

    for kind, flags in _MEASURED_AS.items():
        if kind not in claimed and any(flag in item.flags for item in items for flag in flags):
            note = f"The audio model missed {flags[0]} {at}, measured {_measured(kind, items)}."
            disagreements.append(_record("missed", start_s, end_s, note))
    return disagreements
