"""One finding per thing to look at, however many times it happened.

A reviewer deciding on the same complaint eight times is a reviewer not doing anything else, so
repeats become a single finding that keeps every occurrence as evidence. Its clip plays the worst
occurrence, which comes first in the evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from voicelab.core.models import Finding

_RANK = {"info": 0, "minor": 1, "major": 2, "critical": 3}


def roll_up(
    findings: Iterable[Finding],
    *,
    key: Callable[[Finding], str],
    title: Callable[[list[Finding]], str],
) -> list[Finding]:
    """Merge findings that share `key`, in order of first appearance. Lone ones pass unchanged."""
    groups: dict[str, list[Finding]] = {}
    for finding in findings:
        groups.setdefault(key(finding), []).append(finding)
    return [_merged(group, title) if len(group) > 1 else group[0] for group in groups.values()]


def _merged(group: list[Finding], title: Callable[[list[Finding]], str]) -> Finding:
    ordered = sorted(group, key=lambda f: _RANK[f.severity], reverse=True)
    worst = ordered[0]
    return worst.model_copy(
        update={
            "title": title(group),
            "detail": "; ".join(dict.fromkeys(f.detail for f in ordered if f.detail)),
            "evidence": [e for f in ordered for e in f.evidence],
        }
    )
