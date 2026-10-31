"""The call index and human decisions. `call.json` stays the record; this makes lists fast."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

from voicelab.core.models import CallArtifact, EvalResult, Finding, Verdict
from voicelab.store.sqlite import Database, timestamp

_FILTER_COLUMNS = {
    "verdict": "verdict",
    "scenario": "scenario_id",
    "persona": "persona_id",
    "mode": "mode",
    "profile": "profile",
    "run": "batch_id",
    "status": "status",
}
_NEEDS_REVIEW = (
    "status = 'done' AND human_verdict IS NULL AND (open_findings > 0 OR verdict = 'needs_review')"
)
_WHY_TITLES = 2
_PACE_DECISIONS = 50


class BadCursor(ValueError):
    pass


@dataclass
class CallFilters:
    verdict: list[str] = field(default_factory=list)
    scenario: list[str] = field(default_factory=list)
    persona: list[str] = field(default_factory=list)
    mode: list[str] = field(default_factory=list)
    profile: list[str] = field(default_factory=list)
    run: list[str] = field(default_factory=list)
    status: list[str] = field(default_factory=list)
    open_only: bool = False
    since: str | None = None


@dataclass
class CallPage:
    items: list[dict[str, Any]]
    next_cursor: str | None
    total: int
    facets: dict[str, int]


def timing_metrics(result: EvalResult | None) -> dict[str, Any] | None:
    run = next((r for r in result.evaluators if r.name == "timing"), None) if result else None
    return run.metrics if run is not None and run.status == "ok" else None


def reply_p50(artifact: CallArtifact, result: EvalResult | None) -> float | None:
    """Null for turn-based calls, whose pauses are synthetic, and when nothing was measured."""
    metrics = timing_metrics(result)
    if metrics is None or not artifact.timing_is_real:
        return None
    return (metrics.get("response_latency") or {}).get("median_s")


def overlap_count(artifact: CallArtifact, result: EvalResult | None) -> int | None:
    metrics = timing_metrics(result)
    if metrics is None or not artifact.timing_is_real:
        return None
    return (metrics.get("overlap") or {}).get("count")


def invalid_reason(result: EvalResult) -> str | None:
    if result.verdict != Verdict.INVALID:
        return None
    run = next((r for r in result.evaluators if r.name == "caller_validity"), None)
    if run is not None and run.reason:
        return run.reason
    finding = next((f for f in result.findings if f.evaluator == "caller_validity"), None)
    if finding is not None:
        return finding.title
    missed = next((c for c in result.criteria if c.kind == "beat" and c.status == "fail"), None)
    return f"caller missed: {missed.text}" if missed else "caller never reached the scenario"


def why_flagged(result: EvalResult) -> str:
    open_titles = [f.title for f in result.findings if is_open(f)]
    return ", ".join(open_titles[:_WHY_TITLES]) or result.summary


def is_open(finding: Finding) -> bool:
    return not finding.advisory and finding.human_verdict is None


def eval_columns(artifact: CallArtifact, result: EvalResult) -> dict[str, Any]:
    """What a finished evaluation changes on the call's index row."""
    return {
        "verdict": result.verdict.value,
        "invalid_reason": invalid_reason(result),
        "why": why_flagged(result),
        "reply_p50_s": reply_p50(artifact, result),
        "overlaps": overlap_count(artifact, result),
        "duration_s": artifact.duration_s,
    }


def _fingerprint(evaluator: str, category: str, criterion: str | None, title: str) -> tuple:
    return evaluator, category, criterion, title


def _encode_cursor(row: dict[str, Any]) -> str:
    raw = json.dumps([row["created_at"], row["id"]]).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        created_at, call_id = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except (ValueError, TypeError, binascii.Error) as exc:
        raise BadCursor(cursor) from exc
    return str(created_at), str(call_id)


def _where(filters: CallFilters, *, skip: str | None = None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for name, column in _FILTER_COLUMNS.items():
        values = getattr(filters, name)
        if values and name != skip:
            clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
            params.extend(values)
    if filters.open_only:
        clauses.append("open_findings > 0")
    if filters.since:
        clauses.append("created_at >= ?")
        params.append(filters.since)
    return (" AND ".join(clauses) or "1"), params


class CallIndex:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, **columns: Any) -> None:
        columns.setdefault("created_at", timestamp())
        names = ", ".join(columns)
        marks = ", ".join("?" for _ in columns)
        self.db.run(f"INSERT INTO calls ({names}) VALUES ({marks})", list(columns.values()))

    def update(self, call_id: str, **columns: Any) -> None:
        assignments = ", ".join(f"{name} = ?" for name in columns)
        self.db.run(f"UPDATE calls SET {assignments} WHERE id = ?", [*columns.values(), call_id])

    def get(self, call_id: str) -> dict[str, Any] | None:
        return self.db.row("SELECT * FROM calls WHERE id = ?", (call_id,))

    def delete(self, call_id: str) -> bool:
        with self.db.transaction():
            self.db.run("DELETE FROM call_findings WHERE call_id = ?", (call_id,))
            self.db.run("DELETE FROM human_decisions WHERE call_id = ?", (call_id,))
            return self.db.run("DELETE FROM calls WHERE id = ?", (call_id,)) > 0

    def with_status(self, *statuses: str) -> list[dict[str, Any]]:
        marks = ", ".join("?" for _ in statuses)
        return self.db.rows(
            f"SELECT * FROM calls WHERE status IN ({marks}) ORDER BY created_at, id", statuses
        )

    def in_batch(self, batch_id: str) -> list[dict[str, Any]]:
        return self.db.rows(
            "SELECT * FROM calls WHERE batch_id = ? ORDER BY created_at, id", (batch_id,)
        )

    def page(self, filters: CallFilters, cursor: str | None, limit: int) -> CallPage:
        where, params = _where(filters)
        total = self.db.value(f"SELECT count(*) FROM calls WHERE {where}", params)
        # Facets ignore the verdict filter, so the chips still show what picking another would give.
        facet_where, facet_params = _where(filters, skip="verdict")
        facets = {
            row["verdict"]: row["n"]
            for row in self.db.rows(
                f"SELECT verdict, count(*) AS n FROM calls WHERE {facet_where} "
                "AND verdict IS NOT NULL GROUP BY verdict",
                facet_params,
            )
        }
        if cursor:
            created_at, call_id = _decode_cursor(cursor)
            where += " AND (created_at, id) < (?, ?)"
            params = [*params, created_at, call_id]
        rows = self.db.rows(
            f"SELECT * FROM calls WHERE {where} ORDER BY created_at DESC, id DESC LIMIT ?",
            [*params, limit + 1],
        )
        next_cursor = _encode_cursor(rows[limit - 1]) if len(rows) > limit else None
        return CallPage(rows[:limit], next_cursor, total, facets)

    def recent(self, column: str, value: str, limit: int = 8) -> list[dict[str, Any]]:
        return self.db.rows(
            f"SELECT * FROM calls WHERE {column} = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (value, limit),
        )

    # Counts ------------------------------------------------------------------------------------

    def verdict_counts(self, filters: CallFilters | None = None) -> dict[str, int]:
        where, params = _where(filters or CallFilters())
        rows = self.db.rows(
            f"SELECT verdict, count(*) AS n FROM calls WHERE {where} "
            "AND verdict IS NOT NULL GROUP BY verdict",
            params,
        )
        return {row["verdict"]: row["n"] for row in rows}

    def counts_by(self, *columns: str, filters: CallFilters | None = None) -> list[dict[str, Any]]:
        """Verdict tallies per group. Rows hold the group columns plus a `counts` dict."""
        where, params = _where(filters or CallFilters())
        names = ", ".join(columns)
        rows = self.db.rows(
            f"SELECT {names}, verdict, count(*) AS n FROM calls WHERE {where} "
            f"AND verdict IS NOT NULL GROUP BY {names}, verdict",
            params,
        )
        grouped: dict[tuple, dict[str, Any]] = {}
        for row in rows:
            key = tuple(row[column] for column in columns)
            group = grouped.setdefault(key, {**{c: row[c] for c in columns}, "counts": {}})
            group["counts"][row["verdict"]] = row["n"]
        return list(grouped.values())

    def usage(self, column: str) -> dict[str, dict[str, Any]]:
        """Calls and verdict tallies per persona or scenario id."""
        totals = self.db.rows(
            f"SELECT {column} AS key, count(*) AS n FROM calls "
            f"WHERE {column} IS NOT NULL GROUP BY {column}"
        )
        usage = {row["key"]: {"calls": row["n"], "counts": {}} for row in totals}
        for group in self.counts_by(column):
            if group[column] in usage:
                usage[group[column]]["counts"] = group["counts"]
        return usage

    def calls_per_hour(self, profile: str | None, now: datetime | None = None) -> list[int]:
        """Calls in each of the last 24 hours, oldest first, with the current hour last."""
        now = now or datetime.now(UTC)
        filters = CallFilters(
            profile=[profile] if profile else [], since=timestamp(now - timedelta(hours=24))
        )
        where, params = _where(filters)
        bins = [0] * 24
        for row in self.db.rows(f"SELECT created_at FROM calls WHERE {where}", params):
            age = now - datetime.fromisoformat(row["created_at"])
            index = 23 - int(age.total_seconds() // 3600)
            if 0 <= index < 24:
                bins[index] += 1
        return bins

    def mean_call_seconds(self, batch_id: str) -> float | None:
        return self.db.value(
            "SELECT avg((julianday(finished_at) - julianday(started_at)) * 86400) FROM calls "
            "WHERE batch_id = ? AND status = 'done' AND started_at IS NOT NULL "
            "AND finished_at IS NOT NULL",
            (batch_id,),
        )

    def batch_progress(self, batch_id: str) -> dict[str, Any]:
        rows = self.db.rows(
            "SELECT status, count(*) AS n FROM calls WHERE batch_id = ? GROUP BY status",
            (batch_id,),
        )
        by_status = {row["status"]: row["n"] for row in rows}
        return {
            "calls": sum(by_status.values()),
            "done": by_status.get("done", 0) + by_status.get("failed", 0),
            "running": by_status.get("running", 0) + by_status.get("evaluating", 0),
            "queued": by_status.get("queued", 0),
            "counts": self.verdict_counts(CallFilters(run=[batch_id])),
        }

    # Review queue ------------------------------------------------------------------------------

    def review_queue(self, profile: str | None = None) -> list[dict[str, Any]]:
        """Calls waiting for a person, oldest first."""
        where = _NEEDS_REVIEW
        params: list[Any] = []
        if profile:
            where += " AND profile = ?"
            params.append(profile)
        return self.db.rows(f"SELECT * FROM calls WHERE {where} ORDER BY created_at, id", params)

    def next_review(self, after: str | None) -> str | None:
        """The call after `after` in the queue, wrapping round so `]` keeps moving."""
        ids = [row["id"] for row in self.review_queue()]
        if after in ids:
            position = ids.index(after)
            return next(iter(ids[position + 1 :] + ids[:position]), None)
        return ids[0] if ids else None

    def previous_review(self, call_id: str) -> str | None:
        ids = [row["id"] for row in self.review_queue()]
        if call_id in ids:
            position = ids.index(call_id)
            return ids[position - 1] if position > 0 else None
        return ids[-1] if ids else None

    # Findings and decisions --------------------------------------------------------------------

    def record_eval(self, artifact: CallArtifact, result: EvalResult) -> None:
        """Index a finished evaluation.

        Re-evaluating gives every finding a new id. A decision moves to the new finding that says
        the same thing, so a re-run doesn't throw away the reviewer's work.
        """
        with self.db.transaction():
            kept = {
                _fingerprint(r["evaluator"], r["category"], r["criterion_id"], r["title"]): r
                for r in self.db.rows(
                    "SELECT * FROM call_findings WHERE call_id = ? AND human_verdict IS NOT NULL",
                    (artifact.id,),
                )
            }
            self.db.run("DELETE FROM call_findings WHERE call_id = ?", (artifact.id,))
            for finding in result.findings:
                key = _fingerprint(
                    finding.evaluator, finding.category, finding.criterion_id, finding.title
                )
                decision = kept.get(key)
                self.db.run(
                    "INSERT INTO call_findings (call_id, finding_id, evaluator, category, "
                    "severity, advisory, title, criterion_id, human_verdict, human_note, "
                    "human_by, human_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        artifact.id,
                        finding.id,
                        finding.evaluator,
                        finding.category,
                        finding.severity,
                        int(finding.advisory),
                        finding.title,
                        finding.criterion_id,
                        decision["human_verdict"] if decision else None,
                        decision["human_note"] if decision else "",
                        decision["human_by"] if decision else None,
                        decision["human_at"] if decision else None,
                    ),
                )
            self.update(artifact.id, **eval_columns(artifact, result))
            self._refresh_finding_counts(artifact.id)

    def decide_finding(
        self, call_id: str, finding_id: str, verdict: str | None, note: str, by: str
    ) -> dict[str, Any] | None:
        at = timestamp()
        with self.db.transaction():
            changed = self.db.run(
                "UPDATE call_findings SET human_verdict = ?, human_note = ?, human_by = ?, "
                "human_at = ? WHERE call_id = ? AND finding_id = ?",
                (
                    verdict,
                    note,
                    by if verdict else None,
                    at if verdict else None,
                    call_id,
                    finding_id,
                ),
            )
            if not changed:
                return None
            self._log_decision(call_id, finding_id, verdict, note, by, at)
            self._refresh_finding_counts(call_id)
        return self.finding_decisions(call_id).get(finding_id)

    def decide_call(self, call_id: str, verdict: str, note: str, by: str) -> None:
        at = timestamp()
        with self.db.transaction():
            self.update(call_id, human_verdict=verdict, human_note=note, human_by=by, human_at=at)
            self._log_decision(call_id, None, verdict, note, by, at)

    def finding_decisions(self, call_id: str) -> dict[str, dict[str, Any]]:
        rows = self.db.rows(
            "SELECT finding_id, human_verdict, human_note, human_by, human_at "
            "FROM call_findings WHERE call_id = ?",
            (call_id,),
        )
        return {row["finding_id"]: row for row in rows}

    def decision_history(self, call_id: str) -> list[dict[str, Any]]:
        return self.db.rows(
            "SELECT * FROM human_decisions WHERE call_id = ? ORDER BY id", (call_id,)
        )

    def _log_decision(
        self, call_id: str, finding_id: str | None, verdict: str | None, note: str, by: str, at: str
    ) -> None:
        self.db.run(
            "INSERT INTO human_decisions (call_id, finding_id, verdict, note, decided_by, "
            "decided_at) VALUES (?, ?, ?, ?, ?, ?)",
            (call_id, finding_id, verdict, note, by, at),
        )

    def _refresh_finding_counts(self, call_id: str) -> None:
        self.db.run(
            "UPDATE calls SET "
            "findings = (SELECT count(*) FROM call_findings WHERE call_id = calls.id), "
            "open_findings = (SELECT count(*) FROM call_findings WHERE call_id = calls.id "
            "AND advisory = 0 AND human_verdict IS NULL) WHERE id = ?",
            (call_id,),
        )

    def top_open_findings(
        self, profile: str | None, since: str | None, limit: int = 5
    ) -> list[dict[str, Any]]:
        where, params = _where(CallFilters(profile=[profile] if profile else [], since=since))
        return self.db.rows(
            "SELECT title, evaluator, count(*) AS count FROM call_findings "
            "WHERE advisory = 0 AND human_verdict IS NULL AND call_id IN "
            f"(SELECT id FROM calls WHERE {where}) "
            "GROUP BY title, evaluator ORDER BY count DESC, title LIMIT ?",
            [*params, limit],
        )

    def review_pace(self) -> float | None:
        """Median seconds between the reviewer's recent finding decisions."""
        rows = self.db.rows(
            "SELECT decided_at FROM human_decisions WHERE finding_id IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (_PACE_DECISIONS,),
        )
        times = [datetime.fromisoformat(row["decided_at"]) for row in rows]
        gaps = [(a - b).total_seconds() for a, b in zip(times, times[1:], strict=False)]
        return round(median(gaps), 1) if gaps else None
