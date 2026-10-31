"""Runs (the API's name for a batch of calls)."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from voicelab.store.calls import BadCursor
from voicelab.store.sqlite import Database

ACTIVE_STATUSES = ("queued", "running")
UNFINISHED_STATUSES = ("queued", "running", "paused")


def _load(row: dict[str, Any]) -> dict[str, Any]:
    row["pairs"] = json.loads(row["pairs"])
    row["line"] = json.loads(row["line"]) if row["line"] else None
    return row


class RunStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, **columns: Any) -> None:
        columns["pairs"] = json.dumps(columns["pairs"])
        columns["line"] = json.dumps(columns["line"]) if columns.get("line") else None
        names = ", ".join(columns)
        marks = ", ".join("?" for _ in columns)
        self.db.run(f"INSERT INTO runs ({names}) VALUES ({marks})", list(columns.values()))

    def update(self, run_id: str, **columns: Any) -> None:
        assignments = ", ".join(f"{name} = ?" for name in columns)
        self.db.run(f"UPDATE runs SET {assignments} WHERE id = ?", [*columns.values(), run_id])

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.row("SELECT * FROM runs WHERE id = ?", (run_id,))
        return _load(row) if row else None

    def with_status(self, *statuses: str) -> list[dict[str, Any]]:
        marks = ", ".join("?" for _ in statuses)
        rows = self.db.rows(
            f"SELECT * FROM runs WHERE status IN ({marks}) ORDER BY started_at, id", statuses
        )
        return [_load(row) for row in rows]

    def active(self) -> list[dict[str, Any]]:
        """Runs that may start calls, oldest first."""
        return self.with_status(*ACTIVE_STATUSES)

    def unfinished(self) -> list[dict[str, Any]]:
        return self.with_status(*UNFINISHED_STATUSES)

    def uses(self, column: str, item_id: str) -> bool:
        """Whether an unfinished run has this persona or scenario."""
        return any(item_id in [p[column] for p in run["pairs"]] for run in self.unfinished())

    def page(self, cursor: str | None, limit: int) -> tuple[list[dict[str, Any]], str | None]:
        where, params = "1", []
        if cursor:
            started_at, run_id = _decode(cursor)
            where, params = "(started_at, id) < (?, ?)", [started_at, run_id]
        rows = self.db.rows(
            f"SELECT * FROM runs WHERE {where} ORDER BY started_at DESC, id DESC LIMIT ?",
            [*params, limit + 1],
        )
        next_cursor = _encode(rows[limit - 1]) if len(rows) > limit else None
        return [_load(row) for row in rows[:limit]], next_cursor


def _encode(row: dict[str, Any]) -> str:
    raw = json.dumps([row["started_at"], row["id"]]).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode(cursor: str) -> tuple[str, str]:
    try:
        started_at, run_id = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except (ValueError, TypeError, binascii.Error) as exc:
        raise BadCursor(cursor) from exc
    return str(started_at), str(run_id)
