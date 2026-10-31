"""The SQLite connection and schema behind `data/voicelab.sqlite3`.

Call audio and JSON stay in `ArtifactStore`. This file holds what the dashboard has to query fast
or edit: personas, scenarios, runs, a call index, and human decisions. One connection guarded by a
lock is enough for a single-user app, and every statement here takes milliseconds.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenarios (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    version INTEGER NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenario_versions (
    scenario_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    data TEXT NOT NULL,
    saved_at TEXT NOT NULL,
    PRIMARY KEY (scenario_id, version)
);

-- Outlives the row it describes, so a seed the user deleted stays deleted after a reseed.
CREATE TABLE IF NOT EXISTS seeded (
    kind TEXT NOT NULL,
    id TEXT NOT NULL,
    PRIMARY KEY (kind, id)
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    profile TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    reps INTEGER NOT NULL,
    concurrency INTEGER NOT NULL,
    pairs TEXT NOT NULL,
    line TEXT,
    untested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_by_start ON runs (started_at, id);

CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    batch_id TEXT,
    rep INTEGER,
    seed INTEGER NOT NULL DEFAULT 0,
    scenario_id TEXT,
    scenario_title TEXT,
    persona_id TEXT,
    persona_name TEXT,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    profile TEXT,
    timing_is_real INTEGER NOT NULL,
    verdict TEXT,
    human_verdict TEXT,
    human_note TEXT NOT NULL DEFAULT '',
    human_by TEXT,
    human_at TEXT,
    invalid_reason TEXT,
    why TEXT NOT NULL DEFAULT '',
    findings INTEGER NOT NULL DEFAULT 0,
    open_findings INTEGER NOT NULL DEFAULT 0,
    duration_s REAL NOT NULL DEFAULT 0,
    reply_p50_s REAL,
    overlaps INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,  -- how the call itself failed. Evaluation never writes or clears this.
    eval_error TEXT,
    untested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS calls_by_time ON calls (created_at, id);
CREATE INDEX IF NOT EXISTS calls_by_batch ON calls (batch_id, status);
CREATE INDEX IF NOT EXISTS calls_by_scenario ON calls (scenario_id, persona_id);
CREATE INDEX IF NOT EXISTS calls_by_persona ON calls (persona_id);
CREATE INDEX IF NOT EXISTS calls_by_verdict ON calls (verdict);
CREATE INDEX IF NOT EXISTS calls_by_status ON calls (status);

CREATE TABLE IF NOT EXISTS call_findings (
    call_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    advisory INTEGER NOT NULL,
    title TEXT NOT NULL,
    criterion_id TEXT,
    human_verdict TEXT,
    human_note TEXT NOT NULL DEFAULT '',
    human_by TEXT,
    human_at TEXT,
    anchor_s REAL,  -- where in the call this finding is, for telling identical titles apart
    occurrence INTEGER NOT NULL DEFAULT 0,  -- its place among same-titled findings with no time
    times TEXT,  -- JSON list of every moment it covers, so a decision only follows those moments
    PRIMARY KEY (call_id, finding_id)
);
CREATE INDEX IF NOT EXISTS findings_open ON call_findings (advisory, human_verdict);

-- Append-only record of every human decision, for judge calibration. `finding_id` is null for a
-- decision about the whole call, and `verdict` is null when a finding's decision was cleared.
CREATE TABLE IF NOT EXISTS human_decisions (
    id INTEGER PRIMARY KEY,
    call_id TEXT NOT NULL,
    finding_id TEXT,
    verdict TEXT,
    note TEXT NOT NULL,
    decided_by TEXT NOT NULL,
    decided_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_by_call ON human_decisions (call_id, id);

CREATE TABLE IF NOT EXISTS presets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


# Columns added since the first release. CREATE TABLE IF NOT EXISTS leaves an older file as it is.
_ADDED_COLUMNS = (
    ("calls", "eval_error", "TEXT"),
    ("calls", "untested", "INTEGER NOT NULL DEFAULT 0"),
    ("runs", "untested", "INTEGER NOT NULL DEFAULT 0"),
    ("call_findings", "anchor_s", "REAL"),
    ("call_findings", "occurrence", "INTEGER NOT NULL DEFAULT 0"),
    ("call_findings", "times", "TEXT"),
)


def timestamp(moment: datetime | None = None) -> str:
    """A UTC instant with fixed precision, so text order in the database is time order."""
    moment = moment or datetime.now(UTC)
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


class Database:
    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # The app touches this from the event loop and from FastAPI's worker threads.
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._add_missing_columns()
        self._conn.executescript(SCHEMA)

    def _add_missing_columns(self) -> None:
        tables = {row[0] for row in self._conn.execute("SELECT name FROM sqlite_master")}
        for table, column, definition in _ADDED_COLUMNS:
            if table not in tables:
                continue  # a new database gets the column from SCHEMA
            present = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column in present:
                continue
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            if column == "eval_error":
                # These two failures used to share one column, and only this one can be retried.
                self._conn.execute(
                    "UPDATE calls SET eval_error = error, error = NULL "
                    "WHERE status = 'failed' AND error LIKE 'evaluation failed:%'"
                )

    def rows(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def row(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        found = self.rows(sql, params)
        return found[0] if found else None

    def value(self, sql: str, params: Sequence[Any] = ()) -> Any:
        found = self.row(sql, params)
        return next(iter(found.values())) if found else None

    def run(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute a write and return how many rows it touched."""
        with self._lock:
            return self._conn.execute(sql, params).rowcount

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
