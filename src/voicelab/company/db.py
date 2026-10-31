"""The Fernhill database: schema, reproducible seeding, per-call copies, and diffing.

Every call gets a private database built from the same seed, so calls can't see each other's
changes and an eval can ask exactly what one call did. The company's calendar is frozen at the
seed's `as_of` date, which keeps refund windows and new row dates identical between runs.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from voicelab.core.models import DbChange, RowRef

SEEDS_DIR = Path(__file__).resolve().parents[3] / "seeds"
SEED_FILE = SEEDS_DIR / "company.yaml"

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;

CREATE TABLE plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('email', 'web')),
    price_cents INTEGER NOT NULL,
    mailbox_quota_mb INTEGER
) STRICT;

CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT NOT NULL,
    billing_zip TEXT NOT NULL,
    pin TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'past_due', 'closed')),
    card_last4 TEXT NOT NULL,
    card_status TEXT NOT NULL CHECK (card_status IN ('valid', 'expired'))
) STRICT;

CREATE TABLE subscriptions (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    plan_id TEXT NOT NULL REFERENCES plans(id),
    status TEXT NOT NULL CHECK (status IN ('active', 'past_due', 'cancelled')),
    started_on TEXT NOT NULL,
    renews_on TEXT NOT NULL
) STRICT;

CREATE TABLE domains (
    name TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    expires_on TEXT NOT NULL,
    auto_renew INTEGER NOT NULL,
    renewal_price_cents INTEGER NOT NULL
) STRICT;

CREATE TABLE mailboxes (
    address TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    quota_mb INTEGER NOT NULL,
    used_mb INTEGER NOT NULL,
    locked INTEGER NOT NULL DEFAULT 0,
    last_login TEXT,
    password_reset_on TEXT
) STRICT;

CREATE TABLE websites (
    domain TEXT PRIMARY KEY REFERENCES domains(name),
    customer_id TEXT NOT NULL REFERENCES customers(id),
    status TEXT NOT NULL CHECK (status IN ('online', 'suspended')),
    suspended_reason TEXT,
    ssl_expires_on TEXT NOT NULL
) STRICT;

-- One value per domain, type, and name keeps "update the MX record" unambiguous for the bot.
CREATE TABLE dns_records (
    id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL REFERENCES domains(name),
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL,
    priority INTEGER,
    ttl INTEGER NOT NULL DEFAULT 3600,
    UNIQUE (domain, type, name)
) STRICT;

CREATE TABLE invoices (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    description TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'unpaid', 'refunded', 'void')),
    issued_on TEXT NOT NULL,
    due_on TEXT NOT NULL
) STRICT;

-- No uniqueness on invoice_id: a double charge is two rows, and the tools can create one.
CREATE TABLE payments (
    id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(id),
    amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    card_last4 TEXT NOT NULL,
    paid_on TEXT NOT NULL
) STRICT;

CREATE TABLE refunds (
    id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(id),
    payment_id TEXT REFERENCES payments(id),
    amount_cents INTEGER NOT NULL,
    reason TEXT NOT NULL,
    created_on TEXT NOT NULL
) STRICT;

CREATE TABLE tickets (
    id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(id),
    subject TEXT NOT NULL,
    summary TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    created_on TEXT NOT NULL
) STRICT;
"""

# Parents before children, so foreign keys hold while rows load.
SEED_TABLES = (
    "plans",
    "customers",
    "subscriptions",
    "domains",
    "mailboxes",
    "websites",
    "dns_records",
    "invoices",
    "payments",
    "refunds",
    "tickets",
)


@dataclass
class CompanyDb:
    """One call's private database. All tool calls for that call run against this."""

    conn: sqlite3.Connection
    today: date
    seed_file: Path = SEED_FILE

    def rows(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute(sql, params)]

    def row(self, sql: str, *params: Any) -> dict[str, Any] | None:
        found = self.conn.execute(sql, params).fetchone()
        return dict(found) if found else None

    def close(self) -> None:
        self.conn.close()


def _column_value(value: Any) -> Any:
    # PyYAML turns unquoted dates into date objects, and STRICT tables store dates as TEXT.
    return value.isoformat() if isinstance(value, date) else value


def _build(target: str | Path, seed_file: Path) -> sqlite3.Connection:
    seed = yaml.safe_load(Path(seed_file).read_text())
    # check_same_thread is off because the engine may diff from a different thread than the one
    # that ran the tools. Calls are serial, so a connection never has two users at once.
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO meta VALUES ('as_of', ?)", (_column_value(seed["as_of"]),))
    for table in SEED_TABLES:
        for row in seed.get(table, []):
            columns = ", ".join(row)
            marks = ", ".join("?" for _ in row)
            values = [_column_value(value) for value in row.values()]
            conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", values)
    conn.commit()
    return conn


def create_call_db(path: str | Path = ":memory:", *, seed_file: Path = SEED_FILE) -> CompanyDb:
    """Build a fresh private database for one call, in memory or at `path` for keeping."""
    if path != ":memory:" and Path(path).exists():
        raise FileExistsError(f"call database already exists: {path}")
    conn = _build(path, seed_file)
    as_of = conn.execute("SELECT value FROM meta WHERE key = 'as_of'").fetchone()[0]
    return CompanyDb(conn=conn, today=date.fromisoformat(as_of), seed_file=seed_file)


def _primary_key(conn: sqlite3.Connection, table: str) -> list[str]:
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [column["name"] for column in info if column["pk"]]


def _keyed_rows(
    conn: sqlite3.Connection, table: str, key_columns: list[str]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows = (dict(row) for row in conn.execute(f"SELECT * FROM {table}"))
    return {tuple(row[column] for column in key_columns): row for row in rows}


def table_rows(db: CompanyDb) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    """Every data row by table and primary key, to compare before and after a tool runs."""
    return {
        table: _keyed_rows(db.conn, table, _primary_key(db.conn, table)) for table in SEED_TABLES
    }


def changed_rows(
    before: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    after: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    db: CompanyDb,
) -> list[RowRef]:
    """The rows that differ between two `table_rows` snapshots."""
    refs: list[RowRef] = []
    for table in SEED_TABLES:
        key_columns = _primary_key(db.conn, table)
        for key in sorted(before[table].keys() | after[table].keys()):
            if before[table].get(key) != after[table].get(key):
                refs.append(RowRef(table=table, key=dict(zip(key_columns, key, strict=True))))
    return refs


def diff_against_seed(db: CompanyDb) -> list[DbChange]:
    """Return every row a call inserted, updated, or deleted, ordered by table then key."""
    seed = _build(":memory:", db.seed_file)
    changes: list[DbChange] = []
    try:
        for table in SEED_TABLES:
            key_columns = _primary_key(seed, table)
            before = _keyed_rows(seed, table, key_columns)
            after = _keyed_rows(db.conn, table, key_columns)
            for key in sorted(before.keys() | after.keys()):
                old, new = before.get(key), after.get(key)
                if old == new:
                    continue
                kind = "insert" if old is None else "delete" if new is None else "update"
                changes.append(
                    DbChange(
                        table=table,
                        key=dict(zip(key_columns, key, strict=True)),
                        kind=kind,
                        before=old,
                        after=new,
                    )
                )
    finally:
        seed.close()
    return changes


def table_columns(seed_file: Path = SEED_FILE) -> dict[str, list[str]]:
    """Columns per data table, for validating scenarios and describing the schema to an LLM."""
    conn = _build(":memory:", seed_file)
    try:
        return {
            table: [column["name"] for column in conn.execute(f"PRAGMA table_info({table})")]
            for table in SEED_TABLES
        }
    finally:
        conn.close()


def customer_notes(seed_file: Path = SEED_FILE) -> dict[str, str]:
    """What is unusual about each seeded customer, keyed by customer id."""
    return yaml.safe_load(Path(seed_file).read_text())["customer_notes"]
