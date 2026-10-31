"""Seeding, per-call isolation, and diffing."""

import sqlite3

import pytest

from voicelab.company.db import (
    SEED_TABLES,
    create_call_db,
    customer_notes,
    diff_against_seed,
    table_columns,
)
from voicelab.core.models import DbChange


@pytest.fixture
def db():
    call_db = create_call_db()
    yield call_db
    call_db.close()


def dump(database):
    return {table: database.rows(f"SELECT * FROM {table} ORDER BY 1, 2") for table in SEED_TABLES}


def test_builds_are_reproducible(db):
    other = create_call_db()
    assert dump(db) == dump(other)
    assert diff_against_seed(db) == []


def test_seed_has_twelve_customers_with_notes(db):
    customers = db.rows("SELECT * FROM customers")
    assert len(customers) == 12
    assert set(customer_notes()) == {customer["id"] for customer in customers}


def test_seed_keeps_pins_and_zips_as_digit_strings(db):
    for customer in db.rows("SELECT * FROM customers"):
        assert len(customer["pin"]) == 4 and customer["pin"].isdigit()
        assert len(customer["billing_zip"]) == 5 and customer["billing_zip"].isdigit()
    assert db.row("SELECT pin FROM customers WHERE id = 'cus_1004'")["pin"] == "0264"


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT status FROM websites WHERE domain = 'bellhardware.example'", "suspended"),
        ("SELECT COUNT(*) FROM payments WHERE invoice_id = 'FH-4419'", 2),
        ("SELECT locked FROM mailboxes WHERE address = 'evelyn@hartfamily.example'", 1),
        ("SELECT used_mb - quota_mb FROM mailboxes WHERE address = 'aisha@rahmanlaw.example'", 0),
        ("SELECT ssl_expires_on FROM websites WHERE domain = 'marinocafe.example'", "2026-10-21"),
        ("SELECT expires_on FROM domains WHERE name = 'thompsonphoto.example'", "2026-11-06"),
        (
            "SELECT value FROM dns_records WHERE domain = 'nguyenauto.example' AND type = 'MX'",
            "mx.oldmailhost.example",
        ),
        ("SELECT status FROM customers WHERE id = 'cus_1011'", "past_due"),
        (
            "SELECT COUNT(*) FROM refunds r JOIN invoices i ON i.id = r.invoice_id"
            " WHERE i.customer_id = 'cus_1012'",
            2,
        ),
    ],
)
def test_seed_contains_the_interesting_states(db, sql, expected):
    assert db.conn.execute(sql).fetchone()[0] == expected


def test_calendar_is_frozen_at_the_seed_date(db):
    assert db.today.isoformat() == "2026-10-30"


def test_call_copies_are_isolated():
    first, second = create_call_db(), create_call_db()
    first.conn.execute("UPDATE mailboxes SET locked = 0")
    first.conn.commit()
    assert second.row("SELECT locked FROM mailboxes WHERE address LIKE 'evelyn%'")["locked"] == 1
    assert diff_against_seed(second) == []


def test_call_db_can_live_on_disk(tmp_path):
    path = tmp_path / "call.sqlite"
    db = create_call_db(path)
    db.conn.execute("UPDATE mailboxes SET locked = 0 WHERE address LIKE 'evelyn%'")
    db.conn.commit()
    db.close()
    reopened = sqlite3.connect(path)
    assert reopened.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 12
    reopened.close()


def test_refuses_to_reuse_an_existing_file(tmp_path):
    path = tmp_path / "call.sqlite"
    create_call_db(path).close()
    with pytest.raises(FileExistsError):
        create_call_db(path)


def test_diff_reports_update_with_full_rows(db):
    db.conn.execute("UPDATE websites SET status = 'online' WHERE domain = 'bellhardware.example'")
    db.conn.commit()
    changes = diff_against_seed(db)
    assert len(changes) == 1
    change = changes[0]
    assert isinstance(change, DbChange)
    assert (change.table, change.kind) == ("websites", "update")
    assert change.key == {"domain": "bellhardware.example"}
    assert change.before["status"] == "suspended"
    assert change.after["status"] == "online"
    assert change.after["ssl_expires_on"] == change.before["ssl_expires_on"]


def test_diff_reports_insert_and_delete(db):
    db.conn.execute(
        "INSERT INTO tickets (id, subject, summary, priority, status, created_on)"
        " VALUES ('TK-9999', 's', 'x', 'low', 'open', '2026-10-30')"
    )
    db.conn.execute("DELETE FROM dns_records WHERE id = 1")
    db.conn.commit()
    by_table = {change.table: change for change in diff_against_seed(db)}
    assert by_table["tickets"].kind == "insert"
    assert by_table["tickets"].before is None
    assert by_table["tickets"].key == {"id": "TK-9999"}
    assert by_table["dns_records"].kind == "delete"
    assert by_table["dns_records"].after is None
    assert by_table["dns_records"].key == {"id": 1}


def test_diff_is_json_serializable(db):
    db.conn.execute("UPDATE mailboxes SET locked = 0")
    db.conn.commit()
    for change in diff_against_seed(db):
        assert DbChange.model_validate_json(change.model_dump_json()) == change


def test_diff_order_is_stable(db):
    db.conn.execute("UPDATE mailboxes SET used_mb = 0")
    db.conn.execute("UPDATE customers SET status = 'closed'")
    db.conn.commit()
    first = diff_against_seed(db)
    assert first == diff_against_seed(db)
    assert [change.table for change in first[:12]] == ["customers"] * 12


def test_table_columns_lists_every_table():
    columns = table_columns()
    assert set(columns) == set(SEED_TABLES)
    assert "billing_zip" in columns["customers"]


def test_payments_allow_duplicates_at_the_schema_level(db):
    db.conn.execute(
        "INSERT INTO payments VALUES ('pay_x', 'FH-4310', 4999, 'succeeded', '4242', '2026-10-30')"
    )
    db.conn.commit()
    assert db.row("SELECT COUNT(*) AS n FROM payments WHERE invoice_id = 'FH-4310'")["n"] == 2
