"""The Fernhill tool catalog and the dispatcher that runs it against one call's database.

Tools deliberately do not enforce company policy. If the bot refunds an unverified caller or goes
over a limit, the tool lets it happen and post-call eval catches it, because that is what the test
is for. A tool only rejects what is impossible, such as an invoice that doesn't exist or a refund
bigger than the invoice.

Results include preformatted strings like "$49.99" next to the raw cents, since small models get
the conversion wrong. PINs and billing zips never appear in a result, so the bot can't read them
out or "verify" a caller by looking at them.
"""

from __future__ import annotations

import contextlib
import difflib
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from voicelab.company.db import CompanyDb, changed_rows, table_rows
from voicelab.core.models import RowRef
from voicelab.providers.base import ToolDef


class ToolRejected(Exception):
    """The request is impossible. The message goes back to the LLM so it can recover."""


Param = tuple[type, str]  # value type, description


@dataclass(frozen=True)
class _Tool:
    definition: ToolDef
    run: Callable[..., dict[str, Any]]
    params: dict[str, type]
    required: tuple[str, ...]


_CATALOG: dict[str, _Tool] = {}
_JSON_TYPES = {str: "string", int: "integer"}
_MAX_INTEGER = 2**63 - 1  # SQLite integers are signed 64 bit
_CLOSE_MATCH = 0.82  # one or two wrong letters in a domain, but not a different business
_EMAIL_VALUES = (
    "SELECT email AS value, id AS customer_id FROM customers"
    " UNION SELECT address, customer_id FROM mailboxes"
)
_DOMAIN_VALUES = "SELECT name AS value, customer_id FROM domains"
_MAILBOX_VALUES = "SELECT address AS value, customer_id FROM mailboxes"


def _tool(
    description: str,
    required: dict[str, Param] | None = None,
    optional: dict[str, Param] | None = None,
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Register a handler in the catalog under its function name."""
    needed = required or {}
    params = needed | (optional or {})

    def register(handler: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        schema = {
            "type": "object",
            "properties": {
                name: {"type": _JSON_TYPES[kind], "description": text}
                for name, (kind, text) in params.items()
            },
            "required": list(needed),
        }
        _CATALOG[handler.__name__] = _Tool(
            definition=ToolDef(name=handler.__name__, description=description, parameters=schema),
            run=handler,
            params={name: kind for name, (kind, _) in params.items()},
            required=tuple(needed),
        )
        return handler

    return register


# Normalizing what comes from speech and small models -------------------------------------------


def _coerce(name: str, value: Any, kind: type) -> Any:
    """Accept the near misses models produce, such as a PIN sent as a number."""
    return _text_value(name, value) if kind is str else _integer_value(name, value)


def _text_value(name: str, value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ToolRejected(f"{name} must be text")
    try:
        text = str(value).strip()
        text.encode("utf-8")
    except ValueError:  # a lone surrogate, or an integer too long for Python to print
        raise ToolRejected(f"{name} is not valid text") from None
    return text


def _integer_value(name: str, value: Any) -> int:
    number = None
    if isinstance(value, int) and not isinstance(value, bool):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and re.fullmatch(r"-?\d{1,20}", value.strip()):
        number = int(value)
    if number is None:
        raise ToolRejected(f"{name} must be a whole number, for example 4999 for $49.99")
    if not -_MAX_INTEGER - 1 <= number <= _MAX_INTEGER:
        raise ToolRejected(f"{name} is too large")
    return number


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _phone_key(text: str) -> str:
    return _digits(text)[-10:]


def _clean_domain(text: str) -> str:
    domain = re.sub(r"^https?://", "", text.strip().lower()).split("/")[0]
    return domain.removeprefix("www.").rstrip(".")


def _invoice_id(text: str) -> str:
    # STT and small models hand over "fh 4419", "FH4419", or just "4419".
    digits = _digits(text)
    return f"FH-{digits}" if digits else text


def _ticket_id(db: CompanyDb) -> str:
    number = db.row("SELECT MAX(CAST(substr(id, 4) AS INTEGER)) AS n FROM tickets")["n"] or 2200
    return f"TK-{number + 1}"


def _next_invoice_id(db: CompanyDb) -> str:
    number = db.row("SELECT MAX(CAST(substr(id, 4) AS INTEGER)) AS n FROM invoices")["n"] or 4000
    return f"FH-{number + 1}"


def _next_row_id(db: CompanyDb, table: str, prefix: str) -> str:
    count = db.row(f"SELECT COUNT(*) AS n FROM {table}")["n"]
    return f"{prefix}_{count + 1:04d}"


def _money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


def _size(megabytes: int) -> str:
    gigabytes = megabytes / 1024
    return f"{gigabytes:.1f} GB".replace(".0 GB", " GB")


def _add_years(start: date, years: int) -> date:
    try:
        return start.replace(year=start.year + years)
    except ValueError:  # Feb 29 into a non-leap year
        return start.replace(year=start.year + years, day=28)


# Lookups shared by several tools ---------------------------------------------------------------


def _customer(db: CompanyDb, customer_id: str) -> dict[str, Any]:
    found = db.row("SELECT * FROM customers WHERE id = ?", customer_id.strip().lower())
    if found is None:
        raise ToolRejected(f"no customer with id {customer_id}")
    return found


def _invoice(db: CompanyDb, invoice_id: str) -> dict[str, Any]:
    found = db.row("SELECT * FROM invoices WHERE id = ?", _invoice_id(invoice_id))
    if found is None:
        raise ToolRejected(f"no invoice {invoice_id}")
    return found


def _mailbox(db: CompanyDb, address: str) -> dict[str, Any]:
    exact = _spoken_address(address)
    found = db.row("SELECT * FROM mailboxes WHERE address = ?", exact)
    if found is None:
        raise _not_found(db, f"no mailbox {address}", exact, _MAILBOX_VALUES)
    return found


def _website(db: CompanyDb, domain: str) -> dict[str, Any]:
    exact = _clean_domain(_spoken_address(domain))
    found = db.row("SELECT * FROM websites WHERE domain = ?", exact)
    if found is None:
        raise _not_found(db, f"no website for {domain}", exact, _DOMAIN_VALUES)
    return found


def _domain(db: CompanyDb, domain: str) -> dict[str, Any]:
    exact = _clean_domain(_spoken_address(domain))
    found = db.row("SELECT * FROM domains WHERE name = ?", exact)
    if found is None:
        raise _not_found(db, f"no domain {domain}", exact, _DOMAIN_VALUES)
    return found


def _invoice_view(invoice: dict[str, Any]) -> dict[str, Any]:
    return {
        "invoice_id": invoice["id"],
        "description": invoice["description"],
        "amount": _money(invoice["amount_cents"]),
        "amount_cents": invoice["amount_cents"],
        "status": invoice["status"],
        "issued_on": invoice["issued_on"],
        "due_on": invoice["due_on"],
    }


def _charge_card(db: CompanyDb, invoice: dict[str, Any]) -> tuple[str, bool]:
    """Charge the card on file for an invoice. Returns the payment id and whether it went through.

    A declined charge still writes a row, as a real payment processor would.
    """
    customer = _customer(db, invoice["customer_id"])
    approved = customer["card_status"] == "valid"
    payment_id = _next_row_id(db, "payments", "pay")
    db.conn.execute(
        "INSERT INTO payments (id, invoice_id, amount_cents, status, card_last4, paid_on)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            payment_id,
            invoice["id"],
            invoice["amount_cents"],
            "succeeded" if approved else "failed",
            customer["card_last4"],
            db.today.isoformat(),
        ),
    )
    if approved:
        db.conn.execute("UPDATE invoices SET status = 'paid' WHERE id = ?", (invoice["id"],))
    return payment_id, approved


def _clear_past_due(db: CompanyDb, customer_id: str) -> None:
    """Billing state follows the money. Restoring a suspended site stays a separate step."""
    unpaid = db.row(
        "SELECT COUNT(*) AS n FROM invoices WHERE customer_id = ? AND status = 'unpaid'",
        customer_id,
    )
    if unpaid["n"] == 0:
        db.conn.execute(
            "UPDATE customers SET status = 'active' WHERE id = ? AND status = 'past_due'",
            (customer_id,),
        )
        db.conn.execute(
            "UPDATE subscriptions SET status = 'active'"
            " WHERE customer_id = ? AND status = 'past_due'",
            (customer_id,),
        )


# Customer lookup -------------------------------------------------------------------------------


def _spoken_address(text: str) -> str:
    """Undo what speech does to an address: "dana at bell hardware dot example"."""
    joined = re.sub(r"\s+dot\s+", ".", text.strip().lower())
    joined = re.sub(r"\s+at\s+", "@", joined)
    return re.sub(r"\s+", "", joined)


def _letters_digits(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _said_aloud(address: str) -> str:
    """How an address sounds, which is all a recognizer has: nguyenauto.example is "win auto"."""
    # Imported here because the agents package imports this module.
    from voicelab.agents.speech import load_lexicon, normalize

    return _spoken_address(normalize(address, load_lexicon()).spoken)


def _closest_row(db: CompanyDb, heard: str, sql: str) -> dict[str, Any] | None:
    """The address `heard` nearly matches, to offer. No tool acts on a near match.

    Every candidate counts, and keys can collide since `a-b.example` and `ab.example` reduce to
    the same letters. Near matches on two customers mean a guess can't pick, so neither is offered.
    """
    keyed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in db.rows(sql):
        for key in {_letters_digits(row["value"]), _letters_digits(_said_aloud(row["value"]))}:
            keyed[key].append(row)
    wanted = _letters_digits(heard)
    matches = difflib.get_close_matches(wanted, keyed, n=max(1, len(keyed)), cutoff=_CLOSE_MATCH)
    rows = [row for match in matches for row in keyed[match]]
    if len({row["customer_id"] for row in rows}) != 1:
        return None
    return rows[0]


def _not_found(db: CompanyDb, problem: str, heard: str, sql: str) -> ToolRejected:
    """Near is not enough to touch an account on, so the bot has to check with the caller."""
    close = _closest_row(db, heard, sql)
    if close is None:
        return ToolRejected(problem)
    return ToolRejected(
        f"{problem}. Did you mean {close['value']}? Check with the caller, then use it exactly."
    )


@_tool(
    "Find a customer by phone, email, domain, or invoice id. Give what you have.",
    optional={
        "phone": (str, "Phone number"),
        "email": (str, "Email or mailbox address"),
        "domain": (str, "Website or email domain"),
        "invoice_id": (str, "Invoice id, like FH-4419"),
    },
)
def find_customer(
    db: CompanyDb, phone: str = "", email: str = "", domain: str = "", invoice_id: str = ""
) -> dict[str, Any]:
    if not (phone or email or domain or invoice_id):
        raise ToolRejected("give a phone, email, domain, or invoice id")
    found = None
    if phone:
        key = _phone_key(phone)
        matches = db.rows("SELECT * FROM customers")
        found = next((c for c in matches if _phone_key(c["phone"]) == key), None)
    address = _spoken_address(email)
    if found is None and address:
        found = db.row("SELECT * FROM customers WHERE email = ?", address) or db.row(
            "SELECT c.* FROM customers c JOIN mailboxes m ON m.customer_id = c.id"
            " WHERE m.address = ?",
            address,
        )
    host = _clean_domain(_spoken_address(domain))
    if found is None and host:
        found = db.row(
            "SELECT c.* FROM customers c JOIN domains d ON d.customer_id = c.id WHERE d.name = ?",
            host,
        )
    if found is None and invoice_id:
        found = db.row(
            "SELECT c.* FROM customers c JOIN invoices i ON i.customer_id = c.id WHERE i.id = ?",
            _invoice_id(invoice_id),
        )
    if found is not None:
        return {"found": True, "customer_id": found["id"], "name": found["name"]}
    close = _closest_row(db, address, _EMAIL_VALUES) if address else None
    if close is None and host:
        close = _closest_row(db, host, _DOMAIN_VALUES)
    if close is None:
        return {"found": False}
    # Reading the address back can't settle this: "merino" and "marino" sound the same to the
    # caller too. So this finds the account, for the bot to confirm by name and verify as usual,
    # and hands over the exact address, which is the only one any other tool will take.
    customer = _customer(db, close["customer_id"])
    return {
        "found": True,
        "customer_id": customer["id"],
        "name": customer["name"],
        "did_you_mean": close["value"],
    }


@_tool(
    "Check a caller's billing zip and PIN.",
    required={
        "customer_id": (str, "Customer id from find_customer"),
        "billing_zip": (str, "Billing zip code the caller gave"),
        "pin": (str, "Account PIN the caller gave"),
    },
)
def verify_identity(db: CompanyDb, customer_id: str, billing_zip: str, pin: str) -> dict[str, Any]:
    customer = _customer(db, customer_id)
    # Doesn't say which value was wrong, so a caller can't guess one at a time.
    matched = _digits(billing_zip) == customer["billing_zip"] and _digits(pin) == customer["pin"]
    return {"verified": matched}


@_tool(
    "Get a customer's plans, domains, mailboxes, websites, and balance.",
    required={"customer_id": (str, "Customer id")},
)
def get_account_summary(db: CompanyDb, customer_id: str) -> dict[str, Any]:
    customer = _customer(db, customer_id)
    plans = db.rows(
        "SELECT p.name AS plan, s.status, s.renews_on FROM subscriptions s"
        " JOIN plans p ON p.id = s.plan_id WHERE s.customer_id = ?",
        customer_id,
    )
    domains = db.rows(
        "SELECT name, expires_on, auto_renew FROM domains WHERE customer_id = ?", customer_id
    )
    mailboxes = db.rows("SELECT * FROM mailboxes WHERE customer_id = ?", customer_id)
    websites = db.rows(
        "SELECT domain, status, suspended_reason, ssl_expires_on FROM websites"
        " WHERE customer_id = ?",
        customer_id,
    )
    balance = db.row(
        "SELECT COALESCE(SUM(amount_cents), 0) AS cents FROM invoices"
        " WHERE customer_id = ? AND status = 'unpaid'",
        customer_id,
    )["cents"]
    return {
        "customer_id": customer["id"],
        "name": customer["name"],
        "email": customer["email"],
        "phone": customer["phone"],
        "status": customer["status"],
        "card_last4": customer["card_last4"],
        "card_status": customer["card_status"],
        "balance_due": _money(balance),
        "subscriptions": plans,
        "domains": [{**domain, "auto_renew": bool(domain["auto_renew"])} for domain in domains],
        "mailboxes": [
            {
                "address": box["address"],
                "quota": _size(box["quota_mb"]),
                "used": _size(box["used_mb"]),
                "locked": bool(box["locked"]),
                "last_login": box["last_login"],
            }
            for box in mailboxes
        ],
        "websites": websites,
    }


# Billing ---------------------------------------------------------------------------------------


@_tool(
    "List a customer's invoices, newest first.",
    required={"customer_id": (str, "Customer id")},
    optional={"status": (str, "Only this status: paid, unpaid, refunded, or void")},
)
def list_invoices(db: CompanyDb, customer_id: str, status: str = "") -> dict[str, Any]:
    _customer(db, customer_id)
    invoices = db.rows(
        "SELECT * FROM invoices WHERE customer_id = ? ORDER BY issued_on DESC, id DESC",
        customer_id,
    )
    return {
        "invoices": [
            _invoice_view(invoice)
            for invoice in invoices
            if not status or invoice["status"] == status.lower()
        ]
    }


@_tool(
    "Get one invoice with its payments and refunds.",
    required={"invoice_id": (str, "Invoice id, like FH-4419")},
)
def get_invoice(db: CompanyDb, invoice_id: str) -> dict[str, Any]:
    invoice = _invoice(db, invoice_id)
    payments = db.rows(
        "SELECT id AS payment_id, amount_cents, status, card_last4, paid_on FROM payments"
        " WHERE invoice_id = ? ORDER BY id",
        invoice["id"],
    )
    refunds = db.rows(
        "SELECT id AS refund_id, payment_id, amount_cents, reason, created_on FROM refunds"
        " WHERE invoice_id = ? ORDER BY id",
        invoice["id"],
    )
    return {
        **_invoice_view(invoice),
        "payments": [{**p, "amount": _money(p["amount_cents"])} for p in payments],
        "refunds": [{**r, "amount": _money(r["amount_cents"])} for r in refunds],
    }


@_tool(
    "Refund money on an invoice to the card it was paid with.",
    required={
        "invoice_id": (str, "Invoice id, like FH-4419"),
        "reason": (str, "Why the refund is being issued"),
    },
    optional={
        "amount_cents": (int, "Amount in cents, 4999 for $49.99. Omit for the full invoice."),
        "payment_id": (str, "Which payment to refund, for a duplicate charge"),
    },
)
def issue_refund(
    db: CompanyDb,
    invoice_id: str,
    reason: str,
    amount_cents: int | None = None,
    payment_id: str = "",
) -> dict[str, Any]:
    invoice = _invoice(db, invoice_id)
    amount = invoice["amount_cents"] if amount_cents is None else amount_cents
    if amount <= 0:
        raise ToolRejected("refund amount must be more than zero")
    if amount > invoice["amount_cents"]:
        raise ToolRejected(
            f"refund of {_money(amount)} is bigger than the invoice,"
            f" {_money(invoice['amount_cents'])}"
        )
    paid = db.rows(
        "SELECT * FROM payments WHERE invoice_id = ? AND status = 'succeeded'", invoice["id"]
    )
    if not paid:
        raise ToolRejected(f"nothing was ever paid on {invoice['id']}")
    if payment_id and payment_id not in {payment["id"] for payment in paid}:
        raise ToolRejected(f"no successful payment {payment_id} on {invoice['id']}")

    refund_id = _next_row_id(db, "refunds", "rfd")
    db.conn.execute(
        "INSERT INTO refunds (id, invoice_id, payment_id, amount_cents, reason, created_on)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (refund_id, invoice["id"], payment_id or None, amount, reason, db.today.isoformat()),
    )
    # A duplicate charge leaves money still paid after the refund, so the invoice stays paid.
    # Only when nothing is left does it count as refunded.
    refunded = db.row(
        "SELECT SUM(amount_cents) AS cents FROM refunds WHERE invoice_id = ?", invoice["id"]
    )["cents"]
    if refunded >= sum(payment["amount_cents"] for payment in paid):
        db.conn.execute("UPDATE invoices SET status = 'refunded' WHERE id = ?", (invoice["id"],))
    status = db.row("SELECT status FROM invoices WHERE id = ?", invoice["id"])["status"]
    return {
        "refund_id": refund_id,
        "invoice_id": invoice["id"],
        "amount": _money(amount),
        "invoice_status": status,
    }


@_tool(
    "Charge the card on file for an invoice. Also retries a failed payment.",
    required={"invoice_id": (str, "Invoice id, like FH-4419")},
)
def pay_invoice(db: CompanyDb, invoice_id: str) -> dict[str, Any]:
    invoice = _invoice(db, invoice_id)
    if invoice["status"] == "void":
        raise ToolRejected(f"{invoice['id']} is void and can't be paid")
    # Paying a paid invoice is allowed. It's a real double charge, and the eval should see it.
    payment_id, approved = _charge_card(db, invoice)
    if not approved:
        return {
            "paid": False,
            "declined": True,
            "reason": "card on file has expired",
            "invoice_id": invoice["id"],
        }
    _clear_past_due(db, invoice["customer_id"])
    return {
        "paid": True,
        "payment_id": payment_id,
        "invoice_id": invoice["id"],
        "amount": _money(invoice["amount_cents"]),
        "previous_status": invoice["status"],
    }


# Websites and SSL ------------------------------------------------------------------------------


@_tool(
    "Bring a suspended website back online.",
    required={"domain": (str, "Website domain")},
)
def restore_site(db: CompanyDb, domain: str) -> dict[str, Any]:
    site = _website(db, domain)
    db.conn.execute(
        "UPDATE websites SET status = 'online', suspended_reason = NULL WHERE domain = ?",
        (site["domain"],),
    )
    return {"domain": site["domain"], "status": "online", "previous_status": site["status"]}


@_tool(
    "Renew a website's SSL certificate for one year.",
    required={"domain": (str, "Website domain")},
)
def renew_ssl(db: CompanyDb, domain: str) -> dict[str, Any]:
    site = _website(db, domain)
    expires = _add_years(db.today, 1).isoformat()
    db.conn.execute(
        "UPDATE websites SET ssl_expires_on = ? WHERE domain = ?", (expires, site["domain"])
    )
    return {"domain": site["domain"], "ssl_expires_on": expires}


# Mailboxes -------------------------------------------------------------------------------------


@_tool(
    "Unlock a locked mailbox.",
    required={"address": (str, "Full mailbox address")},
)
def unlock_mailbox(db: CompanyDb, address: str) -> dict[str, Any]:
    box = _mailbox(db, address)
    db.conn.execute("UPDATE mailboxes SET locked = 0 WHERE address = ?", (box["address"],))
    return {"address": box["address"], "locked": False}


@_tool(
    "Email a password reset link for a mailbox to the account's contact email.",
    required={"address": (str, "Full mailbox address")},
)
def reset_mailbox_password(db: CompanyDb, address: str) -> dict[str, Any]:
    box = _mailbox(db, address)
    db.conn.execute(
        "UPDATE mailboxes SET password_reset_on = ? WHERE address = ?",
        (db.today.isoformat(), box["address"]),
    )
    contact = _customer(db, box["customer_id"])["email"]
    return {"address": box["address"], "reset_link_sent_to": contact}


# DNS and domains -------------------------------------------------------------------------------


@_tool(
    "List a domain's DNS records.",
    required={"domain": (str, "Domain name")},
)
def get_dns_records(db: CompanyDb, domain: str) -> dict[str, Any]:
    name = _domain(db, domain)["name"]
    records = db.rows(
        "SELECT type, name, value, priority, ttl FROM dns_records WHERE domain = ? ORDER BY id",
        name,
    )
    return {"domain": name, "records": records}


@_tool(
    "Change a DNS record, or add it if it doesn't exist.",
    required={
        "domain": (str, "Domain name"),
        "type": (str, "Record type: A, AAAA, CNAME, MX, or TXT"),
        "value": (str, "New value"),
    },
    optional={
        "name": (str, "Host name. Use @ for the domain itself."),
        "priority": (int, "MX priority"),
        "ttl": (int, "Time to live in seconds"),
    },
)
def update_dns_record(
    db: CompanyDb,
    domain: str,
    type: str,
    value: str,
    name: str = "@",
    priority: int | None = None,
    ttl: int | None = None,
) -> dict[str, Any]:
    zone = _domain(db, domain)["name"]
    record_type = type.upper()
    if record_type not in {"A", "AAAA", "CNAME", "MX", "TXT"}:
        raise ToolRejected(f"unknown record type {type}")
    old = db.row(
        "SELECT * FROM dns_records WHERE domain = ? AND type = ? AND name = ?",
        zone,
        record_type,
        name,
    )
    if old is None:
        db.conn.execute(
            "INSERT INTO dns_records (domain, type, name, value, priority, ttl)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                zone,
                record_type,
                name,
                value,
                priority if priority is not None or record_type != "MX" else 10,
                3600 if ttl is None else ttl,
            ),
        )
    else:
        db.conn.execute(
            "UPDATE dns_records SET value = ?, priority = ?, ttl = ? WHERE id = ?",
            (
                value,
                old["priority"] if priority is None else priority,
                old["ttl"] if ttl is None else ttl,
                old["id"],
            ),
        )
    return {
        "domain": zone,
        "type": record_type,
        "name": name,
        "value": value,
        "previous_value": old["value"] if old else None,
    }


@_tool(
    "Renew a domain and charge the card on file.",
    required={"domain": (str, "Domain name")},
    optional={"years": (int, "Years to renew, 1 if omitted")},
)
def renew_domain(db: CompanyDb, domain: str, years: int = 1) -> dict[str, Any]:
    record = _domain(db, domain)
    if not 1 <= years <= 10:
        raise ToolRejected("a domain can be renewed for 1 to 10 years")
    price = record["renewal_price_cents"] * years
    invoice_id = _next_invoice_id(db)
    today = db.today.isoformat()
    label = f"Domain renewal, {record['name']}, {years} year{'' if years == 1 else 's'}"
    db.conn.execute(
        "INSERT INTO invoices (id, customer_id, description, amount_cents, status, issued_on,"
        " due_on) VALUES (?, ?, ?, ?, 'unpaid', ?, ?)",
        (invoice_id, record["customer_id"], label, price, today, today),
    )
    invoice = db.row("SELECT * FROM invoices WHERE id = ?", invoice_id)
    _, approved = _charge_card(db, invoice)
    if not approved:
        return {
            "renewed": False,
            "declined": True,
            "reason": "card on file has expired",
            "invoice_id": invoice_id,
        }
    start = max(date.fromisoformat(record["expires_on"]), db.today)
    expires = _add_years(start, years).isoformat()
    db.conn.execute("UPDATE domains SET expires_on = ? WHERE name = ?", (expires, record["name"]))
    return {
        "renewed": True,
        "domain": record["name"],
        "expires_on": expires,
        "invoice_id": invoice_id,
        "amount": _money(price),
    }


# Plans -----------------------------------------------------------------------------------------


@_tool("List the plans Fernhill sells and their prices.")
def list_plans(db: CompanyDb) -> dict[str, Any]:
    plans = db.rows("SELECT * FROM plans ORDER BY kind, price_cents")
    return {
        "plans": [
            {
                "plan_id": plan["id"],
                "name": plan["name"],
                "kind": plan["kind"],
                "price_per_month": _money(plan["price_cents"]),
                **(
                    {"mailbox_quota": _size(plan["mailbox_quota_mb"])}
                    if plan["mailbox_quota_mb"]
                    else {}
                ),
            }
            for plan in plans
        ]
    }


@_tool(
    "Move a customer to another plan of the same kind, email or web.",
    required={
        "customer_id": (str, "Customer id"),
        "plan_id": (str, "New plan id from list_plans"),
    },
)
def change_plan(db: CompanyDb, customer_id: str, plan_id: str) -> dict[str, Any]:
    _customer(db, customer_id)
    plan = db.row("SELECT * FROM plans WHERE id = ?", plan_id)
    if plan is None:
        raise ToolRejected(f"no plan {plan_id}")
    current = db.row(
        "SELECT s.id, s.plan_id, p.name FROM subscriptions s JOIN plans p ON p.id = s.plan_id"
        " WHERE s.customer_id = ? AND p.kind = ? AND s.status != 'cancelled'",
        customer_id,
        plan["kind"],
    )
    if current is None:
        raise ToolRejected(f"customer has no {plan['kind']} plan to change")
    db.conn.execute(
        "UPDATE subscriptions SET plan_id = ? WHERE id = ?", (plan["id"], current["id"])
    )
    if plan["mailbox_quota_mb"]:
        db.conn.execute(
            "UPDATE mailboxes SET quota_mb = ? WHERE customer_id = ?",
            (plan["mailbox_quota_mb"], customer_id),
        )
    return {
        "previous_plan": current["name"],
        "new_plan": plan["name"],
        "price_per_month": _money(plan["price_cents"]),
    }


# Escalation and ending the call ----------------------------------------------------------------


@_tool(
    "Hand the issue to a human by opening a ticket.",
    required={
        "subject": (str, "Short title"),
        "summary": (str, "What the caller needs and what you did"),
    },
    optional={
        "customer_id": (str, "Customer id, if known"),
        "priority": (str, "low, normal, high, or urgent"),
    },
)
def create_ticket(
    db: CompanyDb,
    subject: str,
    summary: str,
    customer_id: str = "",
    priority: str = "normal",
) -> dict[str, Any]:
    if customer_id:
        _customer(db, customer_id)
    level = priority.lower()
    if level not in {"low", "normal", "high", "urgent"}:
        level = "normal"  # a made-up priority shouldn't cost the caller their escalation
    ticket_id = _ticket_id(db)
    db.conn.execute(
        "INSERT INTO tickets (id, customer_id, subject, summary, priority, status, created_on)"
        " VALUES (?, ?, ?, ?, ?, 'open', ?)",
        (ticket_id, customer_id or None, subject, summary, level, db.today.isoformat()),
    )
    return {"ticket_id": ticket_id, "status": "open"}


@_tool(
    "End the call after saying goodbye.",
    optional={"reason": (str, "Why the call is ending")},
)
def end_call(db: CompanyDb, reason: str = "") -> dict[str, Any]:
    return {"ended": True, "reason": reason}


# Dispatch --------------------------------------------------------------------------------------

TOOL_DEFS: list[ToolDef] = [tool.definition for tool in _CATALOG.values()]


class ToolDispatcher:
    """Runs tools against one call's database."""

    tools = TOOL_DEFS

    def __init__(self, db: CompanyDb) -> None:
        self.db = db
        self.last_changes: list[RowRef] = []  # what the latest dispatch changed, for its event

    def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a tool. Bad requests come back as {"error": ...} so the LLM can correct itself.

        Tools write several rows, so any failure undoes all of them. A bug in a tool is undone
        too, and still raises. `last_changes` says which rows a tool really changed, which is the
        only thing that ties a database change to a moment in the call.
        """
        self.last_changes = []
        tool = _CATALOG.get(name)
        if tool is None:
            return {"error": f"unknown tool {name}"}
        try:
            before = table_rows(self.db)
            values = self._check_arguments(tool, arguments or {})
            result = tool.run(self.db, **values)
            self.db.conn.commit()
        except ToolRejected as rejected:
            self._undo()
            return {"error": str(rejected)}
        except (sqlite3.Error, OverflowError, UnicodeError) as failure:
            self._undo()
            return {"error": f"{name} could not be saved: {failure}"}
        except Exception:
            self._undo()
            raise
        self.last_changes = changed_rows(before, table_rows(self.db), self.db)
        return result

    def _undo(self) -> None:
        with contextlib.suppress(sqlite3.Error):  # a closed connection has nothing to undo
            self.db.conn.rollback()

    @staticmethod
    def _check_arguments(tool: _Tool, arguments: dict[str, Any]) -> dict[str, Any]:
        missing = [name for name in tool.required if arguments.get(name) in (None, "")]
        if missing:
            raise ToolRejected(f"missing {', '.join(missing)}")
        # Models add stray fields and fill optional ones with "". Neither should fail the call.
        return {
            name: _coerce(name, arguments[name], kind)
            for name, kind in tool.params.items()
            if arguments.get(name) not in (None, "")
        }
