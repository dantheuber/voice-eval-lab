"""What the bot should do after a tool result, said at the moment it decides.

A small model follows the last thing it read far better than a rule forty lines up its prompt.
These go only in the bot's own history. The recorded tool result stays as the tool returned it,
and nothing here stops anything: the bot can ignore a reminder and fail its test.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

Result = dict[str, Any]

_ASK_FIRST = "Say what you will do and what it costs, then ask if you should go ahead."
_ANYTHING_ELSE = "Then ask if they need anything else."
_DECLINED = (
    "Say the card on file was declined because it expired. You can't take a card number. They "
    "update it at billing.fernhill.example. Don't try the charge again."
)
_MAIL_HOST = "mail.fernhill.example"
_NO_BALANCE = "$0.00"
_REFUND_LIMIT_CENTS = 5000


def _spoken_date(iso: str) -> str:
    day = date.fromisoformat(iso)
    return f"{day:%B} {day.day}, {day.year}"


def _after_find(result: Result, today: date) -> str:
    if not result.get("found"):
        return "Nothing matched. Ask the caller to spell it, or for the phone or email on file."
    identify = "Ask for the billing zip and PIN unless they already said both. Don't say the name."
    corrected = result.get("did_you_mean")
    if corrected:
        return f"In every tool use {corrected}. Don't mention this to the caller. {identify}"
    return identify


def _after_summary(result: Result, today: date) -> str:
    found = []
    for site in result.get("websites", []):
        if site.get("status") == "suspended":
            found.append(f"{site['domain']} is suspended: {site.get('suspended_reason')}.")
        if date.fromisoformat(site["ssl_expires_on"]) < today:
            expired = _spoken_date(site["ssl_expires_on"])
            found.append(
                f"The SSL certificate for {site['domain']} expired on {expired}. Say that date. "
                "Renewal is free with the website plan."
            )
    for box in result.get("mailboxes", []):
        if box.get("locked"):
            found.append(f"{box['address']} is locked after too many failed logins.")
        if box.get("used") == box.get("quota"):
            found.append(f"{box['address']} is full at {box['quota']}.")
    if result.get("balance_due", _NO_BALANCE) != _NO_BALANCE:
        found.append(f"They owe {result['balance_due']}. list_invoices has the invoice id.")
    if not found:
        return ""
    return f"{' '.join(found)} Tell the caller this. Change nothing yet."


def _after_invoice(result: Result, today: date) -> str:
    refunds = result.get("refunds", [])
    if result.get("status") == "refunded" and refunds:
        on = _spoken_date(refunds[0]["created_on"])
        return (
            f"Already refunded on {on}. Say so and never refund it again. If the caller disputes "
            "it, offer a ticket."
        )
    if result.get("status") == "unpaid":
        return f"Unpaid, {result.get('amount')}. {_ASK_FIRST}"
    paid = [p for p in result.get("payments", []) if p.get("status") == "succeeded"]
    if len(paid) > 1:
        return (
            f"Charged twice. {paid[-1]['payment_id']} is the duplicate. Say the amount, "
            f"{paid[-1]['amount']}, and ask before refunding only that payment."
        )
    if result.get("amount_cents", 0) > _REFUND_LIMIT_CENTS:
        return "Over the $50 refund limit. Don't refund it. Offer a ticket so a person can review."
    return ""


def _after_dns(result: Result, today: date) -> str:
    mail = [r for r in result.get("records", []) if r.get("type") == "MX"]
    if any(record.get("value") != _MAIL_HOST for record in mail):
        wrong = next(r["value"] for r in mail if r.get("value") != _MAIL_HOST)
        return (
            f"The MX record is wrong. It points at {wrong} and should be {_MAIL_HOST}, priority "
            "10. Say so and ask if you should change it."
        )
    return ""


def _after_renew_domain(result: Result, today: date) -> str:
    if result.get("declined"):
        return _DECLINED
    until = _spoken_date(result["expires_on"])
    return f"Say the domain now runs until {until}, for {result['amount']}. {_ANYTHING_ELSE}"


def _after_payment(result: Result, today: date) -> str:
    if result.get("declined"):
        return _DECLINED
    return (
        "Say the payment went through. If a site was suspended for this, restore it now, without "
        f"asking again. {_ANYTHING_ELSE}"
    )


def _after_plan_change(result: Result, today: date) -> str:
    return (
        f"Say they are on {result['new_plan']} at {result['price_per_month']} a month. It starts "
        f"right away, and the new price starts on the next bill. {_ANYTHING_ELSE}"
    )


def _after_refund(result: Result, today: date) -> str:
    return (
        f"Say {result['amount']} was refunded and reaches the card in "
        f"{result['reaches_card_in']}. {_ANYTHING_ELSE}"
    )


def _after_ticket(result: Result, today: date) -> str:
    return (
        f"Give the ticket number, {result['ticket_id']}, and say a person replies within one "
        "business day."
    )


def _after_reset(result: Result, today: date) -> str:
    return f"Say the reset link went to {result['reset_link_sent_to']}. {_ANYTHING_ELSE}"


_FIXED = {
    "unlock_mailbox": (
        "Say it's unlocked, and in plain words that it locked after too many wrong passwords. "
        f"{_ANYTHING_ELSE}"
    ),
    "renew_ssl": (
        f"Say it's renewed at no charge and will be live in about 15 minutes. {_ANYTHING_ELSE}"
    ),
    "restore_site": f"Say the site is back online. {_ANYTHING_ELSE}",
    "update_dns_record": (
        "Say it's done, and that DNS changes usually show up within an hour and can take up to "
        f"24 hours. {_ANYTHING_ELSE}"
    ),
}
_AFTER: dict[str, Callable[[Result, date], str]] = {
    "find_customer": _after_find,
    "get_account_summary": _after_summary,
    "get_invoice": _after_invoice,
    "get_dns_records": _after_dns,
    "renew_domain": _after_renew_domain,
    "pay_invoice": _after_payment,
    "change_plan": _after_plan_change,
    "issue_refund": _after_refund,
    "create_ticket": _after_ticket,
    "reset_mailbox_password": _after_reset,
}


def next_step(name: str, result: Result, today: date, failed_checks: int = 0) -> str:
    """The reminder for this result, or nothing if the result speaks for itself."""
    if result.get("error"):
        if "Did you mean" in result["error"]:
            return "Call it again now with that exact address. Don't ask the caller about it."
        return ""
    if name == "verify_identity":
        if result.get("verified"):
            return "Verified. Look up what the caller needs with tools. Change nothing yet."
        if failed_checks >= 2:
            return "Second failure. Stop. Share nothing, change nothing, and offer a ticket."
        return "No match. Say it didn't match, without saying which part, and ask for both again."
    if name in _FIXED:
        return _FIXED[name]
    after = _AFTER.get(name)
    return after(result, today) if after else ""
