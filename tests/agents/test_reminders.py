"""What the bot is reminded of after a tool, and that the record of the tool stays clean."""

from __future__ import annotations

from datetime import date

import pytest
from scripted_llm import ScriptedLLM, call, uses

from voicelab.agents.reminders import next_step
from voicelab.agents.support import SupportBot
from voicelab.company import ToolDispatcher, create_call_db
from voicelab.core.models import ToolCallEvent

TODAY = date(2026, 10, 30)


@pytest.fixture
def db():
    database = create_call_db()
    yield database
    database.close()


def test_a_misheard_address_is_corrected_without_asking_about_it():
    hint = next_step(
        "find_customer", {"found": True, "name": "A", "did_you_mean": "hart.example"}, TODAY
    )
    assert "In every tool use hart.example" in hint
    assert "Don't say the name" in hint


def test_a_failed_check_asks_again_once_then_stops():
    failed = {"verified": False}
    assert "ask for both again" in next_step("verify_identity", failed, TODAY, 1)
    assert "offer a ticket" in next_step("verify_identity", failed, TODAY, 2)


def test_verifying_doesnt_clear_the_bot_to_change_anything():
    assert "Change nothing yet" in next_step("verify_identity", {"verified": True}, TODAY)


def test_a_declined_charge_points_at_the_billing_site():
    assert "billing.fernhill.example" in next_step("pay_invoice", {"declined": True}, TODAY)


def test_an_expired_certificate_comes_with_its_date():
    summary = {
        "websites": [
            {"domain": "a.example", "status": "online", "ssl_expires_on": "2026-10-21"},
            {"domain": "b.example", "status": "online", "ssl_expires_on": "2027-01-01"},
        ]
    }
    hint = next_step("get_account_summary", summary, TODAY)
    assert "a.example expired on October 21, 2026" in hint
    assert "b.example" not in hint


def test_a_full_mailbox_is_named():
    summary = {"mailboxes": [{"address": "a@b.example", "quota": "5 GB", "used": "5 GB"}]}
    assert "a@b.example is full at 5 GB" in next_step("get_account_summary", summary, TODAY)


def test_a_duplicate_charge_names_the_later_payment():
    invoice = {
        "status": "paid",
        "payments": [
            {"payment_id": "pay_1", "status": "succeeded", "amount": "$49.99"},
            {"payment_id": "pay_2", "status": "succeeded", "amount": "$49.99"},
        ],
    }
    hint = next_step("get_invoice", invoice, TODAY)
    assert "pay_2 is the duplicate" in hint


def test_an_invoice_over_the_limit_points_at_a_ticket():
    invoice = {"status": "paid", "amount_cents": 18900, "payments": []}
    assert "ticket" in next_step("get_invoice", invoice, TODAY)


def test_a_refunded_invoice_gives_the_date():
    invoice = {"status": "refunded", "refunds": [{"created_on": "2026-09-12"}]}
    assert "Already refunded on September 12, 2026" in next_step("get_invoice", invoice, TODAY)


def test_a_wrong_mx_record_says_what_it_should_be():
    records = {"records": [{"type": "MX", "value": "mx.old.example"}]}
    hint = next_step("get_dns_records", records, TODAY)
    assert "mx.old.example" in hint and "mail.fernhill.example" in hint


def test_a_right_mx_record_needs_no_reminder():
    records = {"records": [{"type": "MX", "value": "mail.fernhill.example"}]}
    assert next_step("get_dns_records", records, TODAY) == ""


def test_an_error_needs_no_reminder():
    assert next_step("unlock_mailbox", {"error": "no mailbox"}, TODAY) == ""


async def test_the_reminder_reaches_the_model_but_not_the_record(db):
    events = []
    llm = ScriptedLLM(uses(call("find_customer", phone="(503) 555-0101")), "What's your PIN?")
    bot = SupportBot(llm, ToolDispatcher(db), on_event=events.append)
    bot.greet()
    async for _ in bot.respond("I'm on (503) 555-0101."):
        pass

    assert '"next"' in next(m for m in bot.messages if m.role == "tool").content
    event = next(e for e in events if isinstance(e, ToolCallEvent))
    assert "next" not in event.result


async def test_the_second_failed_check_gets_the_stop_reminder(db):
    check = call("verify_identity", customer_id="cus_1001", billing_zip="11111", pin="2222")
    llm = ScriptedLLM(uses(check), "No match.", uses(check), "Sorry.")
    bot = SupportBot(llm, ToolDispatcher(db))
    bot.greet()
    for _ in range(2):
        async for _ in bot.respond("My zip is 11111 and my pin is 2222."):
            pass

    reminders = [m.content for m in bot.messages if m.role == "tool"]
    assert "Second failure" not in reminders[0]
    assert "Second failure" in reminders[1]
