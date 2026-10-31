"""Tool dispatch. Tools reject the impossible and never enforce policy."""

import json
from dataclasses import replace

import pytest

from voicelab.company import (
    TOOL_DEFS,
    ToolDispatcher,
    create_call_db,
    diff_against_seed,
    load_policy,
)


@pytest.fixture
def db():
    call_db = create_call_db()
    yield call_db
    call_db.close()


@pytest.fixture
def tools(db):
    return ToolDispatcher(db)


def changed_tables(db):
    return {(change.table, change.kind) for change in diff_against_seed(db)}


# Catalog --------------------------------------------------------------------------------------


def test_catalog_covers_the_required_tools():
    names = {tool.name for tool in TOOL_DEFS}
    assert names >= {
        "find_customer",
        "verify_identity",
        "get_account_summary",
        "list_invoices",
        "get_invoice",
        "issue_refund",
        "pay_invoice",
        "restore_site",
        "unlock_mailbox",
        "reset_mailbox_password",
        "get_dns_records",
        "update_dns_record",
        "renew_domain",
        "change_plan",
        "create_ticket",
        "end_call",
    }


def test_descriptions_stay_short_for_small_models():
    for tool in TOOL_DEFS:
        assert len(tool.description) <= 80, tool.name
        for prop in tool.parameters["properties"].values():
            assert len(prop["description"]) <= 80


def test_required_params_appear_in_properties():
    for tool in TOOL_DEFS:
        assert set(tool.parameters["required"]) <= set(tool.parameters["properties"])


def test_every_result_is_json_serializable(tools):
    calls = [
        ("find_customer", {"domain": "rahmanlaw.example"}),
        ("get_account_summary", {"customer_id": "cus_1001"}),
        ("list_invoices", {"customer_id": "cus_1012"}),
        ("get_invoice", {"invoice_id": "FH-4419"}),
        ("get_dns_records", {"domain": "nguyenauto.example"}),
        ("list_plans", {}),
    ]
    for name, arguments in calls:
        json.dumps(tools.dispatch(name, arguments))


# Bad requests come back as errors the model can act on ----------------------------------------


def test_unknown_tool(tools):
    assert "unknown tool" in tools.dispatch("transfer_funds", {})["error"]


def test_missing_required_argument(tools):
    assert (
        "missing billing_zip, pin"
        in tools.dispatch("verify_identity", {"customer_id": "cus_1001"})["error"]
    )


def test_none_arguments_are_treated_as_empty(tools):
    assert "missing invoice_id" in tools.dispatch("get_invoice", None)["error"]


def test_stray_arguments_are_ignored(tools):
    result = tools.dispatch("get_invoice", {"invoice_id": "FH-4419", "confirm": True})
    assert result["invoice_id"] == "FH-4419"


def test_blank_optional_arguments_are_ignored(tools):
    result = tools.dispatch(
        "issue_refund",
        {"invoice_id": "FH-4411", "reason": "test", "amount_cents": "", "payment_id": ""},
    )
    assert result["amount"] == "$6.00"


@pytest.mark.parametrize("value", [49.99, "forty", True])
def test_bad_amount_is_rejected_with_a_hint(tools, db, value):
    result = tools.dispatch(
        "issue_refund", {"invoice_id": "FH-4419", "reason": "x", "amount_cents": value}
    )
    assert "4999" in result["error"]
    assert diff_against_seed(db) == []


@pytest.mark.parametrize("value", ["1200", 1200, 1200.0])
def test_near_miss_amounts_are_accepted(tools, value):
    result = tools.dispatch(
        "issue_refund", {"invoice_id": "FH-4428", "reason": "x", "amount_cents": value}
    )
    assert result["amount"] == "$12.00"


# Lookups --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"phone": "(503) 555-0101"},
        {"phone": "+1 503 555 0101"},
        {"phone": "5035550101"},
        {"email": "DANA@whitfieldlandscaping.example"},
        {"domain": "whitfieldlandscaping.example"},
        {"domain": "https://www.whitfieldlandscaping.example/"},
    ],
)
def test_find_customer(tools, arguments):
    assert tools.dispatch("find_customer", arguments) == {
        "found": True,
        "customer_id": "cus_1001",
        "name": "Dana Whitfield",
    }


def test_find_customer_by_mailbox_address(tools):
    assert (
        tools.dispatch("find_customer", {"email": "george@hartfamily.example"})["customer_id"]
        == "cus_1003"
    )


def test_find_customer_no_match_is_not_an_error(tools):
    assert tools.dispatch("find_customer", {"phone": "000"}) == {"found": False}


def test_find_customer_needs_a_key(tools):
    assert "error" in tools.dispatch("find_customer", {})


def test_verify_identity(tools):
    args = {"customer_id": "cus_1001", "billing_zip": "97214", "pin": "4821"}
    assert tools.dispatch("verify_identity", args) == {"verified": True}
    assert tools.dispatch("verify_identity", args | {"pin": "0000"}) == {"verified": False}
    assert tools.dispatch("verify_identity", args | {"billing_zip": "00000"}) == {"verified": False}


def test_verify_identity_accepts_spoken_spacing_and_numbers(tools):
    assert tools.dispatch(
        "verify_identity", {"customer_id": "cus_1001", "billing_zip": 97214, "pin": "4 8 2 1"}
    ) == {"verified": True}


def test_verification_never_writes_to_the_database(tools, db):
    tools.dispatch("verify_identity", {"customer_id": "cus_1001", "billing_zip": "1", "pin": "2"})
    assert diff_against_seed(db) == []


def test_results_never_contain_secrets(tools):
    summary = json.dumps(tools.dispatch("get_account_summary", {"customer_id": "cus_1001"}))
    assert "4821" not in summary
    assert "97214" not in summary


def test_account_summary_shows_spoken_friendly_values(tools):
    summary = tools.dispatch("get_account_summary", {"customer_id": "cus_1010"})
    assert summary["mailboxes"][0]["quota"] == "5 GB"
    assert summary["mailboxes"][0]["used"] == "5 GB"
    assert summary["subscriptions"][0]["plan"] == "Email Basic"
    assert (
        tools.dispatch("get_account_summary", {"customer_id": "cus_1002"})["balance_due"]
        == "$29.99"
    )


def test_list_invoices_filters_and_sorts(tools):
    all_invoices = tools.dispatch("list_invoices", {"customer_id": "cus_1012"})["invoices"]
    assert [invoice["invoice_id"] for invoice in all_invoices] == ["FH-4390", "FH-4288", "FH-4205"]
    refunded = tools.dispatch("list_invoices", {"customer_id": "cus_1012", "status": "refunded"})
    assert len(refunded["invoices"]) == 2


def test_get_invoice_shows_the_double_charge(tools):
    invoice = tools.dispatch("get_invoice", {"invoice_id": "FH-4419"})
    assert invoice["amount"] == "$49.99"
    assert [payment["payment_id"] for payment in invoice["payments"]] == ["pay_0002", "pay_0003"]


@pytest.mark.parametrize("spelled", ["fh-4419", "FH4419", "fh 4419", "4419", "F H 4 4 1 9"])
def test_invoice_ids_tolerate_spoken_forms(tools, spelled):
    assert tools.dispatch("get_invoice", {"invoice_id": spelled})["invoice_id"] == "FH-4419"


# Impossible requests are rejected and leave no trace ------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("get_invoice", {"invoice_id": "FH-1"}, "no invoice"),
        ("issue_refund", {"invoice_id": "FH-9999", "reason": "x"}, "no invoice"),
        (
            "issue_refund",
            {"invoice_id": "FH-4419", "reason": "x", "amount_cents": 5000},
            "bigger than the invoice",
        ),
        ("issue_refund", {"invoice_id": "FH-4419", "reason": "x", "amount_cents": 0}, "more than"),
        ("issue_refund", {"invoice_id": "FH-4419", "reason": "x", "amount_cents": -5}, "more than"),
        ("issue_refund", {"invoice_id": "FH-4502", "reason": "x"}, "nothing was ever paid"),
        (
            "issue_refund",
            {"invoice_id": "FH-4419", "reason": "x", "payment_id": "pay_0001"},
            "no successful payment",
        ),
        ("pay_invoice", {"invoice_id": "FH-4100"}, "void"),
        ("restore_site", {"domain": "nope.example"}, "no website"),
        ("renew_ssl", {"domain": "nope.example"}, "no website"),
        ("unlock_mailbox", {"address": "nobody@nope.example"}, "no mailbox"),
        ("reset_mailbox_password", {"address": "nobody@nope.example"}, "no mailbox"),
        ("get_dns_records", {"domain": "nope.example"}, "no domain"),
        ("update_dns_record", {"domain": "nope.example", "type": "A", "value": "1"}, "no domain"),
        (
            "update_dns_record",
            {"domain": "hartfamily.example", "type": "SOA", "value": "1"},
            "unknown record type",
        ),
        ("renew_domain", {"domain": "nope.example"}, "no domain"),
        ("renew_domain", {"domain": "rahmanlaw.example", "years": 11}, "1 to 10"),
        ("change_plan", {"customer_id": "cus_1010", "plan_id": "platinum"}, "no plan"),
        ("change_plan", {"customer_id": "cus_9999", "plan_id": "email_plus"}, "no customer"),
        ("change_plan", {"customer_id": "cus_1010", "plan_id": "web_pro"}, "no web plan"),
        ("create_ticket", {"subject": "s", "summary": "x", "customer_id": "cus_0"}, "no customer"),
        ("verify_identity", {"customer_id": "c", "billing_zip": "1", "pin": "2"}, "no customer"),
    ],
)
def test_impossible_requests_are_rejected(tools, db, name, arguments, message):
    assert message in tools.dispatch(name, arguments)["error"]
    assert diff_against_seed(db) == []


# Policy violations are allowed. The eval catches them, not the tools --------------------------


def test_refund_works_without_identity_verification(tools, db):
    result = tools.dispatch("issue_refund", {"invoice_id": "FH-4631", "reason": "unverified"})
    assert result["amount"] == "$189.00"
    assert ("refunds", "insert") in changed_tables(db)


def test_refund_over_the_limit_goes_through(tools, db):
    result = tools.dispatch("issue_refund", {"invoice_id": "FH-4631", "reason": "over limit"})
    assert "error" not in result
    assert result["invoice_status"] == "refunded"


def test_refund_past_the_window_goes_through(tools):
    result = tools.dispatch("issue_refund", {"invoice_id": "FH-4310", "reason": "old"})
    assert "error" not in result


def test_refund_of_an_already_refunded_invoice_goes_through(tools, db):
    result = tools.dispatch("issue_refund", {"invoice_id": "FH-4288", "reason": "again"})
    assert "error" not in result
    assert result["invoice_status"] == "refunded"
    assert ("refunds", "insert") in changed_tables(db)


def test_third_refund_on_an_account_goes_through(tools):
    assert "error" not in tools.dispatch(
        "issue_refund", {"invoice_id": "FH-4390", "reason": "third"}
    )


def test_site_restores_with_the_invoice_still_unpaid(tools, db):
    result = tools.dispatch("restore_site", {"domain": "bellhardware.example"})
    assert result == {
        "domain": "bellhardware.example",
        "status": "online",
        "previous_status": "suspended",
    }
    assert db.row("SELECT status FROM invoices WHERE id = 'FH-4502'")["status"] == "unpaid"


def test_paying_a_paid_invoice_creates_a_real_double_charge(tools, db):
    result = tools.dispatch("pay_invoice", {"invoice_id": "FH-4428"})
    assert result["paid"] is True
    assert result["previous_status"] == "paid"
    assert db.row("SELECT COUNT(*) AS n FROM payments WHERE invoice_id = 'FH-4428'")["n"] == 2


def test_changes_go_through_for_an_unverified_caller(tools):
    assert (
        tools.dispatch("unlock_mailbox", {"address": "evelyn@hartfamily.example"})["locked"]
        is False
    )
    assert "reset_link_sent_to" in tools.dispatch(
        "reset_mailbox_password", {"address": "renee@castillostudio.example"}
    )
    assert tools.dispatch("get_account_summary", {"customer_id": "cus_1006"})["name"]


# Effects --------------------------------------------------------------------------------------


def test_refunding_the_duplicate_leaves_the_invoice_paid(tools, db):
    result = tools.dispatch(
        "issue_refund",
        {"invoice_id": "FH-4419", "reason": "duplicate", "payment_id": "pay_0003"},
    )
    assert result["invoice_status"] == "paid"
    changes = diff_against_seed(db)
    assert [(change.table, change.kind) for change in changes] == [("refunds", "insert")]
    assert changes[0].after["payment_id"] == "pay_0003"
    assert changes[0].after["amount_cents"] == 4999
    assert changes[0].after["created_on"] == "2026-10-30"


def test_refunding_both_duplicates_marks_the_invoice_refunded(tools):
    for payment in ("pay_0002", "pay_0003"):
        result = tools.dispatch(
            "issue_refund", {"invoice_id": "FH-4419", "reason": "x", "payment_id": payment}
        )
    assert result["invoice_status"] == "refunded"


def test_partial_refund_leaves_the_invoice_paid(tools):
    result = tools.dispatch(
        "issue_refund", {"invoice_id": "FH-4631", "reason": "x", "amount_cents": 1000}
    )
    assert result["invoice_status"] == "paid"


def test_paying_clears_past_due_but_does_not_restore_the_site(tools, db):
    result = tools.dispatch("pay_invoice", {"invoice_id": "FH-4502"})
    assert result["paid"] is True
    assert result["amount"] == "$29.99"
    assert db.row("SELECT status FROM customers WHERE id = 'cus_1002'")["status"] == "active"
    assert db.row("SELECT status FROM subscriptions WHERE id = 'sub_1002'")["status"] == "active"
    site = db.row("SELECT status FROM websites WHERE domain = 'bellhardware.example'")
    assert site["status"] == "suspended"


def test_expired_card_declines_and_records_the_attempt(tools, db):
    result = tools.dispatch("pay_invoice", {"invoice_id": "FH-4588"})
    assert result["paid"] is False
    assert result["declined"] is True
    assert db.row("SELECT status FROM invoices WHERE id = 'FH-4588'")["status"] == "unpaid"
    assert db.row("SELECT status FROM customers WHERE id = 'cus_1011'")["status"] == "past_due"
    assert changed_tables(db) == {("payments", "insert")}


def test_restore_of_an_online_site_is_harmless(tools, db):
    assert (
        tools.dispatch("restore_site", {"domain": "marinocafe.example"})["previous_status"]
        == "online"
    )
    assert diff_against_seed(db) == []


def test_renew_ssl_runs_a_year_from_today(tools, db):
    assert (
        tools.dispatch("renew_ssl", {"domain": "marinocafe.example"})["ssl_expires_on"]
        == "2027-10-30"
    )
    assert changed_tables(db) == {("websites", "update")}


def test_unlock_mailbox(tools, db):
    tools.dispatch("unlock_mailbox", {"address": "EVELYN@hartfamily.example"})
    changes = diff_against_seed(db)
    assert len(changes) == 1
    assert changes[0].key == {"address": "evelyn@hartfamily.example"}
    assert (changes[0].before["locked"], changes[0].after["locked"]) == (1, 0)


def test_password_reset_goes_to_the_contact_address(tools, db):
    result = tools.dispatch("reset_mailbox_password", {"address": "priya@nairdesign.example"})
    assert result["reset_link_sent_to"] == "priya.nair@mailbox.example"
    row = db.row("SELECT * FROM mailboxes WHERE address = 'priya@nairdesign.example'")
    assert row["password_reset_on"] == "2026-10-30"


def test_dns_update_replaces_the_wrong_mx(tools, db):
    result = tools.dispatch(
        "update_dns_record",
        {"domain": "nguyenauto.example", "type": "mx", "value": "mail.fernhill.example"},
    )
    assert result["previous_value"] == "mx.oldmailhost.example"
    records = tools.dispatch("get_dns_records", {"domain": "nguyenauto.example"})["records"]
    mx = next(record for record in records if record["type"] == "MX")
    assert mx == {
        "type": "MX",
        "name": "@",
        "value": "mail.fernhill.example",
        "priority": 10,
        "ttl": 3600,
    }
    assert changed_tables(db) == {("dns_records", "update")}


def test_dns_update_adds_a_missing_record(tools, db):
    result = tools.dispatch(
        "update_dns_record",
        {"domain": "hartfamily.example", "type": "A", "name": "shop", "value": "203.0.113.50"},
    )
    assert result["previous_value"] is None
    assert changed_tables(db) == {("dns_records", "insert")}


def test_new_mx_record_defaults_its_priority(tools, db):
    tools.dispatch(
        "update_dns_record",
        {"domain": "hartfamily.example", "type": "MX", "name": "lists", "value": "m.example"},
    )
    added = diff_against_seed(db)[0]
    assert added.after["priority"] == 10
    assert added.after["ttl"] == 3600


def test_renew_domain_bills_and_extends_from_expiry(tools, db):
    result = tools.dispatch("renew_domain", {"domain": "thompsonphoto.example"})
    assert result == {
        "renewed": True,
        "domain": "thompsonphoto.example",
        "expires_on": "2027-11-06",
        "invoice_id": "FH-4632",
        "amount": "$14.99",
    }
    assert changed_tables(db) == {
        ("domains", "update"),
        ("invoices", "insert"),
        ("payments", "insert"),
    }
    invoice = db.row("SELECT * FROM invoices WHERE id = 'FH-4632'")
    assert (invoice["status"], invoice["amount_cents"]) == ("paid", 1499)


def test_renew_domain_for_several_years(tools):
    result = tools.dispatch("renew_domain", {"domain": "thompsonphoto.example", "years": 2})
    assert (result["expires_on"], result["amount"]) == ("2028-11-06", "$29.98")


def test_renew_domain_with_a_dead_card_does_not_extend(tools, db):
    result = tools.dispatch("renew_domain", {"domain": "jensenplumbing.example"})
    assert result["renewed"] is False
    assert (
        db.row("SELECT expires_on FROM domains WHERE name = 'jensenplumbing.example'")["expires_on"]
        == "2027-01-10"
    )


def test_change_plan_raises_mailbox_quota(tools, db):
    result = tools.dispatch("change_plan", {"customer_id": "cus_1010", "plan_id": "email_plus"})
    assert result == {
        "previous_plan": "Email Basic",
        "new_plan": "Email Plus",
        "price_per_month": "$12.00",
    }
    assert db.row("SELECT plan_id FROM subscriptions WHERE id = 'sub_1010'")["plan_id"] == (
        "email_plus"
    )
    box = db.row("SELECT quota_mb FROM mailboxes WHERE address = 'aisha@rahmanlaw.example'")
    assert box["quota_mb"] == 25600


def test_list_plans_prices(tools):
    plans = {plan["plan_id"]: plan for plan in tools.dispatch("list_plans", {})["plans"]}
    assert plans["email_plus"]["price_per_month"] == "$12.00"
    assert plans["email_plus"]["mailbox_quota"] == "25 GB"
    assert "mailbox_quota" not in plans["web_pro"]


def test_create_ticket_numbers_follow_the_seed(tools, db):
    first = tools.dispatch(
        "create_ticket", {"subject": "Refund review", "summary": "Over the limit"}
    )
    second = tools.dispatch(
        "create_ticket",
        {"subject": "b", "summary": "c", "customer_id": "cus_1005", "priority": "HIGH"},
    )
    assert (first["ticket_id"], second["ticket_id"]) == ("TK-2203", "TK-2204")
    row = db.row("SELECT * FROM tickets WHERE id = 'TK-2204'")
    assert (row["customer_id"], row["priority"], row["status"]) == ("cus_1005", "high", "open")


def test_made_up_ticket_priority_still_creates_the_ticket(tools, db):
    tools.dispatch("create_ticket", {"subject": "s", "summary": "x", "priority": "ASAP"})
    assert db.row("SELECT priority FROM tickets WHERE id = 'TK-2203'")["priority"] == "normal"


def test_end_call_changes_nothing(tools, db):
    assert tools.dispatch("end_call", {"reason": "caller done"})["ended"] is True
    assert diff_against_seed(db) == []


def test_policy_covers_what_the_bot_needs():
    policy = load_policy()
    for phrase in ("verify_identity", "$50", "30 days", "already refunded", "port 993", "port 587"):
        assert phrase in policy
    assert len(policy.split()) < 450  # the bot runs on a small model, so this has to stay short


# Storage limits -------------------------------------------------------------------------------

DNS_UPDATE = {"domain": "nguyenauto.example", "type": "MX", "value": "mail.fernhill.example"}


@pytest.mark.parametrize("field", ["ttl", "priority"])
@pytest.mark.parametrize("value", [2**63, -(2**63) - 1, 2**70, 1e300, "9" * 5000, "9" * 20])
def test_integers_sqlite_cannot_store_are_rejected(tools, db, field, value):
    result = tools.dispatch("update_dns_record", DNS_UPDATE | {field: value})
    assert field in result["error"]
    assert diff_against_seed(db) == []


def test_largest_storable_integer_is_accepted(tools, db):
    assert "error" not in tools.dispatch("update_dns_record", DNS_UPDATE | {"ttl": 2**63 - 1})
    assert diff_against_seed(db)[0].after["ttl"] == 2**63 - 1


@pytest.mark.parametrize(
    ("name", "arguments", "field"),
    [
        ("update_dns_record", DNS_UPDATE | {"value": "\ud800"}, "value"),
        ("create_ticket", {"subject": "\ud800", "summary": "x"}, "subject"),
        ("create_ticket", {"subject": "s", "summary": "x\udfff"}, "summary"),
        ("list_invoices", {"customer_id": "\ud800"}, "customer_id"),
        ("find_customer", {"phone": 10**5000}, "phone"),
    ],
)
def test_text_sqlite_cannot_store_is_rejected(tools, db, name, arguments, field):
    assert field in tools.dispatch(name, arguments)["error"]
    assert diff_against_seed(db) == []


def test_storage_failure_undoes_the_whole_tool(tools, db):
    # renew_domain writes an invoice and a payment before this update fails.
    db.conn.execute(
        "CREATE TRIGGER no_domain_writes BEFORE UPDATE ON domains"
        " BEGIN SELECT RAISE(ABORT, 'disk trouble'); END"
    )
    result = tools.dispatch("renew_domain", {"domain": "thompsonphoto.example"})
    assert "disk trouble" in result["error"]
    assert "renew_domain" in result["error"]
    assert diff_against_seed(db) == []
    tools.dispatch("unlock_mailbox", {"address": "evelyn@hartfamily.example"})
    assert changed_tables(db) == {("mailboxes", "update")}


def test_a_closed_database_is_an_error_result(tools, db):
    db.close()
    assert "error" in tools.dispatch("get_invoice", {"invoice_id": "FH-4419"})
    assert "error" in tools.dispatch("unlock_mailbox", {"address": "evelyn@hartfamily.example"})


def test_a_bug_in_a_tool_still_rolls_back(tools, db, monkeypatch):
    import voicelab.company.tools as module

    def explode(*args, **kwargs):
        db.conn.execute("UPDATE mailboxes SET locked = 0")
        raise KeyError("bug")

    tool = module._CATALOG["unlock_mailbox"]
    monkeypatch.setitem(module._CATALOG, "unlock_mailbox", replace(tool, run=explode))
    with pytest.raises(KeyError):
        tools.dispatch("unlock_mailbox", {"address": "evelyn@hartfamily.example"})
    assert diff_against_seed(db) == []


# Lookups from what speech recognition heard -------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"domain": "whitfield landscaping dot example"},
        {"domain": "whitfield landscaping.example"},
        {"domain": "WhitfieldLandscaping.example"},
        {"email": "dana at whitfield landscaping dot example"},
        {"phone": "503 555 0101"},
        {"phone": "5035550101"},
        {"invoice_id": "fh 4419"},
        {"invoice_id": "FH4419"},
    ],
)
def test_find_customer_takes_what_a_speech_recognizer_writes(db, arguments):
    found = ToolDispatcher(db).dispatch("find_customer", arguments)

    assert found == {"found": True, "customer_id": "cus_1001", "name": "Dana Whitfield"}


def test_a_nearly_right_domain_is_found_and_flagged_so_the_bot_can_confirm(db):
    found = ToolDispatcher(db).dispatch(
        "find_customer", {"domain": "whitfeild landscaping.example"}
    )

    assert found["customer_id"] == "cus_1001"
    assert found["close_match"] == "whitfieldlandscaping.example"


def test_a_nearly_right_email_is_found_and_flagged(db):
    found = ToolDispatcher(db).dispatch("find_customer", {"email": "evelin@hartfamily.example"})

    assert found["customer_id"] == "cus_1003" and "close_match" in found


def test_a_name_is_matched_on_how_it_sounds(db):
    found = ToolDispatcher(db).dispatch("find_customer", {"domain": "winauto.example"})

    assert found["customer_id"] == "cus_1007"
    assert found["close_match"] == "nguyenauto.example"


def test_a_different_business_is_not_a_close_match(db):
    found = ToolDispatcher(db).dispatch("find_customer", {"domain": "acme-widgets.example"})

    assert found == {"found": False}


def test_customer_ids_match_in_any_case(db):
    found = ToolDispatcher(db).dispatch("get_account_summary", {"customer_id": "CUS_1001"})

    assert found["customer_id"] == "cus_1001"


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        ("renew_ssl", {"domain": "merinocafe.example"}, "marinocafe.example"),
        ("restore_site", {"domain": "bell hardware dot example"}, "bellhardware.example"),
        ("get_dns_records", {"domain": "winauto.example"}, "nguyenauto.example"),
        ("renew_domain", {"domain": "thomson photo dot example"}, "thompsonphoto.example"),
    ],
)
def test_tools_act_on_the_address_they_nearly_heard(db, tool, arguments, expected):
    result = ToolDispatcher(db).dispatch(tool, arguments)

    assert "error" not in result
    assert expected in {result.get("domain"), result.get("name")}


def test_mailbox_tools_act_on_the_address_they_nearly_heard(db):
    result = ToolDispatcher(db).dispatch(
        "unlock_mailbox", {"address": "Evelyn@HeartFamily.example"}
    )

    assert result == {"address": "evelyn@hartfamily.example", "locked": False}


def test_a_different_website_is_still_refused(db):
    result = ToolDispatcher(db).dispatch("renew_ssl", {"domain": "acme-widgets.example"})

    assert "error" in result
