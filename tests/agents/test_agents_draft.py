"""Drafting scenarios from a description."""

import json

import pytest
from scripted_llm import ScriptedLLM

from voicelab.agents.draft import DraftError, draft_prompt, draft_scenario, draft_schema
from voicelab.core.models import Scenario

PRIYA_FACTS = {
    "name": "Priya Nair",
    "billing_zip": "98103",
    "pin": "0264",
    "mailbox": "priya@nairdesign.example",
}
NO_LOOKUP_KEY = {"name": "Priya Nair", "billing_zip": "98103", "pin": "0264"}


def draft(**overrides) -> str:
    base = {
        "title": "Password reset for Priya",
        "description": "Forgotten password.",
        "goal": "I forgot my password.",
        "customer_id": "cus_1004",
        "facts": PRIYA_FACTS,
        "beats": [{"id": "ask", "instruction": "Ask for a reset.", "checkpoint": "Asked."}],
        "must": [{"id": "verify", "text": "Verify first.", "severity": "critical"}],
        "must_not": [{"id": "speak", "text": "Say a password aloud."}],
        "expected_state": [
            {
                "id": "reset",
                "table": "mailboxes",
                "where": {"address": "priya@nairdesign.example"},
                "op": "changed",
                "changes": {"password_reset_on": "2026-10-30"},
            }
        ],
        "pronunciation_watch": ["priya@nairdesign.example"],
        "tags": ["email"],
    }
    return json.dumps(base | overrides)


async def test_returns_a_validated_scenario():
    llm = ScriptedLLM(draft())
    scenario = await draft_scenario(llm, "A caller forgot their email password")
    assert isinstance(scenario, Scenario)
    assert scenario.customer_id == "cus_1004"
    assert scenario.beats[0].id == "ask"
    assert len(llm.requests) == 1


async def test_prompt_carries_the_description_and_the_schema():
    llm = ScriptedLLM(draft())
    await draft_scenario(llm, "A caller forgot their email password")
    system, user = llm.requests[0]["messages"]
    assert "A caller forgot their email password" in user.content
    assert "mailboxes(address, customer_id, quota_mb" in system.content
    assert "cus_1001: Dana Whitfield" in system.content
    assert "billing_zip 97214" in system.content
    assert "Charged twice" in system.content
    assert "$50" in system.content


async def test_requests_json_against_the_scenario_schema():
    llm = ScriptedLLM(draft())
    await draft_scenario(llm, "x")
    assert llm.requests[0]["json_schema"] == draft_schema()


def test_schema_leaves_out_system_assigned_fields():
    properties = draft_schema()["properties"]
    assert not {"id", "created_at", "updated_at"} & set(properties)
    assert {"title", "goal", "facts", "beats", "expected_state"} <= set(properties)


def test_prompt_states_what_the_caller_needs_up_front():
    prompt = draft_prompt()
    assert "phone, email, or domain that finds the customer" in prompt
    assert "mailbox, domain, or invoice id at issue" in prompt


def test_prompt_formats_without_stray_braces():
    assert "{tables}" not in draft_prompt()
    assert "{policy}" not in draft_prompt()


async def test_model_supplied_ids_are_ignored():
    scenario = await draft_scenario(ScriptedLLM(draft(id="scn_hijack")), "x")
    assert scenario.id != "scn_hijack"
    assert scenario.id.startswith("scn_")


@pytest.mark.parametrize(
    ("bad", "complaint"),
    [
        ("not json at all", "not valid JSON"),
        ("[1, 2, 3]", "JSON object"),
        ('{"title": ' + "9" * 5000 + "}", "not valid JSON"),
        ("[" * 200000, "not valid JSON"),
        (json.dumps({"title": "no goal"}), "did not match the schema"),
        (draft(customer_id="cus_9999"), "not a seeded customer"),
        (draft(facts=NO_LOOKUP_KEY), "phone, email, or domain that finds cus_1004"),
        (
            draft(facts=NO_LOOKUP_KEY | {"phone": "(503) 555-0101"}),
            "phone, email, or domain that finds cus_1004",
        ),
        (
            draft(facts=NO_LOOKUP_KEY | {"phone": "(206) 555-0104"}),
            "name priya@nairdesign.example",
        ),
        (
            draft(facts=PRIYA_FACTS | {"pin": "9999"}),
            "the correct PIN",
        ),
        (
            draft(facts={"name": "Priya", "mailbox": "priya@nairdesign.example"}),
            "the correct billing zip",
        ),
        (
            draft(
                customer_id="cus_1001",
                facts={
                    "domain": "whitfieldlandscaping.example",
                    "billing_zip": "97214",
                    "pin": "4821",
                },
                expected_state=[
                    {
                        "table": "refunds",
                        "op": "row_added",
                        "where": {"invoice_id": "FH-4419", "amount_cents": 4999},
                    }
                ],
            ),
            "name FH-4419",
        ),
        (draft(beats=[]), "beats must not be empty"),
        (draft(facts={}), "facts must not be empty"),
        (
            draft(expected_state=[{"table": "wizards", "where": {}, "op": "changed"}]),
            "unknown table wizards",
        ),
        (
            draft(expected_state=[{"table": "mailboxes", "where": {"nope": 1}, "op": "changed"}]),
            "no column nope",
        ),
        (
            draft(
                expected_state=[
                    {"table": "mailboxes", "where": {"address": "x@y.example"}, "op": "changed"}
                ]
            ),
            "no seeded mailboxes row",
        ),
        (
            draft(expected_state=[{"table": "mailboxes", "where": {"used_mb": 2**70}}]),
            "can't be matched",
        ),
        (
            draft(expected_state=[{"table": "mailboxes", "where": {"address": ["a", "b"]}}]),
            "can't be matched",
        ),
    ],
)
async def test_bad_drafts_are_sent_back_with_the_reason(bad, complaint):
    llm = ScriptedLLM(bad, draft())
    scenario = await draft_scenario(llm, "x")
    assert scenario.title == "Password reset for Priya"
    retry = llm.requests[1]["messages"]
    assert retry[-2].role == "assistant"
    assert retry[-1].role == "user"
    assert complaint in retry[-1].content


async def test_gives_up_after_the_attempt_limit():
    llm = ScriptedLLM(default="nope")
    with pytest.raises(DraftError, match="after 2 attempts"):
        await draft_scenario(llm, "x", attempts=2)
    assert len(llm.requests) == 2


@pytest.mark.parametrize(
    "facts",
    [
        NO_LOOKUP_KEY | {"email": "PRIYA.NAIR@mailbox.example", "box": "priya@nairdesign.example"},
        NO_LOOKUP_KEY
        | {"my_website": "https://www.nairdesign.example/", "box": "priya@nairdesign.example"},
        PRIYA_FACTS,
    ],
)
async def test_any_fact_that_finds_the_customer_counts(facts):
    assert (await draft_scenario(ScriptedLLM(draft(facts=facts)), "x")).facts == facts


async def test_a_refusal_scenario_can_have_wrong_credentials():
    wrong = {"mailbox": "priya@nairdesign.example", "billing_zip": "00000", "pin": "1111"}
    no_change = [
        {"table": "mailboxes", "op": "no_change", "where": {"address": "priya@nairdesign.example"}}
    ]
    llm = ScriptedLLM(draft(facts=wrong, expected_state=no_change))
    assert (await draft_scenario(llm, "x")).facts == wrong


async def test_a_ticket_needs_no_verification():
    stranger = {"mailbox": "priya@nairdesign.example"}
    ticket = [{"table": "tickets", "op": "row_added", "where": {"customer_id": "cus_1004"}}]
    llm = ScriptedLLM(draft(facts=stranger, expected_state=ticket))
    assert (await draft_scenario(llm, "x")).facts == stranger


async def test_the_only_unpaid_invoice_needs_no_id_on_the_sheet():
    facts = {"domain": "bellhardware.example", "billing_zip": "97035", "pin": "7316"}
    paid = [
        {"table": "invoices", "where": {"id": "FH-4502"}, "changes": {"status": "paid"}},
        {
            "table": "payments",
            "op": "row_added",
            "where": {"invoice_id": "FH-4502", "status": "succeeded"},
        },
    ]
    llm = ScriptedLLM(draft(customer_id="cus_1002", facts=facts, expected_state=paid))
    assert (await draft_scenario(llm, "x")).customer_id == "cus_1002"


async def test_scenario_without_a_customer_is_allowed():
    scenario = await draft_scenario(ScriptedLLM(draft(customer_id=None, expected_state=[])), "x")
    assert scenario.customer_id is None
