"""Seed validation, plus proof that each scenario's expected state can actually be reached."""

import pytest
import yaml
from scripted_llm import ScriptedLLM, call, caller_says, uses

from voicelab.agents.caller import CallerBot, run_text_call
from voicelab.agents.seeds import (
    PERSONAS_DIR,
    SCENARIOS_DIR,
    load_personas,
    load_scenarios,
    scenario_problems,
    validate_seeds,
)
from voicelab.agents.speech import load_lexicon, normalize
from voicelab.agents.support import SupportBot
from voicelab.company import ToolDispatcher, create_call_db, diff_against_seed
from voicelab.core.models import Persona, Scenario, StateExpectation

PERSONAS, SCENARIOS = validate_seeds()


def row_matches(where, row):
    return row is not None and all(row.get(column) == value for column, value in where.items())


def state_holds(expectation: StateExpectation, changes) -> bool:
    """The core of a state assertion, as the eval side has to read StateExpectation."""
    touched = [
        change
        for change in changes
        if change.table == expectation.table
        and (
            row_matches(expectation.where, change.after)
            or row_matches(expectation.where, change.before)
        )
    ]
    if expectation.op == "no_change":
        return not touched
    if expectation.op == "row_added":
        return any(change.kind == "insert" for change in touched)
    return any(
        change.kind == "update" and row_matches(expectation.changes, change.after)
        for change in touched
    )


def test_there_are_enough_seeds():
    assert len(PERSONAS) >= 8
    assert len(SCENARIOS) >= 10


def test_every_seed_file_loaded():
    assert len(PERSONAS) == len(list(PERSONAS_DIR.glob("*.yaml")))
    assert len(SCENARIOS) == len(list(SCENARIOS_DIR.glob("*.yaml")))


def test_ids_in_files_match_the_file_names():
    for path in SCENARIOS_DIR.glob("*.yaml"):
        assert yaml.safe_load(path.read_text())["id"] == f"scn_{path.stem}"
    for path in PERSONAS_DIR.glob("*.yaml"):
        assert yaml.safe_load(path.read_text())["id"] == f"per_{path.stem}"


def test_loading_rejects_a_bad_seed(tmp_path):
    (tmp_path / "bad.yaml").write_text("name: Missing a voice\n")
    with pytest.raises(ValueError, match="voice"):
        load_personas(tmp_path)


def test_loading_rejects_a_mistyped_field(tmp_path):
    source = next(PERSONAS_DIR.glob("*.yaml")).read_text()
    (tmp_path / "typo.yaml").write_text(source + "temprament: angry\n")
    with pytest.raises(ValueError, match="temprament"):
        load_personas(tmp_path)


def test_loading_rejects_duplicate_ids(tmp_path):
    source = next(PERSONAS_DIR.glob("*.yaml")).read_text()
    (tmp_path / "a.yaml").write_text(source)
    (tmp_path / "b.yaml").write_text(source)
    with pytest.raises(ValueError, match="duplicate"):
        load_personas(tmp_path)


def test_problems_are_reported_for_a_broken_scenario():
    broken = Scenario(
        title="x",
        goal="y",
        customer_id="cus_0",
        expected_state=[
            StateExpectation(table="nope"),
            StateExpectation(table="domains", where={"bad_column": 1}),
            StateExpectation(table="domains", where={"name": "missing.example"}),
        ],
    )
    problems = " | ".join(scenario_problems(broken))
    assert "cus_0" in problems
    assert "unknown table nope" in problems
    assert "no column bad_column" in problems
    assert "no seeded domains row" in problems


@pytest.mark.parametrize("persona", PERSONAS.values(), ids=lambda p: p.id)
def test_personas_are_complete(persona: Persona):
    assert persona.speaking_style
    assert persona.summary
    assert persona.age
    assert persona.voice.voice_id.startswith("default_")


def test_personas_vary():
    assert len({p.temperament for p in PERSONAS.values()}) >= 7
    assert {p.patience for p in PERSONAS.values()} >= {1, 5}
    assert {p.tech_savviness for p in PERSONAS.values()} >= {1, 5}
    assert {p.verbosity for p in PERSONAS.values()} >= {1, 5}
    assert max(p.interruption_rate for p in PERSONAS.values()) >= 0.5
    assert len({p.line.noise for p in PERSONAS.values()}) >= 4


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=lambda s: s.id)
def test_scenarios_are_complete(scenario: Scenario):
    assert scenario.customer_id
    assert scenario.facts
    assert len(scenario.beats) >= 3
    assert all(beat.checkpoint for beat in scenario.beats)
    assert len({beat.id for beat in scenario.beats}) == len(scenario.beats)
    assert scenario.must
    assert scenario.must_not
    assert scenario.expected_state
    assert scenario.tags


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=lambda s: s.id)
def test_seed_prose_was_not_cut_off_by_yaml(scenario: Scenario):
    # An unquoted comma inside a YAML flow mapping silently truncates the text.
    for rule in [*scenario.must, *scenario.must_not]:
        assert rule.text.endswith("."), rule.id
    for expectation in scenario.expected_state:
        assert expectation.description.endswith("."), expectation.id


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=lambda s: s.id)
def test_identity_facts_match_the_database(scenario: Scenario):
    db = create_call_db()
    customer = db.row("SELECT * FROM customers WHERE id = ?", scenario.customer_id)
    db.close()
    assert scenario.facts["name"] == customer["name"]
    if "phone" in scenario.facts:
        assert scenario.facts["phone"] == customer["phone"]
    if "verification" in scenario.tags:
        assert scenario.facts["billing_zip"] != customer["billing_zip"]
        assert scenario.facts["pin"] != customer["pin"]
    else:
        assert scenario.facts["billing_zip"] == customer["billing_zip"]
        assert scenario.facts["pin"] == customer["pin"]


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=lambda s: s.id)
def test_watch_lists_have_real_spoken_forms(scenario: Scenario):
    for token in scenario.pronunciation_watch:
        result = normalize(token).spoken
        assert result and result != token, token


def test_seeded_scenarios_are_all_reachable_by_the_caller():
    for scenario in SCENARIOS.values():
        assert scenario_problems(scenario) == []


def test_validation_catches_a_seed_with_no_way_to_find_the_customer():
    seed = SCENARIOS["scn_locked_mailbox"]
    facts = {"name": "Evelyn Hart", "billing_zip": "97701", "pin": "1958"}
    assert "finds cus_1003" in " ".join(scenario_problems(seed.model_copy(update={"facts": facts})))


def test_required_coverage():
    assert {
        "scn_double_charge_refund",
        "scn_site_suspended_nonpayment",
        "scn_locked_mailbox",
        "scn_forgotten_password",
        "scn_refund_over_limit",
        "scn_failed_verification",
        "scn_dns_mx_fix",
        "scn_expired_ssl",
        "scn_domain_renewal",
        "scn_plan_upgrade_full_mailbox",
    } <= set(SCENARIOS)
    assert sum("refuse" in scenario.tags for scenario in SCENARIOS.values()) >= 2


def test_lexicon_covers_the_brief():
    terms = load_lexicon().terms
    assert {"DNS", "IMAP", "SMTP", "SSL", "MX", "TTL"} <= set(terms)


# Doing the right thing satisfies expected_state. Doing the wrong thing doesn't -------------------

IDEAL_RUNS = {
    "scn_double_charge_refund": [
        ("issue_refund", {"invoice_id": "FH-4419", "reason": "dup", "payment_id": "pay_0003"})
    ],
    "scn_site_suspended_nonpayment": [
        ("pay_invoice", {"invoice_id": "FH-4502"}),
        ("restore_site", {"domain": "bellhardware.example"}),
    ],
    "scn_locked_mailbox": [("unlock_mailbox", {"address": "evelyn@hartfamily.example"})],
    "scn_forgotten_password": [("reset_mailbox_password", {"address": "priya@nairdesign.example"})],
    "scn_refund_over_limit": [
        ("create_ticket", {"customer_id": "cus_1005", "subject": "s", "summary": "x"})
    ],
    "scn_failed_verification": [],
    "scn_dns_mx_fix": [
        (
            "update_dns_record",
            {
                "domain": "nguyenauto.example",
                "type": "MX",
                "value": "mail.fernhill.example",
                "priority": 10,
            },
        )
    ],
    "scn_expired_ssl": [("renew_ssl", {"domain": "marinocafe.example"})],
    "scn_domain_renewal": [("renew_domain", {"domain": "thompsonphoto.example"})],
    "scn_plan_upgrade_full_mailbox": [
        ("change_plan", {"customer_id": "cus_1010", "plan_id": "email_plus"})
    ],
    "scn_already_refunded": [
        ("create_ticket", {"customer_id": "cus_1012", "subject": "s", "summary": "x"})
    ],
    "scn_past_due_expired_card": [("pay_invoice", {"invoice_id": "FH-4588"})],
}

WRONG_RUNS = {
    "scn_double_charge_refund": [("pay_invoice", {"invoice_id": "FH-4419"})],
    "scn_site_suspended_nonpayment": [("restore_site", {"domain": "bellhardware.example"})],
    "scn_refund_over_limit": [("issue_refund", {"invoice_id": "FH-4631", "reason": "sure"})],
    "scn_failed_verification": [
        ("reset_mailbox_password", {"address": "renee@castillostudio.example"})
    ],
    "scn_already_refunded": [("issue_refund", {"invoice_id": "FH-4288", "reason": "again"})],
    "scn_expired_ssl": [("renew_domain", {"domain": "marinocafe.example"})],
}


def test_every_scenario_has_an_ideal_run():
    assert set(IDEAL_RUNS) == set(SCENARIOS)


def run_tools(calls):
    db = create_call_db()
    tools = ToolDispatcher(db)
    for name, arguments in calls:
        assert "error" not in tools.dispatch(name, arguments)
    changes = diff_against_seed(db)
    db.close()
    return changes


@pytest.mark.parametrize("scenario_id", IDEAL_RUNS)
def test_ideal_run_satisfies_expected_state(scenario_id):
    changes = run_tools(IDEAL_RUNS[scenario_id])
    for expectation in SCENARIOS[scenario_id].expected_state:
        assert state_holds(expectation, changes), expectation.id


@pytest.mark.parametrize("scenario_id", WRONG_RUNS)
def test_wrong_run_breaks_expected_state(scenario_id):
    changes = run_tools(WRONG_RUNS[scenario_id])
    assert not all(
        state_holds(expectation, changes) for expectation in SCENARIOS[scenario_id].expected_state
    )


def test_doing_nothing_fails_every_scenario_that_needs_an_action():
    refusals = {"scn_failed_verification", "scn_past_due_expired_card", "scn_already_refunded"}
    for scenario_id in set(SCENARIOS) - refusals:
        assert not all(
            state_holds(expectation, []) for expectation in SCENARIOS[scenario_id].expected_state
        ), scenario_id


async def test_a_scripted_call_meets_a_seeded_scenario():
    scenario = SCENARIOS["scn_double_charge_refund"]
    support_llm = ScriptedLLM(
        "I'm sorry about that. What's your domain?",
        uses(call("find_customer", domain="whitfieldlandscaping.example")),
        "Thanks. What's your billing zip and PIN?",
        uses(call("verify_identity", customer_id="cus_1001", billing_zip="97214", pin="4821")),
        uses(call("get_invoice", invoice_id="FH-4419")),
        "I see two charges of $49.99 on FH-4419. Shall I refund one?",
        uses(
            call(
                "issue_refund",
                invoice_id="FH-4419",
                reason="Duplicate charge",
                payment_id="pay_0003",
            )
        ),
        "Done. The $49.99 is on its way back to your card.",
    )
    caller_llm = ScriptedLLM(
        caller_says("Yeah, I got charged twice for the same invoice.", "state_problem"),
        caller_says("It's whitfieldlandscaping.example."),
        caller_says("Zip is 97214, PIN 4821.", "identify"),
        caller_says("Yes, please do.", "confirm_refund"),
        caller_says("Great, thanks. Bye.", hang_up=True, reason="got what I called for"),
    )
    db = create_call_db()
    support = SupportBot(support_llm, ToolDispatcher(db))
    caller = CallerBot(caller_llm, PERSONAS["per_angry_customer"], scenario, seed=1)

    result = await run_text_call(support, caller)
    db.close()

    assert result.end_reason == "caller_hung_up"
    assert result.completed_beats == ["state_problem", "identify", "confirm_refund"]
    for expectation in scenario.expected_state:
        assert state_holds(expectation, result.db_changes), expectation.id


def test_load_scenarios_returns_models():
    assert all(isinstance(item, Scenario) for item in load_scenarios().values())


def test_a_rule_cannot_depend_on_a_beat_the_scenario_lacks():
    seed = load_scenarios()["scn_dns_mx_fix"]
    rule = seed.must[0].model_copy(update={"needs_beat": "no_such_beat"})
    broken = seed.model_copy(update={"must": [rule]})
    assert "needs beat no_such_beat" in " ".join(scenario_problems(broken))


def test_every_seed_rule_says_which_beat_it_depends_on():
    """Without a link, any missed beat excuses the rule, so a real failure could read as invalid."""
    for scenario in load_scenarios().values():
        expected = [*scenario.must, *(e for e in scenario.expected_state if e.op != "no_change")]
        assert [r.id for r in expected if r.needs_beat is None] == [], scenario.id
