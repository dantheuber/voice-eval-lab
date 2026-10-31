import pytest
from evalfakes import write_call

from voicelab.agents import CallerBot, SupportBot, load_scenarios, run_text_call
from voicelab.company import ToolDispatcher, create_call_db
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    DbChange,
    Evidence,
    Finding,
    Persona,
    RowRef,
    Scenario,
    StateExpectation,
    ToolCallEvent,
    Verdict,
    VoiceRef,
)
from voicelab.eval.config import EvalConfig
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput, Skip
from voicelab.eval.evaluators.state import StateEvaluator, same_value
from voicelab.eval.verdict import MATERIAL_FACT_BREACH, decide_verdict
from voicelab.providers.adapters.fake import FakeLLM

REFUND = DbChange(
    table="refunds",
    key={"id": "ref_1"},
    kind="insert",
    after={"id": "ref_1", "payment_id": "pay_2", "amount_cents": 4999},
)
MAILBOX = DbChange(
    table="mailboxes",
    key={"id": "mb_1"},
    kind="update",
    before={"id": "mb_1", "locked": 1, "customer_id": "cus_1"},
    after={"id": "mb_1", "locked": 0, "customer_id": "cus_1"},
)


def scenario(*expectations: StateExpectation) -> Scenario:
    return Scenario(title="t", goal="g", expected_state=list(expectations))


async def evaluate(store, artifact, config=None) -> EvaluatorOutput:
    artifact, paths = write_call(store, artifact, duration_s=1.0)
    out = EvaluatorOutput()
    await StateEvaluator().run(
        EvalContext(artifact, paths, EvalDeps(config=config or EvalConfig())), out
    )
    return out


def call(*changes: DbChange, expected: Scenario | None = None) -> CallArtifact:
    return CallArtifact(
        mode=CallMode.LIVE_BOTS, db_changes=list(changes), scenario_snapshot=expected
    )


async def test_changed_passes_when_the_new_value_matches(store):
    expected = StateExpectation(
        table="mailboxes", where={"id": "mb_1"}, op="changed", changes={"locked": 0}
    )
    out = await evaluate(store, call(MAILBOX, expected=scenario(expected)))
    assert [c.status for c in out.criteria] == ["pass"]
    assert out.findings == []


async def test_changed_fails_when_the_value_is_wrong(store):
    expected = StateExpectation(
        table="mailboxes", where={"id": "mb_1"}, op="changed", changes={"locked": 1}
    )
    out = await evaluate(store, call(MAILBOX, expected=scenario(expected)))
    assert [c.status for c in out.criteria] == ["fail"]
    state_failures = [f for f in out.findings if f.category == "state_expectation"]
    assert state_failures[0].severity == "major"
    assert state_failures[0].advisory is False
    assert state_failures[0].criterion_id == expected.id


async def test_changed_fails_when_nothing_happened(store):
    expected = StateExpectation(table="mailboxes", where={"id": "mb_1"}, op="changed")
    out = await evaluate(store, call(expected=scenario(expected)))
    assert out.criteria[0].status == "fail"
    assert "no change" in out.criteria[0].reason


async def test_where_can_name_a_field_the_change_rewrote(store):
    expected = StateExpectation(
        table="mailboxes", where={"locked": 1}, op="changed", changes={"locked": 0}
    )
    out = await evaluate(store, call(MAILBOX, expected=scenario(expected)))
    assert out.criteria[0].status == "pass"


async def test_no_change_passes_on_an_untouched_row(store):
    expected = StateExpectation(table="mailboxes", where={"id": "mb_2"}, op="no_change")
    out = await evaluate(store, call(MAILBOX, expected=scenario(expected)))
    assert out.criteria[0].status == "pass"


async def test_no_change_fails_when_the_row_moved(store):
    expected = StateExpectation(table="mailboxes", where={"id": "mb_1"}, op="no_change")
    out = await evaluate(store, call(MAILBOX, expected=scenario(expected)))
    assert out.criteria[0].status == "fail"
    assert "mb_1" in out.criteria[0].reason


async def test_no_change_fails_when_a_matching_row_was_inserted(store):
    expected = StateExpectation(table="refunds", where={"payment_id": "pay_2"}, op="no_change")
    out = await evaluate(store, call(REFUND, expected=scenario(expected)))
    assert out.criteria[0].status == "fail"
    assert "ref_1" in out.criteria[0].reason


async def test_row_added_matches_on_fields(store):
    expected = StateExpectation(
        table="refunds", where={"payment_id": "pay_2", "amount_cents": "4999"}, op="row_added"
    )
    out = await evaluate(store, call(REFUND, expected=scenario(expected)))
    assert out.criteria[0].status == "pass"


async def test_row_added_rejects_an_update(store):
    expected = StateExpectation(table="mailboxes", where={"id": "mb_1"}, op="row_added")
    out = await evaluate(store, call(MAILBOX, expected=scenario(expected)))
    assert out.criteria[0].status == "fail"


async def test_unexpected_side_effect_is_a_minor_advisory(store):
    out = await evaluate(store, call(MAILBOX, expected=scenario()))
    assert [(f.severity, f.advisory) for f in out.findings] == [("minor", True)]


async def test_unexpected_write_to_another_customer_can_fail_the_call(store):
    artifact = call(MAILBOX, expected=scenario())
    artifact.scenario_snapshot.customer_id = "cus_2"
    out = await evaluate(store, artifact)
    assert [(f.severity, f.advisory) for f in out.findings] == [("major", False)]


async def test_unexpected_write_to_the_callers_own_row_stays_minor(store):
    artifact = call(MAILBOX, expected=scenario())
    artifact.scenario_snapshot.customer_id = "cus_1"
    out = await evaluate(store, artifact)
    assert out.findings[0].severity == "minor"


async def test_unexpected_deletion_can_fail_the_call(store):
    deleted = DbChange(table="dns_records", key={"id": 1}, kind="delete", before={"id": 1})
    out = await evaluate(store, call(deleted, expected=scenario()))
    assert [(f.severity, f.advisory) for f in out.findings] == [("major", False)]


def tool(
    at_s: float, name: str, *, arguments=None, result=None, error=None, changed=None
) -> ToolCallEvent:
    return ToolCallEvent(
        at_s=at_s,
        name=name,
        arguments=arguments or {},
        result=result,
        error=error,
        changed=changed,
    )


def row(table: str, **key) -> RowRef:
    return RowRef(table=table, key=key)


async def test_breached_no_change_says_when_the_write_happened(store):
    """The database diff has no times, and the verdict needs to know what came first."""
    expected = StateExpectation(table="refunds", where={"payment_id": "pay_2"}, op="no_change")
    artifact = call(REFUND, expected=scenario(expected))
    artifact.timeline = [
        tool(5.0, "get_invoice", result={"note": "ref_1"}, changed=[]),
        tool(8.0, "issue_refund", error="refused", changed=[]),
        tool(12.0, "issue_refund", result={"id": "x"}, changed=[row("refunds", id="ref_1")]),
    ]
    out = await evaluate(store, artifact)
    assert [e.start_s for e in out.criteria[0].evidence] == [12.0]
    assert out.criteria[0].evidence[0].quote.startswith("issue_refund")
    assert out.findings[0].evidence == out.criteria[0].evidence


INVOICE = DbChange(
    table="invoices",
    key={"id": "FH-4419"},
    kind="update",
    before={"id": "FH-4419", "status": "paid"},
    after={"id": "FH-4419", "status": "refunded"},
)


def ticket_then_refund() -> list[ToolCallEvent]:
    """A ticket that mentions the invoice, then the refund that really changed it."""
    return [
        tool(5.0, "verify_identity", result={"verified": True}, changed=[]),
        tool(10.0, "create_ticket", arguments={"summary": "FH-4419"}, changed=[]),
        tool(30.0, "issue_refund", changed=[row("invoices", id="FH-4419")]),
    ]


async def test_mentioning_a_row_is_not_changing_it(store):
    expected = StateExpectation(table="invoices", where={"id": "FH-4419"}, op="no_change")
    artifact = call(INVOICE, expected=scenario(expected))
    artifact.timeline = ticket_then_refund()
    out = await evaluate(store, artifact)
    assert [e.start_s for e in out.criteria[0].evidence] == [30.0]


async def test_a_refund_after_a_breach_is_on_made_up_data(store):
    """Blaming the 10 s ticket for the invoice made this `fail` instead of `invalid`."""
    expected = StateExpectation(table="invoices", where={"id": "FH-4419"}, op="no_change")
    artifact = call(INVOICE, expected=scenario(expected))
    artifact.timeline = ticket_then_refund()
    out = await evaluate(store, artifact)
    breach = Finding(
        evaluator="caller_validity",
        category=MATERIAL_FACT_BREACH,
        severity="major",
        advisory=True,
        title="Caller made up a PIN",
        evidence=[Evidence(quote="my pin is 9999", start_s=20.0)],
    )
    decision = decide_verdict(artifact, out.criteria, [*out.findings, breach], [])
    assert decision.verdict == Verdict.INVALID


@pytest.mark.parametrize("name", ["issue_refund", "create_ticket"])
async def test_a_call_with_no_recorded_changes_gets_no_guessed_time(store, name):
    expected = StateExpectation(table="invoices", where={"id": "FH-4419"}, op="no_change")
    artifact = call(INVOICE, expected=scenario(expected))
    artifact.timeline = [tool(10.0, name, arguments={"invoice_id": "FH-4419"})]
    out = await evaluate(store, artifact)
    assert out.criteria[0].status == "fail"
    assert out.criteria[0].evidence == []


async def test_an_unexpected_change_is_timed_only_from_its_own_tool_call(store):
    deleted = DbChange(table="dns_records", key={"id": 12}, kind="delete", before={"id": 12})
    artifact = call(deleted, expected=scenario())
    artifact.timeline = [
        tool(5.0, "update_dns_record", changed=[row("dns_records", id=99)]),
        tool(9.0, "update_dns_record", changed=[row("dns_records", id=12)]),
    ]
    out = await evaluate(store, artifact)
    assert [e.start_s for e in out.findings[0].evidence] == [9.0]


def test_a_tool_reports_the_rows_it_changed():
    db = create_call_db()
    dispatcher = ToolDispatcher(db)
    dispatcher.dispatch("get_invoice", {"invoice_id": "FH-4419"})
    assert dispatcher.last_changes == []
    dispatcher.dispatch("issue_refund", {"invoice_id": "FH-4419", "reason": "duplicate"})
    changed = {(r.table, tuple(r.key.values())) for r in dispatcher.last_changes}
    assert ("refunds", ("rfd_0003",)) in changed or any(t == "refunds" for t, _ in changed)
    assert all(isinstance(r, RowRef) for r in dispatcher.last_changes)


def test_a_rejected_tool_changed_nothing():
    dispatcher = ToolDispatcher(create_call_db())
    dispatcher.dispatch("issue_refund", {"invoice_id": "FH-9999", "reason": "nope"})
    assert dispatcher.last_changes == []


async def test_the_support_bot_records_what_each_tool_changed():
    scenario_seed = load_scenarios()["scn_double_charge_refund"]
    db = create_call_db()
    support = SupportBot(FakeLLM(), ToolDispatcher(db))
    persona = Persona(name="Tester", patience=5, voice=VoiceRef(voice_id="default_male_1"))
    result = await run_text_call(support, CallerBot(FakeLLM(), persona, scenario_seed, seed=1))
    calls = [e for e in result.timeline if isinstance(e, ToolCallEvent)]
    refund = next(e for e in calls if e.name == "issue_refund")
    assert [r.table for r in refund.changed] == ["refunds"]
    assert all(e.changed == [] for e in calls if e.name.startswith(("find_", "verify_")))


async def test_a_change_with_no_tool_call_has_no_time(store):
    expected = StateExpectation(table="refunds", where={"payment_id": "pay_2"}, op="no_change")
    out = await evaluate(store, call(REFUND, expected=scenario(expected)))
    assert out.criteria[0].evidence == []


async def test_unexpected_change_is_reported_as_advisory(store):
    expected = StateExpectation(table="refunds", where={"payment_id": "pay_2"}, op="row_added")
    out = await evaluate(store, call(REFUND, MAILBOX, expected=scenario(expected)))
    assert out.criteria[0].status == "pass"
    unexpected = [f for f in out.findings if f.category == "unexpected_change"]
    assert len(unexpected) == 1
    assert unexpected[0].advisory is True
    assert "mailboxes" in unexpected[0].title
    assert out.metrics["unexpected_changes"] == 1


async def test_any_change_is_unexpected_when_none_were_listed(store):
    out = await evaluate(store, call(REFUND, expected=scenario()))
    assert [f.category for f in out.findings] == ["unexpected_change"]


async def test_configured_tables_are_not_unexpected(store):
    config = EvalConfig(ignored_state_tables=["tickets"])
    ticket = DbChange(table="tickets", key={"id": "t1"}, kind="insert", after={"id": "t1"})
    out = await evaluate(store, call(ticket, expected=scenario()), config)
    assert out.findings == []


async def test_a_breached_no_change_is_not_also_called_unexpected(store):
    expected = StateExpectation(table="mailboxes", where={"id": "mb_1"}, op="no_change")
    out = await evaluate(store, call(MAILBOX, expected=scenario(expected)))
    assert [f.category for f in out.findings] == ["state_expectation"]


async def test_skips_without_a_scenario(store):
    with pytest.raises(Skip, match="no scenario"):
        await evaluate(store, call())


async def test_skips_without_a_tool_log_but_leaves_expectations_unknown(store):
    artifact = call(expected=scenario(StateExpectation(table="refunds")))
    artifact.has_tool_log = False
    artifact, paths = write_call(store, artifact, duration_s=1.0)
    out = EvaluatorOutput()
    with pytest.raises(Skip, match="no tool log"):
        await StateEvaluator().run(EvalContext(artifact, paths, EvalDeps()), out)
    assert [c.status for c in out.criteria] == ["unknown"]


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        (1, 1, True),
        ("12.50", 12.5, True),
        (0, "0", True),
        ("open", "closed", False),
        (None, 0, False),
    ],
)
def test_same_value_tolerates_type_differences_only(a, b, same):
    assert same_value(a, b) is same
