import pytest
from evalfakes import write_call

from voicelab.core.models import (
    CallArtifact,
    CallMode,
    DbChange,
    Scenario,
    StateExpectation,
    ToolCallEvent,
)
from voicelab.eval.config import EvalConfig
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput, Skip
from voicelab.eval.evaluators.state import StateEvaluator, same_value

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


def tool(at_s: float, name: str, *, arguments=None, result=None, error=None) -> ToolCallEvent:
    return ToolCallEvent(
        at_s=at_s, name=name, arguments=arguments or {}, result=result, error=error
    )


async def test_breached_no_change_says_when_the_write_happened(store):
    """The database diff has no times, and the verdict needs to know what came first."""
    expected = StateExpectation(table="refunds", where={"payment_id": "pay_2"}, op="no_change")
    artifact = call(REFUND, expected=scenario(expected))
    artifact.timeline = [
        tool(5.0, "get_invoice", result={"note": "ref_1"}),
        tool(8.0, "issue_refund", error="refused"),
        tool(12.0, "issue_refund", result={"refund_id": "ref_1"}),
    ]
    out = await evaluate(store, artifact)
    assert [e.start_s for e in out.criteria[0].evidence] == [12.0]
    assert out.criteria[0].evidence[0].quote.startswith("issue_refund")
    assert out.findings[0].evidence == out.criteria[0].evidence


async def test_unexpected_change_with_an_unmentioned_key_points_at_the_first_write(store):
    deleted = DbChange(table="dns_records", key={"id": 12}, kind="delete", before={"id": 12})
    artifact = call(deleted, expected=scenario())
    artifact.timeline = [
        tool(5.0, "get_dns_records"),
        tool(9.0, "update_dns_record", arguments={"domain": "a.example"}),
    ]
    out = await evaluate(store, artifact)
    assert [e.start_s for e in out.findings[0].evidence] == [9.0]


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
