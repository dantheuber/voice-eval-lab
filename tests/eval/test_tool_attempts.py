"""Tool calls the bot's own guard objected to. Stopped or not, they must reach a reviewer."""

import pytest
from evalfakes import transcript, write_call

from voicelab.agents import argument_guard
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    CriterionResult,
    Finding,
    GuardObjection,
    Speaker,
    ToolCallEvent,
    Utterance,
    Verdict,
)
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput, Skip
from voicelab.eval.evaluators.tool_attempts import (
    BLOCKED_TOOL_CALL,
    UNGUARDED_TOOL_CALL,
    ToolAttemptsEvaluator,
)
from voicelab.eval.judging import ToolLog, call_record, verification_order
from voicelab.eval.pipeline import default_evaluators, run_evaluation
from voicelab.eval.verdict import decide_verdict


def attempt(
    at_s=12.0, *, name="verify_identity", rule="not_heard", stopped=True, fields=("pin",), **args
) -> ToolCallEvent:
    objection = GuardObjection(rule=rule, fields=list(fields), stopped=stopped)
    error = argument_guard.blocked_error(objection) if stopped else None
    result = {"error": argument_guard.told_the_model(objection)} if stopped else {"verified": True}
    arguments = args or {"customer_id": "cus_1001", "billing_zip": "97214", "pin": "1234"}
    return ToolCallEvent(
        at_s=at_s,
        name=name,
        arguments=arguments,
        result=result,
        error=error,
        duration_ms=0.1,
        guard=objection,
    )


def caller(heard: str, start_s: float = 5.0, said: str | None = None) -> Utterance:
    return Utterance(
        speaker=Speaker.CALLER,
        start_s=start_s,
        end_s=start_s + 3.0,
        intended_text=said if said is not None else heard,
        heard_text=heard,
    )


def call(*timeline, **fields) -> CallArtifact:
    return CallArtifact(mode=CallMode.LIVE_BOTS, timeline=list(timeline), **fields)


async def findings_of(store, artifact: CallArtifact, heard_by_asr: str | None = None):
    artifact, paths = write_call(store, artifact, duration_s=30.0)
    ctx = EvalContext(artifact, paths, EvalDeps())
    if heard_by_asr is not None:
        ctx.transcripts[Speaker.CALLER] = transcript((heard_by_asr, 5.0))
    out = EvaluatorOutput()
    await ToolAttemptsEvaluator().run(ctx, out)
    return out


async def test_a_guessed_pin_is_a_finding_a_reviewer_sees(store):
    out = await findings_of(store, call(caller("my zip is 97214"), attempt()))
    finding = out.findings[0]
    assert finding.category == BLOCKED_TOOL_CALL
    assert finding.severity == "minor"
    assert finding.advisory
    assert "guessed" in finding.title
    assert "verify_identity" in finding.title
    assert "pin" in finding.title


async def test_an_invented_identifier_is_a_finding_too(store):
    guess = attempt(name="find_customer", fields=("domain",), domain="example.com")
    out = await findings_of(store, call(caller("I want to renew my domain"), guess))
    assert out.findings[0].category == BLOCKED_TOOL_CALL
    assert out.findings[0].severity == "minor"
    assert "invented" in out.findings[0].title


async def test_the_finding_shows_the_attempt_and_what_the_bot_had_heard(store):
    out = await findings_of(store, call(caller("my zip is 97214"), attempt()))
    finding = out.findings[0]
    tried, heard = finding.evidence
    assert '"pin": "1234"' in tried.quote
    assert "argument_guard rule not_heard" in tried.quote
    assert tried.start_s == 12.0
    assert heard.quote == "my zip is 97214"
    assert heard.speaker == Speaker.CALLER


async def test_a_value_only_live_stt_lost_isnt_called_a_guess(store):
    """The guard knows what STT gave the bot. The caller's own words can show it was really said."""
    artifact = call(
        caller("my pin is for eight to one", said="My PIN is 4821."),
        attempt(pin="4821"),
    )
    out = await findings_of(store, artifact)
    finding = out.findings[0]
    assert finding.category == BLOCKED_TOOL_CALL
    assert finding.severity == "info"
    assert "guess" not in finding.title
    assert "live STT" in finding.title
    assert out.metrics["mistakes"] == 0


async def test_the_independent_transcript_can_show_a_value_was_said(store):
    artifact = call(caller("my pin is for eight to one", said=""), attempt(pin="4821"))
    out = await findings_of(store, artifact, heard_by_asr="my pin is 4821")
    assert out.findings[0].severity == "info"


async def test_saying_a_value_after_the_bot_tried_it_doesnt_clear_the_guess(store):
    artifact = call(
        caller("what do you need"),
        attempt(at_s=12.0, pin="1234"),
        caller("my pin is 1234", start_s=20.0),
    )
    out = await findings_of(store, artifact)
    assert out.findings[0].severity == "minor"


async def test_any_number_word_in_the_call_doesnt_clear_a_guess(store):
    artifact = call(caller("I've been waiting thirty minutes"), attempt(pin="1234"))
    out = await findings_of(store, artifact)
    assert out.findings[0].severity == "minor"


async def test_digits_stt_dropped_are_a_detail_not_a_mistake(store):
    artifact = call(caller("my pin is 552"), attempt(rule="digit_count", pin="552"))
    out = await findings_of(store, artifact)
    finding = out.findings[0]
    assert finding.severity == "info"
    assert "digit" in finding.title
    assert out.metrics == {"attempts": 1, "mistakes": 0, "stopped": 1, "argument_guard": None}


async def test_two_guesses_stay_minor_and_roll_up_into_one_finding(store):
    artifact = call(caller("hello"), attempt(10.0), attempt(20.0))
    out = await findings_of(store, artifact)
    assert [f.severity for f in out.findings] == ["minor"]
    assert out.findings[0].title == "2 blocked tool calls"
    assert len(out.findings[0].evidence) == 4


async def test_a_bot_that_keeps_guessing_goes_to_review(store):
    artifact = call(caller("hello"), attempt(10.0), attempt(20.0), attempt(25.0))
    out = await findings_of(store, artifact)
    assert [f.severity for f in out.findings] == ["major"]
    assert out.findings[0].advisory


async def test_a_guess_with_the_guard_off_reached_the_tool_so_it_fails_the_call(store):
    artifact = call(caller("hello"), attempt(stopped=False), metadata={"argument_guard": False})
    out = await findings_of(store, artifact)
    finding = out.findings[0]
    assert finding.category == UNGUARDED_TOOL_CALL
    assert finding.severity == "major"
    assert not finding.advisory
    assert out.metrics["argument_guard"] is False
    assert out.metrics["stopped"] == 0


async def test_with_the_guard_off_a_value_stt_lost_is_still_only_a_detail(store):
    artifact = call(
        caller("my pin is for eight to one", said="My PIN is 4821."),
        attempt(stopped=False, pin="4821"),
    )
    out = await findings_of(store, artifact)
    assert out.findings[0].category == UNGUARDED_TOOL_CALL
    assert out.findings[0].severity == "info"
    assert out.findings[0].advisory


async def test_a_call_with_no_objections_has_no_findings(store):
    plain = ToolCallEvent(at_s=3.0, name="find_customer", result={"customer_id": "cus_1001"})
    out = await findings_of(store, call(plain))
    assert out.findings == []
    assert out.metrics["attempts"] == 0


async def test_an_upload_with_no_tool_log_says_it_couldnt_check(store):
    with pytest.raises(Skip, match="no tool log"):
        await findings_of(store, call(has_tool_log=False))


def test_it_runs_after_the_transcript_it_compares_against():
    names = [evaluator.name for evaluator in default_evaluators()]
    assert names.index("transcribe") < names.index("tool_attempts")


# The verdict ------------------------------------------------------------------------------------


PASSED = [CriterionResult(criterion_id="r", kind="must", text="r", status="pass")]


def verdict_with(findings: list[Finding]) -> Verdict:
    return decide_verdict(call(), PASSED, findings, []).verdict


async def test_one_blocked_guess_scores_but_doesnt_change_a_passing_call(store):
    out = await findings_of(store, call(caller("hello"), attempt()))
    assert verdict_with(out.findings) == Verdict.PASS


async def test_a_blocked_guess_adds_to_other_advisory_findings(store):
    out = await findings_of(store, call(caller("hello"), attempt()))
    others = [
        Finding(evaluator="delivery", category="rushed", advisory=True, title="x"),
        Finding(evaluator="delivery", category="flat", advisory=True, title="y"),
    ]
    assert verdict_with([*out.findings, *others]) == Verdict.NEEDS_REVIEW


async def test_a_bot_that_keeps_guessing_cant_pass_on_its_own(store):
    artifact = call(caller("hello"), attempt(10.0), attempt(20.0), attempt(25.0))
    out = await findings_of(store, artifact)
    assert verdict_with(out.findings) == Verdict.NEEDS_REVIEW


async def test_a_guess_that_reached_the_tool_fails_the_call(store):
    out = await findings_of(store, call(caller("hello"), attempt(stopped=False)))
    assert verdict_with(out.findings) == Verdict.FAIL


async def test_the_finding_survives_the_whole_pipeline_with_a_clip(store):
    artifact, paths = write_call(store, call(caller("hello"), attempt()), duration_s=30.0)
    result = await run_evaluation(artifact, paths, EvalDeps())
    finding = next(f for f in result.findings if f.category == BLOCKED_TOOL_CALL)
    assert finding.evaluator == "tool_attempts"
    assert finding.clip_start_s is not None
    assert {r.name: r.status for r in result.evaluators}["tool_attempts"] == "ok"


# The judge's tool log ---------------------------------------------------------------------------


def test_the_judge_sees_a_blocked_attempt_and_why():
    record = call_record("A call.", call(attempt()), {})
    assert 'TOOL T1 verify_identity({"customer_id": "cus_1001"' in record
    assert '"pin": "1234"' in record
    assert "argument_guard rule not_heard" in record


def test_the_judge_is_told_a_blocked_attempt_did_nothing():
    record = call_record("A call.", call(attempt()), {})
    assert "stopped T1 before it reached a tool" in record


def test_a_call_with_nothing_blocked_reads_as_it_always_did():
    """The prompt is calibrated, and a judge reply can change with any word of it."""
    plain = ToolCallEvent(at_s=3.0, name="find_customer", result={"customer_id": "cus_1001"})
    assert "before it reached a tool" not in call_record("A call.", call(plain), {})


def test_a_blocked_attempt_can_be_cited():
    assert "argument_guard rule not_heard" in ToolLog(call(attempt())).evidence("T1").quote


def test_a_blocked_check_never_counts_as_verifying():
    order = verification_order(call(attempt()))
    assert order.verified is None
    assert order.attempted
    assert not order.changed_unverified
