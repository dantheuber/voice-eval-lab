import pytest

from voicelab.core.models import (
    Beat,
    CallArtifact,
    CallMode,
    CriterionResult,
    EvaluatorRun,
    Evidence,
    Finding,
    Rule,
    Scenario,
    Speaker,
    StateExpectation,
    SystemEvent,
    ToolCallEvent,
    Verdict,
)
from voicelab.eval.config import VerdictConfig
from voicelab.eval.evaluators.caller_validity import FACTS_CRITERION_ID, MATERIAL_FACT_BREACH
from voicelab.eval.verdict import decide_verdict, unchecked_items, write_summary

REQUIRED = Beat(id="beat_required", instruction="state the problem")
OPTIONAL = Beat(id="beat_optional", instruction="mention the weather", required=False)


def scripted() -> CallArtifact:
    scenario = Scenario(title="t", goal="g", beats=[REQUIRED, OPTIONAL])
    return CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario)


def criterion(
    kind: str, status: str, criterion_id: str = "c1", text: str = "rule"
) -> CriterionResult:
    return CriterionResult(criterion_id=criterion_id, kind=kind, text=text, status=status)


def finding(severity: str, *, advisory: bool = False, criterion_id=None) -> Finding:
    return Finding(
        evaluator="e",
        category="c",
        severity=severity,
        advisory=advisory,
        title=f"{severity} issue",
        criterion_id=criterion_id,
    )


def run(status: str = "ok", reason: str = "", name: str = "e", **metrics) -> EvaluatorRun:
    return EvaluatorRun(name=name, status=status, reason=reason, metrics=metrics)


PASSED = criterion("general", "pass")


def decide(artifact=None, criteria=(PASSED,), findings=(), evaluators=(), config=None):
    return decide_verdict(
        artifact or scripted(), list(criteria), list(findings), list(evaluators), config
    )


def test_clean_call_passes():
    decision = decide()
    assert decision.verdict == Verdict.PASS
    assert decision.reasons == []


def test_missed_required_beat_is_invalid():
    decision = decide(criteria=[PASSED, criterion("beat", "fail", REQUIRED.id, "state it")])
    assert decision.verdict == Verdict.INVALID
    assert "state it" in decision.reasons[0]


def test_invalid_outranks_a_bot_failure():
    decision = decide(
        criteria=[criterion("beat", "fail", REQUIRED.id), criterion("must", "fail")],
        findings=[finding("critical")],
    )
    assert decision.verdict == Verdict.INVALID


def test_missed_optional_beat_is_not_invalid():
    decision = decide(criteria=[PASSED, criterion("beat", "fail", OPTIONAL.id)])
    assert decision.verdict == Verdict.PASS


def test_unknown_required_beat_is_review_not_invalid():
    decision = decide(criteria=[PASSED, criterion("beat", "unknown", REQUIRED.id)])
    assert decision.verdict == Verdict.NEEDS_REVIEW


def test_fact_sheet_breach_is_review_not_invalid():
    decision = decide(
        criteria=[PASSED, criterion("beat", "fail", FACTS_CRITERION_ID)],
        findings=[finding("major", advisory=True, criterion_id=FACTS_CRITERION_ID)],
    )
    assert decision.verdict == Verdict.NEEDS_REVIEW


@pytest.mark.parametrize("kind", ["must", "must_not", "state"])
def test_failed_rule_or_state_expectation_fails(kind):
    decision = decide(criteria=[PASSED, criterion(kind, "fail", text="do the thing")])
    assert decision.verdict == Verdict.FAIL
    assert f"failed {kind}: do the thing" in decision.reasons


@pytest.mark.parametrize("severity", ["major", "critical"])
def test_serious_non_advisory_finding_fails(severity):
    assert decide(findings=[finding(severity)]).verdict == Verdict.FAIL


def test_failed_general_criterion_fails_only_through_its_finding_severity():
    minor = decide(criteria=[PASSED, criterion("general", "fail")], findings=[finding("minor")])
    assert minor.verdict == Verdict.PASS
    major = decide(criteria=[PASSED, criterion("general", "fail")], findings=[finding("major")])
    assert major.verdict == Verdict.FAIL


def test_failed_rule_is_listed_once_with_its_finding():
    decision = decide(
        criteria=[criterion("must", "fail", "r1")], findings=[finding("major", criterion_id="r1")]
    )
    assert len(decision.reasons) == 1


@pytest.mark.parametrize("severity", ["info", "minor"])
def test_lesser_non_advisory_findings_do_not_fail(severity):
    assert decide(findings=[finding(severity)] * 5).verdict == Verdict.PASS


@pytest.mark.parametrize("severity", ["info", "minor", "major", "critical"])
def test_advisory_findings_never_fail(severity):
    decision = decide(findings=[finding(severity, advisory=True)] * 20)
    assert decision.verdict != Verdict.FAIL


def test_advisory_score_below_threshold_passes():
    decision = decide(findings=[finding("minor", advisory=True)] * 2)
    assert decision.verdict == Verdict.PASS


def test_advisory_score_at_threshold_goes_to_review():
    decision = decide(findings=[finding("minor", advisory=True)] * 3)
    assert decision.verdict == Verdict.NEEDS_REVIEW
    assert "advisory findings add up to 3" in decision.reasons[0]


def test_one_major_advisory_is_enough_for_review():
    assert decide(findings=[finding("major", advisory=True)]).verdict == Verdict.NEEDS_REVIEW


def test_info_advisories_do_not_count():
    assert decide(findings=[finding("info", advisory=True)] * 50).verdict == Verdict.PASS


def test_advisory_threshold_is_configurable():
    config = VerdictConfig(review_score=1)
    decision = decide(findings=[finding("minor", advisory=True)], config=config)
    assert decision.verdict == Verdict.NEEDS_REVIEW


def test_unknown_criterion_goes_to_review():
    decision = decide(criteria=[PASSED, criterion("must", "unknown", text="can't tell")])
    assert decision.verdict == Verdict.NEEDS_REVIEW
    assert "can't tell" in decision.reasons[0]


def test_evaluator_error_goes_to_review():
    decision = decide(evaluators=[run("error", "stt down", "transcribe")])
    assert decision.verdict == Verdict.NEEDS_REVIEW
    assert "transcribe errored: stt down" in decision.reasons[0]


def test_skipped_evaluator_alone_does_not_block_a_pass():
    assert decide(evaluators=[run("skipped", "no analyst")]).verdict == Verdict.PASS


def test_failure_outranks_review():
    decision = decide(
        criteria=[criterion("must", "fail"), criterion("must", "unknown", "c2")],
        evaluators=[run("error", "x")],
    )
    assert decision.verdict == Verdict.FAIL


def test_no_scenario_call_can_pass_on_general_criteria():
    artifact = CallArtifact(mode=CallMode.HUMAN)
    assert decide(artifact).verdict == Verdict.PASS


def test_call_with_nothing_confirmed_cannot_pass():
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_tool_log=False)
    decision = decide(artifact, criteria=[])
    assert decision.verdict == Verdict.NEEDS_REVIEW
    assert "no criterion could be confirmed" in decision.reasons


def test_no_scenario_call_with_failed_general_finding_fails():
    artifact = CallArtifact(mode=CallMode.HUMAN)
    decision = decide(artifact, findings=[finding("major")])
    assert decision.verdict == Verdict.FAIL


@pytest.mark.parametrize("findings", [(), (finding("major"),)])
def test_an_upload_with_no_tool_log_goes_to_a_person_either_way(findings):
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_tool_log=False)
    decision = decide(artifact, findings=list(findings))
    assert decision.verdict == Verdict.NEEDS_REVIEW
    assert any("no tool log" in reason for reason in decision.reasons)


def test_an_upload_with_a_tool_log_can_pass_and_fail():
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_tool_log=True)
    assert decide(artifact).verdict == Verdict.PASS
    assert decide(artifact, findings=[finding("major")]).verdict == Verdict.FAIL


def summary_for(artifact, criteria=(PASSED,), findings=(), evaluators=()) -> str:
    decision = decide(artifact, criteria, findings, evaluators)
    return write_summary(decision, artifact, list(criteria), list(findings), list(evaluators))


def test_summary_states_verdict_and_counts():
    text = summary_for(
        scripted(),
        criteria=[PASSED, criterion("must", "fail", text="say goodbye")],
        findings=[finding("major", criterion_id="c1"), finding("minor", advisory=True)],
    )
    assert text.startswith("Verdict: fail.")
    assert "failed must: say goodbye" in text
    assert "Criteria: 1 passed, 1 failed, 0 unknown." in text
    assert "Findings: 2, of which 1 advisory." in text


def test_summary_for_a_call_without_scenario_says_what_was_not_checked():
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_tool_log=False)
    evaluators = [
        run("skipped", "no scenario, so there is no expected state", "state"),
        run("skipped", "no audio analyst configured", "audio_listening"),
        run(name="transcript_judge", not_checked=["facts against tool results"]),
        run(name="timing", skipped={"overlap": "channels were inferred"}),
    ]
    text = summary_for(artifact, evaluators=evaluators)
    assert text.startswith("Verdict: needs_review.")
    assert "no tool log" in text
    assert "Not checked: scenario rules" in text
    assert "audio_listening (no audio analyst configured)" in text
    assert "facts against tool results" in text
    assert "overlap (channels were inferred)" in text
    assert text.count("no expected state") == 0


def test_summary_carries_the_pronunciation_limitation():
    text = summary_for(scripted(), evaluators=[run(limitation="ASR can hide errors.")])
    assert text.endswith("ASR can hide errors.")


def test_summary_caps_long_reason_lists():
    criteria = [criterion("must", "fail", f"c{i}", f"rule {i}") for i in range(8)]
    text = summary_for(scripted(), criteria=criteria)
    assert "and 3 more" in text


def test_unchecked_items_empty_for_a_full_call():
    assert unchecked_items(scripted(), [run()]) == []


def breach_finding(category: str) -> Finding:
    return Finding(
        evaluator="caller_validity",
        category=category,
        severity="major",
        advisory=True,
        title="Caller gave the wrong invoice",
        criterion_id=FACTS_CRITERION_ID,
    )


def test_material_fact_breach_invalidates_the_run():
    decision = decide(
        criteria=[PASSED, criterion("beat", "fail", FACTS_CRITERION_ID)],
        findings=[breach_finding(MATERIAL_FACT_BREACH)],
    )
    assert decision.verdict == Verdict.INVALID
    assert "Caller gave the wrong invoice" in decision.reasons[0]


def test_material_fact_breach_outranks_a_bot_failure():
    decision = decide(
        criteria=[criterion("must", "fail"), criterion("beat", "fail", FACTS_CRITERION_ID)],
        findings=[breach_finding(MATERIAL_FACT_BREACH), finding("critical")],
    )
    assert decision.verdict == Verdict.INVALID


def test_non_material_slip_reaches_review_however_the_score_is_configured():
    decision = decide(
        criteria=[PASSED, criterion("beat", "fail", FACTS_CRITERION_ID)],
        findings=[breach_finding("fact_sheet_slip")],
        config=VerdictConfig(review_score=100),
    )
    assert decision.verdict == Verdict.NEEDS_REVIEW
    assert any("fact sheet" in reason for reason in decision.reasons)


def test_non_material_slip_does_not_hide_a_bot_failure():
    decision = decide(
        criteria=[criterion("must", "fail"), criterion("beat", "fail", FACTS_CRITERION_ID)],
        findings=[breach_finding("fact_sheet_slip")],
    )
    assert decision.verdict == Verdict.FAIL


# Commission and omission ------------------------------------------------------------------------

REPORT = Beat(id="report", instruction="report the problem")
APPROVE = Beat(id="approve", instruction="approve the fix")
SAID_REFUND = Evidence(quote="the refund has been processed", speaker=Speaker.BOT, start_s=40.0)


def linked_scenario() -> Scenario:
    return Scenario(
        title="t",
        goal="g",
        beats=[REPORT, APPROVE],
        must=[
            Rule(id="greet", text="greet the caller", needs_beat="report"),
            Rule(id="fix", text="make the fix", needs_beat="approve"),
            Rule(id="unlinked", text="be helpful"),
        ],
        must_not=[Rule(id="no_refund", text="issue a refund")],
        expected_state=[
            StateExpectation(id="done", table="t", changes={"x": 1}, needs_beat="approve"),
            StateExpectation(id="keep", table="invoices", op="no_change", description="left alone"),
        ],
    )


def linked(*events) -> CallArtifact:
    return CallArtifact(
        mode=CallMode.LIVE_BOTS, scenario_snapshot=linked_scenario(), timeline=list(events)
    )


def beat(beat_id: str, status: str) -> CriterionResult:
    return criterion("beat", status, beat_id, f"beat {beat_id}")


def must_not_failed(*evidence: Evidence) -> CriterionResult:
    result = criterion("must_not", "fail", "no_refund", "issue a refund")
    return result.model_copy(update={"evidence": list(evidence)})


def no_change_failed(at_s: float | None = 30.0) -> CriterionResult:
    evidence = [Evidence(quote="issue_refund", start_s=at_s)] if at_s is not None else []
    result = criterion("state", "fail", "keep", "left alone")
    return result.model_copy(update={"evidence": evidence})


def serious_change() -> Finding:
    return Finding(
        evaluator="state",
        category="unexpected_change",
        severity="major",
        title="Unexpected delete on invoices",
    )


def test_breached_no_change_fails_even_when_the_caller_missed_a_beat():
    decision = decide(
        linked(), criteria=[beat("report", "pass"), beat("approve", "fail"), no_change_failed()]
    )
    assert decision.verdict == Verdict.FAIL
    assert "left alone" in decision.reasons[0]


def test_serious_unexpected_change_fails_even_when_the_caller_missed_a_beat():
    decision = decide(linked(), criteria=[beat("approve", "fail")], findings=[serious_change()])
    assert decision.verdict == Verdict.FAIL
    assert "Unexpected delete on invoices" in decision.reasons[0]


def test_minor_unexpected_change_is_a_side_effect_not_a_failure():
    side_effect = serious_change().model_copy(update={"severity": "minor", "advisory": True})
    decision = decide(linked(), criteria=[beat("approve", "fail")], findings=[side_effect])
    assert decision.verdict == Verdict.INVALID


def test_failed_must_not_with_a_verified_quote_fails_despite_a_missed_beat():
    decision = decide(linked(), criteria=[beat("approve", "fail"), must_not_failed(SAID_REFUND)])
    assert decision.verdict == Verdict.FAIL


def test_failed_must_not_without_evidence_is_not_a_proven_commission():
    decision = decide(linked(), criteria=[beat("approve", "fail"), must_not_failed()])
    assert decision.verdict == Verdict.INVALID


def test_omission_after_a_reached_beat_is_a_bot_failure():
    decision = decide(
        linked(),
        criteria=[
            beat("report", "pass"),
            beat("approve", "pass"),
            criterion("must", "fail", "fix", "make the fix"),
        ],
    )
    assert decision.verdict == Verdict.FAIL
    assert "failed must: make the fix" in decision.reasons


def test_omission_before_a_missed_beat_is_a_bot_failure():
    """The caller did reach what this rule depends on, so a later miss doesn't excuse it."""
    decision = decide(
        linked(),
        criteria=[
            beat("report", "pass"),
            beat("approve", "fail"),
            criterion("must", "fail", "greet", "greet the caller"),
        ],
    )
    assert decision.verdict == Verdict.FAIL
    assert "failed must: greet the caller" in decision.reasons


def test_omission_of_an_unreached_beat_is_invalid_and_says_why():
    decision = decide(
        linked(),
        criteria=[
            beat("report", "pass"),
            beat("approve", "fail"),
            criterion("must", "fail", "fix", "make the fix"),
            criterion("state", "fail", "done", "the row changed"),
        ],
    )
    assert decision.verdict == Verdict.INVALID
    assert any("beat approve" in reason for reason in decision.reasons)
    excused = next(reason for reason in decision.reasons if "not counted against" in reason)
    assert "make the fix" in excused and "the row changed" in excused


def test_rule_with_no_link_is_excused_by_any_missed_required_beat():
    decision = decide(
        linked(),
        criteria=[beat("approve", "fail"), criterion("must", "fail", "unlinked", "be helpful")],
    )
    assert decision.verdict == Verdict.INVALID


def test_omission_with_an_undecided_beat_goes_to_review():
    decision = decide(
        linked(),
        criteria=[
            PASSED,
            beat("report", "pass"),
            beat("approve", "unknown"),
            criterion("must", "fail", "fix", "make the fix"),
        ],
    )
    assert decision.verdict == Verdict.NEEDS_REVIEW


def test_call_with_no_beat_results_still_blames_the_bot():
    """A human caller has no script, so every failed rule stands."""
    artifact = linked()
    artifact.mode = CallMode.HUMAN
    decision = decide(artifact, criteria=[criterion("must", "fail", "fix", "make the fix")])
    assert decision.verdict == Verdict.FAIL


def test_rule_links_are_kept_apart_by_kind():
    """A state expectation and a rule can share an id, and each has its own link."""
    scenario = linked_scenario()
    scenario.must = [Rule(id="done", text="say it's done", needs_beat="report")]
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario)
    decision = decide(
        artifact,
        criteria=[
            beat("report", "pass"),
            beat("approve", "fail"),
            criterion("state", "fail", "done", "the row changed"),
        ],
    )
    assert decision.verdict == Verdict.INVALID


# Who ended the call ------------------------------------------------------------------------------


def test_bot_ending_the_call_before_a_beat_is_a_bot_failure():
    artifact = linked(SystemEvent(at_s=90.0, kind="bot_ended", detail="refund processed"))
    decision = decide(artifact, criteria=[beat("report", "pass"), beat("approve", "fail")])
    assert decision.verdict == Verdict.FAIL
    assert "the bot ended the call before the caller reached" in decision.reasons[0]
    assert "beat approve" in decision.reasons[0]


def test_bot_ending_the_call_after_only_an_optional_beat_is_fine():
    artifact = scripted()
    artifact.timeline.append(SystemEvent(at_s=90.0, kind="bot_ended"))
    decision = decide(artifact, criteria=[PASSED, criterion("beat", "fail", OPTIONAL.id)])
    assert decision.verdict == Verdict.PASS


def test_caller_hanging_up_before_a_beat_is_invalid_and_says_so():
    artifact = linked(SystemEvent(at_s=97.0, kind="caller_hung_up", detail="decided to hang up"))
    decision = decide(artifact, criteria=[beat("report", "pass"), beat("approve", "fail")])
    assert decision.verdict == Verdict.INVALID
    assert any("the caller hung up" in r and "decided to hang up" in r for r in decision.reasons)


def test_hitting_the_turn_limit_before_a_beat_is_invalid_and_says_so():
    artifact = linked(SystemEvent(at_s=97.0, kind="max_turns"))
    decision = decide(artifact, criteria=[beat("approve", "fail")])
    assert decision.verdict == Verdict.INVALID
    assert any("turn limit" in reason for reason in decision.reasons)


# Fact sheet breaches around a commission ----------------------------------------------------------


def timed_breach(at_s: float | None) -> Finding:
    evidence = [Evidence(quote="my pin is 9999", speaker=Speaker.CALLER, start_s=at_s)]
    return Finding(
        evaluator="caller_validity",
        category=MATERIAL_FACT_BREACH,
        severity="major",
        advisory=True,
        title="Caller made up a PIN",
        evidence=evidence if at_s is not None else [],
        criterion_id=FACTS_CRITERION_ID,
    )


def test_material_breach_stays_invalid_when_the_bot_did_nothing_forbidden():
    decision = decide(
        linked(),
        criteria=[beat("approve", "pass"), criterion("must", "fail", "fix", "make the fix")],
        findings=[timed_breach(50.0)],
    )
    assert decision.verdict == Verdict.INVALID


def test_commission_before_the_breach_fails_and_mentions_the_breach():
    decision = decide(linked(), criteria=[no_change_failed(30.0)], findings=[timed_breach(50.0)])
    assert decision.verdict == Verdict.FAIL
    assert any("before the caller" in reason for reason in decision.reasons)


def test_commission_after_the_breach_is_invalid_because_the_data_was_made_up():
    decision = decide(linked(), criteria=[no_change_failed(60.0)], findings=[timed_breach(50.0)])
    assert decision.verdict == Verdict.INVALID
    assert any("after" in reason and "made-up" in reason for reason in decision.reasons)


def test_commission_with_no_time_cannot_be_put_before_the_breach():
    decision = decide(linked(), criteria=[no_change_failed(None)], findings=[timed_breach(50.0)])
    assert decision.verdict == Verdict.INVALID


def test_summary_explains_a_commission_past_a_missed_beat():
    artifact = linked()
    text = summary_for(artifact, criteria=[beat("approve", "fail"), no_change_failed()])
    assert text.startswith("Verdict: fail.")
    assert "Why: the bot did something forbidden" in text


def test_a_broken_no_change_says_what_was_broken():
    decision = decide(linked(), criteria=[no_change_failed()])
    assert decision.reasons[0] == "the bot did something forbidden, which broke: left alone"


def test_a_missed_beat_is_named_when_it_doesn_t_excuse_a_commission():
    decision = decide(linked(), criteria=[beat("approve", "fail"), no_change_failed()])
    assert decision.reasons[-1] == (
        "the caller also missed a required checkpoint, which doesn't excuse this: beat approve"
    )


def test_a_summary_reads_cleanly_when_criteria_end_in_a_period():
    failed = criterion("must", "fail", "c1", "Say goodbye.")
    decision = decide(criteria=[failed])
    text = write_summary(decision, scripted(), [failed], [], [])
    assert "Why: failed must: Say goodbye." in text
    assert ".;" not in text and ".." not in text


def changed_unverified(at_s: float = 48.0, *events) -> CallArtifact:
    tool = ToolCallEvent(at_s=at_s, name="reset_mailbox_password", result={"ok": True})
    return linked(tool, *events)


def test_changing_the_account_unverified_fails_despite_a_missed_beat():
    """Seen on a real call: a password reset, with no PIN given because nobody asked for one."""
    decision = decide(changed_unverified(), criteria=[beat("report", "fail")])
    assert decision.verdict == Verdict.FAIL
    assert "reset_mailbox_password" in decision.reasons[0]
    assert "before verifying" in decision.reasons[0]


def test_changing_the_account_after_verifying_is_no_commission():
    tool = ToolCallEvent(at_s=50.0, name="reset_mailbox_password", result={"ok": True})
    check = ToolCallEvent(at_s=40.0, name="verify_identity", result={"verified": True})
    decision = decide(linked(check, tool), criteria=[beat("report", "fail")])
    assert decision.verdict == Verdict.INVALID


def test_an_unverified_change_before_a_breach_is_still_the_bots_doing():
    decision = decide(changed_unverified(30.0), criteria=[PASSED], findings=[timed_breach(50.0)])
    assert decision.verdict == Verdict.FAIL


def test_a_rule_whose_beat_never_got_a_result_goes_to_review():
    """Caller validity died before this beat, so nobody knows if the caller reached it."""
    failed = criterion("must", "fail", "fix", "make the fix")
    decision = decide(
        linked(), criteria=[failed], evaluators=[run("error", "judge down", "caller_validity")]
    )
    assert decision.verdict == Verdict.NEEDS_REVIEW
    assert any("caller_validity errored" in reason for reason in decision.reasons)


def test_an_absent_beat_says_what_could_not_be_settled():
    decision = decide(linked(), criteria=[PASSED, criterion("must", "fail", "fix", "make the fix")])
    assert decision.verdict == Verdict.NEEDS_REVIEW
    assert any("make the fix" in reason for reason in decision.reasons)


def test_only_a_passed_beat_blames_the_bot():
    decision = decide(
        linked(), criteria=[beat("approve", "pass"), criterion("must", "fail", "fix", "make it")]
    )
    assert decision.verdict == Verdict.FAIL


def test_an_unlinked_rule_is_not_blamed_when_caller_validity_died():
    decision = decide(
        linked(),
        criteria=[criterion("must", "fail", "unlinked", "be helpful")],
        evaluators=[run("error", "judge down", "caller_validity")],
    )
    assert decision.verdict == Verdict.NEEDS_REVIEW


def test_a_human_call_has_no_beats_to_wait_for():
    artifact = linked()
    artifact.mode = CallMode.UPLOAD
    decision = decide(artifact, criteria=[criterion("must", "fail", "fix", "make the fix")])
    assert decision.verdict == Verdict.FAIL
