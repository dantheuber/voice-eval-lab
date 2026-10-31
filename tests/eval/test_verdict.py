import pytest

from voicelab.core.models import (
    Beat,
    CallArtifact,
    CallMode,
    CriterionResult,
    EvaluatorRun,
    Finding,
    Scenario,
    Verdict,
)
from voicelab.eval.config import VerdictConfig
from voicelab.eval.evaluators.caller_validity import FACTS_CRITERION_ID
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
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_tool_log=False)
    decision = decide(artifact, findings=[finding("major")])
    assert decision.verdict == Verdict.FAIL


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
    assert text.startswith("Verdict: pass.")
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
