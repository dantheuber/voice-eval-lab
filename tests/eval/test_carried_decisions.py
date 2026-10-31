"""A person's decision on a finding, across a re-evaluation that changed what the finding covers."""

from __future__ import annotations

import json

from voicelab.core.models import CallArtifact, CallMode, EvalResult, Evidence, Finding
from voicelab.store.calls import CallIndex, carry_decisions, finding_times
from voicelab.store.sqlite import Database

TITLE = "2 slow responses, the slowest 6.0 s"


def finding(*times: float, severity: str = "minor", title: str = TITLE) -> Finding:
    return Finding(
        evaluator="timing",
        category="slow_response",
        severity=severity,
        title=title,
        evidence=[Evidence(start_s=t, end_s=t + 1.0) for t in times],
    )


def decided(*times: float, severity: str = "minor", stored: bool = True) -> dict:
    return {
        "finding_id": "old",
        "evaluator": "timing",
        "category": "slow_response",
        "criterion_id": None,
        "title": TITLE,
        "severity": severity,
        "anchor_s": min(times),
        "occurrence": 0,
        "times": json.dumps(list(times)) if stored else None,
        "human_verdict": "dismissed",
    }


def test_a_rollup_covering_the_same_moments_keeps_its_decision():
    new = finding(10.0, 20.0)
    assert new.id in carry_decisions([decided(10.0, 20.0)], [new])


def test_the_same_moments_in_another_order_are_the_same_moments():
    new = finding(20.0, 10.0)
    assert new.id in carry_decisions([decided(10.0, 20.0)], [new])


def test_a_small_drift_in_timing_is_still_the_same_moments():
    new = finding(10.4, 19.7)
    assert new.id in carry_decisions([decided(10.0, 20.0)], [new])


def test_a_decision_does_not_reach_an_occurrence_nobody_saw():
    """Same title, same earliest moment, and one moment nobody dismissed."""
    assert carry_decisions([decided(10.0, 20.0)], [finding(10.0, 100.0)]) == {}


def test_a_rollup_that_grew_is_not_the_one_that_was_decided():
    assert carry_decisions([decided(10.0, 20.0)], [finding(10.0, 20.0, 30.0)]) == {}


def test_a_rollup_that_shrank_is_not_the_one_that_was_decided():
    assert carry_decisions([decided(10.0, 20.0)], [finding(10.0)]) == {}


def test_one_new_moment_cannot_stand_for_two_old_ones():
    assert carry_decisions([decided(10.0, 10.5)], [finding(10.2, 80.0)]) == {}


def test_a_worse_finding_needs_a_new_decision():
    new = finding(10.0, 20.0, severity="major")
    assert carry_decisions([decided(10.0, 20.0, severity="minor")], [new]) == {}


def test_a_decision_from_before_moments_were_stored_only_reaches_one_occurrence():
    single = finding(10.0)
    assert single.id in carry_decisions([decided(10.0, stored=False)], [single])
    assert carry_decisions([decided(10.0, stored=False)], [finding(10.0, 20.0)]) == {}


def test_a_finding_lists_its_moments_in_order():
    assert finding_times(finding(20.0, 10.0)) == [10.0, 20.0]
    assert finding_times(Finding(evaluator="e", category="c", title="t")) == []


def stored_call(tmp_path) -> tuple[CallIndex, CallArtifact, Finding]:
    index = CallIndex(Database(tmp_path / "calls.sqlite3"))
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS)
    index.add(id=artifact.id, mode=artifact.mode.value, status="done", timing_is_real=1, seed=0)
    first = finding(10.0, 20.0)
    index.record_eval(artifact, EvalResult(call_id=artifact.id, verdict="pass", findings=[first]))
    return index, artifact, first


def decision_on(index: CallIndex, call_id: str) -> str | None:
    rows = index.db.rows("SELECT human_verdict FROM call_findings WHERE call_id = ?", (call_id,))
    return rows[0]["human_verdict"]


def test_a_stored_decision_survives_a_rerun_covering_the_same_moments(tmp_path):
    index, artifact, first = stored_call(tmp_path)
    index.decide_finding(artifact.id, first.id, "dismissed", "", "dan")
    again = finding(10.0, 20.0)
    index.record_eval(artifact, EvalResult(call_id=artifact.id, verdict="pass", findings=[again]))
    assert decision_on(index, artifact.id) == "dismissed"


def test_a_stored_decision_stays_off_a_rerun_with_a_moment_nobody_saw(tmp_path):
    index, artifact, first = stored_call(tmp_path)
    index.decide_finding(artifact.id, first.id, "dismissed", "", "dan")
    again = finding(10.0, 100.0)
    index.record_eval(artifact, EvalResult(call_id=artifact.id, verdict="pass", findings=[again]))
    assert decision_on(index, artifact.id) is None
