"""The mapping is pure, so these run without an app."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from voicelab.api import mapping
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    DbChange,
    EvalResult,
    EvaluatorRun,
    Finding,
    LineConditions,
    Persona,
    Rule,
    Scenario,
    Speaker,
    ToolCallEvent,
    Utterance,
    Verdict,
    VoiceRef,
)
from voicelab.services.errors import Invalid


def _row(**overrides):
    row = {
        "id": "call_1",
        "batch_id": "batch_1",
        "rep": 2,
        "scenario_id": "scn_1",
        "scenario_title": "Title",
        "persona_id": "per_1",
        "persona_name": "Name",
        "mode": "live_bots",
        "status": "done",
        "profile": "local",
        "verdict": "invalid",
        "human_verdict": None,
        "invalid_reason": "no zip",
        "findings": 3,
        "open_findings": 1,
        "duration_s": 12.5,
        "reply_p50_s": 1.4,
        "overlaps": 2,
        "started_at": "2026-10-30T09:17:08.000000+00:00",
        "timing_is_real": 1,
    }
    return {**row, **overrides}


def test_field_names_are_camelized_but_dict_contents_are_not():
    change = DbChange(table="invoices", key={"invoice_id": 1}, kind="update", after={"paid_on": 2})
    assert mapping.to_api(change)["key"] == {"invoice_id": 1}
    assert mapping.to_api(change)["after"] == {"paid_on": 2}
    scenario = Scenario(
        title="t",
        goal="g",
        facts={"billing_zip": "97214"},
        must_not=[Rule(id="r", text="x")],
    )
    shaped = mapping.to_api(scenario)
    assert shaped["facts"] == {"billing_zip": "97214"}
    assert shaped["mustNot"][0]["id"] == "r"
    assert "must_not" not in shaped


def test_tool_arguments_keep_their_snake_case():
    event = ToolCallEvent(at_s=1.0, name="x", arguments={"billing_zip": "1"}, duration_ms=2.0)
    shaped = mapping.to_api(event)
    assert shaped["atS"] == 1.0
    assert shaped["durationMs"] == 2.0
    assert shaped["arguments"] == {"billing_zip": "1"}


def test_enums_and_times_become_strings():
    shaped = mapping.to_api(
        CallArtifact(
            mode=CallMode.LIVE_BOTS, started_at=datetime(2026, 10, 30, 9, 17, 8, tzinfo=UTC)
        )
    )
    assert shaped["mode"] == "live_bots"
    assert shaped["startedAt"] == "2026-10-30T09:17:08Z"


def test_from_api_accepts_camel_case_and_drops_what_clients_echo_back():
    persona = mapping.from_api(
        Persona,
        {
            "name": "Ann",
            "speakingStyle": "dry",
            "interruptionRate": 0.5,
            "voice": {"voiceId": "v1", "provider": "p"},
            "line": {"snrDb": 12, "packetLossPct": 3, "oneWayLatencyMs": 100, "noise": "cafe"},
            "usage": {"calls": 3},
        },
    )
    assert persona.speaking_style == "dry"
    assert persona.voice.voice_id == "v1"
    assert persona.line.snr_db == 12
    assert persona.line.one_way_latency_ms == 100


def test_from_api_handles_nested_lists_and_leaves_their_dicts_alone():
    scenario = mapping.from_api(
        Scenario,
        {
            "title": "t",
            "goal": "g",
            "mustNot": [{"text": "x", "severity": "minor"}],
            "expectedState": [{"table": "invoices", "where": {"customer_id": "c"}}],
        },
    )
    assert scenario.must_not[0].severity == "minor"
    assert scenario.expected_state[0].where == {"customer_id": "c"}


def test_from_api_round_trips():
    persona = Persona(name="n", voice=VoiceRef(voice_id="v"), line=LineConditions(snr_db=9))
    assert mapping.from_api(Persona, mapping.to_api(persona)) == persona


def test_from_api_reports_bad_fields():
    with pytest.raises(Invalid) as caught:
        mapping.from_api(Persona, {"name": "n", "patience": 9, "voice": {"voiceId": "v"}})
    assert caught.value.detail["problems"][0]["field"] == "patience"


def test_counts_keep_invalid_separate_and_fill_gaps():
    tally = mapping.counts({"pass": 2, "invalid": 1, "needs_review": 3})
    assert tally == {"pass": 2, "fail": 0, "needsReview": 3, "invalid": 1}


def test_call_summary_shape():
    summary = mapping.call_summary(_row())
    assert summary["batchId"] == "batch_1"
    assert summary["invalidReason"] == "no zip"
    assert summary["startedAt"] == "2026-10-30T09:17:08.000000Z"
    assert summary["replyP50S"] == 1.4
    assert summary["timingIsReal"] is True


def test_timing_is_null_not_zero_when_timing_is_synthetic():
    summary = mapping.call_summary(_row(timing_is_real=0))
    assert summary["replyP50S"] is None
    assert summary["overlaps"] is None


def _evaluated(metrics=None, status="ok") -> EvalResult:
    return EvalResult(
        call_id="call_1",
        verdict=Verdict.PASS,
        evaluators=[EvaluatorRun(name="timing", status=status, metrics=metrics or {})],
    )


METRICS = {
    "response_latency": {"median_s": 1.42, "p95_s": 2.9},
    "overlap": {"count": 1},
    "dead_air": {"count_over_threshold": 1},
    "barge_in": {"median_reaction_s": 1.5},
}


def test_timing_block_reads_the_evaluators_metrics():
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS)
    assert mapping.timing_block(artifact, _evaluated(METRICS)) == {
        "replyP50S": 1.42,
        "replyP95S": 2.9,
        "overlaps": 1,
        "deadAirEvents": 1,
        "bargeStopS": 1.5,
    }


def test_unmeasured_timing_stays_null():
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS)
    block = mapping.timing_block(artifact, _evaluated({"response_latency": None, "overlap": None}))
    assert block["replyP50S"] is None
    assert block["overlaps"] is None


def test_timing_block_is_null_for_turn_based_or_skipped():
    turn_based = CallArtifact(mode=CallMode.TURN_BASED, timing_is_real=False)
    assert mapping.timing_block(turn_based, _evaluated(METRICS)) is None
    live = CallArtifact(mode=CallMode.LIVE_BOTS)
    assert mapping.timing_block(live, _evaluated(status="skipped")) is None
    assert mapping.timing_block(live, None) is None


def test_evidence_follows_the_artifact():
    artifact = CallArtifact(
        mode=CallMode.UPLOAD,
        has_tool_log=False,
        has_intended_text=False,
        timeline=[Utterance(speaker=Speaker.BOT, start_s=0, end_s=1)],
    )
    assert mapping.evidence_flags(artifact, None, audio_available=True) == {
        "audio": True,
        "transcript": False,
        "intendedText": False,
        "toolLog": False,
        "dbSnapshots": False,
        "scenario": False,
    }


def test_apply_decisions_overlays_without_touching_the_original():
    finding = Finding(id="fnd_1", evaluator="timing", category="c", title="t")
    result = EvalResult(call_id="c", verdict=Verdict.PASS, findings=[finding])
    decided = mapping.apply_decisions(
        result,
        {
            "fnd_1": {
                "human_verdict": "dismissed",
                "human_note": "fine",
                "human_at": "2026-10-30T09:00:00.000000+00:00",
            }
        },
    )
    assert decided.findings[0].human_verdict == "dismissed"
    assert decided.findings[0].human_note == "fine"
    assert result.findings[0].human_verdict is None


def test_timeline_gets_words_and_text_from_the_independent_transcript():
    artifact = CallArtifact(
        mode=CallMode.UPLOAD, timeline=[Utterance(speaker=Speaker.BOT, start_s=1.0, end_s=2.0)]
    )
    words = [
        {"text": "hello", "start_s": 1.1, "end_s": 1.4, "confidence": 0.9},
        {"text": "later", "start_s": 5.0, "end_s": 5.4, "confidence": 0.9},
    ]
    result = EvalResult(
        call_id=artifact.id,
        verdict=Verdict.PASS,
        evaluators=[
            EvaluatorRun(
                name="transcribe", status="ok", metrics={"transcript": {"bot": {"words": words}}}
            )
        ],
    )
    utterance = mapping.timeline_json(artifact, result)[0]
    assert utterance["heardText"] == "hello"
    assert utterance["words"] == [{"text": "hello", "startS": 1.1, "endS": 1.4, "confidence": 0.9}]


def test_heard_text_stays_null_without_a_transcript():
    artifact = CallArtifact(
        mode=CallMode.UPLOAD, timeline=[Utterance(speaker=Speaker.BOT, start_s=1.0, end_s=2.0)]
    )
    assert mapping.timeline_json(artifact, None)[0]["heardText"] is None


def test_engine_messages_are_normalized_whether_models_or_dumps():
    from_model = mapping.stream_message(
        {"type": "turn", "utterance": Utterance(speaker=Speaker.BOT, start_s=1, end_s=2)}, "call_1"
    )
    assert from_model["callId"] == "call_1"
    assert from_model["utterance"]["startS"] == 1
    from_dump = mapping.stream_message(
        {"type": "event", "call_id": "call_2", "event": {"at_s": 3, "arguments": {"a_b": 1}}},
        "call_1",
    )
    assert from_dump["callId"] == "call_2"
    assert from_dump["event"] == {"atS": 3, "arguments": {"a_b": 1}}


def test_eta_uses_the_runs_own_rate_and_stops_when_it_stops():
    run = {"status": "running", "mode": "turn_based", "concurrency": 2}
    progress = {"calls": 10, "done": 4}
    assert mapping.eta_s(run, progress, 30.0) == 90.0
    assert mapping.eta_s(run, progress, None) == 630.0
    assert mapping.eta_s({**run, "status": "stopped"}, progress, 30.0) is None
