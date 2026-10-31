from __future__ import annotations

import base64

import pytest

from voicelab.core.models import (
    CallMode,
    CallStatus,
    EvaluatorRun,
    Evidence,
    Finding,
    Verdict,
)
from voicelab.store.calls import CallFilters
from voicelab.store.sqlite import timestamp


def add_call(rig, call_id: str, **columns):
    defaults = dict(
        id=call_id,
        mode="turn_based",
        status="done",
        profile="local-default",
        timing_is_real=0,
    )
    rig.services.calls.add(**{**defaults, **columns})


@pytest.fixture
def many(rig):
    for i in range(7):
        add_call(
            rig,
            f"call_{i}",
            verdict=["pass", "fail", "needs_review", "invalid"][i % 4],
            scenario_id="scn_a" if i % 2 else "scn_b",
            persona_id="per_a",
            batch_id="batch_a" if i < 3 else None,
            open_findings=i % 3,
            mode="live_bots" if i == 0 else "turn_based",
            created_at=f"2026-10-30T09:00:0{i}.000000+00:00",
        )
    return rig


def ids(page):
    return {item["id"] for item in page["items"]}


def test_list_pages_through_everything_without_repeats(client, many):
    seen = []
    cursor = None
    while True:
        params = {"limit": 3, **({"cursor": cursor} if cursor else {})}
        page = client.get("/api/calls", params=params).json()
        assert page["total"] == 7
        seen.extend(item["id"] for item in page["items"])
        cursor = page["nextCursor"]
        if cursor is None:
            break
    assert len(seen) == 7
    assert len(set(seen)) == 7


def test_last_page_has_null_cursor(client, many):
    assert client.get("/api/calls", params={"limit": 7}).json()["nextCursor"] is None


def test_bad_cursor_is_invalid(client):
    garbage = base64.urlsafe_b64encode(b"not json").decode()
    response = client.get("/api/calls", params={"cursor": garbage})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("verdict=pass", {"call_0", "call_4"}),
        ("verdict=pass&verdict=fail", {"call_0", "call_4", "call_1", "call_5"}),
        ("scenario=scn_a", {"call_1", "call_3", "call_5"}),
        ("run=batch_a", {"call_0", "call_1", "call_2"}),
        ("mode=live_bots", {"call_0"}),
        ("persona=per_nobody", set()),
        ("status=done&profile=other", set()),
        ("openOnly=true", {"call_1", "call_2", "call_4", "call_5"}),
        ("verdict=pass&openOnly=true", {"call_4"}),
    ],
)
def test_filters(client, many, query, expected):
    assert ids(client.get(f"/api/calls?{query}").json()) == expected


def test_facets_ignore_the_verdict_filter_but_keep_the_others(client, many):
    page = client.get("/api/calls?verdict=pass&scenario=scn_a").json()
    assert page["facets"]["verdict"] == {"fail": 2, "invalid": 1}


def test_limit_is_bounded(client):
    assert client.get("/api/calls?limit=0").status_code == 400


def test_review_payload_shape(client, rig, finished_call):
    call = finished_call()
    body = client.get(f"/api/calls/{call['id']}").json()
    assert set(body) == {
        "call",
        "eval",
        "summary",
        "audio",
        "peaks",
        "speech",
        "overlaps",
        "timing",
        "evidence",
        "prevReviewId",
        "nextReviewId",
    }
    assert body["call"]["personaSnapshot"]["id"] == "per_sysadmin"
    assert body["audio"]["stereoUrl"] == f"/api/calls/{call['id']}/audio/recording.wav"
    assert body["audio"]["sampleRate"] == 16000
    assert body["peaks"]["perSecond"] == 50
    assert len(body["peaks"]["caller"]) == len(body["peaks"]["bot"])
    assert body["speech"]["caller"] == [[1.5, 3.0]]
    assert body["speech"]["bot"] == [[0.0, 1.0]]
    assert body["overlaps"] == []
    assert body["timing"] is None  # the fake run was turn-based
    assert body["summary"]["id"] == call["id"]
    assert body["evidence"]["toolLog"] is True


def test_speech_regions_are_cached_beside_the_peaks(client, rig, finished_call):
    call = finished_call()
    client.get(f"/api/calls/{call['id']}")
    assert (rig.services.store.paths(call["id"]).dir / "speech.json").exists()


def test_live_call_has_timing(client, rig, finished_call):
    call = finished_call(mode="live_bots")
    row = rig.services.calls.get(call["id"])
    assert row["reply_p50_s"] == 1.4
    body = client.get(f"/api/calls/{call['id']}").json()
    assert body["timing"]["replyP50S"] == 1.4
    assert body["summary"]["overlaps"] == 1


def test_unknown_timing_stays_null_through_the_list(client, rig, finished_call):
    rig.evals.reply_p50_s = None
    call = finished_call(mode="live_bots")
    assert client.get(f"/api/calls/{call['id']}").json()["summary"]["replyP50S"] is None


def test_queued_call_says_it_has_no_audio(client, rig, start_run):
    rig.runner.hold = True  # the first call blocks, so the second stays queued
    run = start_run(reps=2)["run"]
    calls = client.get(f"/api/runs/{run['id']}").json()["calls"]
    queued = next(c for c in calls if c["status"] == "queued")
    body = client.get(f"/api/calls/{queued['id']}").json()
    assert body["audio"] == {
        "stereoUrl": None,
        "callerUrl": None,
        "botUrl": None,
        "sampleRate": 16000,
        "available": False,
    }
    assert body["peaks"] is None
    assert body["eval"] is None
    assert body["speech"] == {"caller": [], "bot": []}


def test_eval_is_null_while_evaluating(client, rig, finished_call, waiter):
    call = finished_call()
    rig.evals.hold = True
    assert client.post(f"/api/calls/{call['id']}/reevaluate", json={}).status_code == 200
    assert client.get(f"/api/calls/{call['id']}").json()["eval"] is None
    rig.run_on_loop(_release(rig))
    waiter(lambda: client.get(f"/api/calls/{call['id']}").json()["eval"] is not None)


def _release(rig):
    async def release():
        rig.evals.hold = False
        rig.evals.release.set()

    return release


def test_audio_serves_ranges(client, finished_call):
    call = finished_call()
    url = f"/api/calls/{call['id']}/audio/recording.wav"
    whole = client.get(url)
    assert whole.status_code == 200
    assert whole.headers["content-type"] == "audio/wav"
    assert whole.headers["accept-ranges"] == "bytes"
    part = client.get(url, headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.content == whole.content[100:200]


def test_unknown_audio_file_is_404(client, finished_call):
    call = finished_call()
    assert client.get(f"/api/calls/{call['id']}/audio/call.json").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/api/calls/..%2F..%2Fetc%2Fpasswd",
        "/api/calls/..%2f..%2fsecrets",
        "/api/calls/call_1%00",
        "/api/uploads/..%2F..%2Fetc/preview",
        "/api/calls/%2e%2e/audio/recording.wav",
    ],
)
def test_traversal_in_ids_fails(client, path):
    response = client.get(path)
    assert response.status_code == 404


def test_traversal_in_audio_name_fails(client, finished_call):
    call = finished_call()
    for name in ("..%2Fcall.json", "..%2F..%2Fvoicelab.sqlite3", "recording.wav%00.png"):
        assert client.get(f"/api/calls/{call['id']}/audio/{name}").status_code == 404


def test_crafted_id_on_a_mutation_fails(client):
    # The decoded path no longer fits the route, so the dashboard's GET route refuses the method.
    response = client.post("/api/calls/..%2F..%2Fx/verdict", json={"verdict": "pass"})
    assert response.status_code in (404, 405)
    assert client.delete("/api/calls/..%2F..%2Fx").status_code in (404, 405)


def test_missing_call_uses_the_error_shape(client):
    response = client.get("/api/calls/call_missing")
    assert response.status_code == 404
    body = response.json()["error"]
    assert body["code"] == "not_found"
    assert body["message"]
    assert "detail" in body


def with_findings(rig, *titles, advisory=False):
    rig.evals.findings = [
        Finding(evaluator="timing", category="c", title=title, advisory=advisory)
        for title in titles
    ]


def test_finding_decision_persists_with_who_and_when(client, rig, finished_call):
    with_findings(rig, "Wrong amount")
    call = finished_call()
    finding = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"][0]
    url = f"/api/calls/{call['id']}/findings/{finding['id']}"
    decided = client.post(url, json={"humanVerdict": "confirmed", "humanNote": "yes"}).json()
    assert decided["humanVerdict"] == "confirmed"
    assert decided["humanNote"] == "yes"
    assert decided["humanAt"]
    reread = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"][0]
    assert reread["humanVerdict"] == "confirmed"
    stored = rig.services.calls.finding_decisions(call["id"])[finding["id"]]
    assert stored["human_by"] == "tester"


def test_a_note_with_no_decision_is_kept(client, rig, finished_call):
    with_findings(rig, "Wrong amount")
    call = finished_call()
    finding = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"][0]
    url = f"/api/calls/{call['id']}/findings/{finding['id']}"
    saved = client.post(url, json={"humanVerdict": None, "humanNote": "listen again"}).json()
    assert saved["humanNote"] == "listen again"
    assert saved["humanVerdict"] is None
    reread = client.get(f"/api/calls/{call['id']}").json()
    assert reread["eval"]["findings"][0]["humanNote"] == "listen again"
    assert reread["summary"]["openFindings"] == 1


def test_clearing_a_decision_keeps_the_history(client, rig, finished_call):
    with_findings(rig, "Wrong amount")
    call = finished_call()
    finding = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"][0]
    url = f"/api/calls/{call['id']}/findings/{finding['id']}"
    client.post(url, json={"humanVerdict": "confirmed", "humanNote": "first"})
    client.post(url, json={"humanVerdict": "dismissed", "humanNote": "second"})
    cleared = client.post(url, json={"humanVerdict": None, "humanNote": ""}).json()
    assert cleared["humanVerdict"] is None
    assert cleared["humanAt"] is None
    history = rig.services.calls.decision_history(call["id"])
    assert [(h["verdict"], h["note"]) for h in history] == [
        ("confirmed", "first"),
        ("dismissed", "second"),
        (None, ""),
    ]
    assert all(h["decided_by"] == "tester" and h["decided_at"] for h in history)


def test_decisions_drive_open_counts(client, rig, finished_call):
    with_findings(rig, "One", "Two")
    call = finished_call()
    assert call["openFindings"] == 2
    findings = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"]
    url = f"/api/calls/{call['id']}/findings/{findings[0]['id']}"
    client.post(url, json={"humanVerdict": "dismissed", "humanNote": ""})
    summary = client.get(f"/api/calls/{call['id']}").json()["summary"]
    assert summary["openFindings"] == 1
    assert summary["findings"] == 2


def test_details_are_in_the_eval_but_not_the_count(client, rig, finished_call):
    rig.evals.findings = [
        Finding(evaluator="timing", category="dead_air", title="a"),
        Finding(evaluator="delivery", category="noisy", severity="info", advisory=True, title="b"),
    ]
    call = finished_call()
    summary = client.get(f"/api/calls/{call['id']}").json()["summary"]
    assert summary["findings"] == 1
    assert len(client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"]) == 2


def test_advisory_findings_are_listed_but_never_open(client, rig, finished_call):
    with_findings(rig, "Odd tone", advisory=True)
    call = finished_call()
    assert call["findings"] == 1
    assert call["openFindings"] == 0


def test_unknown_finding_is_404(client, finished_call):
    call = finished_call()
    url = f"/api/calls/{call['id']}/findings/fnd_missing"
    assert client.post(url, json={"humanVerdict": "confirmed"}).status_code == 404


def test_bad_human_verdict_is_rejected(client, rig, finished_call):
    with_findings(rig, "One")
    call = finished_call()
    finding = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"][0]
    url = f"/api/calls/{call['id']}/findings/{finding['id']}"
    assert client.post(url, json={"humanVerdict": "maybe"}).status_code == 400


def test_call_verdict_sits_beside_the_machine_verdict(client, rig, finished_call):
    rig.evals.verdicts = [Verdict.FAIL]
    call = finished_call()
    summary = client.post(
        f"/api/calls/{call['id']}/verdict", json={"verdict": "pass", "note": "judge is wrong"}
    ).json()
    assert summary["humanVerdict"] == "pass"
    assert summary["verdict"] == "fail"
    row = rig.services.calls.get(call["id"])
    assert (row["human_note"], row["human_by"]) == ("judge is wrong", "tester")
    assert row["human_at"]
    assert rig.services.store.read_eval(call["id"]).verdict == Verdict.FAIL
    assert rig.services.calls.decision_history(call["id"])[0]["finding_id"] is None


def test_decisions_survive_reevaluation(client, rig, finished_call, waiter):
    with_findings(rig, "Wrong amount")
    call = finished_call()
    old = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"][0]
    client.post(
        f"/api/calls/{call['id']}/findings/{old['id']}",
        json={"humanVerdict": "confirmed", "humanNote": "real"},
    )
    assert client.post(f"/api/calls/{call['id']}/reevaluate", json={}).status_code == 200
    waiter(lambda: client.get(f"/api/calls/{call['id']}").json()["eval"] is not None)
    new = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"][0]
    assert new["id"] != old["id"]
    assert new["humanVerdict"] == "confirmed"
    assert client.get(f"/api/calls/{call['id']}").json()["summary"]["openFindings"] == 0


def _dead_air(at_s, title="3.0 s of dead air"):
    return Finding(
        evaluator="timing",
        category="dead_air",
        title=title,
        evidence=[Evidence(start_s=at_s, end_s=at_s + 3)],
    )


def _reevaluate(client, rig, call_id, waiter, findings):
    rig.evals.findings = findings
    jobs = len(rig.evals.jobs)
    client.post(f"/api/calls/{call_id}/reevaluate", json={})
    waiter(lambda: len(rig.evals.jobs) == jobs + 1)
    waiter(lambda: client.get(f"/api/calls/{call_id}").json()["summary"]["status"] == "done")
    return client.get(f"/api/calls/{call_id}").json()["eval"]["findings"]


def _decide(client, call_id, finding, verdict):
    client.post(
        f"/api/calls/{call_id}/findings/{finding['id']}",
        json={"humanVerdict": verdict, "humanNote": verdict},
    )


def _at(findings, start_s):
    return next(f for f in findings if f["evidence"][0]["startS"] == start_s)


def test_dismissing_one_occurrence_leaves_its_twin_open(client, rig, finished_call, waiter):
    rig.evals.findings = [_dead_air(10.0), _dead_air(50.0)]
    call = finished_call()
    first = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"]
    _decide(client, call["id"], _at(first, 10.0), "dismissed")
    again = _reevaluate(client, rig, call["id"], waiter, [_dead_air(10.0), _dead_air(50.0)])
    assert _at(again, 10.0)["humanVerdict"] == "dismissed"
    assert _at(again, 50.0)["humanVerdict"] is None
    assert client.get(f"/api/calls/{call['id']}").json()["summary"]["openFindings"] == 1


def test_conflicting_decisions_keep_to_their_own_occurrence(client, rig, finished_call, waiter):
    rig.evals.findings = [_dead_air(10.0), _dead_air(50.0)]
    call = finished_call()
    first = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"]
    _decide(client, call["id"], _at(first, 10.0), "dismissed")
    _decide(client, call["id"], _at(first, 50.0), "confirmed")
    again = _reevaluate(client, rig, call["id"], waiter, [_dead_air(50.0), _dead_air(10.0)])
    assert _at(again, 10.0)["humanVerdict"] == "dismissed"
    assert _at(again, 50.0)["humanVerdict"] == "confirmed"


def test_a_decision_does_not_jump_to_a_different_moment(client, rig, finished_call, waiter):
    rig.evals.findings = [_dead_air(10.0)]
    call = finished_call()
    first = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"]
    _decide(client, call["id"], first[0], "dismissed")
    again = _reevaluate(client, rig, call["id"], waiter, [_dead_air(50.0)])
    assert again[0]["humanVerdict"] is None


def test_a_slightly_moved_occurrence_keeps_its_decision(client, rig, finished_call, waiter):
    rig.evals.findings = [_dead_air(10.0)]
    call = finished_call()
    first = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"]
    _decide(client, call["id"], first[0], "dismissed")
    again = _reevaluate(client, rig, call["id"], waiter, [_dead_air(10.2)])
    assert again[0]["humanVerdict"] == "dismissed"


def test_findings_without_times_match_one_to_one_in_order(client, rig, finished_call, waiter):
    plain = [Finding(evaluator="transcript_judge", category="c", title="Rude") for _ in range(2)]
    rig.evals.findings = plain
    call = finished_call()
    first = client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"]
    _decide(client, call["id"], first[0], "dismissed")
    again = _reevaluate(client, rig, call["id"], waiter, plain)
    assert [f["humanVerdict"] for f in again] == ["dismissed", None]


DELIVERY_METRICS = {
    "bot": [{"utterance_id": "utt_1", "words_per_minute": 142.5, "pitch_range_st": 7.1}],
    "medians": {"words_per_minute": 141.0},
    "flag_counts": {"too_fast": 1},
    "limitation": "Pitch is rough on a phone line.",
    "wpm_note": "Measured on the bot channel.",
}


def test_evaluator_metrics_reach_the_client_exactly_as_written(client, rig, finished_call):
    rig.evals.more_evaluators = [
        EvaluatorRun(name="delivery", status="ok", metrics=DELIVERY_METRICS)
    ]
    call = finished_call()
    evaluators = client.get(f"/api/calls/{call['id']}").json()["eval"]["evaluators"]
    assert next(e for e in evaluators if e["name"] == "delivery")["metrics"] == DELIVERY_METRICS


def test_skipped_and_errored_evaluators_are_listed_in_server_order(client, rig, finished_call):
    rig.evals.more_evaluators = [
        EvaluatorRun(name="audio_listening", status="skipped", reason="no analyst"),
        EvaluatorRun(name="delivery", status="error", reason="ValueError: bad clip"),
    ]
    call = finished_call()
    evaluators = client.get(f"/api/calls/{call['id']}").json()["eval"]["evaluators"]
    assert [(e["name"], e["status"]) for e in evaluators] == [
        ("timing", "ok"),
        ("caller_validity", "ok"),
        ("audio_listening", "skipped"),
        ("delivery", "error"),
    ]
    assert evaluators[3]["reason"] == "ValueError: bad clip"


def test_reevaluate_passes_chosen_evaluators(client, rig, finished_call, waiter):
    call = finished_call()
    body = {"evaluators": ["timing"]}
    assert client.post(f"/api/calls/{call['id']}/reevaluate", json=body).json() == {
        "accepted": True
    }
    waiter(lambda: len(rig.evals.jobs) == 2)
    assert rig.evals.jobs[-1].evaluators == ["timing"]


def test_reevaluate_rejects_unknown_evaluators(client, finished_call):
    call = finished_call()
    body = {"evaluators": ["no_such_evaluator"]}
    response = client.post(f"/api/calls/{call['id']}/reevaluate", json=body)
    assert response.status_code == 400


def test_reevaluate_conflicts_while_evaluating(client, rig, finished_call, waiter):
    call = finished_call()
    rig.evals.hold = True
    client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    response = client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    assert response.status_code == 409
    rig.run_on_loop(_release(rig))
    waiter(lambda: client.get(f"/api/calls/{call['id']}").json()["eval"] is not None)


def test_delete_removes_call_files_and_decisions(client, rig, finished_call):
    call = finished_call()
    client.post(f"/api/calls/{call['id']}/verdict", json={"verdict": "pass", "note": ""})
    assert client.delete(f"/api/calls/{call['id']}").status_code == 204
    assert client.get(f"/api/calls/{call['id']}").status_code == 404
    assert not rig.services.store.paths(call["id"]).dir.exists()
    assert rig.services.calls.decision_history(call["id"]) == []


def test_delete_conflicts_while_evaluating(client, rig, finished_call, waiter):
    call = finished_call()
    rig.evals.hold = True
    client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    assert client.delete(f"/api/calls/{call['id']}").status_code == 409
    rig.run_on_loop(_release(rig))
    waiter(lambda: client.get(f"/api/calls/{call['id']}").json()["eval"] is not None)


def test_next_review_walks_the_queue_and_wraps(client, rig):
    for i, verdict in enumerate(["needs_review", "fail", "pass"]):
        add_call(
            rig,
            f"call_{i}",
            verdict=verdict,
            open_findings=1 if verdict == "fail" else 0,
            created_at=f"2026-10-30T09:0{i}:00.000000+00:00",
        )
    assert client.get("/api/calls/next-review").json() == {"callId": "call_0"}
    assert client.get("/api/calls/next-review?after=call_0").json() == {"callId": "call_1"}
    assert client.get("/api/calls/next-review?after=call_1").json() == {"callId": "call_0"}


def test_next_review_skips_human_verdicted_calls(client, rig):
    add_call(
        rig,
        "call_0",
        verdict="needs_review",
        created_at="2026-10-30T09:00:00.000000+00:00",
    )
    add_call(
        rig,
        "call_1",
        verdict="needs_review",
        created_at="2026-10-30T09:01:00.000000+00:00",
    )
    client.post("/api/calls/call_0/verdict", json={"verdict": "pass", "note": ""})
    assert client.get("/api/calls/next-review").json() == {"callId": "call_1"}


def test_next_review_with_a_single_call_excludes_it(client, rig):
    add_call(rig, "call_0", verdict="needs_review")
    assert client.get("/api/calls/next-review?after=call_0").json() == {"callId": None}


def test_review_payload_links_neighbours(client, rig, finished_call):
    rig.evals.verdicts = [Verdict.NEEDS_REVIEW, Verdict.NEEDS_REVIEW]
    first = finished_call()
    second = finished_call()
    body = client.get(f"/api/calls/{second['id']}").json()
    assert body["prevReviewId"] == first["id"]


def test_overview_counts_invalid_apart_and_excludes_it_from_open_totals(client, rig, finished_call):
    rig.evals.verdicts = [Verdict.PASS, Verdict.INVALID, Verdict.NEEDS_REVIEW]
    finished_call()
    finished_call()
    finished_call()
    body = client.get("/api/overview").json()
    assert body["profile"] == "local-default"
    assert body["counts"] == {"pass": 1, "fail": 0, "needsReview": 1, "invalid": 1}
    assert body["queueTotal"] == 1
    assert body["queue"][0]["why"]
    assert len(body["callsPerHour"]) == 24
    assert body["callsPerHour"][-1] >= 3
    assert len(body["recentRuns"]) == 3
    assert body["runningRuns"] == 0
    assert body["scenarioRates"][0]["counts"]["invalid"] == 1
    assert isinstance(body["health"], list)


def test_overview_review_pace(client, rig, finished_call):
    assert client.get("/api/overview").json()["reviewPaceSPerFinding"] is None
    with_findings(rig, "One", "Two", "Three")
    call = finished_call()
    for finding in client.get(f"/api/calls/{call['id']}").json()["eval"]["findings"]:
        client.post(
            f"/api/calls/{call['id']}/findings/{finding['id']}",
            json={"humanVerdict": "dismissed", "humanNote": ""},
        )
    assert client.get("/api/overview").json()["reviewPaceSPerFinding"] is not None


def test_overview_top_open_findings(client, rig, finished_call):
    with_findings(rig, "Mispronounced", "Mispronounced", "Dead air")
    finished_call()
    top = client.get("/api/overview").json()["topOpenFindings"]
    assert top[0] == {"title": "Mispronounced", "evaluator": "timing", "count": 2}


def test_overview_rejects_a_bad_window(client):
    assert client.get("/api/overview?window=soon").status_code == 400


def test_overview_window_excludes_old_calls(client, rig):
    add_call(rig, "call_old", verdict="pass", created_at="2020-01-01T00:00:00.000000+00:00")
    add_call(rig, "call_new", verdict="fail", created_at=timestamp())
    assert client.get("/api/overview?window=24h").json()["counts"]["pass"] == 0
    assert client.get("/api/overview?window=24h").json()["counts"]["fail"] == 1


def test_index_query_filters_with_lists(rig):
    add_call(rig, "call_1", verdict="pass")
    filters = CallFilters(verdict=["pass", "fail"], status=["done"])
    assert rig.services.calls.verdict_counts(filters) == {"pass": 1}


def test_call_status_values_match_the_contract():
    assert {s.value for s in CallStatus} == {"queued", "running", "evaluating", "done", "failed"}
    assert {m.value for m in CallMode} == {"live_bots", "turn_based", "human", "upload"}
