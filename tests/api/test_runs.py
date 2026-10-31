from __future__ import annotations

import asyncio
import shutil
import time

import pytest
from conftest import PAIR
from fakes import build_rig, write_recording
from starlette.websockets import WebSocketDisconnect

from voicelab.api.routes import live
from voicelab.core.models import CallStatus, LineConditions, Verdict
from voicelab.services.errors import Conflict
from voicelab.services.runs import RunPlan


def released(rig):
    async def release():
        rig.runner.hold = False
        rig.runner.release.set()

    return release


def release_evals(rig):
    async def release():
        rig.evals.hold = False
        rig.evals.release.set()

    return release


def run_of(client, run_id):
    return client.get(f"/api/runs/{run_id}").json()


def statuses(client, run_id):
    return sorted(c["status"] for c in run_of(client, run_id)["calls"])


def test_pairs_times_reps_expand_into_calls(client, start_run, rig, waiter):
    second = {"scenarioId": "scn_expired_ssl", "personaId": "per_sysadmin"}
    created = start_run(pairs=[PAIR, second], reps=3)
    run = created["run"]
    assert created["effectiveMode"] == "turn_based"
    assert created["estimateS"] > 0
    assert run["calls"] == 6
    assert run["scenarioIds"] == ["scn_dns_mx_fix", "scn_expired_ssl"]
    assert run["personaIds"] == ["per_sysadmin"]
    assert run["reps"] == 3
    assert run["label"] == "2 scenarios x 1 personas x 3"
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    calls = run_of(client, run["id"])["calls"]
    assert sorted(c["rep"] for c in calls if c["scenarioId"] == "scn_dns_mx_fix") == [1, 2, 3]
    assert {request.seed for request in rig.runner.requests} == {1, 2, 3}


def test_auto_mode_follows_the_profile(start_run):
    assert start_run(profile="live")["effectiveMode"] == "live_bots"
    assert start_run(profile="local-default")["effectiveMode"] == "turn_based"


def test_explicit_mode_wins(start_run):
    assert start_run(mode="live_bots")["effectiveMode"] == "live_bots"


@pytest.mark.parametrize(
    "overrides",
    [
        {"pairs": []},
        {"reps": 0},
        {"concurrency": 0},
        {"profile": "no-such-profile"},
        {"mode": "human"},
        {"pairs": [{"scenarioId": "scn_missing", "personaId": "per_sysadmin"}]},
        {"pairs": [{"scenarioId": "scn_dns_mx_fix", "personaId": "per_missing"}]},
        {"line": "whatever"},
    ],
)
def test_bad_runs_are_rejected(client, overrides):
    body = {"pairs": [PAIR], "reps": 1, "profile": "local-default", **overrides}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid"


def test_line_override_reaches_the_runner(client, start_run, rig, waiter):
    line = LineConditions(noise="cafe", snr_db=10)
    body = {
        "codec": "mulaw_8k",
        "noise": "cafe",
        "snrDb": 10,
        "packetLossPct": 0,
        "oneWayLatencyMs": 80,
    }
    start_run(line=body)
    waiter(lambda: rig.runner.requests)
    assert rig.runner.requests[0].line == line


def test_persona_line_means_no_override(start_run, rig, waiter):
    start_run()
    waiter(lambda: rig.runner.requests)
    assert rig.runner.requests[0].line is None


def test_run_lifecycle_reports_counts_with_invalid_apart(client, start_run, rig, waiter):
    rig.evals.verdicts = [Verdict.PASS, Verdict.INVALID, Verdict.FAIL]
    run = start_run(reps=3)["run"]
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    body = run_of(client, run["id"])
    assert body["run"]["counts"] == {"pass": 1, "fail": 1, "needsReview": 0, "invalid": 1}
    assert body["run"]["done"] == 3
    assert body["run"]["running"] == 0
    assert body["run"]["etaS"] is None
    assert body["run"]["endedAt"]
    invalid = next(c for c in body["calls"] if c["verdict"] == "invalid")
    assert invalid["invalidReason"] == "never gave the zip"


def test_concurrency_one_runs_each_call_through_its_eval_first(client, start_run, rig, waiter):
    rig.evals.hold = True
    run = start_run(reps=2)["run"]
    waiter(lambda: "evaluating" in statuses(client, run["id"]))
    assert statuses(client, run["id"]) == ["evaluating", "queued"]
    assert len(rig.runner.requests) == 1
    rig.run_on_loop(release_evals(rig))
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    assert len(rig.runner.requests) == 2


def test_a_profile_with_one_session_runs_one_call_at_a_time(client, start_run, rig, waiter):
    profiles = rig.services.profiles
    profiles.items["local-default"] = profiles.require("local-default").model_copy(
        update={"max_calls": 1}
    )
    rig.runner.hold = True
    run = start_run(reps=2, concurrency=2)["run"]
    assert rig.services.runs.get(run["id"])["concurrency"] == 1
    waiter(lambda: "running" in statuses(client, run["id"]))
    assert statuses(client, run["id"]) == ["queued", "running"]
    assert len(rig.runner.requests) == 1
    rig.run_on_loop(released(rig))
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    assert len(rig.runner.requests) == 2


def test_a_stored_concurrency_cannot_get_past_the_profiles_limit(client, rig, waiter):
    """A run keeps its concurrency across restarts, and the profile may have been capped since."""
    profiles = rig.services.profiles
    rig.runner.hold = True
    plan = RunPlan(
        pairs=[(PAIR["scenarioId"], PAIR["personaId"])],
        reps=2,
        profile="local-default",
        mode="auto",
        concurrency=2,
    )
    run_id, _ = rig.services.worker.create(plan)
    assert rig.services.runs.get(run_id)["concurrency"] == 2
    profiles.items["local-default"] = profiles.require("local-default").model_copy(
        update={"max_calls": 1}
    )

    async def wake() -> None:
        rig.services.worker.wake()

    rig.run_on_loop(wake)
    waiter(lambda: "running" in statuses(client, run_id))
    time.sleep(0.05)
    assert statuses(client, run_id) == ["queued", "running"]
    rig.run_on_loop(released(rig))
    waiter(lambda: run_of(client, run_id)["run"]["status"] == "done")


def test_concurrency_two_runs_two_at_once(client, start_run, rig, waiter):
    rig.runner.hold = True
    run = start_run(reps=3, concurrency=2)["run"]
    waiter(lambda: statuses(client, run["id"]).count("running") == 2)
    assert statuses(client, run["id"]).count("queued") == 1
    rig.run_on_loop(released(rig))
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")


def test_later_run_waits_for_the_earlier_one(client, start_run, rig, waiter):
    rig.runner.hold = True
    first = start_run()["run"]
    second = start_run()["run"]
    waiter(lambda: "running" in statuses(client, first["id"]))
    assert run_of(client, second["id"])["run"]["status"] == "queued"
    rig.run_on_loop(released(rig))
    waiter(lambda: run_of(client, second["id"])["run"]["status"] == "done")


def test_pause_finishes_in_flight_and_resume_carries_on(client, start_run, rig, waiter):
    rig.runner.hold = True
    run = start_run(reps=2)["run"]
    waiter(lambda: "running" in statuses(client, run["id"]))
    assert client.post(f"/api/runs/{run['id']}/pause").json()["status"] == "paused"
    rig.run_on_loop(released(rig))
    waiter(lambda: "done" in statuses(client, run["id"]))
    assert statuses(client, run["id"]) == ["done", "queued"]
    assert run_of(client, run["id"])["run"]["status"] == "paused"
    assert len(rig.runner.requests) == 1
    assert client.post(f"/api/runs/{run['id']}/resume").json()["status"] == "running"
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")


def test_stop_cuts_running_calls_and_drops_queued_ones(client, start_run, rig, waiter):
    rig.runner.hold = True
    run = start_run(reps=2)["run"]
    waiter(lambda: "running" in statuses(client, run["id"]))
    stopped = client.post(f"/api/runs/{run['id']}/stop").json()
    assert stopped["status"] == "stopped"
    assert stopped["endedAt"]
    waiter(lambda: statuses(client, run["id"]) == ["failed", "failed"])
    for call in run_of(client, run["id"])["calls"]:
        assert rig.services.store.read_artifact(call["id"]).error == "stopped"
        assert rig.services.calls.get(call["id"])["error"] == "stopped"
    assert len(rig.runner.requests) == 1
    assert rig.evals.jobs == []


def test_stop_lets_a_recorded_call_finish_evaluating(client, start_run, rig, waiter):
    rig.evals.hold = True
    run = start_run()["run"]
    waiter(lambda: "evaluating" in statuses(client, run["id"]))
    client.post(f"/api/runs/{run['id']}/stop")
    rig.run_on_loop(release_evals(rig))
    waiter(lambda: statuses(client, run["id"]) == ["done"])
    assert run_of(client, run["id"])["run"]["status"] == "stopped"


@pytest.mark.parametrize("action", ["pause", "stop"])
def test_actions_on_a_finished_run_conflict(client, start_run, waiter, action):
    run = start_run()["run"]
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    response = client.post(f"/api/runs/{run['id']}/{action}")
    assert response.status_code == 409


def test_resume_of_a_running_run_conflicts(client, start_run, rig, waiter):
    rig.runner.hold = True
    run = start_run()["run"]
    waiter(lambda: "running" in statuses(client, run["id"]))
    assert client.post(f"/api/runs/{run['id']}/resume").status_code == 409


def test_missing_run_is_404(client):
    assert client.get("/api/runs/batch_missing").status_code == 404
    assert client.post("/api/runs/batch_missing/pause").status_code == 404


def test_runner_exception_fails_only_that_call(client, start_run, rig, waiter):
    rig.runner.fail_with = RuntimeError("line dropped")
    run = start_run(reps=2)["run"]
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    calls = run_of(client, run["id"])["calls"]
    assert [c["status"] for c in calls] == ["failed", "failed"]
    assert rig.services.store.read_artifact(calls[0]["id"]).error == "RuntimeError: line dropped"
    assert rig.evals.jobs == []


def test_failed_artifact_is_not_evaluated(client, start_run, rig, waiter):
    rig.runner.status = CallStatus.FAILED
    rig.runner.error = "stt timed out"
    run = start_run()["run"]
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    assert statuses(client, run["id"]) == ["failed"]
    assert rig.evals.jobs == []
    call = run_of(client, run["id"])["calls"][0]
    assert client.get(f"/api/calls/{call['id']}").json()["call"]["error"] == "stt timed out"


def test_failed_calls_stay_out_of_the_counts(client, start_run, rig, waiter):
    rig.runner.fail_with = RuntimeError("no")
    run = start_run()["run"]
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    body = run_of(client, run["id"])["run"]
    assert body["counts"] == {"pass": 0, "fail": 0, "needsReview": 0, "invalid": 0}
    assert body["done"] == 1


def test_eval_failure_marks_the_call_and_the_run_moves_on(client, start_run, rig, waiter):
    rig.evals.fail_with = RuntimeError("worker died")
    run = start_run(reps=2)["run"]
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    assert statuses(client, run["id"]) == ["failed", "failed"]
    call = run_of(client, run["id"])["calls"][0]
    assert "worker died" in rig.services.calls.get(call["id"])["eval_error"]


def test_runs_list_pages(client, start_run, waiter):
    ids = {start_run()["run"]["id"] for _ in range(3)}
    first = client.get("/api/runs", params={"limit": 2}).json()
    assert len(first["items"]) == 2
    second = client.get("/api/runs", params={"cursor": first["nextCursor"], "limit": 2}).json()
    assert second["nextCursor"] is None
    assert {r["id"] for r in first["items"] + second["items"]} == ids


def test_matrix_groups_by_pair(client, rig, finished_call):
    rig.evals.verdicts = [Verdict.PASS, Verdict.INVALID]
    finished_call()
    finished_call()
    cells = client.get("/api/matrix", params={"profile": "local-default"}).json()["cells"]
    assert cells == [
        {
            "scenarioId": PAIR["scenarioId"],
            "personaId": PAIR["personaId"],
            "counts": {"pass": 1, "fail": 0, "needsReview": 0, "invalid": 1},
        }
    ]


def test_presets(client, start_run):
    body = {
        "name": "nightly",
        "pairs": [PAIR],
        "reps": 3,
        "profile": "local-default",
        "mode": "auto",
        "line": "persona",
        "concurrency": 2,
    }
    created = client.post("/api/presets", json=body)
    assert created.status_code == 201
    assert created.json()["pairs"] == [PAIR]
    assert client.get("/api/presets").json()["items"][0]["name"] == "nightly"
    run = start_run(reps=2)["run"]
    derived = client.get(f"/api/runs/{run['id']}/presets").json()["items"][0]
    assert derived["reps"] == 2
    assert derived["pairs"] == [PAIR]
    assert derived["line"] == "persona"


# Streams -----------------------------------------------------------------------------------------


def collect(socket, until: str) -> list[dict]:
    messages = []
    while True:
        message = socket.receive_json()
        messages.append(message)
        if message["type"] == until:
            return messages


def test_run_stream_follows_a_call_from_start_to_verdict(client, start_run, rig, waiter):
    rig.runner.hold = True
    run = start_run()["run"]
    with client.websocket_connect(f"/ws/runs/{run['id']}") as socket:
        assert socket.receive_json()["type"] == "run"
        waiter(lambda: "running" in statuses(client, run["id"]))
        rig.run_on_loop(released(rig))
        messages = collect(socket, "runDone")
    kinds = [m["type"] for m in messages]
    assert kinds.count("callDone") == 1
    assert "callEvaluated" in kinds
    assert kinds[-1] == "runDone"
    done = next(m for m in messages if m["type"] == "callDone")
    evaluated = next(m for m in messages if m["type"] == "callEvaluated")
    assert done["call"]["status"] == "evaluating"
    assert evaluated["call"]["verdict"] == "pass"
    assert messages[-1]["run"]["status"] == "done"


def test_run_stream_of_a_finished_run_says_so_and_closes(client, start_run, waiter):
    run = start_run()["run"]
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    with client.websocket_connect(f"/ws/runs/{run['id']}") as socket:
        assert socket.receive_json()["type"] == "run"
        assert socket.receive_json()["type"] == "runDone"


def test_run_stream_of_a_missing_run(client):
    with client.websocket_connect("/ws/runs/batch_missing") as socket:
        assert socket.receive_json()["type"] == "error"


def test_call_summaries_on_the_stream_are_the_full_ones(client, start_run, rig):
    """The engine's own callStarted has no rep or verdict, so the worker's must be what arrives."""

    async def hold():
        rig.services.gate.hold("test")

    async def release():
        rig.services.gate.release("test")

    rig.run_on_loop(hold)  # keeps the call from starting before the socket is listening
    run = start_run()["run"]
    with client.websocket_connect(f"/ws/runs/{run['id']}") as socket:
        socket.receive_json()
        rig.run_on_loop(release)
        messages = collect(socket, "runDone")
    started = [m for m in messages if m["type"] == "callStarted"]
    assert len(started) == 1
    assert started[0]["call"]["rep"] == 1
    assert started[0]["call"]["status"] == "running"


def test_partials_stream_only_for_the_watched_call(client, rig):
    from voicelab.api.routes.live import _CallWatch

    watch = _CallWatch(["call_a", "call_b"])
    partial = {"type": "partial", "callId": "call_a"}
    other = {"type": "partial", "callId": "call_b"}
    assert watch.wanted(partial)  # defaults to the earliest running call
    assert not watch.wanted(other)
    watch.on_client({"type": "watch", "callId": "call_b"})
    assert watch.wanted(other)
    assert not watch.wanted(partial)
    assert watch.wanted({"type": "run"})


def test_watch_default_moves_on_when_the_call_ends():
    from voicelab.api.routes.live import _CallWatch

    watch = _CallWatch(["call_a"])
    watch.wanted({"type": "callStarted", "call": {"id": "call_b"}})
    watch.wanted({"type": "callDone", "call": {"id": "call_a"}})
    assert watch.wanted({"type": "turn", "callId": "call_b"})


def test_call_stream_sends_eval_progress_and_closes(client, rig, finished_call, waiter):
    from voicelab.core.models import Finding

    rig.evals.findings = [Finding(evaluator="timing", category="c", title="Dead air")]
    call = finished_call()
    rig.evals.hold = True
    client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    with client.websocket_connect(f"/ws/calls/{call['id']}") as socket:
        assert socket.receive_json() == {"type": "status", "status": "evaluating"}
        rig.run_on_loop(release_evals(rig))
        messages = collect(socket, "eval")
    kinds = [m["type"] for m in messages]
    assert "evaluator" in kinds
    assert "finding" in kinds
    assert kinds[-1] == "eval"
    assert messages[-1]["eval"]["callId"] == call["id"]
    evaluator = next(m for m in messages if m["type"] == "evaluator")["evaluator"]
    assert evaluator["durationS"] == 0.0
    assert "metrics" in evaluator


def test_call_stream_of_a_finished_call_sends_status_and_closes(client, finished_call):
    call = finished_call()
    with client.websocket_connect(f"/ws/calls/{call['id']}") as socket:
        assert socket.receive_json() == {"type": "status", "status": "done"}
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()


def test_call_stream_of_a_failed_call_sends_status_and_closes(client, rig, finished_call):
    rig.runner.fail_with = RuntimeError("no")
    call = finished_call()
    with client.websocket_connect(f"/ws/calls/{call['id']}") as socket:
        assert socket.receive_json() == {"type": "status", "status": "failed"}
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()


def test_call_stream_reports_a_failed_evaluation(client, rig, finished_call):
    call = finished_call()
    rig.evals.hold = True
    rig.evals.fail_with = RuntimeError("worker died")
    client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    with client.websocket_connect(f"/ws/calls/{call['id']}") as socket:
        assert socket.receive_json()["status"] == "evaluating"
        rig.run_on_loop(release_evals(rig))
        messages = collect(socket, "error")
    assert "worker died" in messages[-1]["message"]


class _LeftAlready:
    """A socket whose browser has gone, as uvicorn presents it."""

    async def accept(self):
        pass

    async def send_json(self, message):
        raise WebSocketDisconnect(1006)

    async def close(self):
        raise WebSocketDisconnect(1006)


@pytest.mark.parametrize(
    ("handler", "item"),
    [
        (live.health_stream, None),
        (live.call_stream, "call_missing"),
        (live.run_stream, "batch_missing"),
    ],
)
def test_a_client_leaving_is_not_an_error(rig, handler, item):
    arguments = [_LeftAlready(), *([item] if item else []), rig.services]
    rig.run_on_loop(lambda: handler(*arguments))


def test_call_with_vanished_files_is_404(client, rig, finished_call):
    call = finished_call()
    shutil.rmtree(rig.services.store.paths(call["id"]).dir)
    assert client.get(f"/api/calls/{call['id']}").status_code == 404


def test_a_queued_call_with_vanished_files_fails_and_the_run_goes_on(
    client, rig, start_run, waiter
):
    rig.runner.hold = True
    run = start_run(reps=2)["run"]
    waiter(lambda: "running" in statuses(client, run["id"]))
    queued = next(c for c in run_of(client, run["id"])["calls"] if c["status"] == "queued")
    shutil.rmtree(rig.services.store.paths(queued["id"]).dir)
    rig.run_on_loop(released(rig))
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    assert statuses(client, run["id"]) == ["done", "failed"]


def test_a_bad_listener_does_not_end_the_eval_queue(client, rig, finished_call, waiter):
    call = finished_call()

    def broken(_):
        raise RuntimeError("listener bug")

    rig.services.evals.on_settled = broken
    client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    waiter(lambda: len(rig.evals.jobs) == 2)
    waiter(lambda: client.get(f"/api/calls/{call['id']}").json()["eval"] is not None)
    client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    waiter(lambda: len(rig.evals.jobs) == 3)


def test_call_stream_of_a_missing_call(client):
    with client.websocket_connect("/ws/calls/call_missing") as socket:
        assert socket.receive_json()["type"] == "error"


# Execution outcome and evaluation ----------------------------------------------------------------


def _stopped_with_a_recording(client, rig, start_run, waiter):
    """A stopped call whose engine still wrote what it had recorded."""
    rig.runner.hold = True
    run = start_run()["run"]
    waiter(lambda: "running" in statuses(client, run["id"]))
    call_id = run_of(client, run["id"])["calls"][0]["id"]
    write_recording(rig.services.store, rig.services.store.read_artifact(call_id))
    client.post(f"/api/runs/{run['id']}/stop")
    waiter(lambda: statuses(client, run["id"]) == ["failed"])
    return run["id"], call_id


def test_a_stopped_call_cannot_be_evaluated_into_a_pass(client, rig, start_run, waiter):
    run_id, call_id = _stopped_with_a_recording(client, rig, start_run, waiter)
    response = client.post(f"/api/calls/{call_id}/reevaluate", json={})
    assert response.status_code == 409
    assert "stopped" in response.json()["error"]["message"]
    row = rig.services.calls.get(call_id)
    assert (row["status"], row["error"], row["verdict"]) == ("failed", "stopped", None)
    assert run_of(client, run_id)["run"]["counts"]["pass"] == 0


def test_the_service_itself_refuses_to_evaluate_an_aborted_call(client, rig, start_run, waiter):
    """`voicelab eval` reaches the service without going through the route."""
    _, call_id = _stopped_with_a_recording(client, rig, start_run, waiter)

    async def submit():
        rig.services.evals.submit(call_id)

    with pytest.raises(Conflict):
        rig.run_on_loop(submit)
    assert rig.services.calls.get(call_id)["error"] == "stopped"
    assert rig.services.store.read_artifact(call_id).error == "stopped"


def test_aborted_calls_stay_out_of_every_aggregate(client, rig, finished_call, waiter):
    """Even a verdict left on an aborted row by the old behavior must not count."""
    call = finished_call()
    rig.services.calls.update(call["id"], status="failed", error="stopped")
    assert client.get("/api/overview").json()["counts"]["pass"] == 0
    assert client.get("/api/calls").json()["facets"]["verdict"] == {}
    assert client.get("/api/matrix").json()["cells"] == []
    usage = client.get("/api/personas/per_sysadmin").json()["usage"]
    assert usage["counts"]["pass"] == 0


def test_an_evaluation_failure_can_be_retried(client, rig, finished_call, waiter):
    rig.evals.fail_with = RuntimeError("worker died")
    call = finished_call()
    row = rig.services.calls.get(call["id"])
    assert row["status"] == "failed"
    assert row["error"] is None  # the call itself went fine
    assert "worker died" in row["eval_error"]

    rig.evals.fail_with = None
    assert client.post(f"/api/calls/{call['id']}/reevaluate", json={}).status_code == 200
    waiter(lambda: rig.services.calls.get(call["id"])["status"] == "done")
    row = rig.services.calls.get(call["id"])
    assert row["eval_error"] is None
    assert row["verdict"] == "pass"
    assert rig.services.store.read_artifact(call["id"]).error is None


def test_a_failed_evaluation_keeps_its_reason_visible(client, rig, finished_call):
    rig.evals.fail_with = RuntimeError("worker died")
    call = finished_call()
    body = client.get(f"/api/calls/{call['id']}").json()
    assert "worker died" in body["call"]["error"]


# Run creation ------------------------------------------------------------------------------------


def test_a_run_is_not_schedulable_until_all_its_calls_exist(client, rig, waiter):
    """Waking the scheduler halfway through expansion used to finish the run with a third of its
    calls, and strand the rest."""
    store = rig.services.store
    real_write = store.write_artifact
    loop = rig.client.portal.call(asyncio.get_running_loop)
    writes = []

    def slow_write(artifact):
        real_write(artifact)
        writes.append(artifact.id)
        if len(writes) == 2:
            loop.call_soon_threadsafe(rig.services.worker.wake)
            time.sleep(0.2)

    store.write_artifact = slow_write
    body = {"pairs": [PAIR], "reps": 3, "profile": "local-default"}
    run = client.post("/api/runs", json=body).json()["run"]
    assert run["calls"] == 3
    store.write_artifact = real_write
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    assert len(rig.runner.requests) == 3
    assert statuses(client, run["id"]) == ["done", "done", "done"]


def test_a_failed_expansion_leaves_nothing_behind(client, rig):
    store = rig.services.store
    real_write = store.write_artifact
    calls_dir = rig.settings.data_dir / "calls"
    writes = []

    def failing_write(artifact):
        writes.append(artifact.id)
        if len(writes) == 2:
            raise OSError("disk full")
        real_write(artifact)

    store.write_artifact = failing_write
    with pytest.raises(OSError, match="disk full"):
        client.post("/api/runs", json={"pairs": [PAIR], "reps": 3, "profile": "local-default"})
    store.write_artifact = real_write
    assert client.get("/api/runs").json()["items"] == []
    assert client.get("/api/calls").json()["total"] == 0
    assert not list(calls_dir.glob("*")) if calls_dir.exists() else True


# The live gate during model preparation -----------------------------------------------------


def _hold_the_gate_during_preparation(rig):
    """A mic call that starts while models load, which is the gap the first gate check can't see."""
    profiles = rig.services.profiles
    real = profiles.wait_for_servers

    async def preparing(profile, timeout_s):
        profiles.wait_for_servers = real  # one mic call, not one per preparation
        rig.services.gate.hold("mic")
        await real(profile, timeout_s)

    profiles.wait_for_servers = preparing


def _let_go(rig):
    async def release():
        rig.services.gate.release("mic")

    return release


def test_a_bot_call_waits_if_a_mic_call_began_during_preparation(client, rig, start_run, waiter):
    _hold_the_gate_during_preparation(rig)
    run = start_run()["run"]
    waiter(lambda: "running" in statuses(client, run["id"]))
    time.sleep(0.1)
    assert rig.runner.requests == []
    rig.run_on_loop(_let_go(rig))
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    assert len(rig.runner.requests) == 1


def test_an_eval_waits_if_a_mic_call_began_during_preparation(client, rig, finished_call, waiter):
    call = finished_call()
    jobs = len(rig.evals.jobs)
    _hold_the_gate_during_preparation(rig)
    client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    time.sleep(0.1)
    assert len(rig.evals.jobs) == jobs
    rig.run_on_loop(_let_go(rig))
    waiter(lambda: len(rig.evals.jobs) == jobs + 1)


# Restart -----------------------------------------------------------------------------------------


def test_restart_fails_calls_cut_off_and_finishes_the_rest(tmp_path, waiter):
    rig = build_rig(tmp_path)
    with rig.client:
        rig.runner.hold = True
        run_id = rig.client.post("/api/runs", json={"pairs": [PAIR], "reps": 3}).json()["run"]["id"]
        waiter(lambda: "running" in statuses(rig.client, run_id))
    # Shutdown cancelled the call on the line and left it marked running, as a crash would.
    after = build_rig(tmp_path)
    with after.client:
        waiter(lambda: run_of(after.client, run_id)["run"]["status"] == "done")
        calls = run_of(after.client, run_id)["calls"]
        assert sorted(c["status"] for c in calls) == ["done", "done", "failed"]
        failed = next(c for c in calls if c["status"] == "failed")
        assert after.services.calls.get(failed["id"])["error"] == "interrupted by restart"
        assert len(after.runner.requests) == 2


def test_restart_requeues_evaluations(tmp_path, waiter):
    rig = build_rig(tmp_path)
    with rig.client:
        rig.evals.hold = True
        run_id = rig.client.post("/api/runs", json={"pairs": [PAIR]}).json()["run"]["id"]
        waiter(lambda: "evaluating" in statuses(rig.client, run_id))
    after = build_rig(tmp_path)
    with after.client:
        waiter(lambda: run_of(after.client, run_id)["run"]["status"] == "done")
        assert statuses(after.client, run_id) == ["done"]
        assert len(after.evals.jobs) == 1
        assert after.runner.requests == []


def test_restart_keeps_paused_runs_paused(tmp_path, waiter):
    rig = build_rig(tmp_path)
    with rig.client:
        rig.runner.hold = True
        run_id = rig.client.post("/api/runs", json={"pairs": [PAIR], "reps": 2}).json()["run"]["id"]
        waiter(lambda: "running" in statuses(rig.client, run_id))
        rig.client.post(f"/api/runs/{run_id}/pause")
    after = build_rig(tmp_path)
    with after.client:
        assert run_of(after.client, run_id)["run"]["status"] == "paused"
        assert after.runner.requests == []
        after.client.post(f"/api/runs/{run_id}/resume")
        waiter(lambda: run_of(after.client, run_id)["run"]["status"] == "done")
        assert len(after.runner.requests) == 1


def test_restart_survives_a_call_whose_files_vanished(tmp_path, waiter):
    import shutil

    rig = build_rig(tmp_path)
    with rig.client:
        rig.runner.hold = True
        run_id = rig.client.post("/api/runs", json={"pairs": [PAIR]}).json()["run"]["id"]
        waiter(lambda: "running" in statuses(rig.client, run_id))
        call_id = rig.services.calls.in_batch(run_id)[0]["id"]
    shutil.rmtree(rig.services.store.paths(call_id).dir)
    after = build_rig(tmp_path)
    with after.client:
        assert after.client.get("/api/health").status_code == 200
        assert after.services.calls.get(call_id)["error"] == "call files are missing"


def test_snapshots_are_taken_at_queue_time(client, start_run, rig, waiter):
    rig.runner.hold = True
    run = start_run(reps=2)["run"]
    waiter(lambda: "running" in statuses(client, run["id"]))
    persona = client.get("/api/personas/per_sysadmin").json()
    client.put("/api/personas/per_sysadmin", json={**persona, "name": "Renamed"})
    rig.run_on_loop(released(rig))
    waiter(lambda: run_of(client, run["id"])["run"]["status"] == "done")
    assert {r.persona.name for r in rig.runner.requests} == {persona["name"]}
