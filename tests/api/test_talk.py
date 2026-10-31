from __future__ import annotations

import asyncio
import time

from conftest import PAIR
from fakes import build_rig

from voicelab.core.profiles import LocalService
from voicelab.services.errors import Conflict

BODY = {"scenarioId": None, "personaId": None, "line": None, "record": True, "profile": None}


def open_session(client, **overrides):
    response = client.post("/api/talk/sessions", json={**BODY, **overrides})
    assert response.status_code == 201, response.text
    return response.json()


def test_session_flow_ends_in_evaluation(client, rig, waiter):
    session = open_session(client)
    session_id, call_id = session["sessionId"], session["callId"]
    assert session["offerUrl"] == f"/api/talk/sessions/{session_id}/offer"
    assert session["iceServers"] == []  # nothing leaves this machine
    assert session["sheet"] is None
    state = client.get(f"/api/talk/sessions/{session_id}").json()
    assert state["state"] == "connecting"
    assert state["callId"] == call_id
    assert state["startedAt"].endswith("Z")

    offer = {"sdp": "v=0", "type": "offer", "pc_id": None, "restart_pc": False}
    answer = client.post(session["offerUrl"], json=offer).json()
    assert answer == {"sdp": "answer:v=0", "type": "answer", "pc_id": session_id}
    assert client.get(f"/api/talk/sessions/{session_id}").json()["state"] == "live"
    assert rig.talk.sessions[0].offers == [("v=0", "offer")]

    assert client.post(f"/api/talk/sessions/{session_id}/end").json() == {"callId": call_id}
    assert client.get(f"/api/talk/sessions/{session_id}").json()["state"] == "ended"
    waiter(lambda: client.get(f"/api/calls/{call_id}").json()["summary"]["status"] == "done")
    body = client.get(f"/api/calls/{call_id}").json()
    assert body["call"]["mode"] == "human"
    assert body["eval"]["verdict"] == "pass"
    assert body["summary"]["durationS"] == 4.0


def test_offer_keeps_the_clients_snake_case_and_pc_id(client):
    session = open_session(client)
    offer = {"sdp": "v=0", "type": "offer", "pc_id": "pc_1", "restart_pc": True}
    assert client.post(session["offerUrl"], json=offer).json()["pc_id"] == "pc_1"


def test_ice_candidates_reach_the_session(client, rig):
    session = open_session(client)
    candidate = {"candidate": "c", "sdp_mid": "0", "sdp_mline_index": 0}
    body = {"pc_id": "pc_1", "candidates": [candidate]}
    assert client.patch(session["offerUrl"], json=body).status_code == 200
    assert rig.talk.sessions[0].candidates == [candidate]


def test_session_without_ice_support_still_accepts_the_patch(client, rig):
    session = open_session(client)
    rig.talk.sessions[0].__class__ = type("Plain", (), {})
    assert client.patch(session["offerUrl"], json={"candidates": []}).status_code == 200


def test_scenario_builds_the_sheet_and_reaches_the_factory(client, rig):
    session = open_session(client, scenarioId=PAIR["scenarioId"], personaId=PAIR["personaId"])
    sheet = session["sheet"]
    assert "billing zip: 95112" in sheet["youAre"]
    assert sheet["problem"].startswith("Mail to your domain")
    assert sheet["beats"][0]["id"] == "report_problem"
    assert sheet["mustNotSay"]
    request = rig.talk.requests[0]
    assert request.scenario.id == PAIR["scenarioId"]
    assert request.persona.id == PAIR["personaId"]


def test_line_is_passed_through(client, rig):
    line = {"codec": "none", "noise": "car", "snrDb": 9, "packetLossPct": 0, "oneWayLatencyMs": 0}
    open_session(client, line=line)
    assert rig.talk.requests[0].line.noise == "car"


def test_second_session_conflicts_while_one_is_live(client):
    open_session(client)
    assert client.post("/api/talk/sessions", json=BODY).status_code == 409


def test_unknown_scenario_is_invalid(client):
    response = client.post("/api/talk/sessions", json={**BODY, "scenarioId": "scn_missing"})
    assert response.status_code == 400


def test_loading_model_blocks_the_session_and_names_the_role(client, rig):
    profile = rig.services.profiles.require("local-default")
    profile.roles["llm_support"]["url"] = "http://127.0.0.1:18998/v1"
    profile.services.append(LocalService(name="llm", command=["sleep", "30"], port=18998))
    response = client.post("/api/talk/sessions", json=BODY)
    assert response.status_code == 503
    assert response.json()["error"]["detail"]["role"] == "llm_support"
    assert rig.talk.requests == []


def test_end_twice_hangs_up_once(client, rig, waiter):
    session = open_session(client)
    url = f"/api/talk/sessions/{session['sessionId']}/end"
    client.post(url)
    client.post(url)
    assert rig.talk.sessions[0].ended == 1


def test_offer_after_the_end_conflicts(client):
    session = open_session(client)
    client.post(f"/api/talk/sessions/{session['sessionId']}/end")
    offer = {"sdp": "v=0", "type": "offer"}
    assert client.post(session["offerUrl"], json=offer).status_code == 409


def test_unknown_session(client):
    assert client.get("/api/talk/sessions/talk_missing").status_code == 404
    assert client.post("/api/talk/sessions/talk_missing/end").status_code == 404
    offer = {"sdp": "x", "type": "offer"}
    assert client.post("/api/talk/sessions/talk_missing/offer", json=offer).status_code == 404


def test_unrecorded_session_leaves_nothing_behind(client, rig):
    session = open_session(client, record=False)
    client.post(f"/api/talk/sessions/{session['sessionId']}/end")
    assert client.get(f"/api/calls/{session['callId']}").status_code == 404
    assert not rig.services.store.paths(session["callId"]).dir.exists()
    assert rig.evals.jobs == []


def test_factory_failure_marks_the_call_and_frees_the_queue(tmp_path, waiter):
    rig = build_rig(tmp_path, raise_server_exceptions=False)
    rig.talk.fail_with = RuntimeError("no webrtc")
    with rig.client as client:
        response = client.post("/api/talk/sessions", json=BODY)
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal"
        assert not rig.services.gate.active
        failed = rig.services.calls.with_status("failed")
        assert "no webrtc" in failed[0]["error"]
        run = client.post("/api/runs", json={"pairs": [PAIR]}).json()["run"]
        waiter(lambda: client.get(f"/api/runs/{run['id']}").json()["run"]["status"] == "done")


def test_mic_call_holds_back_new_bot_calls_and_evals(client, rig, start_run, waiter):
    session = open_session(client)
    run = start_run()["run"]
    assert client.get(f"/api/runs/{run['id']}").json()["calls"][0]["status"] == "queued"
    assert rig.runner.requests == []
    client.post(f"/api/talk/sessions/{session['sessionId']}/end")
    waiter(lambda: client.get(f"/api/runs/{run['id']}").json()["run"]["status"] == "done")
    assert len(rig.runner.requests) == 1


def test_mic_call_holds_back_a_requested_eval(client, rig, finished_call, waiter):
    call = finished_call()
    session = open_session(client)
    jobs = len(rig.evals.jobs)
    client.post(f"/api/calls/{call['id']}/reevaluate", json={})
    assert len(rig.evals.jobs) == jobs
    assert client.get(f"/api/calls/{call['id']}").json()["summary"]["status"] == "evaluating"
    client.post(f"/api/talk/sessions/{session['sessionId']}/end")
    waiter(lambda: len(rig.evals.jobs) >= jobs + 2)


def test_a_call_already_running_finishes_during_a_mic_call(client, rig, start_run, waiter):
    rig.runner.hold = True
    run = start_run()["run"]
    waiter(lambda: rig.runner.requests)
    session = open_session(client)

    async def release():
        rig.runner.hold = False
        rig.runner.release.set()

    rig.run_on_loop(release)
    # The call ends, and its evaluation waits for the person to hang up.
    waiter(
        lambda: client.get(f"/api/runs/{run['id']}").json()["calls"][0]["status"] == "evaluating"
    )
    assert rig.evals.jobs == []
    client.post(f"/api/talk/sessions/{session['sessionId']}/end")
    waiter(lambda: client.get(f"/api/runs/{run['id']}").json()["run"]["status"] == "done")


def test_session_the_browser_never_connected_is_ended(client, rig, start_run, waiter):
    session = open_session(client)
    rig.settings.talk_connect_timeout_s = 0.0
    rig.run_on_loop(rig.services.talks.sweep)
    assert rig.talk.sessions[0].ended == 1
    assert not rig.services.gate.active
    state = client.get(f"/api/talk/sessions/{session['sessionId']}").json()
    assert state["state"] == "ended"


def test_live_session_is_left_alone_until_the_limit(client, rig):
    session = open_session(client)
    client.post(session["offerUrl"], json={"sdp": "x", "type": "offer"})
    rig.settings.talk_connect_timeout_s = 0.0
    rig.run_on_loop(rig.services.talks.sweep)
    assert rig.talk.sessions[0].ended == 0
    rig.settings.talk_max_s = 0.0
    rig.run_on_loop(rig.services.talks.sweep)
    assert rig.talk.sessions[0].ended == 1


def test_engine_reported_disconnect_ends_the_session(client, rig, waiter):
    session = open_session(client)

    async def drop():
        await rig.talk.sessions[0].emit({"type": "disconnected"})

    rig.run_on_loop(drop)
    waiter(
        lambda: client.get(f"/api/talk/sessions/{session['sessionId']}").json()["state"] == "ended"
    )
    waiter(lambda: rig.talk.sessions[0].ended == 1)


def test_shutdown_ends_open_sessions(tmp_path):
    rig = build_rig(tmp_path)
    with rig.client as client:
        open_session(client)
    assert rig.talk.sessions[0].ended == 1


# A profile with one session ----------------------------------------------------------------------


def one_session(rig, profile_id="local-default"):
    profiles = rig.services.profiles
    profiles.items[profile_id] = profiles.require(profile_id).model_copy(update={"max_calls": 1})


def let_the_bot_call_finish(rig):
    async def release():
        rig.runner.hold = False
        rig.runner.release.set()

    rig.run_on_loop(release)


def test_a_mic_call_is_refused_while_a_bot_call_is_on_the_line(client, rig, start_run, waiter):
    one_session(rig)
    rig.runner.hold = True
    start_run()
    waiter(lambda: rig.runner.requests)

    response = client.post("/api/talk/sessions", json=BODY)

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "profile_busy"
    assert "1 call at a time" in error["message"]
    assert "wait" in error["message"].lower()
    assert error["detail"]["maxCalls"] == 1
    assert rig.talk.requests == []
    assert not rig.services.gate.active
    let_the_bot_call_finish(rig)


def test_the_refusal_says_how_long_when_the_run_has_timed_a_call(client, rig, start_run, waiter):
    one_session(rig)
    rig.runner.hold = True
    run = start_run(reps=2)["run"]
    waiter(lambda: rig.runner.requests)
    calls = client.get(f"/api/runs/{run['id']}").json()["calls"]
    rig.services.calls.update(calls[1]["id"], status="done", duration_s=180.0)

    error = client.post("/api/talk/sessions", json=BODY).json()["error"]

    assert 0 < error["detail"]["waitS"] <= 180
    assert f"about {round(error['detail']['waitS'])} s" in error["message"]
    let_the_bot_call_finish(rig)


def test_a_mic_call_can_start_once_the_bot_call_has_finished(client, rig, start_run, waiter):
    one_session(rig)
    rig.runner.hold = True
    run = start_run()["run"]
    waiter(lambda: rig.runner.requests)
    assert client.post("/api/talk/sessions", json=BODY).status_code == 409
    let_the_bot_call_finish(rig)
    waiter(lambda: client.get(f"/api/runs/{run['id']}").json()["run"]["status"] == "done")
    open_session(client)


def test_a_bot_call_being_evaluated_leaves_the_session_free(client, rig, start_run, waiter):
    """Eval uses other servers, so a person doesn't have to wait for it."""
    one_session(rig)
    rig.evals.hold = True
    run = start_run()["run"]
    calls = lambda: client.get(f"/api/runs/{run['id']}").json()["calls"]  # noqa: E731
    waiter(lambda: calls()[0]["status"] == "evaluating")
    open_session(client)


def test_a_bot_call_still_loading_gives_way_to_a_mic_call(client, rig, start_run, waiter):
    """Nothing is on the line yet, and mic calls go first."""
    one_session(rig)
    profiles = rig.services.profiles
    real, loaded, loading = profiles.ready, asyncio.Event(), []

    async def slow_ready(profile, timeout_s):
        loading.append(profile.id)
        await loaded.wait()
        return await real(profile, timeout_s)

    async def finish_loading():
        loaded.set()

    profiles.ready = slow_ready
    run = start_run()["run"]
    waiter(lambda: loading)
    session = open_session(client)
    rig.run_on_loop(finish_loading)
    time.sleep(0.05)
    assert rig.runner.requests == []
    client.post(f"/api/talk/sessions/{session['sessionId']}/end")
    waiter(lambda: client.get(f"/api/runs/{run['id']}").json()["run"]["status"] == "done")
    assert len(rig.runner.requests) == 1


def test_a_failed_mic_setup_gives_the_session_back(tmp_path, waiter):
    rig = build_rig(tmp_path, raise_server_exceptions=False)
    one_session(rig)
    rig.talk.fail_with = RuntimeError("no webrtc")
    with rig.client as client:
        assert client.post("/api/talk/sessions", json=BODY).status_code == 500
        run = client.post("/api/runs", json={"pairs": [PAIR]}).json()["run"]
        waiter(lambda: client.get(f"/api/runs/{run['id']}").json()["run"]["status"] == "done")
        rig.talk.fail_with = None
        assert client.post("/api/talk/sessions", json=BODY).status_code == 201


def test_a_second_person_starting_during_setup_is_refused(client, rig):
    """Models can take a moment to check, and the mic call's claim has to come before that."""
    one_session(rig)
    profile = rig.services.profiles.require("local-default")
    real = rig.services.profiles.require_ready
    outcomes = []

    async def and_try_again(*args, **options):
        stack = await real(*args, **options)
        try:
            await rig.services.talks.create(
                scenario_id=None, persona_id=None, line=None, record=True, profile_id=profile.id
            )
        except Conflict as exc:
            outcomes.append(exc.message)
        return stack

    rig.services.profiles.require_ready = and_try_again
    open_session(client)
    assert outcomes == ["a mic call is already live"]
