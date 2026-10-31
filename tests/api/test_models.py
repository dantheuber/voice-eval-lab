from __future__ import annotations

import socket
import sqlite3
import sys

import pytest
import yaml
from fakes import PROFILE_YAML, build_rig, write_profiles

from voicelab.core.models import (
    CallArtifact,
    CallMode,
    CriterionResult,
    EvalResult,
    EvaluatorRun,
    Finding,
    Verdict,
)
from voicelab.core.profiles import LocalService
from voicelab.services import system
from voicelab.services.compare import ComparedCall, compare_runs
from voicelab.store.sqlite import Database


def test_profiles_list(client):
    body = client.get("/api/profiles").json()
    assert body["activeId"] == "local-default"
    local = next(p for p in body["items"] if p["id"] == "local-default")
    assert local["label"] == "Local"
    assert local["configPath"].endswith("local-default.yaml")
    assert local["liveCapable"] is False
    assert local["benchmark"] is None
    assert local["health"] == "ok"


def test_untested_profile_is_not_the_default_fallback(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    untested = PROFILE_YAML.format(id="aaa-cloud", label="Cloud", live="false", llm="x")
    (profiles / "a.yaml").write_text(f"{untested}untested: true\n")
    (profiles / "b.yaml").write_text(
        PROFILE_YAML.format(id="zzz-fake", label="Fake", live="false", llm="x")
    )
    rig = build_rig(tmp_path)
    with rig.client as client:
        assert client.get("/api/profiles").json()["activeId"] == "zzz-fake"


def test_a_profile_that_cannot_build_is_model_not_ready(tmp_path):
    rig = build_rig(tmp_path)

    def broken(profile):
        raise ImportError("boto3 is not installed")

    rig.services.profiles.stack_factory = broken
    with rig.client as client:
        response = client.post("/api/scenarios/draft", json={"description": "x"})
        assert response.status_code == 503
        assert "boto3" in response.json()["error"]["message"]


def test_active_profile_change_persists(client, rig):
    assert client.put("/api/profiles/active", json={"id": "live"}).json()["id"] == "live"
    assert client.get("/api/profiles").json()["activeId"] == "live"
    assert rig.services.catalog.setting("active_profile") == "live"


def test_unknown_profile_is_invalid(client):
    assert client.put("/api/profiles/active", json={"id": "nope"}).status_code == 400


def test_profile_change_conflicts_during_a_run(client, rig, start_run):
    rig.runner.hold = True
    start_run()
    assert client.put("/api/profiles/active", json={"id": "live"}).status_code == 409


def test_profile_change_keeps_servers_that_did_not_change(client, rig):
    kept = LocalService(name="stt", command=["sleep", "1"], port=18991)
    dropped = LocalService(name="old", command=["sleep", "1"], port=18992)
    rig.services.profiles.require("local-default").services.extend([kept, dropped])
    rig.services.profiles.require("live").services.append(kept)
    supervisor = rig.services.supervisor
    rig.run_on_loop(lambda: supervisor.apply([kept, dropped], start=False))
    entry = supervisor._entries["stt"]
    client.put("/api/profiles/active", json={"id": "live"})
    assert supervisor._entries["stt"] is entry
    assert "old" not in supervisor._entries


def test_health_covers_every_role(client):
    body = client.get("/api/health").json()
    roles = {item["role"]: item for item in body["items"]}
    assert roles["llm_support"]["state"] == "ok"
    assert roles["audio_analyst"]["state"] == "off"
    assert body["ready"] is True
    assert set(roles["llm_support"]) == {
        "role",
        "name",
        "model",
        "state",
        "detail",
        "elapsedS",
        "usualS",
    }


def test_health_without_a_profile_is_not_ready(tmp_path):
    (tmp_path / "profiles").mkdir()
    rig = build_rig(tmp_path)
    with rig.client as client:
        body = client.get("/api/health").json()
        assert body["ready"] is False
        assert body["items"][0]["state"] == "down"
        assert client.get("/api/profiles").json() == {"items": [], "activeId": None}
        assert client.get("/api/roles").json() == {"items": []}
        assert client.get("/api/voices").json() == {"items": []}
        assert client.post("/api/runs", json={"pairs": []}).status_code == 400
        response = client.post("/api/scenarios/draft", json={"description": "x"})
        assert response.status_code == 503


def test_health_socket_pushes_changes(client, rig):
    with client.websocket_connect("/ws/health") as socket:
        first = socket.receive_json()
        assert first["type"] == "health"
        assert first["items"][0]["role"]
        rig.run_on_loop(lambda: rig.services.supervisor.apply([], start=False))
        assert socket.receive_json()["type"] == "health"


def test_roles_list_options_from_every_profile(client):
    items = client.get("/api/roles").json()["items"]
    llm = next(item for item in items if item["role"] == "llm_support")
    assert llm["interface"] == "LLM"
    assert llm["label"] == "support bot LLM"
    assert llm["adapter"] == "fake:llm"
    assert {o["id"] for o in llm["options"]} == {"fake:llm", "fake:big-llm"}
    assert llm["health"]["role"] == "llm_support"
    analyst = next(item for item in items if item["role"] == "audio_analyst")
    assert analyst["adapter"] == "none"
    assert analyst["interface"] == "AudioAnalyst"


def test_setting_a_role_writes_the_active_profiles_yaml(client, rig):
    response = client.put("/api/roles/llm_support", json={"adapter": "fake:big-llm"})
    assert response.json()["adapter"] == "fake:big-llm"
    saved = yaml.safe_load((rig.settings.profiles_dir / "local-default.yaml").read_text())
    assert saved["roles"]["llm_support"]["model"] == "big-llm"
    assert saved["roles"]["llm_caller"]["model"] == "llm"


def test_setting_a_role_keeps_the_header_comments(client, rig):
    path = rig.settings.profiles_dir / "local-default.yaml"
    path.write_text("# License terms to keep in view.\n" + path.read_text())
    client.put("/api/roles/llm_support", json={"adapter": "fake:big-llm"})
    assert path.read_text().startswith("# License terms to keep in view.\nid: local-default")


def test_a_broken_profile_file_does_not_hide_the_others(tmp_path, caplog):
    profiles = tmp_path / "profiles"
    write_profiles(profiles)
    (profiles / "aaa-broken.yaml").write_text("label: Local: not valid yaml\n")
    rig = build_rig(tmp_path)
    with rig.client as client:
        ids = {item["id"] for item in client.get("/api/profiles").json()["items"]}
    assert ids == {"local-default", "live"}
    assert "aaa-broken.yaml" in caplog.text


def test_a_required_role_cannot_be_emptied(client):
    assert client.put("/api/roles/llm_support", json={"adapter": "none"}).status_code == 409


def test_unknown_adapter_and_role(client):
    assert client.put("/api/roles/llm_support", json={"adapter": "x"}).status_code == 400
    assert client.put("/api/roles/no_role", json={"adapter": "x"}).status_code == 404


def test_role_change_conflicts_during_a_run(client, rig, start_run):
    rig.runner.hold = True
    start_run()
    response = client.put("/api/roles/llm_support", json={"adapter": "fake:big-llm"})
    assert response.status_code == 409


def test_restart_needs_a_server_behind_the_role(client):
    assert client.post("/api/servers/llm_support/restart").status_code == 404


def test_restart_all_is_accepted(client):
    assert client.post("/api/servers/restart-all").json() == {"accepted": True}


def test_benchmark_lands_on_the_profile(client, waiter):
    assert client.post("/api/benchmark").json() == {"accepted": True}

    def measured():
        items = client.get("/api/profiles").json()["items"]
        return next(p for p in items if p["id"] == "local-default")["benchmark"]

    result = waiter(measured)
    assert result["botReplyP50S"] >= 0
    assert result["at"]


def test_system(client):
    body = client.get("/api/system").json()
    assert set(body["gpu"]) == {"device", "memoryUsedGb", "memoryTotalGb"}
    assert body["urls"]["local"] == "http://127.0.0.1:18080"
    assert body["urls"]["lan"].startswith("https://")
    assert body["urls"]["lan"].endswith(":18443")
    assert body["cert"] == "self-signed"
    assert body["servers"] == []


def test_thresholds_default_to_what_eval_really_uses(client):
    assert client.get("/api/thresholds").json() == {
        "deadAirS": 6.0,
        "bargeStopTargetS": 1.0,
        "replyLatencyWarnS": 5.0,
        "mosBaseline": 3.0,
        "audioListeningMode": "advisory",
    }


def test_thresholds_save_and_reach_evaluation(client, rig, finished_call):
    body = {
        "deadAirS": 2.0,
        "bargeStopTargetS": 0.5,
        "replyLatencyWarnS": 2.5,
        "mosBaseline": 4.0,
        "audioListeningMode": "strict",
    }
    saved = client.put("/api/thresholds", json=body).json()
    assert saved == {**body, "audioListeningMode": "advisory"}  # read-only
    assert client.get("/api/thresholds").json() == saved
    finished_call()
    config = rig.evals.jobs[-1].config
    assert config.timing.dead_air_s == 2.0
    assert config.timing.barge_in_reaction_s == 0.5
    assert config.timing.slow_response_s == 2.5
    assert config.audio_quality.low_mos == 4.0


def test_bad_thresholds_are_rejected(client):
    current = client.get("/api/thresholds").json()
    assert client.put("/api/thresholds", json={**current, "deadAirS": 0}).status_code == 400


def _no_aws(monkeypatch, tmp_path):
    """Nothing set, and no files, so a real ~/.aws can't change a test."""
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))


def _aws_keys(monkeypatch, tmp_path):
    _no_aws(monkeypatch, tmp_path)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shh")


def test_aws_reports_status_without_values(client, monkeypatch, tmp_path):
    _aws_keys(monkeypatch, tmp_path)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    body = client.get("/api/aws").json()
    assert body == {
        "region": "us-west-2",
        "credentials": "found",
        "problem": None,
        "testedAgainstLive": False,
    }


def test_aws_without_credentials(client, monkeypatch, tmp_path):
    _no_aws(monkeypatch, tmp_path)
    body = client.get("/api/aws").json()
    assert body.pop("problem").startswith("no AWS credentials were found")
    assert body == {"region": None, "credentials": "not_found", "testedAgainstLive": False}


def test_aws_named_profile_nobody_selected_is_not_credentials(client, monkeypatch, tmp_path):
    _no_aws(monkeypatch, tmp_path)
    credentials = tmp_path / "credentials"
    credentials.write_text("[spike]\naws_access_key_id = x\naws_secret_access_key = y\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    assert client.get("/api/aws").json()["credentials"] == "not_found"

    credentials.write_text("[default]\naws_access_key_id = x\naws_secret_access_key = y\n")
    assert client.get("/api/aws").json()["credentials"] == "found"


def test_aws_test_answers(client):
    body = client.post("/api/aws/test").json()
    assert body["ok"] is False
    assert body["message"]


def _untested_profile(rig):
    return rig.services.profiles.require("live").model_copy(update={"untested": True})


def test_run_and_calls_say_when_adapters_were_untested(
    client, rig, start_run, waiter, monkeypatch, tmp_path
):
    _aws_keys(monkeypatch, tmp_path)
    rig.services.profiles.items["live"] = _untested_profile(rig)
    run = start_run(profile="live")["run"]
    assert run["untested"] is True
    waiter(lambda: client.get(f"/api/runs/{run['id']}").json()["run"]["status"] == "done")
    body = client.get(f"/api/runs/{run['id']}").json()
    assert body["run"]["untested"] is True
    assert body["calls"][0]["untested"] is True
    assert client.get(f"/api/calls/{body['calls'][0]['id']}").json()["summary"]["untested"] is True
    assert client.get("/api/calls").json()["items"][0]["untested"] is True


def test_tested_runs_say_so(client, start_run):
    run = start_run()["run"]
    assert run["untested"] is False
    assert client.get("/api/calls").json()["items"][0]["untested"] is False


def test_untested_survives_a_later_edit_of_the_profile(
    client, rig, start_run, waiter, monkeypatch, tmp_path
):
    _aws_keys(monkeypatch, tmp_path)
    rig.services.profiles.items["live"] = _untested_profile(rig)
    run = start_run(profile="live")["run"]
    waiter(lambda: client.get(f"/api/runs/{run['id']}").json()["run"]["status"] == "done")
    rig.services.profiles.reload()  # the YAML says tested, as if someone edited it afterwards
    assert rig.services.profiles.require("live").untested is False
    body = client.get(f"/api/runs/{run['id']}").json()
    assert body["run"]["untested"] is True
    assert body["calls"][0]["untested"] is True


def test_a_provider_used_at_run_time_marks_the_call_untested(client, rig, finished_call, waiter):
    """The profile can be tested while one adapter the engine actually used isn't."""
    real = rig.runner.__call__

    async def with_bedrock(request, *, stack, store, emit):
        artifact = await real(request, stack=stack, store=store, emit=emit)
        artifact.providers["llm_support"] = "bedrock/claude"
        return artifact

    rig.services.worker.runner = lambda: with_bedrock
    assert finished_call()["untested"] is True


def test_older_databases_gain_the_new_columns(tmp_path):
    path = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE calls (id TEXT PRIMARY KEY, status TEXT, error TEXT, batch_id TEXT, "
        "created_at TEXT, scenario_id TEXT, persona_id TEXT, verdict TEXT)"
    )
    connection.execute("CREATE TABLE runs (id TEXT PRIMARY KEY, started_at TEXT)")
    connection.execute(
        "CREATE TABLE call_findings (call_id TEXT, finding_id TEXT, advisory INT, "
        "human_verdict TEXT)"
    )
    connection.execute(
        "INSERT INTO calls (id, status, error) VALUES "
        "('call_1', 'failed', 'evaluation failed: boom'), ('call_2', 'failed', 'stopped')"
    )
    connection.commit()
    connection.close()
    database = Database(path)
    rows = {row["id"]: row for row in database.rows("SELECT * FROM calls")}
    assert rows["call_1"]["eval_error"] == "evaluation failed: boom"
    assert rows["call_1"]["error"] is None
    assert rows["call_2"]["error"] == "stopped"
    assert rows["call_2"]["untested"] == 0
    assert database.rows("SELECT untested FROM runs") == []
    assert database.rows("SELECT anchor_s, occurrence FROM call_findings") == []
    database.close()


# Compare -----------------------------------------------------------------------------------------


def _compared(
    call_id, *, verdict=Verdict.PASS, criteria=(), findings=(), metrics=None, mode="turn_based"
):
    runs = [EvaluatorRun(name=name, status="ok", metrics=m) for name, m in (metrics or {}).items()]
    artifact = CallArtifact(id=call_id, mode=CallMode(mode), timing_is_real=mode == "live_bots")
    result = EvalResult(
        call_id=call_id,
        verdict=verdict,
        criteria=list(criteria),
        findings=list(findings),
        evaluators=runs,
    )
    row = {"id": call_id, "scenario_id": "scn_a", "persona_id": "per_a"}
    return ComparedCall(row, artifact, result)


def _criterion(criterion_id="verify", status="pass", kind="must"):
    return CriterionResult(criterion_id=criterion_id, kind=kind, text="Verify", status=status)


def _run(mode="turn_based", scenarios=("scn_a",)):
    return {"id": "batch", "mode": mode, "scenarioIds": list(scenarios)}


def measure(body, key):
    return next(m for m in body["measures"] if m["key"] == key)


def test_invalid_calls_leave_every_denominator():
    good = _compared("call_1", criteria=[_criterion()])
    invalid = _compared("call_2", verdict=Verdict.INVALID, criteria=[_criterion(status="fail")])
    body = compare_runs(_run(), _run(), [good, invalid], [good])
    assert body["criteria"][0]["a"] == {"pass": 1, "n": 1}
    assert body["cells"][0]["a"] == {"pass": 1, "n": 1}
    assert measure(body, "invalid") == {
        "key": "invalid",
        "label": "Invalid runs",
        "a": 1,
        "b": 0,
        "comparable": False,
        "reason": "test, not bot",
    }


def test_unknown_criteria_are_not_counted_as_failures():
    body = compare_runs(
        _run(),
        _run(),
        [_compared("call_1", criteria=[_criterion(status="unknown")])],
        [_compared("call_2", criteria=[_criterion()])],
    )
    assert body["criteria"][0]["a"] == {"pass": 0, "n": 0}


def test_beats_are_left_out_because_they_measure_the_caller():
    calls = [_compared("call_1", criteria=[_criterion(kind="beat")])]
    assert compare_runs(_run(), _run(), calls, calls)["criteria"] == []


def test_a_criterion_in_one_run_shows_zero_on_the_other():
    body = compare_runs(_run(), _run(), [_compared("call_1", criteria=[_criterion()])], [])
    assert body["criteria"][0]["b"] == {"pass": 0, "n": 0}


def test_scenarios_only_in_one_run_are_dropped():
    only_a = _compared("call_1", criteria=[_criterion("gone")])
    only_a.row["scenario_id"] = "scn_b"
    shared = _compared("call_2", criteria=[_criterion("kept")])
    body = compare_runs(_run(scenarios=("scn_a", "scn_b")), _run(), [only_a, shared], [shared])
    assert body["scenarioIds"] == ["scn_a"]
    assert body["droppedScenarioIds"] == ["scn_b"]
    assert [c["criterionId"] for c in body["criteria"]] == ["kept"]


def test_worst_names_the_biggest_drop_and_who_failed():
    before = [_compared(f"call_a{i}", criteria=[_criterion()]) for i in range(2)]
    after = [
        _compared("call_b0", criteria=[_criterion()]),
        _compared("call_b1", criteria=[_criterion(status="fail")]),
    ]
    assert compare_runs(_run(), _run(), before, after)["worst"] == {
        "criterionId": "verify",
        "deltaPts": -50,
        "callIds": ["call_b1"],
    }


def test_no_drop_means_no_worst():
    calls = [_compared("call_1", criteria=[_criterion()])]
    assert compare_runs(_run(), _run(), calls, calls)["worst"] is None


def test_latency_is_not_comparable_on_turn_based_runs():
    timing = {"timing": {"response_latency": {"median_s": 1.0}}}
    live = [_compared("call_1", metrics=timing, mode="live_bots")]
    body = compare_runs(_run("live_bots"), _run(), live, [_compared("call_2")])
    reply = measure(body, "replyP50S")
    assert reply["a"] == 1.0
    assert reply["b"] is None
    assert reply["comparable"] is False
    assert reply["reason"] == "turn-based"


def test_unmeasured_values_are_null_not_zero():
    calls = [_compared("call_1", mode="live_bots")]
    body = compare_runs(_run("live_bots"), _run("live_bots"), calls, calls)
    assert measure(body, "replyP50S")["a"] is None
    assert measure(body, "replyP50S")["reason"] == "not measured"
    assert measure(body, "mos")["a"] is None
    assert measure(body, "mos")["comparable"] is False


def test_mispronunciations_count_only_corroborated_tokens():
    metrics = {"pronunciation": {"critical_mismatches": 5, "critical_corroborated": 2}}
    calls = [_compared("call_1", metrics=metrics)]
    assert (
        measure(compare_runs(_run(), _run(), calls, calls), "criticalMispronunciations")["a"] == 2
    )


def test_advisory_notes_are_counted_but_never_comparable():
    advisory = Finding(evaluator="audio_listening", category="c", title="t", advisory=True)
    plain = Finding(evaluator="timing", category="c", title="t")
    calls = [_compared("call_1", findings=[advisory, plain])]
    note = measure(compare_runs(_run(), _run(), calls, calls), "advisoryNotes")
    assert (note["a"], note["comparable"], note["reason"]) == (1, False, "advisory")


def test_compare_route(client, rig, finished_call):
    rig.evals.verdicts = [Verdict.PASS, Verdict.FAIL]
    a = finished_call()["batchId"]
    b = finished_call()["batchId"]
    body = client.get("/api/compare", params={"a": a, "b": b}).json()
    assert body["a"]["id"] == a
    assert body["b"]["id"] == b
    assert body["cells"][0]["a"] == {"pass": 1, "n": 1}
    assert body["cells"][0]["b"] == {"pass": 0, "n": 1}


def test_compare_of_a_missing_run(client):
    assert client.get("/api/compare", params={"a": "x", "b": "y"}).status_code == 404


# Serving -----------------------------------------------------------------------------------------


@pytest.fixture
def served(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("private")
    rig = build_rig(tmp_path, dist=dist)
    with rig.client:
        yield rig.client


def test_static_files_are_served(served):
    assert served.get("/assets/app.js").text == "console.log(1)"
    assert served.get("/").text == "<html>app</html>"


def test_client_routes_fall_back_to_the_app(served):
    assert served.get("/calls/call_123").text == "<html>app</html>"
    assert served.get("/settings").text == "<html>app</html>"


def test_unknown_api_paths_stay_json_404s(served):
    response = served.get("/api/no/such/route")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert served.get("/ws/no").status_code == 404


@pytest.mark.parametrize(
    "path", ["/..%2Fsecret.txt", "/assets/..%2F..%2Fsecret.txt", "/%2e%2e/secret.txt"]
)
def test_dashboard_cannot_be_walked_out_of(served, path):
    response = served.get(path)
    assert "private" not in response.text
    assert response.status_code == 404


def test_no_build_says_how_to_make_one(client):
    response = client.get("/")
    assert response.status_code == 404
    assert "pnpm build" in response.json()["error"]["message"]


def test_method_not_allowed_uses_the_error_shape(client):
    response = client.put("/api/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "invalid"


def test_malformed_json_is_invalid(client):
    response = client.post(
        "/api/calls/x/verdict", content="{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid"


def test_a_cloud_profile_with_no_credentials_is_refused(client, rig, monkeypatch, tmp_path):
    _no_aws(monkeypatch, tmp_path)
    rig.services.profiles.items["live"] = _untested_profile(rig)
    response = client.put("/api/profiles/active", json={"id": "live"})
    assert response.status_code == 400
    assert "AWS credentials" in response.json()["error"]["message"]


def test_a_run_on_a_cloud_profile_with_no_credentials_is_refused(
    client, rig, monkeypatch, tmp_path
):
    _no_aws(monkeypatch, tmp_path)
    rig.services.profiles.items["live"] = _untested_profile(rig)
    pair = {"scenarioId": "scn_dns_mx_fix", "personaId": "per_impatient_owner"}
    body = {"pairs": [pair], "reps": 1, "profile": "live", "mode": "auto", "concurrency": 1}
    response = client.post("/api/runs", json={**body, "line": "persona", "label": None})
    assert response.status_code == 400
    assert "AWS credentials" in response.json()["error"]["message"]


def _unknown_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_PROFILE", "does-not-exist")


def _unknown_profile_beside_good_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shh")
    monkeypatch.setenv("AWS_PROFILE", "does-not-exist")


def _key_without_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")


def _credentials_file(text):
    def write(monkeypatch, tmp_path):
        path = tmp_path / "credentials"
        path.write_text(text)
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(path))

    return write


def _config_file(text):
    def write(monkeypatch, tmp_path):
        path = tmp_path / "config"
        path.write_text(text)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(path))
        monkeypatch.setenv("AWS_PROFILE", "here")

    return write


UNUSABLE = {
    "unknown profile": (_unknown_profile, "does-not-exist"),
    "unknown profile, good keys": (_unknown_profile_beside_good_keys, "does-not-exist"),
    "key without secret": (_key_without_secret, "AWS_SECRET_ACCESS_KEY"),
    "empty default": (_credentials_file("[default]\n"), "no AWS credentials"),
    "key in file, no secret": (
        _credentials_file("[default]\naws_access_key_id = x\n"),
        "aws_secret_access_key",
    ),
    "blank secret": (
        _credentials_file("[default]\naws_access_key_id = x\naws_secret_access_key =\n"),
        "incomplete",
    ),
    "unreadable file": (_credentials_file("[default\nnot ini"), "Unable to parse"),
    "role with a missing source": (
        _config_file("[profile here]\nrole_arn = arn:aws:iam::1:role/x\nsource_profile = gone\n"),
        "gone",
    ),
}


@pytest.mark.parametrize("name", UNUSABLE)
def test_credentials_boto3_could_not_use_are_refused_and_no_server_stops(
    client, rig, monkeypatch, tmp_path, name
):
    arrange, says = UNUSABLE[name]
    _no_aws(monkeypatch, tmp_path)
    arrange(monkeypatch, tmp_path)
    running = LocalService(name="stt", command=["sleep", "1"], port=18993)
    supervisor = rig.services.supervisor
    rig.run_on_loop(lambda: supervisor.apply([running], start=False))
    rig.services.profiles.items["live"] = _untested_profile(rig)
    entry = supervisor._entries["stt"]

    response = client.put("/api/profiles/active", json={"id": "live"})

    assert response.status_code == 400
    assert says in response.json()["error"]["message"]
    assert supervisor._entries["stt"] is entry
    assert rig.services.catalog.setting("active_profile") is None
    status = client.get("/api/aws").json()
    assert status["credentials"] == "not_found"
    assert says in status["problem"]


def _role(monkeypatch, tmp_path):
    credentials = tmp_path / "credentials"
    credentials.write_text("[base]\naws_access_key_id = x\naws_secret_access_key = y\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    _config_file("[profile here]\nrole_arn = arn:aws:iam::1:role/x\nsource_profile = base\n")(
        monkeypatch, tmp_path
    )


def _sso(monkeypatch, tmp_path):
    _config_file(
        "[profile here]\nsso_start_url = https://x\nsso_region = us-east-1\n"
        "sso_account_id = 1\nsso_role_name = r\n"
    )(monkeypatch, tmp_path)


def _credential_process(monkeypatch, tmp_path):
    marker = tmp_path / "ran"
    _config_file(f"[profile here]\ncredential_process = /usr/bin/touch {marker}\n")(
        monkeypatch, tmp_path
    )


@pytest.mark.parametrize("arrange", [_role, _sso, _credential_process])
def test_credentials_that_need_a_request_count_without_making_it(
    client, monkeypatch, tmp_path, arrange
):
    """The first use of these reaches AWS or runs a program, so the file is all that's checked."""
    _no_aws(monkeypatch, tmp_path)
    arrange(monkeypatch, tmp_path)

    def no_network(*_args, **_kwargs):
        raise AssertionError("the check reached the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    assert system.aws_status()["credentials"] == "found"
    assert not (tmp_path / "ran").exists()


def test_the_check_does_not_ask_the_instance_metadata_service(monkeypatch, tmp_path):
    _no_aws(monkeypatch, tmp_path)

    def no_network(*_args, **_kwargs):
        raise AssertionError("the check reached the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    assert system.aws_status()["credentials"] == "not_found"


def test_a_cloud_profile_without_boto3_says_how_to_get_it(client, rig, monkeypatch, tmp_path):
    _aws_keys(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "boto3", None)
    rig.services.profiles.items["live"] = _untested_profile(rig)
    response = client.put("/api/profiles/active", json={"id": "live"})
    assert response.status_code == 400
    assert "uv sync --extra aws" in response.json()["error"]["message"]
    assert client.get("/api/aws").json()["credentials"] == "not_found"
