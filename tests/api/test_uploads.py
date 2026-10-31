from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fakes import RATE, bursts

from voicelab.core.audio import AudioClip
from voicelab.core.models import Verdict
from voicelab.core.profiles import LocalService


def stereo_wav() -> bytes:
    caller = bursts([(0.5, 1.5)], 4.0).samples()
    bot = bursts([(2.0, 3.0)], 4.0).samples()
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([caller, bot], axis=1), RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def mono_wav() -> bytes:
    return bursts([(0.5, 1.5), (2.5, 3.5)], 4.0).to_wav_bytes()


def upload(client, data=None, *, name="call.wav", **fields):
    files = {"file": (name, data if data is not None else stereo_wav(), "audio/wav")}
    return client.post("/api/uploads", files=files, data=fields)


def test_upload_makes_a_call_with_a_probe_and_evidence(client, rig):
    response = upload(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["probe"] == {
        "durationS": 4.0,
        "channels": 2,
        "sampleRate": 16000,
        "channelMap": "unknown",
        "sizeBytes": len(stereo_wav()),
    }
    assert body["evidence"] == {
        "audio": True,
        "transcript": False,
        "intendedText": False,
        "toolLog": False,
        "dbSnapshots": False,
        "scenario": False,
    }
    review = client.get(f"/api/calls/{body['callId']}").json()
    assert review["call"]["mode"] == "upload"
    assert review["summary"]["status"] == "queued"
    assert rig.evals.jobs == []  # waits for the channel map and the Evaluate button


def test_mono_upload_is_reported_as_mono(client):
    assert upload(client, mono_wav()).json()["probe"]["channelMap"] == "mono"


def test_dual_mono_stereo_is_reported_as_mono(client):
    samples = bursts([(0.5, 1.5)], 2.0).samples()
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([samples, samples], axis=1), RATE, format="WAV", subtype="PCM_16")
    probe = upload(client, buffer.getvalue()).json()["probe"]
    assert probe["channels"] == 2
    assert probe["channelMap"] == "mono"


def test_context_is_recorded(client, rig):
    tool = {"name": "get_invoice", "arguments": {"invoice_id": "FH-1"}, "atS": 2.5, "result": {}}
    response = upload(
        client,
        scenarioId="scn_dns_mx_fix",
        botVersion="v7",
        intendedTranscript="Hello there.\nGoodbye.",
        toolLog=json.dumps(tool),
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["evidence"]["toolLog"] is True
    assert body["evidence"]["intendedText"] is True
    assert body["evidence"]["scenario"] is True
    artifact = rig.services.store.read_artifact(body["callId"])
    assert artifact.providers["bot_version"] == "v7"
    assert artifact.scenario_snapshot.id == "scn_dns_mx_fix"
    assert [e.name for e in artifact.timeline if e.type == "tool_call"] == ["get_invoice"]
    bot_turns = [e for e in artifact.timeline if e.type == "utterance" and e.speaker == "bot"]
    assert bot_turns[-1].intended_text == "Hello there. Goodbye."


def test_intended_transcript_as_json_uses_start_times(client, rig):
    lines = json.dumps([{"text": "Later line", "startS": 2.1}])
    call_id = upload(client, intendedTranscript=lines).json()["callId"]
    artifact = rig.services.store.read_artifact(call_id)
    bot = [e for e in artifact.timeline if e.type == "utterance" and e.speaker == "bot"]
    assert bot[0].intended_text == "Later line"


@pytest.mark.parametrize(
    "fields",
    [
        {"toolLog": "not json"},
        {"toolLog": '{"arguments": {}}'},
        {"intendedTranscript": '[{"nope": 1}]'},
        {"scenarioId": "scn_missing"},
    ],
)
def test_bad_context_is_rejected_and_leaves_nothing_behind(client, rig, fields):
    assert upload(client, **fields).status_code == 400
    assert not list((rig.settings.data_dir / "calls").glob("*"))


@pytest.mark.parametrize(
    "name",
    [
        "../evil.wav",
        "..\\evil.wav",
        "/etc/cron.d/evil.wav",
        "a/../../b.wav",
        "evil..wav",
    ],
)
def test_path_in_the_file_name_fails(client, rig, name):
    response = upload(client, name=name)
    assert response.status_code == 400
    assert not list((rig.settings.data_dir / "calls").glob("*"))
    assert not (rig.settings.data_dir / "evil.wav").exists()


def test_unsupported_type_fails(client):
    assert upload(client, b"<html>", name="notes.html").status_code == 400


def test_unreadable_audio_fails(client):
    response = upload(client, b"this is not audio", name="call.wav")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid"


def test_recording_over_the_length_cap_fails(client, rig):
    rig.services.store.max_upload_s = 1.0
    response = upload(client)
    assert response.status_code == 400
    assert "limit" in response.json()["error"]["message"]


def test_file_over_the_size_limit_fails(client, rig):
    rig.settings.max_upload_bytes = 16
    response = upload(client)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "invalid"


def _left_and_right(rig, call_id):
    paths = rig.services.store.paths(call_id)
    return AudioClip.from_file(paths.caller), AudioClip.from_file(paths.bot)


def _first_speech(clip: AudioClip) -> float:
    return float(np.argmax(np.abs(clip.samples()) > 0.05) / clip.sample_rate)


def test_channel_map_swaps_a_stereo_file(client, rig):
    call_id = upload(client, stereo_wav()).json()["callId"]
    caller, _ = _left_and_right(rig, call_id)
    assert _first_speech(caller) == pytest.approx(0.5, abs=0.1)
    response = client.post(
        f"/api/uploads/{call_id}/channel-map", json={"channelMap": "caller_right"}
    )
    assert response.json() == {"ok": True}
    caller, _ = _left_and_right(rig, call_id)
    assert _first_speech(caller) == pytest.approx(2.0, abs=0.1)


def test_channel_map_flips_speakers_on_a_mono_file(client, rig):
    call_id = upload(client, mono_wav()).json()["callId"]
    before, _ = _left_and_right(rig, call_id)
    client.post(f"/api/uploads/{call_id}/channel-map", json={"channelMap": "caller_right"})
    after, _ = _left_and_right(rig, call_id)
    assert _first_speech(before) != pytest.approx(_first_speech(after), abs=0.1)
    artifact = rig.services.store.read_artifact(call_id)
    note = next(e for e in artifact.timeline if e.type == "system" and "Labels" in e.detail)
    assert "swapped" in note.detail


def test_channel_map_mono_treats_a_stereo_file_as_one_mix(client, rig):
    call_id = upload(client, stereo_wav()).json()["callId"]
    client.post(f"/api/uploads/{call_id}/channel-map", json={"channelMap": "mono"})
    artifact = rig.services.store.read_artifact(call_id)
    assert "speaker_separation" in artifact.providers
    assert client.get(f"/api/calls/{call_id}").json()["overlaps"] == []


def test_channel_map_keeps_context_and_clears_old_results(client, rig, waiter):
    call_id = upload(client, botVersion="v7", scenarioId="scn_dns_mx_fix").json()["callId"]
    client.post(f"/api/uploads/{call_id}/evaluate")
    waiter(lambda: client.get(f"/api/calls/{call_id}").json()["summary"]["status"] == "done")
    client.post(f"/api/uploads/{call_id}/channel-map", json={"channelMap": "caller_right"})
    review = client.get(f"/api/calls/{call_id}").json()
    assert review["eval"] is None
    assert review["summary"]["verdict"] is None
    assert review["summary"]["status"] == "queued"
    artifact = rig.services.store.read_artifact(call_id)
    assert artifact.id == call_id
    assert artifact.providers["bot_version"] == "v7"
    assert artifact.scenario_id == "scn_dns_mx_fix"
    assert len(list((rig.settings.data_dir / "calls").iterdir())) == 1


def test_bad_channel_map_is_rejected(client):
    call_id = upload(client).json()["callId"]
    response = client.post(f"/api/uploads/{call_id}/channel-map", json={"channelMap": "sideways"})
    assert response.status_code == 400


def test_upload_routes_only_touch_uploads(client, finished_call):
    call = finished_call()
    assert client.post(f"/api/uploads/{call['id']}/evaluate").status_code == 404
    assert client.get(f"/api/uploads/{call['id']}/preview").status_code == 404
    response = client.post(f"/api/uploads/{call['id']}/channel-map", json={"channelMap": "mono"})
    assert response.status_code == 404


def test_evaluate_then_review(client, rig, waiter):
    call_id = upload(client).json()["callId"]
    response = client.post(f"/api/uploads/{call_id}/evaluate")
    assert response.json()["accepted"] is True
    assert response.json()["estimateS"] > 0
    waiter(lambda: client.get(f"/api/calls/{call_id}").json()["summary"]["status"] == "done")
    assert rig.evals.jobs[0].call_id == call_id
    assert client.get(f"/api/calls/{call_id}").json()["eval"] is not None


def test_evaluation_starts_the_models_it_needs_or_says_why_not(client, rig, waiter):
    profile = rig.services.profiles.require("local-default")
    profile.services.append(LocalService(name="never", command=["sleep", "30"], port=18997))
    rig.settings.model_wait_s = 0.05
    call_id = upload(client).json()["callId"]
    client.post(f"/api/uploads/{call_id}/evaluate")
    waiter(lambda: client.get(f"/api/calls/{call_id}").json()["summary"]["status"] == "failed")
    assert "not ready" in rig.services.calls.get(call_id)["eval_error"]
    assert "never" in rig.services.supervisor.statuses()
    assert rig.evals.jobs == []


def test_evaluate_twice_conflicts_while_running(client, rig, waiter):
    rig.evals.hold = True
    call_id = upload(client).json()["callId"]
    client.post(f"/api/uploads/{call_id}/evaluate")
    assert client.post(f"/api/uploads/{call_id}/evaluate").status_code == 409

    async def release():
        rig.evals.hold = False
        rig.evals.release.set()

    rig.run_on_loop(release)
    waiter(lambda: client.get(f"/api/calls/{call_id}").json()["summary"]["status"] == "done")


def test_preview_is_a_clip(client):
    call_id = upload(client).json()["callId"]
    response = client.get(f"/api/uploads/{call_id}/preview", params={"startS": 1, "durationS": 2})
    assert response.headers["content-type"] == "audio/wav"
    info = sf.info(io.BytesIO(response.content))
    assert info.duration == pytest.approx(2.0)
    assert info.channels == 2


def test_preview_past_the_end_is_invalid(client):
    call_id = upload(client).json()["callId"]
    response = client.get(f"/api/uploads/{call_id}/preview", params={"startS": 99})
    assert response.status_code == 400


def test_negative_preview_is_invalid(client):
    call_id = upload(client).json()["callId"]
    assert client.get(f"/api/uploads/{call_id}/preview?startS=-1").status_code == 400


def test_uploads_do_not_count_as_verdict_free_of_review(client, rig, waiter):
    rig.evals.verdicts = [Verdict.NEEDS_REVIEW]
    call_id = upload(client).json()["callId"]
    client.post(f"/api/uploads/{call_id}/evaluate")
    waiter(lambda: client.get(f"/api/calls/{call_id}").json()["summary"]["status"] == "done")
    assert client.get("/api/calls/next-review").json() == {"callId": call_id}
