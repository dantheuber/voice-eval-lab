from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import soundfile as sf
from fakes import fake_stack

from voicelab.core.models import Persona, Verdict, VoiceRef
from voicelab.providers.base import VoiceInfo
from voicelab.services import authoring

PERSONA = {
    "name": "Ann",
    "summary": "Busy",
    "age": 40,
    "temperament": "calm",
    "speakingStyle": "dry",
    "patience": 2,
    "verbosity": 2,
    "techSavviness": 4,
    "interruptionRate": 0.25,
    "voice": {"provider": None, "voiceId": "default_female_1", "style": None},
    "line": {
        "codec": "mulaw_8k",
        "noise": "cafe",
        "snrDb": 15,
        "packetLossPct": 1,
        "oneWayLatencyMs": 90,
    },
}

SCENARIO = {
    "title": "Refund",
    "description": "d",
    "goal": "Get a refund",
    "customerId": "cus_1001",
    "facts": {"phone": "(503) 555-0101", "billing_zip": "97214"},
    "beats": [{"id": "b1", "instruction": "Ask", "checkpoint": "Asked", "required": True}],
    "must": [{"id": "m1", "text": "Verify", "severity": "critical"}],
    "mustNot": [],
    "expectedState": [],
    "maxTurns": 12,
    "pronunciationWatch": ["FH-4419"],
    "tags": ["billing"],
}


def test_seeds_are_listed_camel_cased_with_usage(client):
    items = client.get("/api/personas").json()["items"]
    sysadmin = next(p for p in items if p["id"] == "per_sysadmin")
    assert sysadmin["speakingStyle"]
    assert sysadmin["line"]["snrDb"] == 22
    assert sysadmin["usage"] == {
        "calls": 0,
        "counts": {"pass": 0, "fail": 0, "needsReview": 0, "invalid": 0},
    }


def test_interruption_rate_stays_a_fraction_on_the_wire(client):
    """The frontend's own helpers turn this into the 1 to 5 scale, so sending it would double up."""
    created = client.post("/api/personas", json={**PERSONA, "interruptionRate": 0.75}).json()
    assert created["interruptionRate"] == 0.75
    sysadmin = client.get("/api/personas/per_sysadmin").json()
    assert sysadmin["interruptionRate"] == 0.15


def test_persona_create_read_update_delete(client):
    response = client.post("/api/personas", json=PERSONA)
    assert response.status_code == 201
    created = response.json()
    assert created["id"].startswith("per_")
    assert created["createdAt"].endswith("Z")
    assert created["line"]["oneWayLatencyMs"] == 90

    detail = client.get(f"/api/personas/{created['id']}").json()
    assert detail["recentCalls"] == []
    assert detail["usage"]["calls"] == 0

    updated = client.put(
        f"/api/personas/{created['id']}", json={**detail, "name": "Anne", "id": "per_other"}
    ).json()
    assert updated["id"] == created["id"]
    assert updated["name"] == "Anne"
    assert updated["createdAt"] == created["createdAt"]
    assert updated["updatedAt"] >= created["updatedAt"]

    assert client.delete(f"/api/personas/{created['id']}").status_code == 204
    assert client.get(f"/api/personas/{created['id']}").status_code == 404


def test_persona_body_is_validated(client):
    response = client.post("/api/personas", json={**PERSONA, "patience": 9})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid"
    assert response.json()["error"]["detail"]["problems"][0]["field"] == "patience"


def test_persona_in_an_unfinished_run_cannot_be_deleted(client, rig, start_run):
    rig.runner.hold = True
    start_run()
    response = client.delete("/api/personas/per_sysadmin")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_missing_persona(client):
    assert client.get("/api/personas/per_missing").status_code == 404
    assert client.put("/api/personas/per_missing", json=PERSONA).status_code == 404
    assert client.delete("/api/personas/per_missing").status_code == 404
    assert client.get("/api/personas/per_missing/prompt").status_code == 404


def test_persona_prompt_reflects_the_persona(client):
    prompt = client.get("/api/personas/per_sysadmin/prompt").json()["prompt"]
    assert "You are 38 years old." in prompt
    assert "Dry and precise" in prompt
    assert "(from the scenario)" in prompt


def test_usage_counts_invalid_apart(client, rig, finished_call):
    rig.evals.verdicts = [Verdict.PASS, Verdict.INVALID]
    call = finished_call()
    finished_call()
    persona = client.get("/api/personas/per_sysadmin").json()
    assert persona["usage"] == {
        "calls": 2,
        "counts": {"pass": 1, "fail": 0, "needsReview": 0, "invalid": 1},
    }
    assert len(persona["recentCalls"]) == 2
    assert call["id"] in {c["id"] for c in persona["recentCalls"]}
    scenario = client.get("/api/scenarios/scn_dns_mx_fix").json()
    assert scenario["usage"]["counts"]["invalid"] == 1


def test_voices_come_from_the_tts_adapters(client):
    assert client.get("/api/voices").json()["items"] == [
        {
            "provider": "fake-tts",
            "voiceId": "default_male_1",
            "label": "Default male 1",
            "language": "en-US",
        }
    ]


def _preview(client, line):
    voice = {"provider": None, "voiceId": "default_male_1", "style": None}
    return client.post("/api/tts/preview", json={"text": "Hello", "voice": voice, "line": line})


def test_preview_goes_through_the_line(client):
    response = _preview(client, PERSONA["line"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    _, rate = sf.read(io.BytesIO(response.content))
    assert rate == 8000  # mu-law line


def test_preview_without_a_line_keeps_the_tts_rate(client):
    _, rate = sf.read(io.BytesIO(_preview(client, None).content))
    assert rate == 16000


def test_preview_rejects_empty_text(client):
    voice = {"voiceId": "v"}
    response = client.post("/api/tts/preview", json={"text": "", "voice": voice, "line": None})
    assert response.status_code == 400


def test_scenario_versions(client, rig):
    created = client.post("/api/scenarios", json=SCENARIO)
    assert created.status_code == 201
    scenario = created.json()
    assert scenario["mustNot"] == []
    assert client.get(f"/api/scenarios/{scenario['id']}").json()["version"] == 1

    client.put(f"/api/scenarios/{scenario['id']}", json={**scenario, "title": "Second"})
    detail = client.get(f"/api/scenarios/{scenario['id']}").json()
    assert detail["version"] == 2
    assert detail["title"] == "Second"
    versions = rig.services.catalog.scenario_versions(scenario["id"])
    assert [(v, s.title) for v, s in versions] == [(1, "Refund"), (2, "Second")]
    listed = client.get("/api/scenarios").json()["items"]
    assert next(s for s in listed if s["id"] == scenario["id"])["version"] == 2


def test_editing_a_scenario_leaves_existing_calls_their_snapshot(client, finished_call):
    call = finished_call()
    scenario = client.get("/api/scenarios/scn_dns_mx_fix").json()
    client.put("/api/scenarios/scn_dns_mx_fix", json={**scenario, "title": "Rewritten"})
    body = client.get(f"/api/calls/{call['id']}").json()
    assert body["call"]["scenarioSnapshot"]["title"] == scenario["title"]


def test_scenario_with_unknown_references_is_rejected(client):
    bad = {**SCENARIO, "customerId": "cus_missing"}
    response = client.post("/api/scenarios", json=bad)
    assert response.status_code == 400
    assert "cus_missing" in response.json()["error"]["message"]


def test_scenario_delete_conflicts_while_a_run_uses_it(client, rig, start_run):
    rig.runner.hold = True
    start_run()
    assert client.delete("/api/scenarios/scn_dns_mx_fix").status_code == 409


def test_customers(client):
    items = client.get("/api/customers").json()["items"]
    first = next(c for c in items if c["id"] == "cus_1001")
    assert first["name"]
    assert first["note"]
    assert isinstance(first["plans"], list)
    assert first["domain"]


def test_customer_detail_keeps_column_names(client):
    body = client.get("/api/customers/cus_1001").json()
    assert set(body) == {
        "customer",
        "subscriptions",
        "invoices",
        "payments",
        "refunds",
        "domains",
        "mailboxes",
        "websites",
        "dnsRecords",
        "tickets",
    }
    assert "billing_zip" in body["customer"]
    assert body["invoices"][0]["amount_cents"] > 0


def test_missing_customer(client):
    assert client.get("/api/customers/cus_missing").status_code == 404


DRAFTED = {
    "title": "Refund for a double charge",
    "goal": "I was charged twice.",
    "customer_id": "cus_1001",
    "facts": {"name": "Dana Whitfield", "phone": "(503) 555-0101", "amount": "$12,345.67"},
    "beats": [{"id": "b", "instruction": "Ask for a refund", "checkpoint": "Asked"}],
    "must": [{"id": "m", "text": "Verify the caller"}],
    "must_not": [],
    "expected_state": [],
}


def _judge(rig, reply):
    stack = rig.services.profiles.stack(rig.services.profiles.require("local-default"))
    stack.llm_judge.reply = reply
    return stack.llm_judge


def test_draft_returns_a_scenario_without_ids_and_names_guesses(client, rig):
    _judge(rig, DRAFTED)
    response = client.post(
        "/api/scenarios/draft",
        json={"description": "double charge", "customerId": "cus_1001", "baseScenarioId": None},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scenario"]["title"] == DRAFTED["title"]
    assert "id" not in body["scenario"]
    assert "createdAt" not in body["scenario"]
    assert body["scenario"]["mustNot"] == []
    assert body["guessedFields"] == ["facts.amount"]
    assert body["model"] == "judge-llm"


def test_draft_names_a_customer_the_model_picked(client, rig):
    _judge(rig, DRAFTED)
    body = client.post(
        "/api/scenarios/draft",
        json={"description": "double charge", "customerId": None, "baseScenarioId": None},
    ).json()
    assert "customerId" in body["guessedFields"]


def test_draft_gives_the_model_its_base_scenario(client, rig):
    judge = _judge(rig, DRAFTED)
    body = client.post(
        "/api/scenarios/draft",
        json={
            "description": "like this",
            "customerId": None,
            "baseScenarioId": "scn_dns_mx_fix",
        },
    ).json()
    assert "scn_dns_mx_fix" in body["notes"]
    assert "mx.oldmailhost.example" in judge.prompts[0]


def test_draft_that_never_validates_is_an_error(client, rig):
    _judge(rig, "not json")
    response = client.post(
        "/api/scenarios/draft",
        json={"description": "x", "customerId": None, "baseScenarioId": None},
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "internal"


def test_dry_run_is_text_only_with_reached_checkpoints(client, rig):
    stack = rig.services.profiles.stack(rig.services.profiles.require("local-default"))
    beats = client.get("/api/scenarios/scn_dns_mx_fix").json()["beats"]
    stack.llm_caller.reply = [
        {
            "say": "My mail stopped.",
            "completed_beat": beats[0]["id"],
            "hang_up": False,
            "hang_up_reason": "",
        },
        *[{"say": "Hm.", "completed_beat": None, "hang_up": False, "hang_up_reason": ""}] * 8,
        {"say": "Forget it.", "completed_beat": None, "hang_up": True, "hang_up_reason": "fed up"},
    ]
    response = client.post(
        "/api/scenarios/scn_dns_mx_fix/dry-run", json={"personaId": "per_sysadmin"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["turns"][0]["speaker"] == "bot"
    assert {"speaker": "caller", "text": "My mail stopped."} in body["turns"]
    assert body["reachedCheckpoints"] == [beats[0]["id"]]
    assert body["model"] == "support-llm"


def test_dry_run_without_a_persona_uses_a_neutral_caller(client, rig):
    stack = rig.services.profiles.stack(rig.services.profiles.require("local-default"))
    stack.llm_caller.reply = {
        "say": "Bye.",
        "completed_beat": None,
        "hang_up": True,
        "hang_up_reason": "done",
    }
    response = client.post("/api/scenarios/scn_dns_mx_fix/dry-run", json={"personaId": None})
    assert response.status_code == 200, response.text


def test_dry_run_of_a_missing_scenario(client):
    assert client.post("/api/scenarios/scn_nope/dry-run", json={}).status_code == 404


@pytest.mark.parametrize(
    ("role_blocks", "route", "body"),
    [
        ("tts_caller", "/api/tts/preview", {"text": "x", "voice": {"voiceId": "v"}, "line": None}),
        ("llm_judge", "/api/scenarios/draft", {"description": "x"}),
        ("llm_support", "/api/scenarios/scn_dns_mx_fix/dry-run", {}),
    ],
)
def test_loading_models_answer_model_not_ready(client, rig, role_blocks, route, body):
    from voicelab.core.profiles import LocalService

    profile = rig.services.profiles.require("local-default")
    profile.roles[role_blocks]["url"] = "http://127.0.0.1:18999/v1"
    profile.services.append(
        LocalService(name="never", command=["sleep", "30"], port=18999, startup_timeout_s=30)
    )
    response = client.post(route, json=body)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "model_not_ready"
    assert error["detail"]["role"] == role_blocks
    assert set(error["detail"]) == {"role", "model", "elapsedS", "usualS"}


def test_stack_factory_is_the_injected_one(rig):
    stack = rig.services.profiles.stack(rig.services.profiles.require("local-default"))
    assert stack.llm_support.name == fake_stack(stack.profile).llm_support.name


def test_a_voice_the_caller_tts_refuses_is_a_clear_400(client, rig, monkeypatch):
    async def refuse(*_args, **_kwargs):
        raise ValueError("'Easton' is not a license-safe Pocket TTS voice")

    stack = rig.services.profiles.stack(rig.services.profiles.resolve(None))
    monkeypatch.setattr(type(stack.tts_caller), "synthesize", refuse)
    voice = {"provider": None, "voiceId": "Easton", "style": None}
    response = client.post("/api/tts/preview", json={"text": "Hi", "voice": voice, "line": None})
    assert response.status_code == 400
    assert "license-safe" in response.json()["error"]["message"]


def test_prompt_preview_is_the_real_prompt_for_an_unsaved_form(client):
    created = client.post("/api/personas", json=PERSONA).json()
    saved = client.get(f"/api/personas/{created['id']}/prompt").json()["prompt"]
    assert client.post("/api/personas/prompt", json=PERSONA).json()["prompt"] == saved
    assert "fact sheet" in saved


async def test_a_persona_with_another_engines_voice_is_named_before_any_call():
    class PhononLike:
        voice_map = {"default_female_1": "Juniper"}

        async def voices(self):
            return [VoiceInfo(id="Easton", label="Easton")]

    def persona(name, voice_id):
        return Persona(name=name, voice=VoiceRef(voice_id=voice_id))

    personas = [persona("Ann", "default_female_1"), persona("Bo", "Easton"), persona("Cy", "vera")]
    stack = SimpleNamespace(tts_caller=PhononLike())
    assert await authoring.voiceless_personas(stack, personas) == ["Cy (vera)"]
