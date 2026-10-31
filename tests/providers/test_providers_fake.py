import json

import numpy as np
import pytest

from voicelab.agents import (
    CallerBot,
    SupportBot,
    draft_scenario,
    load_personas,
    load_scenarios,
    run_text_call,
)
from voicelab.company.db import create_call_db
from voicelab.company.tools import ToolDispatcher
from voicelab.core.audio import AudioClip
from voicelab.core.models import LineConditions, Persona, VoiceRef
from voicelab.core.profiles import build_stack, load_profiles
from voicelab.eval.evaluators.audio_listening import LISTENING_SCHEMA
from voicelab.eval.evaluators.state import satisfying_changes
from voicelab.eval.vad import speech_regions
from voicelab.providers.adapters.fake import FakeAnalyst, FakeLLM, FakeSTT, FakeTTS
from voicelab.providers.adapters.fake_brains import judge_reply
from voicelab.providers.adapters.fake_speech import (
    SYMBOL_S,
    decode_speech,
    encode_speech,
    speech_duration_s,
)
from voicelab.providers.base import LLM, STT, TTS, AudioAnalyst, ChatMessage
from voicelab.telephony import PhoneLine

SENTENCE = "My zip is 97214 and invoice FH-4419 is at whitfieldlandscaping.example today."


def heard(clip: AudioClip) -> str:
    return " ".join(word.text for word in decode_speech(clip))


# Speech ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [8000, 16000, 24000])
def test_text_survives_a_round_trip_at_any_rate(rate):
    assert heard(encode_speech(SENTENCE, rate)) == SENTENCE


def test_length_follows_the_text():
    short = encode_speech("Okay.", 16000)
    long = encode_speech("Okay. " * 20, 16000)

    assert short.duration_s == pytest.approx(speech_duration_s("Okay."), abs=0.01)
    assert long.duration_s == pytest.approx(20 * short.duration_s, rel=0.01)


def test_the_audio_is_shaped_like_speech_for_the_vad():
    clip = encode_speech(SENTENCE, 16000)
    padded = AudioClip.silence(1, 16000) + clip + AudioClip.silence(1, 16000)

    regions = speech_regions(padded)

    assert len(regions) == 1
    assert regions[0].start_s == pytest.approx(1, abs=0.1)
    assert regions[0].duration_s == pytest.approx(clip.duration_s, abs=0.15)
    levels = [np.abs(clip.slice(t, t + 0.02).samples()).max() for t in np.arange(0, 3, 0.02)]
    assert min(levels) < 0.01 < max(levels) < 0.5


@pytest.mark.parametrize(
    "line",
    [
        LineConditions(),
        LineConditions(noise="office", snr_db=20),
    ],
)
def test_text_survives_the_phone_line(line):
    assert heard(PhoneLine(line, seed=3).process(encode_speech(SENTENCE, 16000))) == SENTENCE


def test_mains_hum_costs_a_word_at_most():
    line = LineConditions(noise="hum", snr_db=15)
    words = heard(PhoneLine(line, seed=3).process(encode_speech(SENTENCE, 16000))).split()

    assert len(set(SENTENCE.split()) - set(words)) <= 1


def test_heavy_loss_garbles_the_audio_as_it_would_real_speech():
    bad = LineConditions(noise="street", snr_db=8, packet_loss_pct=10)
    result = heard(PhoneLine(bad, seed=1).process(encode_speech(SENTENCE, 16000)))

    assert result != SENTENCE
    assert len(result.split()) >= 3


def test_a_word_cut_off_partway_is_not_heard():
    clip = encode_speech("alpha bravo charlie", 16000)
    whole_words = (len("alpha") + 1 + len("bravo") + 1) * SYMBOL_S

    assert heard(clip.slice(0, whole_words)) == "alpha bravo"
    assert heard(clip.slice(0, whole_words - 0.1)) == "alpha"
    assert heard(clip.slice(0, whole_words + 0.2)) == "alpha bravo"


def test_words_are_timed_where_they_sit_in_a_channel():
    gap = AudioClip.silence(2.0, 16000)
    channel = gap + encode_speech("first turn", 16000) + gap + encode_speech("second", 16000)

    words = decode_speech(channel)

    assert [w.text for w in words] == ["first", "turn", "second"]
    assert words[0].start_s == pytest.approx(2.03, abs=0.03)
    assert words[2].start_s > words[1].end_s + 1.9
    assert all(0.9 < w.confidence <= 1.0 for w in words)


def test_silence_and_empty_audio_hear_nothing():
    assert decode_speech(AudioClip.silence(1, 16000)) == []
    assert decode_speech(AudioClip(b"", 16000)) == []


def test_encoding_is_deterministic():
    assert encode_speech(SENTENCE, 16000).pcm == encode_speech(SENTENCE, 16000).pcm


# Adapters ----------------------------------------------------------------------------------------


def test_fake_adapters_satisfy_the_interfaces():
    assert isinstance(FakeSTT(), STT)
    assert isinstance(FakeTTS(), TTS)
    assert isinstance(FakeLLM(), LLM)
    assert isinstance(FakeAnalyst(), AudioAnalyst)


async def test_tts_stream_adds_up_to_the_whole_clip():
    tts = FakeTTS()
    whole = await tts.synthesize(SENTENCE, voice_id="default_male_1")

    chunks = [chunk async for chunk in tts.stream(SENTENCE, voice_id="default_male_1")]

    assert len(chunks) > 5
    assert b"".join(chunk.pcm for chunk in chunks) == whole.pcm


async def test_voices_differ_and_all_seeded_voices_exist():
    tts = FakeTTS()
    known = {voice.id for voice in await tts.voices()}
    one = await tts.synthesize("Hello", voice_id="default_male_1")
    two = await tts.synthesize("Hello", voice_id="default_female_3")

    assert one.pcm != two.pcm
    assert {persona.voice.voice_id for persona in load_personas().values()} <= known


async def test_stt_returns_timed_words_with_confidence():
    clip = encode_speech("hello there", 16000)

    transcript = await FakeSTT(name="fake-eval").transcribe(clip)

    assert transcript.text == "hello there"
    assert transcript.model == "fake-eval" and transcript.language == "en"
    assert [w.text for w in transcript.words] == ["hello", "there"]
    assert transcript.words[0].confidence > 0.9


async def test_analyst_answers_in_the_listening_schema():
    reply = json.loads(await FakeAnalyst().analyze(AudioClip.silence(1, 16000), "listen"))

    assert set(LISTENING_SCHEMA["required"]) <= set(reply)


async def test_stream_chat_assembles_the_same_answer_as_chat():
    messages = [
        ChatMessage(role="system", content="You are Fern."),
        ChatMessage(role="user", content="I forgot my password."),
    ]
    tools = ToolDispatcher.tools
    llm = FakeLLM()

    whole = await llm.chat(messages, tools=tools)
    deltas = [delta async for delta in llm.stream_chat(messages, tools=tools)]

    assert "".join(d.text for d in deltas) == whole.text
    assert deltas[-1].final.text == whole.text


async def test_plain_chat_with_no_job_still_answers():
    reply = await FakeLLM(name="anything").chat([ChatMessage(role="user", content="Hello?")])

    assert reply.text and reply.model == "anything"


# Conversations -----------------------------------------------------------------------------------


def _quiet_caller() -> Persona:
    return Persona(name="Tester", patience=5, voice=VoiceRef(voice_id="default_male_1"))


@pytest.mark.parametrize("scenario_id", sorted(load_scenarios()))
async def test_the_fake_bots_reach_every_seeded_scenarios_outcome(scenario_id):
    scenario = load_scenarios()[scenario_id]
    db = create_call_db()
    support = SupportBot(FakeLLM(), ToolDispatcher(db))
    caller = CallerBot(FakeLLM(), _quiet_caller(), scenario, seed=1)

    result = await run_text_call(support, caller)

    assert result.end_reason == "caller_hung_up"
    assert result.completed_beats == [beat.id for beat in scenario.beats]
    for expectation in scenario.expected_state:
        hits = satisfying_changes(expectation, result.db_changes)
        assert not hits if expectation.op == "no_change" else hits, expectation.description


async def test_the_fake_support_bot_will_not_act_before_verifying():
    db = create_call_db()
    support = SupportBot(FakeLLM(), ToolDispatcher(db))
    support.greet()

    spoken = [s.spoken async for s in support.respond("Refund invoice FH-4419 right now.")]
    spoken += [s.spoken async for s in support.respond("My domain is bellhardware.example.")]

    assert "billing zip" in " ".join(spoken)
    assert db.row("SELECT COUNT(*) AS n FROM refunds WHERE invoice_id = 'FH-4419'")["n"] == 0


async def test_the_fake_drafter_produces_a_scenario_that_passes_validation():
    scenario = await draft_scenario(FakeLLM(), "A caller wants a refund")

    assert scenario.beats and scenario.customer_id == "cus_1001"


def test_the_fake_judge_quotes_the_transcript_word_for_word():
    prompt = (
        "Transcript:\n[0.5s] BOT: Thanks for calling.\n[3.0s] CALLER: I want a refund today.\n\n"
        "Steps of the script:\n- ask_refund: Say you want a refund.\n\n"
    )

    [reply] = judge_reply(prompt)["results"]

    assert reply["id"] == "ask_refund"
    assert reply["status"] == "pass" and reply["quotes"] == ["I want a refund today."]


def test_the_fake_judge_has_nothing_to_say_about_an_empty_transcript():
    prompt = "Transcript:\n\nCriteria:\n- nice: Be nice.\n\n"
    assert judge_reply(prompt)["results"][0]["status"] == "unknown"


def test_the_fake_profile_fills_every_role():
    built = build_stack(load_profiles()["fake"])

    assert built.stt_eval.name != built.stt_live.name
    assert built.profile.roles["tts_bot"]["voice_id"] in ("default_female_1",)
