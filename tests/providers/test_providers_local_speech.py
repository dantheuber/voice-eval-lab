import json
import struct
from urllib.parse import parse_qs

import httpx
import numpy as np
import pytest
import respx

from voicelab.agents import load_personas
from voicelab.core.audio import AudioClip
from voicelab.core.profiles import build_stack, default_profiles_dir, load_profiles
from voicelab.providers.adapters.local_speech import (
    PHONON_PERSONA_VOICES,
    PHONON_VOICES,
    POCKET_PERSONA_VOICES,
    POCKET_VOICES,
    HttpTranscriptionSTT,
    LlamaAudioAnalyst,
    LlamaTTS,
    PhononTTS,
    PocketTTS,
    SpeechServerError,
    WhisperCppSTT,
)
from voicelab.providers.base import STT, TTS, AudioAnalyst

BASE = "http://speech.test"
CLIP = AudioClip.silence(0.2, 16000)


def form_fields(request: httpx.Request) -> bytes:
    return request.content


# Parakeet and other /v1/audio/transcriptions servers ---------------------------------------------


@respx.mock
async def test_transcription_reads_timed_words():
    route = respx.post(f"{BASE}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={
                "text": " Your invoice",
                "words": [
                    {"word": "Your", "start": 0.0, "end": 0.24, "confidence": 0.9},
                    {"word": "invoice", "start": 0.24, "end": 0.56, "confidence": 0.8},
                ],
            },
        )
    )
    stt = HttpTranscriptionSTT(BASE, "parakeet", name="parakeet-live")

    transcript = await stt.transcribe(CLIP)

    assert isinstance(stt, STT)
    assert transcript.text == "Your invoice" and transcript.model == "parakeet-live"
    assert [(w.text, w.end_s, w.confidence) for w in transcript.words] == [
        ("Your", 0.24, 0.9),
        ("invoice", 0.56, 0.8),
    ]
    body = form_fields(route.calls[0].request)
    assert b'name="timestamp_granularities[]"' in body and b"verbose_json" in body
    assert b"parakeet" in body and b"RIFF" in body


@respx.mock
async def test_meaningless_confidence_is_reported_as_unknown():
    respx.post(f"{BASE}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={
                "text": "Hi",
                "words": [{"word": "Hi", "start": 0, "end": 1, "confidence": 1}],
            },
        )
    )

    transcript = await HttpTranscriptionSTT(BASE, "m", trust_confidence=False).transcribe(CLIP)

    assert transcript.words[0].confidence is None


@respx.mock
async def test_the_model_id_is_looked_up_once_when_not_configured():
    listing = respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "parakeet-oss"}]})
    )
    route = respx.post(f"{BASE}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "Hi"})
    )
    stt = HttpTranscriptionSTT(BASE)

    await stt.transcribe(CLIP)
    await stt.transcribe(CLIP)

    assert listing.call_count == 1
    assert b"parakeet-oss" in form_fields(route.calls[0].request)


@respx.mock
async def test_the_language_tag_qwen_asr_writes_first_is_removed():
    respx.post(f"{BASE}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "language English<asr_text>Hello there."})
    )

    transcript = await HttpTranscriptionSTT(BASE, "m").transcribe(CLIP)

    assert transcript.text == "Hello there."


async def test_transcribing_nothing_makes_no_request():
    transcript = await HttpTranscriptionSTT(BASE, "m").transcribe(AudioClip(b"", 16000))

    assert transcript.text == ""


@respx.mock
async def test_phone_audio_is_sent_at_16_khz():
    route = respx.post(f"{BASE}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": ""})
    )

    await HttpTranscriptionSTT(BASE, "m").transcribe(AudioClip.silence(1.0, 8000))

    body = form_fields(route.calls[0].request)
    header = body[body.index(b"RIFF") :]
    assert struct.unpack("<I", header[24:28])[0] == 16000


@respx.mock
async def test_a_server_error_is_reported():
    respx.post(f"{BASE}/v1/audio/transcriptions").mock(return_value=httpx.Response(500, text="no"))

    with pytest.raises(SpeechServerError, match="500"):
        await HttpTranscriptionSTT(BASE, "m").transcribe(CLIP)


@respx.mock
async def test_an_unreachable_server_is_reported():
    respx.post(f"{BASE}/v1/audio/transcriptions").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(SpeechServerError, match="unreachable"):
        await HttpTranscriptionSTT(BASE, "m").transcribe(CLIP)


# Whisper -----------------------------------------------------------------------------------------


def whisper_reply() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "text": " Your invoice",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.1,
                    "words": [
                        {"word": " Your", "start": 0.03, "end": 0.41, "probability": 0.93},
                        {"word": " invoice", "start": 0.41, "end": 1.13, "probability": 0.98},
                        {"word": "[_BEG_]", "start": 1.13, "end": 1.13, "probability": 0.1},
                    ],
                }
            ],
        },
    )


@respx.mock
async def test_whisper_returns_real_word_probabilities():
    route = respx.post(f"{BASE}/inference").mock(return_value=whisper_reply())
    stt = WhisperCppSTT(BASE, name="whisper-large-v3")

    transcript = await stt.transcribe(CLIP)

    assert isinstance(stt, STT)
    assert [(w.text, w.confidence) for w in transcript.words] == [
        ("Your", 0.93),
        ("invoice", 0.98),
    ]
    assert transcript.text == "Your invoice"
    assert b"verbose_json" in form_fields(route.calls[0].request)


def _token(word: str, start: float, end: float, probability: float = 0.9) -> dict:
    return {"word": word, "start": start, "end": end, "probability": probability}


@respx.mock
async def test_whisper_tokens_are_joined_into_the_words_they_spell():
    """Whisper's "words" are tokens. Only a leading space starts a new word."""
    tokens = [
        _token(" I", 0.0, 0.1, 0.87),
        _token("'m", 0.1, 0.3, 0.96),
        _token(" Mar", 0.4, 0.6, 0.8),
        _token("ino", 0.6, 0.9, 0.7),
        _token(",", 0.9, 1.0, 0.2),
        _token(" zip", 1.1, 1.3),
        _token(" 60", 1.3, 1.6),
        _token("6", 1.6, 1.9),
        _token("14", 1.9, 2.2),
        _token(".", 2.2, 2.3),
    ]
    segment = {"start": 0.0, "end": 2.3, "words": tokens}
    respx.post(f"{BASE}/inference").mock(
        return_value=httpx.Response(200, json={"text": "x", "segments": [segment]})
    )

    words = (await WhisperCppSTT(BASE).transcribe(CLIP)).words

    assert [w.text for w in words] == ["I'm", "Marino,", "zip", "60614."]
    assert (words[0].start_s, words[0].end_s) == (0.0, 0.3)
    assert (words[3].start_s, words[3].end_s) == (1.3, 2.3)
    # The weakest letter is the word's confidence, and a comma says nothing about the word.
    assert words[1].confidence == pytest.approx(0.7)


@respx.mock
async def test_whisper_word_split_across_segments_stays_one_word():
    first = {"start": 0.0, "end": 1.0, "words": [_token(" Fern", 0.0, 1.0)]}
    second = {"start": 1.0, "end": 2.0, "words": [_token("hill", 1.0, 1.5), _token(" mail", 1.5, 2)]}
    respx.post(f"{BASE}/inference").mock(
        return_value=httpx.Response(200, json={"text": "x", "segments": [first, second]})
    )

    words = (await WhisperCppSTT(BASE).transcribe(CLIP)).words

    assert [w.text for w in words] == ["Fernhill", "mail"]


@respx.mock
async def test_whisper_never_gets_a_prompt_even_when_handed_hints():
    route = respx.post(f"{BASE}/inference").mock(return_value=whisper_reply())

    await WhisperCppSTT(BASE).transcribe(CLIP, hints=["Fernhill"])

    assert b"Fernhill" not in form_fields(route.calls[0].request)


@respx.mock
async def test_whisper_failures_are_reported():
    respx.post(f"{BASE}/inference").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(SpeechServerError):
        await WhisperCppSTT(BASE).transcribe(CLIP)


# LFM3-Audio /tts ---------------------------------------------------------------------------------


class Split(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


@respx.mock
async def test_llama_tts_streams_float_pcm_across_chunk_boundaries():
    data = struct.pack("<3f", 0.5, -0.5, 0.25)
    route = respx.post(f"{BASE}/tts").mock(
        return_value=httpx.Response(200, stream=Split(data[:6], data[6:]))
    )
    tts = LlamaTTS(BASE, voice_map={"bot": "female"})

    clips = [clip async for clip in tts.stream("Hello", voice_id="bot")]

    assert isinstance(tts, TTS)
    samples = np.concatenate([clip.samples() for clip in clips])
    assert samples == pytest.approx([0.5, -0.5, 0.25], abs=0.001)
    assert all(clip.sample_rate == 24000 for clip in clips)
    assert json.loads(route.calls[0].request.content) == {
        "input": "Hello",
        "voice": "female",
        "lang": "en",
        "stream": True,
        "response_format": "pcm",
    }


@respx.mock
async def test_llama_tts_synthesize_joins_the_stream():
    respx.post(f"{BASE}/tts").mock(
        return_value=httpx.Response(200, content=struct.pack("<2f", 0.1, 0.2))
    )

    clip = await LlamaTTS(BASE).synthesize("Hello", voice_id="female")

    assert clip.n_samples == 2


@respx.mock
async def test_llama_tts_errors_are_reported():
    respx.post(f"{BASE}/tts").mock(return_value=httpx.Response(500, text="model busted"))

    with pytest.raises(SpeechServerError, match="model busted"):
        await LlamaTTS(BASE).synthesize("Hello", voice_id="female")


async def test_llama_tts_offers_both_voices():
    assert {v.id for v in await LlamaTTS(BASE).voices()} == {"female", "male"}


# Phonon ------------------------------------------------------------------------------------------


@respx.mock
async def test_phonon_streams_pcm_and_maps_persona_voices():
    data = b"\x01\x00\x02\x00\x03\x00"
    route = respx.post(f"{BASE}/tts").mock(
        return_value=httpx.Response(200, stream=Split(data[:3], data[3:]))
    )
    tts = PhononTTS(BASE)

    clip = await tts.synthesize("Hello", voice_id="default_male_3")

    assert isinstance(tts, TTS)
    assert clip.pcm == data and clip.sample_rate == 24000
    assert json.loads(route.calls[0].request.content) == {"text": "Hello", "voice": "Miles"}


@respx.mock
async def test_phonon_errors_are_reported():
    respx.post(f"{BASE}/tts").mock(return_value=httpx.Response(500, text="no such voice"))

    with pytest.raises(SpeechServerError, match="no such voice"):
        await PhononTTS(BASE).synthesize("Hello", voice_id="Easton")


def test_every_seeded_persona_has_its_own_phonon_voice_of_the_right_gender():
    for voice in {p.voice.voice_id for p in load_personas().values()}:
        assert PHONON_VOICES[PHONON_PERSONA_VOICES[voice]][0] == voice.split("_")[1]
    assert len(set(PHONON_PERSONA_VOICES.values())) == len(PHONON_PERSONA_VOICES)


# Pocket TTS --------------------------------------------------------------------------------------


def wav_header(rate: int = 24000) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (
        b"RIFF\xff\xff\xff\xffWAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data\xff\xff\xff\xff"
    )


@respx.mock
async def test_pocket_strips_the_wav_header_however_it_is_split():
    wav = wav_header() + b"\x01\x00\x02\x00\x03\x00"
    route = respx.post(f"{BASE}/tts").mock(
        return_value=httpx.Response(200, stream=Split(wav[:10], wav[10:45], wav[45:]))
    )
    tts = PocketTTS(BASE)

    clip = await tts.synthesize("Hello", voice_id="default_female_1")

    assert isinstance(tts, TTS)
    assert clip.pcm == b"\x01\x00\x02\x00\x03\x00" and clip.sample_rate == 24000
    fields = parse_qs(route.calls[0].request.content.decode())
    assert fields == {"text": ["Hello"], "voice_url": ["mary"]}


@pytest.mark.parametrize("voice", ["cosette", "jean", "alba", "bill_boerst"])
async def test_pocket_refuses_voices_outside_the_license_safe_set(voice):
    with pytest.raises(ValueError, match="license-safe"):
        await PocketTTS(BASE).synthesize("Hello", voice_id=voice)


def test_only_cc0_and_cc_by_presets_are_listed():
    assert not {"cosette", "jean"} & set(POCKET_VOICES)
    assert len(POCKET_VOICES) == 14


def test_every_seeded_persona_gets_its_own_voice_of_the_right_gender():
    voices = [p.voice.voice_id for p in load_personas().values()]
    presets = [POCKET_PERSONA_VOICES[voice] for voice in voices]

    assert len(set(presets)) == len(presets)
    for voice, preset in zip(voices, presets, strict=True):
        assert POCKET_VOICES[preset][0] == voice.split("_")[1]


@respx.mock
async def test_pocket_errors_are_reported():
    respx.post(f"{BASE}/tts").mock(return_value=httpx.Response(400, text="bad voice"))

    with pytest.raises(SpeechServerError, match="bad voice"):
        await PocketTTS(BASE).synthesize("Hello", voice_id="anna")


async def test_pocket_lists_its_voices_with_their_details():
    voices = {v.id: v for v in await PocketTTS(BASE).voices()}

    assert voices["eponine"].accent == "Scottish" and voices["george"].gender == "male"


# Audio listener ----------------------------------------------------------------------------------


@respx.mock
async def test_the_analyst_sends_the_clip_as_input_audio():
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    analyst = LlamaAudioAnalyst(f"{BASE}/v1")

    reply = await analyst.analyze(CLIP, "How does it sound?", json_schema={"type": "object"})

    assert isinstance(analyst, AudioAnalyst) and reply == "{}"
    body = json.loads(route.calls[0].request.content)
    audio, text = body["messages"][1]["content"]
    assert audio["input_audio"]["format"] == "wav" and text["text"] == "How does it sound?"
    assert body["response_format"]["type"] == "json_schema"


# Profiles ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["local-default", "local-fast", "local-realtime"])
def test_local_profiles_build_and_are_live_capable(name):
    profile = load_profiles()[name]

    stack = build_stack(profile)

    assert profile.live_capable is True
    assert isinstance(stack.stt_eval, WhisperCppSTT)
    assert isinstance(stack.stt_live, HttpTranscriptionSTT)
    assert isinstance(stack.tts_bot, PhononTTS)
    assert stack.audio_analyst is None  # no listener beat chance, so none is configured
    assert profile.roles["tts_bot"]["voice_id"] in PHONON_VOICES
    ports = [service.port for service in profile.services]
    assert len(ports) == len(set(ports)) and all(port >= 18000 for port in ports)


def test_the_main_profile_uses_the_more_accurate_llm_and_fast_uses_the_quicker_one():
    profiles = load_profiles()

    assert "qwen" in profiles["local-default"].roles["llm_support"]["name"]
    assert "lfm3" in profiles["local-fast"].roles["llm_support"]["name"]


def test_the_main_profile_has_no_lfm_models_so_no_revenue_limit():
    text = (default_profiles_dir() / "local-default.yaml").read_text()

    assert "LFM3" not in text.split("id: local-default")[1]
    for name in ("local-fast", "local-realtime"):
        assert "USD 10" in (default_profiles_dir() / f"{name}.yaml").read_text()


def test_only_the_realtime_profile_plays_the_bot_with_realtime():
    profiles = load_profiles()

    assert profiles["local-realtime"].support_engine == "realtime"
    assert profiles["local-default"].support_engine == "cascade"
    assert build_stack(profiles["local-default"]).realtime is None
    assert "llm_caller" in profiles["local-realtime"].roles


def test_the_realtime_server_has_one_slot_so_a_second_call_cannot_queue_on_it():
    service = next(
        s for s in load_profiles()["local-realtime"].services if s.name == "lfm3-realtime"
    )

    assert service.command[service.command.index("-np") + 1] == "1"
