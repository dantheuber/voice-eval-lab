"""Bedrock adapters against stubbed clients. Request shapes are checked by botocore's own models
and the Smithy SDK classes, and replies follow the AWS docs. None of this has touched live AWS."""

import base64
import io
import json
import subprocess
import sys
from typing import Any

import boto3
import pytest
import yaml
from aws_sdk_bedrock_runtime.models import (
    BidirectionalOutputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOutputChunk,
    InvokeModelWithBidirectionalStreamOutputModelTimeoutException,
    ModelTimeoutException,
)
from aws_sdk_transcribe_streaming.models import (
    Alternative,
    AudioStreamAudioEvent,
    BadRequestException,
    Item,
    Result,
    TranscriptEvent,
    TranscriptResultStreamBadRequestException,
    TranscriptResultStreamTranscriptEvent,
)
from aws_sdk_transcribe_streaming.models import (
    Transcript as AwsTranscript,
)
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber
from botocore.validate import ParamValidator

from voicelab.core.audio import AudioClip
from voicelab.core.profiles import build_stack, default_profiles_dir, load_profiles
from voicelab.providers.adapters import bedrock
from voicelab.providers.adapters.bedrock import (
    BedrockConverseLLM,
    NovaSonicVoice,
    PollyTTS,
    TranscribeSTT,
)
from voicelab.providers.base import (
    LLM,
    STT,
    TTS,
    ChatMessage,
    RealtimeVoice,
    ToolCall,
    ToolDef,
)

MODEL = "global.anthropic.claude-haiku-5-5"
USER = [ChatMessage(role="user", content="Hello")]
LOOKUP = ToolDef(
    name="find_customer",
    description="Find a customer",
    parameters={"type": "object", "properties": {"phone": {"type": "string"}}},
)


def aws_client(service: str) -> Any:
    return boto3.client(
        service, region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y"
    )


def converse_reply(*content: dict, stop: str = "end_turn") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": list(content)}},
        "stopReason": stop,
        "usage": {"inputTokens": 11, "outputTokens": 5, "totalTokens": 16},
        "metrics": {"latencyMs": 100},
    }


async def converse_with(reply: dict, expected: dict, *messages, **options):
    client = aws_client("bedrock-runtime")
    llm = BedrockConverseLLM(MODEL, client=client)
    with Stubber(client) as stub:
        stub.add_response("converse", reply, expected)
        response = await llm.chat(list(messages) or USER, **options)
    return response


class ConverseStreamClient:
    """`converse_stream` can't be stubbed, so this checks the request against the real model."""

    def __init__(self, *batches: list[dict]) -> None:
        self.batches = list(batches)
        self.requests: list[dict] = []
        self.closed = 0
        self._shape = aws_client("bedrock-runtime").meta.service_model.operation_model(
            "ConverseStream"
        )

    def converse_stream(self, **request: Any) -> dict:
        report = ParamValidator().validate(request, self._shape.input_shape)
        assert not report.has_errors(), report.generate_report()
        self.requests.append(request)
        client = self

        class Events(list):
            def close(self) -> None:
                client.closed += 1

        return {"stream": Events(self.batches.pop(0))}


def text_events(*pieces: str) -> list[dict]:
    return [
        {"messageStart": {"role": "assistant"}},
        *({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": p}}} for p in pieces),
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}}},
    ]


def tool_events(tool_id: str, name: str, *pieces: str, index: int = 0) -> list[dict]:
    return [
        {
            "contentBlockStart": {
                "contentBlockIndex": index,
                "start": {"toolUse": {"toolUseId": tool_id, "name": name}},
            }
        },
        *(
            {"contentBlockDelta": {"contentBlockIndex": index, "delta": {"toolUse": {"input": p}}}}
            for p in pieces
        ),
        {"contentBlockStop": {"contentBlockIndex": index}},
    ]


async def collect(stream):
    deltas = [delta async for delta in stream]
    return "".join(d.text for d in deltas), deltas[-1].final


# Marked as what they are -------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", [BedrockConverseLLM, TranscribeSTT, PollyTTS, NovaSonicVoice])
def test_every_adapter_is_labelled_untested(adapter):
    assert adapter.untested is True


def test_the_module_says_it_has_never_run_against_aws():
    assert "NONE OF THIS HAS EVER RUN AGAINST LIVE AWS" in bedrock.__doc__


def test_adapters_satisfy_the_interfaces():
    assert isinstance(BedrockConverseLLM(MODEL), LLM)
    assert isinstance(TranscribeSTT(), STT)
    assert isinstance(PollyTTS(), TTS)
    assert isinstance(NovaSonicVoice(), RealtimeVoice)


def test_the_app_imports_and_runs_with_no_aws_packages():
    script = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"boto3", "botocore", "aiobotocore", "awscrt", "smithy_http",
                                  "aws_sdk_bedrock_runtime", "aws_sdk_transcribe_streaming"}:
            raise ImportError(name)
sys.meta_path.insert(0, Block())
from voicelab.core.profiles import build_stack, load_profiles
stack = build_stack(load_profiles()["fake"])
example = build_stack(load_profiles()["bedrock-example"])
try:
    example.tts_bot._client
except RuntimeError as failure:
    assert "uv sync --extra aws" in str(failure)
else:
    raise SystemExit("expected a clear error")
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_example_profile_builds_and_uses_the_requested_models():
    raw = yaml.safe_load((default_profiles_dir() / "bedrock.example.yaml").read_text())
    profile = load_profiles()["bedrock-example"]

    stack = build_stack(profile)

    assert raw["untested"] is True and profile.untested is True
    assert stack.llm_support.model == "global.anthropic.claude-haiku-5-5"
    assert stack.llm_caller.model == "global.anthropic.claude-haiku-5-5"
    assert stack.llm_judge.model == "global.anthropic.claude-opus-5-5"
    assert isinstance(stack.tts_bot, PollyTTS) and isinstance(stack.stt_live, TranscribeSTT)
    assert isinstance(stack.realtime, NovaSonicVoice)
    assert stack.realtime.model == "amazon.nova-2-5-sonic"
    assert stack.stt_eval.vocabulary_name is None


def test_the_example_gives_each_persona_voice_its_own_polly_voice():
    voices = load_profiles()["bedrock-example"].roles["tts_caller"]["voice_map"]

    assert len(set(voices.values())) == len(voices)


# Converse requests -------------------------------------------------------------------------------


async def test_converse_sends_system_apart_from_messages():
    system = ChatMessage(role="system", content="You are Fern.")

    await converse_with(
        converse_reply({"text": "Hi"}),
        {
            "modelId": MODEL,
            "system": [{"text": "You are Fern."}],
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
            "inferenceConfig": {"maxTokens": 64, "temperature": 0.1},
        },
        system,
        *USER,
        temperature=0.1,
        max_tokens=64,
    )


async def test_tools_are_sent_as_tool_specs():
    await converse_with(
        converse_reply({"text": "Hi"}),
        {
            "modelId": MODEL,
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.3},
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "find_customer",
                            "description": "Find a customer",
                            "inputSchema": {"json": LOOKUP.parameters},
                        }
                    }
                ]
            },
        },
        tools=[LOOKUP],
    )


async def test_temperature_can_be_left_out_and_the_token_limit_has_a_floor():
    client = aws_client("bedrock-runtime")
    llm = BedrockConverseLLM(MODEL, client=client, send_temperature=False, min_max_tokens=2000)
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            converse_reply({"text": "Hi"}),
            {
                "modelId": MODEL,
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
                "inferenceConfig": {"maxTokens": 2000},
            },
        )
        await llm.chat(USER, max_tokens=256)


async def test_json_goes_through_output_config_with_closed_objects():
    client = aws_client("bedrock-runtime")
    llm = BedrockConverseLLM(MODEL, client=client, effort="low")
    schema = {
        "type": "object",
        "properties": {
            "say": {"type": "string"},
            "items": {"type": "array", "items": {"type": "object", "properties": {}}},
        },
    }
    closed = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "say": {"type": "string"},
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    }
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            converse_reply({"text": '{"say": "hi"}'}),
            {
                "modelId": MODEL,
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
                "inferenceConfig": {"maxTokens": 512, "temperature": 0.3},
                "outputConfig": {
                    "effort": "low",
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {"jsonSchema": {"schema": ANY, "name": "reply"}},
                    },
                },
            },
        )
        response = await llm.chat(USER, json_schema=schema)

    assert json.loads(response.text) == {"say": "hi"}
    assert bedrock._strict_schema(schema) == closed
    assert "additionalProperties" not in schema  # the bot's own schema is left as it was


async def test_tool_history_merges_results_into_one_user_turn():
    calls = [
        ToolCall(id="a", name="one", arguments={"x": 1}),
        ToolCall(id="b", name="two", arguments={}),
    ]
    history = [
        *USER,
        ChatMessage(role="assistant", content="Checking.", tool_calls=calls),
        ChatMessage(role="tool", content='{"ok": 1}', tool_call_id="a", name="one"),
        ChatMessage(role="tool", content='{"ok": 2}', tool_call_id="b", name="two"),
    ]

    await converse_with(
        converse_reply({"text": "Done"}),
        {
            "modelId": MODEL,
            "messages": [
                {"role": "user", "content": [{"text": "Hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"text": "Checking."},
                        {"toolUse": {"toolUseId": "a", "name": "one", "input": {"x": 1}}},
                        {"toolUse": {"toolUseId": "b", "name": "two", "input": {}}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "a", "content": [{"text": '{"ok": 1}'}]}},
                        {"toolResult": {"toolUseId": "b", "content": [{"text": '{"ok": 2}'}]}},
                    ],
                },
            ],
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.3},
        },
        *history,
    )


async def test_blank_assistant_text_is_left_out_because_converse_rejects_it():
    call = ToolCall(id="a", name="one", arguments={})
    history = [
        *USER,
        ChatMessage(role="assistant", content="", tool_calls=[call]),
        ChatMessage(role="tool", content="{}", tool_call_id="a", name="one"),
    ]

    await converse_with(
        converse_reply({"text": "Done"}),
        {
            "modelId": MODEL,
            "messages": [
                {"role": "user", "content": [{"text": "Hello"}]},
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "a", "name": "one", "input": {}}}],
                },
                {
                    "role": "user",
                    "content": [{"toolResult": {"toolUseId": "a", "content": [{"text": "{}"}]}}],
                },
            ],
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.3},
        },
        *history,
    )


# Converse replies --------------------------------------------------------------------------------


async def test_a_reply_with_text_tool_use_and_reasoning_parses():
    reply = converse_reply(
        {"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "sig"}}},
        {"text": "Let me look."},
        {"toolUse": {"toolUseId": "t1", "name": "find_customer", "input": {"phone": "555"}}},
        stop="tool_use",
    )
    client = aws_client("bedrock-runtime")
    llm = BedrockConverseLLM(MODEL, client=client)
    with Stubber(client) as stub:
        stub.add_response("converse", reply)
        response = await llm.chat(USER)

    assert response.text == "Let me look."
    assert response.tool_calls == [
        ToolCall(id="t1", name="find_customer", arguments={"phone": "555"})
    ]
    assert response.finish_reason == "tool_use"
    assert (response.input_tokens, response.output_tokens) == (11, 5)
    assert response.model == MODEL


async def test_reasoning_goes_back_only_with_the_tool_round_still_in_progress():
    block = {"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "sig"}}}
    first = converse_reply(
        block, {"toolUse": {"toolUseId": "t1", "name": "one", "input": {}}}, stop="tool_use"
    )
    client = aws_client("bedrock-runtime")
    llm = BedrockConverseLLM(MODEL, client=client)
    call = ToolCall(id="t1", name="one", arguments={})
    in_round = [
        *USER,
        ChatMessage(role="assistant", tool_calls=[call]),
        ChatMessage(role="tool", content="{}", tool_call_id="t1", name="one"),
    ]
    later = [*in_round, ChatMessage(role="user", content="And then?")]

    with Stubber(client) as stub:
        stub.add_response("converse", first)
        stub.add_response(
            "converse",
            converse_reply({"text": "Done"}),
            {
                "modelId": MODEL,
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {
                        "role": "assistant",
                        "content": [
                            block,
                            {"toolUse": {"toolUseId": "t1", "name": "one", "input": {}}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"toolResult": {"toolUseId": "t1", "content": [{"text": "{}"}]}}
                        ],
                    },
                ],
                "inferenceConfig": {"maxTokens": 512, "temperature": 0.3},
            },
        )
        await llm.chat(USER)
        await llm.chat(in_round)

    wire = bedrock._converse_messages(later, llm._reasoning)
    assert all("reasoningContent" not in b for m in wire for b in m["content"])


async def test_stream_yields_text_and_a_final_response():
    client = ConverseStreamClient(text_events("Hel", "lo"))
    llm = BedrockConverseLLM(MODEL, client=client)

    text, final = await collect(llm.stream_chat(USER, tools=[LOOKUP]))

    assert text == "Hello" and final.text == "Hello"
    assert final.finish_reason == "end_turn"
    assert (final.input_tokens, final.output_tokens) == (3, 2)
    assert final.first_token_ms >= 0
    assert client.requests[0]["toolConfig"]["tools"][0]["toolSpec"]["name"] == "find_customer"


async def test_stream_assembles_tool_input_from_pieces():
    events = [
        *tool_events("t1", "find_customer", '{"pho', 'ne": "555"}'),
        *tool_events("t2", "list_plans", index=1),
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    llm = BedrockConverseLLM(MODEL, client=ConverseStreamClient(events))

    _, final = await collect(llm.stream_chat(USER))

    assert final.tool_calls == [
        ToolCall(id="t1", name="find_customer", arguments={"phone": "555"}),
        ToolCall(id="t2", name="list_plans", arguments={}),
    ]
    assert final.finish_reason == "tool_use"


async def test_stream_drops_a_tool_call_with_broken_input():
    llm = BedrockConverseLLM(MODEL, client=ConverseStreamClient(tool_events("t1", "one", '{"x"')))

    _, final = await collect(llm.stream_chat(USER))

    assert final.tool_calls == []


async def test_stream_keeps_reasoning_for_the_tool_round():
    events = [
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "hm"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"signature": "sig"}},
            }
        },
        *tool_events("t1", "one", "{}", index=1),
    ]
    llm = BedrockConverseLLM(MODEL, client=ConverseStreamClient(events))

    text, _ = await collect(llm.stream_chat(USER))

    assert text == ""
    assert llm._reasoning["t1"] == [
        {"reasoningContent": {"reasoningText": {"text": "hm", "signature": "sig"}}}
    ]


async def test_closing_a_stream_early_closes_it_on_the_aws_side():
    client = ConverseStreamClient(text_events("one", "two", "three"))
    stream = BedrockConverseLLM(MODEL, client=client).stream_chat(USER)

    assert (await anext(stream)).text == "one"
    await stream.aclose()

    assert client.closed == 1


async def test_an_error_mid_stream_reaches_the_caller():
    class Failing:
        def converse_stream(self, **request):
            def events():
                yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}}
                raise ConnectionError("throttled")

            return {"stream": events()}

    with pytest.raises(ConnectionError, match="throttled"):
        await collect(BedrockConverseLLM(MODEL, client=Failing()).stream_chat(USER))


# Polly -------------------------------------------------------------------------------------------


def polly_with(pcm: bytes) -> tuple[Any, Stubber]:
    client = aws_client("polly")
    stub = Stubber(client)
    stub.add_response(
        "synthesize_speech",
        {
            "AudioStream": StreamingBody(io.BytesIO(pcm), len(pcm)),
            "ContentType": "audio/pcm",
            "RequestCharacters": 5,
        },
        {
            "Text": "Hello",
            "TextType": "text",
            "VoiceId": "Ruth",
            "Engine": "neural",
            "LanguageCode": "en-US",
            "OutputFormat": "pcm",
            "SampleRate": "16000",
        },
    )
    return client, stub


async def test_polly_asks_for_raw_pcm_and_maps_persona_voices():
    pcm = b"\x01\x00" * 100
    client, stub = polly_with(pcm)
    tts = PollyTTS(client=client, voice_map={"default_female_1": "Ruth"})

    with stub:
        clip = await tts.synthesize("Hello", voice_id="default_female_1")

    assert clip.pcm == pcm and clip.sample_rate == 16000


async def test_polly_passes_an_unmapped_voice_straight_through():
    client, stub = polly_with(b"\x01\x00")

    with stub:
        await PollyTTS(client=client).synthesize("Hello", voice_id="Ruth")


async def test_polly_streaming_never_splits_a_sample():
    class Body:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_chunks(self, size):
            return iter([b"\x01\x00\x02", b"\x00\x03\x00"])

    class Client:
        def synthesize_speech(self, **request):
            return {"AudioStream": Body()}

    chunks = [c async for c in PollyTTS(client=Client()).stream("Hi", voice_id="Ruth")]

    assert [c.pcm for c in chunks] == [b"\x01\x00", b"\x02\x00\x03\x00"]


async def test_polly_lists_voices():
    client = aws_client("polly")
    with Stubber(client) as stub:
        stub.add_response(
            "describe_voices",
            {
                "Voices": [
                    {"Id": "Ruth", "Name": "Ruth", "Gender": "Female", "LanguageCode": "en-US"}
                ]
            },
            {"Engine": "neural", "LanguageCode": "en-US"},
        )
        voices = await PollyTTS(client=client).voices()

    assert [(v.id, v.gender, v.language) for v in voices] == [("Ruth", "Female", "en-US")]


# Transcribe --------------------------------------------------------------------------------------


class FakeInput:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.closed = False

    async def send(self, event: Any) -> None:
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


class FakeStream:
    """Stands for the SDK's duplex event stream."""

    def __init__(self, *output: Any) -> None:
        self.input_stream = FakeInput()
        self.output = output

    async def await_output(self) -> tuple[Any, Any]:
        async def events():
            for item in self.output:
                yield item

        return None, events()


def result(*items: Item, text: str, partial: bool = False) -> Any:
    alternative = Alternative(transcript=text, items=list(items))
    return TranscriptResultStreamTranscriptEvent(
        value=TranscriptEvent(
            transcript=AwsTranscript(
                results=[Result(is_partial=partial, alternatives=[alternative])]
            )
        )
    )


def word(text: str, start: float, end: float, confidence: float) -> Item:
    return Item(type="pronunciation", content=text, start_time=start, end_time=end,
                confidence=confidence)  # fmt: skip


def transcribe_with(stream: FakeStream, **options: Any) -> tuple[TranscribeSTT, list]:
    opened: list = []

    async def open_stream(region: str, request: Any) -> FakeStream:
        opened.append((region, request))
        return stream

    return TranscribeSTT("us-west-2", speed=1000.0, open_stream=open_stream, **options), opened


async def test_transcribe_streams_pcm_chunks_and_reads_words_with_confidence():
    stream = FakeStream(
        result(word("Hello", 0.1, 0.4, 0.5), text="Hello", partial=True),
        result(
            word("Hello", 0.1, 0.4, 0.98),
            Item(type="punctuation", content=","),
            word("there", 0.5, 0.9, 0.71),
            text="Hello, there",
        ),
    )
    stt, opened = transcribe_with(stream)
    clip = AudioClip.silence(0.35, 16000)

    transcript = await stt.transcribe(clip)

    region, request = opened[0]
    assert region == "us-west-2"
    assert request.language_code == "en-US"
    assert request.media_sample_rate_hertz == 16000 and request.media_encoding == "pcm"
    assert request.show_speaker_label is None or request.show_speaker_label is False
    assert isinstance(stream.input_stream.events[0], AudioStreamAudioEvent)
    sizes = [len(e.value.audio_chunk) for e in stream.input_stream.events]
    assert sizes == [3200, 3200, 3200, 1600]  # 100 ms chunks, inside AWS's advised range
    assert stream.input_stream.closed

    assert transcript.text == "Hello, there"
    assert [(w.text, w.start_s, w.end_s, w.confidence) for w in transcript.words] == [
        ("Hello,", 0.1, 0.4, 0.98),
        ("there", 0.5, 0.9, 0.71),
    ]
    assert transcript.language == "en" and transcript.model == "amazon-transcribe"


async def test_transcribe_resamples_what_it_is_given():
    stream = FakeStream()
    stt, _ = transcribe_with(stream)

    await stt.transcribe(AudioClip.silence(0.1, 8000))

    assert sum(len(e.value.audio_chunk) for e in stream.input_stream.events) == 3200


async def test_eval_transcripts_cannot_use_a_custom_vocabulary():
    stream = FakeStream()
    stt, opened = transcribe_with(stream, vocabulary_name="fernhill-terms")

    await stt.transcribe(AudioClip.silence(0.1, 16000), hints=None)
    await stt.transcribe(AudioClip.silence(0.1, 16000), hints=["Fernhill"])

    assert opened[0][1].vocabulary_name is None
    assert opened[1][1].vocabulary_name == "fernhill-terms"


async def test_transcribe_stream_errors_are_raised():
    failure = TranscriptResultStreamBadRequestException(value=BadRequestException(message="no"))
    stt, _ = transcribe_with(FakeStream(failure))

    with pytest.raises(BadRequestException):
        await stt.transcribe(AudioClip.silence(0.1, 16000))


# Nova Sonic --------------------------------------------------------------------------------------


def sonic_out(event: dict) -> InvokeModelWithBidirectionalStreamOutputChunk:
    payload = json.dumps({"event": event}).encode()
    return InvokeModelWithBidirectionalStreamOutputChunk(
        value=BidirectionalOutputPayloadPart(bytes_=payload)
    )


async def sonic_with(stream: FakeStream, **connect: Any):
    async def open_stream(region: str, model: str) -> FakeStream:
        assert (region, model) == ("us-east-1", "amazon.nova-2-5-sonic")
        return stream

    voice = NovaSonicVoice(open_stream=open_stream)
    options = {"instructions": "You are Fern.", "tools": [LOOKUP]} | connect
    return await voice.connect(**options)


def sent_events(stream: FakeStream) -> list[dict]:
    decoded = []
    for chunk in stream.input_stream.events:
        assert isinstance(chunk, InvokeModelWithBidirectionalStreamInputChunk)
        decoded.append(json.loads(chunk.value.bytes_)["event"])
    return decoded


async def test_a_sonic_session_opens_with_the_documented_event_order():
    stream = FakeStream()

    await sonic_with(stream)

    events = sent_events(stream)
    assert [next(iter(e)) for e in events] == [
        "sessionStart", "promptStart", "contentStart", "textInput", "contentEnd", "contentStart",
    ]  # fmt: skip
    start, prompt, system, text, system_end, audio = events
    assert start["sessionStart"]["inferenceConfiguration"]["maxTokens"] == 1024
    configured = prompt["promptStart"]
    assert configured["audioOutputConfiguration"] == {
        "mediaType": "audio/lpcm",
        "sampleRateHertz": 24000,
        "sampleSizeBits": 16,
        "channelCount": 1,
        "voiceId": "matthew",
        "encoding": "base64",
        "audioType": "SPEECH",
    }
    spec = configured["toolConfiguration"]["tools"][0]["toolSpec"]
    assert spec["name"] == "find_customer"
    assert json.loads(spec["inputSchema"]["json"]) == LOOKUP.parameters
    assert configured["toolUseOutputConfiguration"] == {"mediaType": "application/json"}
    assert system["contentStart"]["role"] == "SYSTEM"
    assert text["textInput"]["content"] == "You are Fern."
    assert text["textInput"]["contentName"] == system["contentStart"]["contentName"]
    assert system_end["contentEnd"]["contentName"] == system["contentStart"]["contentName"]
    assert audio["contentStart"]["type"] == "AUDIO" and audio["contentStart"]["role"] == "USER"
    assert audio["contentStart"]["audioInputConfiguration"]["sampleRateHertz"] == 16000


async def test_a_sonic_voice_can_be_chosen_per_call():
    stream = FakeStream()

    await sonic_with(stream, voice_id="tiffany", tools=[])

    prompt = sent_events(stream)[1]["promptStart"]
    assert prompt["audioOutputConfiguration"]["voiceId"] == "tiffany"
    assert "toolConfiguration" not in prompt


async def test_sonic_audio_is_base64_in_the_open_audio_block():
    stream = FakeStream()
    session = await sonic_with(stream)

    await session.send_audio(b"\x01\x02")

    last = sent_events(stream)[-1]["audioInput"]
    assert base64.b64decode(last["content"]) == b"\x01\x02"
    assert last["contentName"] == sent_events(stream)[-2]["contentStart"]["contentName"]


async def test_a_sonic_tool_result_is_its_own_content_block():
    stream = FakeStream()
    session = await sonic_with(stream)

    await session.send_tool_result("t1", '{"found": true}')

    start, result_event, end = sent_events(stream)[-3:]
    config = start["contentStart"]
    assert config["type"] == "TOOL" and config["role"] == "TOOL"
    assert config["toolResultInputConfiguration"]["toolUseId"] == "t1"
    assert result_event["toolResult"]["content"] == '{"found": true}'
    assert end["contentEnd"]["contentName"] == config["contentName"]


async def test_sonic_events_normalize():
    audio = base64.b64encode(b"\x01\x00").decode()
    stream = FakeStream(
        sonic_out({"contentStart": {"role": "USER", "type": "TEXT"}}),
        sonic_out({"textOutput": {"role": "USER", "content": "Hi there"}}),
        sonic_out(
            {
                "contentStart": {
                    "role": "ASSISTANT",
                    "type": "TEXT",
                    "additionalModelFields": '{"generationStage": "SPECULATIVE"}',
                }
            }
        ),
        sonic_out({"textOutput": {"role": "ASSISTANT", "content": "I plan to say this."}}),
        sonic_out({"contentStart": {"role": "ASSISTANT", "type": "AUDIO"}}),
        sonic_out({"audioOutput": {"content": audio}}),
        sonic_out(
            {
                "contentStart": {
                    "role": "ASSISTANT",
                    "type": "TEXT",
                    "additionalModelFields": '{"generationStage": "FINAL"}',
                }
            }
        ),
        sonic_out({"textOutput": {"role": "ASSISTANT", "content": "I said this."}}),
        sonic_out(
            {
                "toolUse": {
                    "toolUseId": "t1",
                    "toolName": "find_customer",
                    "content": '{"phone": "5"}',
                }
            }
        ),
        sonic_out({"completionEnd": {}}),
    )
    session = await sonic_with(stream)

    events = [event async for event in session.events()]

    assert [e.type for e in events] == [
        "user_transcript", "audio", "bot_transcript", "tool_call", "response_done",
    ]  # fmt: skip
    assert events[0].text == "Hi there"
    assert events[1].audio_pcm16 == b"\x01\x00"
    assert events[2].text == "I said this."
    assert events[3].tool_call == ToolCall(id="t1", name="find_customer", arguments={"phone": "5"})


async def test_sonic_barge_in_is_reported_both_ways_it_can_arrive():
    stream = FakeStream(
        sonic_out({"textOutput": {"role": "ASSISTANT", "content": '{ "interrupted" : true }'}}),
        sonic_out({"contentEnd": {"stopReason": "INTERRUPTED"}}),
    )
    session = await sonic_with(stream)

    events = [event async for event in session.events()]

    assert [e.type for e in events] == ["speech_started", "speech_started"]


async def test_sonic_interrupt_drops_audio_until_the_next_response():
    audio = base64.b64encode(b"\x01\x00").decode()
    stream = FakeStream(
        sonic_out({"audioOutput": {"content": audio}}),
        sonic_out({"contentStart": {"role": "ASSISTANT", "type": "AUDIO"}}),
        sonic_out({"audioOutput": {"content": audio}}),
    )
    session = await sonic_with(stream)

    await session.interrupt(response_id=None, played_ms=500)
    events = [event async for event in session.events()]

    assert len(events) == 1 and events[0].type == "audio"


async def test_sonic_bad_tool_input_is_an_error_event_not_a_crash():
    stream = FakeStream(
        sonic_out({"toolUse": {"toolUseId": "t1", "toolName": "one", "content": '{"x": '}}),
    )
    session = await sonic_with(stream)

    events = [event async for event in session.events()]

    assert events[0].type == "error" and "bad tool input" in events[0].text


async def test_sonic_stream_errors_end_the_events():
    failure = InvokeModelWithBidirectionalStreamOutputModelTimeoutException(
        value=ModelTimeoutException(message="timed out")
    )
    session = await sonic_with(FakeStream(failure))

    events = [event async for event in session.events()]

    assert events[0].type == "error" and "timed out" in events[0].text


async def test_closing_a_sonic_session_ends_scopes_in_order_once():
    stream = FakeStream()
    session = await sonic_with(stream)

    await session.close()
    await session.close()

    assert [next(iter(e)) for e in sent_events(stream)[-3:]] == [
        "contentEnd", "promptEnd", "sessionEnd",
    ]  # fmt: skip
    assert stream.input_stream.closed
