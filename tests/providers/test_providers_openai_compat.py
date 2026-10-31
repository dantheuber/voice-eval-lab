import json

import httpx
import pytest
import respx

from voicelab.providers.adapters.openai_compat import LLMError, LLMTimeoutError, OpenAICompatLLM
from voicelab.providers.base import LLM, ChatMessage, ToolCall, ToolDef

BASE = "http://model.test/v1"
URL = f"{BASE}/chat/completions"
USER = [ChatMessage(role="user", content="Hello")]
LOOKUP = ToolDef(
    name="find_customer",
    description="Find a customer",
    parameters={"type": "object", "properties": {"phone": {"type": "string"}}},
)


def llm(**options) -> OpenAICompatLLM:
    return OpenAICompatLLM(BASE, "test-model", retry_backoff_s=0.0, **options)


def reply(message: dict, **fields) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": message}], **fields})


def sse(*events: dict | str) -> httpx.Response:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    return httpx.Response(200, text="".join(lines), headers={"content-type": "text/event-stream"})


def delta(**fields) -> dict:
    return {"choices": [{"delta": fields, "finish_reason": None}]}


def tool_delta(**fields) -> dict:
    return delta(tool_calls=[fields])


def sent(route: respx.Route, index: int = -1) -> dict:
    return json.loads(route.calls[index].request.content)


async def collect(stream):
    deltas = [item async for item in stream]
    final = deltas[-1].final
    return "".join(item.text for item in deltas), final


def test_it_satisfies_the_interface():
    assert isinstance(llm(), LLM)


# Requests ----------------------------------------------------------------------------------------


@respx.mock
async def test_chat_sends_a_standard_request():
    route = respx.post(URL).mock(return_value=reply({"content": "Hi"}))

    response = await llm().chat(USER, tools=[LOOKUP], temperature=0.1, max_tokens=64)

    body = sent(route)
    assert body["model"] == "test-model"
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["temperature"] == 0.1 and body["max_tokens"] == 64
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "find_customer",
                "description": "Find a customer",
                "parameters": LOOKUP.parameters,
            },
        }
    ]
    assert "stream" not in body
    assert response.text == "Hi"


@respx.mock
async def test_tool_history_goes_out_in_openai_form():
    route = respx.post(URL).mock(return_value=reply({"content": "Done"}))
    call = ToolCall(id="c1", name="find_customer", arguments={"phone": "555"})
    history = [
        *USER,
        ChatMessage(role="assistant", tool_calls=[call]),
        ChatMessage(
            role="tool", content='{"found": true}', tool_call_id="c1", name="find_customer"
        ),
    ]

    await llm().chat(history)

    assistant, tool = sent(route)["messages"][1:]
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["function"] == {
        "name": "find_customer",
        "arguments": '{"phone": "555"}',
    }
    assert tool == {
        "role": "tool",
        "content": '{"found": true}',
        "tool_call_id": "c1",
        "name": "find_customer",
    }


@respx.mock
async def test_json_schema_uses_response_format():
    route = respx.post(URL).mock(return_value=reply({"content": '{"say": "hi"}'}))
    schema = {"type": "object", "properties": {"say": {"type": "string"}}}

    response = await llm().chat(USER, json_schema=schema)

    assert sent(route)["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": schema},
    }
    assert json.loads(response.text) == {"say": "hi"}


@respx.mock
async def test_json_object_mode_puts_the_schema_in_the_prompt():
    route = respx.post(URL).mock(return_value=reply({"content": "{}"}))
    system = ChatMessage(role="system", content="Be brief.")

    await llm(structured_output="json_object").chat([system, *USER], json_schema={"type": "object"})

    body = sent(route)
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["content"].startswith("Be brief.\n\nReply with JSON only")


@respx.mock
async def test_prompt_mode_sends_no_response_format():
    route = respx.post(URL).mock(return_value=reply({"content": "{}"}))

    await llm(structured_output="prompt").chat(USER, json_schema={"type": "object"})

    body = sent(route)
    assert "response_format" not in body and body["messages"][0]["role"] == "system"


@respx.mock
async def test_optional_fields_can_be_left_out_or_added():
    route = respx.post(URL).mock(return_value=reply({"content": "Hi"}))
    server = OpenAICompatLLM(
        BASE,
        send_temperature=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 50},
    )

    await server.chat(USER)

    body = sent(route)
    assert "model" not in body and "temperature" not in body
    assert body["top_k"] == 50 and body["chat_template_kwargs"] == {"enable_thinking": False}


@respx.mock
async def test_the_key_comes_from_the_named_environment_variable(monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "sekret")
    route = respx.post(URL).mock(return_value=reply({"content": "Hi"}))

    await llm(api_key_env="TEST_TOKEN").chat(USER)

    assert route.calls[0].request.headers["authorization"] == "Bearer sekret"


@respx.mock
async def test_no_key_means_no_auth_header():
    route = respx.post(URL).mock(return_value=reply({"content": "Hi"}))

    await llm().chat(USER)

    assert "authorization" not in route.calls[0].request.headers


def test_the_name_says_what_is_running():
    assert llm().name == "openai_compat/test-model"
    assert OpenAICompatLLM("http://box:8092/v1").name == "openai_compat/box:8092"
    assert OpenAICompatLLM(BASE, name="support").name == "support"


# Non-streaming replies ---------------------------------------------------------------------------


@respx.mock
async def test_chat_reads_tool_calls_and_usage():
    call = {"id": "c1", "function": {"name": "find_customer", "arguments": '{"phone": "555"}'}}
    respx.post(URL).mock(
        return_value=reply(
            {"content": None, "tool_calls": [call]},
            usage={"prompt_tokens": 12, "completion_tokens": 4},
            model="served-model",
        )
    )

    response = await llm().chat(USER)

    assert response.text == ""
    assert response.tool_calls == [
        ToolCall(id="c1", name="find_customer", arguments={"phone": "555"})
    ]
    assert (response.input_tokens, response.output_tokens) == (12, 4)
    assert response.model == "served-model"


@respx.mock
async def test_llama_cpp_timings_fill_in_tokens_and_first_token_time():
    respx.post(URL).mock(
        return_value=reply(
            {"content": "Hi"},
            timings={"prompt_ms": 42.0, "prompt_n": 100, "predicted_n": 7},
        )
    )

    response = await llm().chat(USER)

    assert response.first_token_ms == 42.0
    assert (response.input_tokens, response.output_tokens) == (100, 7)


@respx.mock
async def test_reasoning_never_reaches_the_text():
    respx.post(URL).mock(
        return_value=reply(
            {"content": "<think>hmm</think>Hello", "reasoning_content": "let me see"}
        )
    )

    assert (await llm().chat(USER)).text == "Hello"


@respx.mock
async def test_empty_content_with_tool_calls_or_nothing_at_all_is_fine():
    respx.post(URL).mock(return_value=reply({"content": ""}))

    assert (await llm().chat(USER)).text == ""


@respx.mock
async def test_structured_output_left_in_the_reasoning_channel_is_recovered():
    respx.post(URL).mock(return_value=reply({"content": "", "reasoning_content": '{"say": "hi"}'}))

    assert (await llm().chat(USER, json_schema={"type": "object"})).text == '{"say": "hi"}'


@respx.mock
async def test_content_given_as_parts_is_joined():
    respx.post(URL).mock(
        return_value=reply({"content": [{"type": "text", "text": "He"}, {"text": "llo"}]})
    )

    assert (await llm().chat(USER)).text == "Hello"


@respx.mock
async def test_a_tool_call_with_bad_arguments_is_dropped_so_the_caller_can_retry():
    bad = {"id": "c1", "function": {"name": "find_customer", "arguments": '{"phone": "5'}}
    respx.post(URL).mock(return_value=reply({"content": "", "tool_calls": [bad]}))

    response = await llm().chat(USER)

    assert response.tool_calls == [] and response.finish_reason == "malformed_tool_call"


@respx.mock
async def test_arguments_that_are_already_an_object_are_accepted():
    call = {"id": "c1", "function": {"name": "find_customer", "arguments": {"phone": "555"}}}
    respx.post(URL).mock(return_value=reply({"content": "", "tool_calls": [call]}))

    assert (await llm().chat(USER)).tool_calls[0].arguments == {"phone": "555"}


@respx.mock
async def test_missing_arguments_mean_no_arguments_and_a_missing_id_is_invented():
    call = {"function": {"name": "list_plans"}}
    respx.post(URL).mock(return_value=reply({"content": "", "tool_calls": [call]}))

    tool_call = (await llm().chat(USER)).tool_calls[0]

    assert tool_call.arguments == {} and tool_call.id.startswith("call_0_")


# Streaming ---------------------------------------------------------------------------------------


@respx.mock
async def test_stream_yields_text_then_a_final_response():
    route = respx.post(URL).mock(
        return_value=sse(
            delta(role="assistant"),
            delta(content="Hel"),
            delta(content="lo"),
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
            "[DONE]",
        )
    )

    text, final = await collect(llm().stream_chat(USER))

    assert text == "Hello" and final.text == "Hello"
    assert final.finish_reason == "stop"
    assert (final.input_tokens, final.output_tokens) == (9, 2)
    assert final.first_token_ms is not None and final.first_token_ms >= 0
    body = sent(route)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


@respx.mock
async def test_stream_options_can_be_turned_off_for_strict_servers():
    route = respx.post(URL).mock(return_value=sse(delta(content="Hi"), "[DONE]"))

    await collect(llm(include_usage=False).stream_chat(USER))

    assert "stream_options" not in sent(route)


@respx.mock
async def test_reasoning_deltas_are_not_text_and_do_not_start_the_first_token_clock():
    respx.post(URL).mock(
        return_value=sse(delta(reasoning_content="thinking"), delta(content="Hi"), "[DONE]")
    )

    deltas = [item async for item in llm().stream_chat(USER)]

    assert [d.text for d in deltas if d.text] == ["Hi"]
    assert deltas[-1].final.text == "Hi"


@respx.mock
async def test_think_tags_stay_in_deltas_but_not_in_the_final_text():
    respx.post(URL).mock(
        return_value=sse(delta(content="<think>hm</think>"), delta(content="Hi"), "[DONE]")
    )

    text, final = await collect(llm().stream_chat(USER))

    assert "<think>" in text and final.text == "Hi"


@respx.mock
async def test_a_tool_call_split_across_deltas_is_assembled():
    respx.post(URL).mock(
        return_value=sse(
            tool_delta(index=0, id="c1", function={"name": "find_", "arguments": ""}),
            tool_delta(index=0, function={"name": "customer", "arguments": '{"pho'}),
            tool_delta(index=0, function={"arguments": 'ne": "555"}'}),
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        )
    )

    _, final = await collect(llm().stream_chat(USER))

    assert final.tool_calls == [ToolCall(id="c1", name="find_customer", arguments={"phone": "555"})]
    assert final.finish_reason == "tool_calls" and final.first_token_ms is not None


@respx.mock
async def test_parallel_calls_keep_apart_by_index():
    respx.post(URL).mock(
        return_value=sse(
            tool_delta(index=0, id="a", function={"name": "one", "arguments": ""}),
            tool_delta(index=1, id="b", function={"name": "two", "arguments": ""}),
            tool_delta(index=1, function={"arguments": '{"x": 2}'}),
            tool_delta(index=0, function={"arguments": '{"x": 1}'}),
            "[DONE]",
        )
    )

    _, final = await collect(llm().stream_chat(USER))

    assert [(c.name, c.arguments) for c in final.tool_calls] == [
        ("one", {"x": 1}),
        ("two", {"x": 2}),
    ]


@respx.mock
async def test_calls_numbered_alike_keep_apart_by_id():
    respx.post(URL).mock(
        return_value=sse(
            tool_delta(index=0, id="a", function={"name": "one", "arguments": "{}"}),
            tool_delta(index=0, id="b", function={"name": "two", "arguments": "{}"}),
            "[DONE]",
        )
    )

    _, final = await collect(llm().stream_chat(USER))

    assert [c.name for c in final.tool_calls] == ["one", "two"]


@respx.mock
async def test_a_tool_call_with_no_index_continues_the_last_one():
    respx.post(URL).mock(
        return_value=sse(
            tool_delta(id="a", function={"name": "one", "arguments": '{"x"'}),
            tool_delta(function={"arguments": ": 1}"}),
            "[DONE]",
        )
    )

    _, final = await collect(llm().stream_chat(USER))

    assert final.tool_calls[0].arguments == {"x": 1}


@respx.mock
async def test_a_name_repeated_on_every_delta_is_not_doubled():
    respx.post(URL).mock(
        return_value=sse(
            tool_delta(index=0, id="a", function={"name": "one", "arguments": "{"}),
            tool_delta(index=0, function={"name": "one", "arguments": "}"}),
            "[DONE]",
        )
    )

    _, final = await collect(llm().stream_chat(USER))

    assert final.tool_calls[0].name == "one"


@respx.mock
async def test_truncated_arguments_in_a_stream_drop_only_that_call():
    respx.post(URL).mock(
        return_value=sse(
            tool_delta(index=0, id="a", function={"name": "good", "arguments": "{}"}),
            tool_delta(index=1, id="b", function={"name": "bad", "arguments": '{"x": '}),
            "[DONE]",
        )
    )

    _, final = await collect(llm().stream_chat(USER))

    assert [c.name for c in final.tool_calls] == ["good"]


@respx.mock
async def test_comments_and_garbage_lines_in_the_stream_are_skipped():
    body = (
        ": keep-alive\n\n"
        'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
        "data: {not json\n\n"
        'data:{"choices": [{"delta": {"content": "!"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(URL).mock(return_value=httpx.Response(200, text=body))

    text, _ = await collect(llm().stream_chat(USER))

    assert text == "Hi!"


@respx.mock
async def test_a_stream_with_no_final_marker_still_finishes():
    respx.post(URL).mock(return_value=sse(delta(content="Hi")))

    text, final = await collect(llm().stream_chat(USER))

    assert text == "Hi" and final.text == "Hi"


@respx.mock
async def test_an_empty_reply_gives_an_empty_final():
    respx.post(URL).mock(return_value=sse("[DONE]"))

    text, final = await collect(llm().stream_chat(USER))

    assert text == "" and final.text == "" and final.tool_calls == []


@respx.mock
async def test_an_error_inside_the_stream_raises():
    respx.post(URL).mock(return_value=sse({"error": {"message": "context full"}}))

    with pytest.raises(LLMError, match="context full"):
        await collect(llm().stream_chat(USER))


@respx.mock
async def test_closing_the_stream_early_closes_the_connection():
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"delta": {"content": "Hi. "}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": "More"}}]}\n\n'

        async def aclose(self):
            closed.append(True)

    respx.post(URL).mock(return_value=httpx.Response(200, stream=Body()))

    stream = llm().stream_chat(USER)
    assert (await anext(stream)).text == "Hi. "
    await stream.aclose()

    assert closed == [True]


# Failures and retries ----------------------------------------------------------------------------


@respx.mock
async def test_connection_errors_are_retried():
    route = respx.post(URL).mock(
        side_effect=[httpx.ConnectError("refused"), reply({"content": "Hi"})]
    )

    assert (await llm().chat(USER)).text == "Hi"
    assert route.call_count == 2


@respx.mock
async def test_connection_errors_are_retried_for_a_stream():
    route = respx.post(URL).mock(
        side_effect=[httpx.RemoteProtocolError("reset"), sse(delta(content="Hi"), "[DONE]")]
    )

    text, _ = await collect(llm().stream_chat(USER))

    assert text == "Hi" and route.call_count == 2


@respx.mock
async def test_it_gives_up_after_the_retries_with_a_clear_error():
    route = respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(LLMError, match="unreachable"):
        await llm(retries=2).chat(USER)

    assert route.call_count == 3


@respx.mock
async def test_a_server_still_loading_is_retried():
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(503, json={"error": {"message": "Loading model"}}),
            reply({"content": "Hi"}),
        ]
    )

    assert (await llm().chat(USER)).text == "Hi"
    assert route.call_count == 2


@respx.mock
async def test_a_bad_request_is_not_retried():
    route = respx.post(URL).mock(return_value=httpx.Response(400, text="bad schema"))

    with pytest.raises(LLMError, match="400.*bad schema"):
        await llm().chat(USER)

    assert route.call_count == 1


@respx.mock
async def test_a_slow_model_times_out_without_a_retry():
    route = respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))

    with pytest.raises(LLMTimeoutError):
        await llm().chat(USER)

    assert route.call_count == 1


@respx.mock
async def test_a_connect_timeout_is_retried_and_reported_as_a_timeout():
    route = respx.post(URL).mock(side_effect=httpx.ConnectTimeout("no answer"))

    with pytest.raises(LLMTimeoutError):
        await llm(retries=1).chat(USER)

    assert route.call_count == 2


@respx.mock
async def test_a_stream_that_stalls_is_not_retried_because_text_is_already_out():
    class Stalls(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
            raise httpx.ReadTimeout("stalled")

        async def aclose(self):
            pass

    route = respx.post(URL).mock(return_value=httpx.Response(200, stream=Stalls()))

    stream = llm().stream_chat(USER)
    assert (await anext(stream)).text == "Hi"
    with pytest.raises(LLMTimeoutError):
        await anext(stream)

    assert route.call_count == 1


@respx.mock
async def test_a_reply_that_is_not_json_is_an_error():
    respx.post(URL).mock(return_value=httpx.Response(200, text="<html>proxy</html>"))

    with pytest.raises(LLMError, match="isn't JSON"):
        await llm().chat(USER)


@respx.mock
async def test_an_error_field_in_a_200_reply_is_an_error():
    respx.post(URL).mock(return_value=httpx.Response(200, json={"error": "out of memory"}))

    with pytest.raises(LLMError, match="out of memory"):
        await llm().chat(USER)
