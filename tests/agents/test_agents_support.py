"""The support bot: streaming, tools, malformed output, barge-in, and hang-up."""

import json

import pytest
from scripted_llm import ScriptedLLM, call, uses

from voicelab.agents.support import (
    GOODBYE,
    GREETING,
    SYSTEM_PROMPT,
    TROUBLE_LINE,
    SupportBot,
    parse_text_tool_calls,
    speakable_text,
    support_instructions,
)
from voicelab.company import ToolDispatcher, create_call_db, diff_against_seed, load_policy
from voicelab.core.models import SystemEvent, ToolCallEvent
from voicelab.providers.base import LLM


@pytest.fixture
def db():
    call_db = create_call_db()
    yield call_db
    call_db.close()


def make_bot(db, *script, **options):
    events = []
    llm = ScriptedLLM(*script, default=options.pop("default", None))
    bot = SupportBot(llm, ToolDispatcher(db), on_event=events.append, **options)
    bot.greet()
    return bot, llm, events


async def turn(bot, text="Hello"):
    return [segment async for segment in bot.respond(text)]


def tool_events(events):
    return [event for event in events if isinstance(event, ToolCallEvent)]


def roles(messages):
    return [message.role for message in messages]


def test_scripted_llm_matches_the_protocol():
    assert isinstance(ScriptedLLM(), LLM)


# Prompt ---------------------------------------------------------------------------------------


def test_system_prompt_is_built_for_voice(db):
    bot, _, _ = make_bot(db)
    prompt = bot.messages[0].content
    assert "read aloud" in prompt
    assert "one question at a time" in prompt.lower()
    assert "No markdown" in prompt
    assert "clear yes" in prompt
    assert "October 30, 2026" in prompt
    assert load_policy() in prompt


def test_prompt_stays_short_for_a_small_model():
    assert len(SYSTEM_PROMPT.split()) < 120


def test_greeting_is_fixed_and_recorded(db):
    llm = ScriptedLLM()
    bot = SupportBot(llm, ToolDispatcher(db))
    segment = bot.greet()
    assert segment.display == GREETING
    assert roles(bot.messages) == ["system", "user", "assistant"]
    assert llm.requests == []


# Streaming ------------------------------------------------------------------------------------


async def test_yields_sentence_by_sentence(db):
    bot, _, _ = make_bot(db, "First sentence here. Second one. And the end")
    segments = await turn(bot)
    assert [s.display for s in segments] == [
        "First sentence here.",
        "Second one.",
        "And the end",
    ]


async def test_first_sentence_arrives_before_the_stream_ends(db):
    text = "Short one. " + " ".join(["word"] * 50)
    bot, llm, _ = make_bot(db, text)
    stream = bot.respond("Hi")
    first = await anext(stream)
    assert first.display == "Short one."
    assert llm.chunks_sent < 5
    await stream.aclose()


async def test_segments_carry_spoken_forms(db):
    bot, _, _ = make_bot(db, "Invoice FH-4419 was $49.99. Thanks.")
    first, second = await turn(bot)
    assert first.display == "Invoice FH-4419 was $49.99."
    assert first.spoken == (
        "Invoice F H dash 4 4 1 9 was forty nine dollars and ninety nine cents."
    )
    assert second.spoken == "Thanks."


async def test_history_keeps_the_reply(db):
    bot, _, _ = make_bot(db, "All set. Anything else?")
    await turn(bot, "Thanks")
    assert bot.messages[-1].role == "assistant"
    assert bot.messages[-1].content == "All set. Anything else?"
    assert bot.messages[-2].content == "Thanks"


async def test_an_adapter_that_only_fills_final_text_still_speaks(db):
    class FinalOnly(ScriptedLLM):
        async def stream_chat(self, messages, **options):
            from voicelab.providers.base import LLMDelta, LLMResponse

            self.requests.append({"messages": messages})
            yield LLMDelta(final=LLMResponse(text="Just the final text."))

    bot = SupportBot(FinalOnly(), ToolDispatcher(db))
    bot.greet()
    assert [s.display for s in await turn(bot)] == ["Just the final text."]


async def test_tools_are_offered_on_every_round(db):
    bot, llm, _ = make_bot(db, uses(call("list_plans")), "Done.")
    await turn(bot)
    assert all(request["tools"] for request in llm.requests)


# Tools ----------------------------------------------------------------------------------------


async def test_runs_tools_until_it_has_something_to_say(db):
    bot, llm, events = make_bot(
        db,
        uses(call("find_customer", phone="503-555-0101")),
        uses(call("verify_identity", customer_id="cus_1001", billing_zip="97214", pin="4821")),
        "Thanks Dana, you're verified.",
    )
    segments = await turn(bot)
    assert [s.display for s in segments] == ["Thanks Dana, you're verified."]
    assert [event.name for event in events] == ["find_customer", "verify_identity"]
    assert len(llm.requests) == 3


async def test_tool_events_carry_timing_and_results(db):
    times = iter([12.5, 13.0])
    bot, _, events = make_bot(
        db,
        uses(call("get_invoice", invoice_id="FH-4419")),
        "Found it.",
        clock=lambda: next(times),
    )
    await turn(bot)
    event = events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.at_s == 12.5
    assert event.arguments == {"invoice_id": "FH-4419"}
    assert event.result["amount"] == "$49.99"
    assert event.error is None
    assert event.duration_ms >= 0


async def test_default_clock_measures_from_the_start_of_the_call(db):
    bot, _, events = make_bot(db, uses(call("list_plans")), "Okay.")
    await turn(bot)
    assert 0 <= events[0].at_s < 5


async def test_tool_errors_reach_the_model_and_the_timeline(db):
    bot, llm, events = make_bot(
        db, uses(call("get_invoice", invoice_id="FH-1")), "I can't find that invoice."
    )
    await turn(bot)
    assert events[0].error == "no invoice FH-1"
    tool_message = llm.requests[1]["messages"][-1]
    assert tool_message.role == "tool"
    assert "no invoice" in tool_message.content


async def test_tool_messages_pair_up_with_their_calls(db):
    first = call("list_plans")
    bot, llm, _ = make_bot(db, uses(first), "Okay.")
    await turn(bot)
    sent = llm.requests[1]["messages"]
    assert roles(sent)[-2:] == ["assistant", "tool"]
    assert sent[-2].tool_calls == [first]
    assert sent[-1].tool_call_id == first.id
    assert sent[-1].name == "list_plans"
    assert json.loads(sent[-1].content)["plans"]


async def test_several_tool_calls_in_one_reply_all_run(db):
    bot, _, events = make_bot(
        db,
        uses(call("list_plans"), call("get_invoice", invoice_id="FH-4419")),
        "Okay.",
    )
    await turn(bot)
    assert [event.name for event in events] == ["list_plans", "get_invoice"]


async def test_text_before_a_tool_call_is_spoken(db):
    bot, _, _ = make_bot(db, uses(call("list_plans"), text="One moment."), "Here you go.")
    segments = await turn(bot)
    assert [s.display for s in segments] == ["One moment.", "Here you go."]
    assert bot.messages[-3].content == "One moment."
    assert bot.messages[-3].tool_calls


async def test_unknown_tool_is_survivable(db):
    bot, _, events = make_bot(db, uses(call("transfer_funds")), "Sorry about that.")
    assert [s.display for s in await turn(bot)] == ["Sorry about that."]
    assert "unknown tool" in events[0].error


async def test_the_bot_runs_on_the_call_database(db):
    bot, _, _ = make_bot(
        db, uses(call("unlock_mailbox", address="evelyn@hartfamily.example")), "Unlocked."
    )
    await turn(bot)
    assert [change.table for change in diff_against_seed(db)] == ["mailboxes"]


async def test_no_event_callback_is_fine(db):
    bot = SupportBot(ScriptedLLM(uses(call("list_plans")), "Okay."), ToolDispatcher(db))
    assert [s.display for s in await turn(bot)] == ["Okay."]


# Malformed output -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name": "list_plans", "arguments": {}}</tool_call>',
        '{"name": "list_plans", "arguments": {}}',
        '{"name": "list_plans", "parameters": {}}',
        '```json\n{"name": "list_plans", "arguments": {}}\n```',
        '[{"name": "list_plans", "arguments": {}}]',
        '{"function": {"name": "list_plans", "arguments": "{}"}}',
        '[TOOL_CALLS] {"name": "list_plans", "arguments": {}}',
        '<think>hmm</think>{"name": "list_plans", "arguments": {}}',
    ],
)
async def test_tool_calls_written_as_text_still_run(db, text):
    bot, _, events = make_bot(db, text, "We have six plans.")
    segments = await turn(bot)
    assert [s.display for s in segments] == ["We have six plans."]
    assert [event.name for event in events] == ["list_plans"]


async def test_arguments_in_text_tool_calls_are_kept(db):
    text = '<tool_call>{"name": "get_invoice", "arguments": {"invoice_id": "FH-4419"}}</tool_call>'
    bot, _, events = make_bot(db, text, "Found it.")
    await turn(bot)
    assert events[0].arguments == {"invoice_id": "FH-4419"}
    assert events[0].error is None


async def test_speech_before_a_text_tool_call_is_kept(db):
    bot, _, _ = make_bot(
        db, 'Let me check. {"name": "list_plans", "arguments": {}}', "We have six plans."
    )
    segments = await turn(bot)
    assert [s.display for s in segments] == ["Let me check.", "We have six plans."]


@pytest.mark.parametrize(
    "garbage",
    [
        '<tool_call>{"name": "list_plans", "arguments": {',
        '{"name": "list_plans", "arguments": "not json"}',
        '{"thought": "I should call a tool"}',
        "```python\nprint('hi')\n```",
        "",
    ],
)
async def test_malformed_output_retries_and_is_never_spoken(db, garbage):
    bot, llm, events = make_bot(db, garbage, "How can I help?")
    segments = await turn(bot)
    assert [s.display for s in segments] == ["How can I help?"]
    assert len(llm.requests) == 2
    assert [event.kind for event in events] == ["malformed_tool_call"]


async def test_retry_note_goes_to_the_model_but_not_into_history(db):
    bot, llm, _ = make_bot(db, "<tool_call>{broken", "Okay.")
    await turn(bot, "Hi there")
    retry_request = llm.requests[1]["messages"]
    assert "not usable" in retry_request[-1].content
    assert "Hi there" in retry_request[-1].content
    assert retry_request[-1].role == "user"
    assert all("not usable" not in message.content for message in bot.messages)
    assert all("broken" not in message.content for message in bot.messages)


async def test_retry_note_after_a_tool_result_stays_valid(db):
    bot, llm, _ = make_bot(db, uses(call("list_plans")), "<tool_call>{broken", "Okay.")
    await turn(bot)
    assert roles(llm.requests[2]["messages"])[-3:] == ["assistant", "tool", "user"]


async def test_speech_then_malformed_call_keeps_alternating_roles(db):
    bot, llm, _ = make_bot(db, "Let me look. <tool_call>{broken", "Here it is.")
    segments = await turn(bot)
    assert [s.display for s in segments] == ["Let me look.", "Here it is."]
    assert roles(bot.messages)[-2:] == ["user", "assistant"]
    assert bot.messages[-1].content == "Let me look. Here it is."


async def test_model_that_never_recovers_gets_a_fixed_line(db):
    bot, llm, events = make_bot(db, default="<tool_call>{broken", max_tool_rounds=3)
    segments = await turn(bot)
    assert [s.display for s in segments] == [TROUBLE_LINE]
    assert len(llm.requests) == 3
    assert events[-1].kind == "tool_loop_limit"
    assert bot.messages[-1].content == TROUBLE_LINE


async def test_tool_loop_limit_stops_a_model_stuck_calling_tools(db):
    bot, llm, events = make_bot(db, default=uses(call("list_plans")), max_tool_rounds=4)
    segments = await turn(bot)
    assert [s.display for s in segments] == [TROUBLE_LINE]
    assert len(llm.requests) == 4
    assert len(tool_events(events)) == 4
    assert isinstance(events[-1], SystemEvent)
    assert events[-1].kind == "tool_loop_limit"


async def test_reasoning_is_never_spoken(db):
    bot, _, _ = make_bot(db, "<think>The user wants. Maybe a refund.</think>Sure thing.")
    assert [s.display for s in await turn(bot)] == ["Sure thing."]


async def test_unclosed_reasoning_is_never_spoken(db):
    bot, _, _ = make_bot(db, "<think>Still pondering. On and on. ", "Sure thing.")
    assert [s.display for s in await turn(bot)] == ["Sure thing."]


def test_speakable_text_cuts_at_tool_markup():
    assert speakable_text('Okay. <tool_call>{"a": 1}') == "Okay. "
    assert speakable_text('{"name": "x"}') == ""
    assert speakable_text("It costs $5 {roughly}") == "It costs $5 {roughly}"


@pytest.mark.parametrize("cut_off", ["Okay. <tool_", "Okay. {", "Okay. <thi", "Okay. [TOOL"])
def test_a_reply_cut_off_inside_markup_stays_silent(cut_off):
    assert speakable_text(cut_off) == "Okay. "


async def test_retried_speech_then_tool_call_keeps_one_assistant_message(db):
    bot, llm, _ = make_bot(
        db, "Let me look. <tool_call>{broken", uses(call("list_plans")), "We have six."
    )
    await turn(bot)
    assert roles(bot.messages)[-3:] == ["assistant", "tool", "assistant"]
    assert bot.messages[-3].content == "Let me look."
    assert bot.messages[-3].tool_calls


def test_parse_ignores_plain_speech():
    assert parse_text_tool_calls("Your site is back online.") == ([], False)


def test_parse_finds_several_calls():
    text = '<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>'
    calls, malformed = parse_text_tool_calls(text)
    assert [item.name for item in calls] == ["a", "b"]
    assert malformed is False


# Barge-in -------------------------------------------------------------------------------------


async def test_interruption_keeps_only_what_was_played(db):
    bot, _, _ = make_bot(db, "Your refund is done. It takes five days. Anything else?")
    await turn(bot)
    bot.on_interrupted("Your refund is done.")
    assert bot.messages[-1].role == "assistant"
    assert bot.messages[-1].content == "Your refund is done."


async def test_interruption_mid_sentence(db):
    bot, _, _ = make_bot(db, "Your invoice FH-4419 was paid on Friday.")
    await turn(bot)
    bot.on_interrupted("Your invoice F H dash 4")
    assert bot.messages[-1].content == "Your invoice F H dash 4"


async def test_the_next_request_sees_only_what_was_heard(db):
    bot, llm, _ = make_bot(db, "Secret tail. Never heard part.", "Okay.")
    await turn(bot, "Hi")
    bot.on_interrupted("Secret tail.")
    await turn(bot, "Wait")
    sent = " ".join(message.content for message in llm.requests[1]["messages"])
    assert "Never heard part" not in sent
    assert "Secret tail." in sent


async def test_nothing_played_removes_the_reply(db):
    bot, _, _ = make_bot(db, "This was never heard.")
    await turn(bot, "Hi")
    bot.on_interrupted("")
    assert bot.messages[-1].role == "user"
    assert bot.messages[-1].content == "Hi"


async def test_nothing_played_then_caller_speaks_keeps_roles_alternating(db):
    bot, llm, _ = make_bot(db, "Never heard.", "Okay.")
    await turn(bot, "Hi")
    bot.on_interrupted("  ")
    await turn(bot, "Hello?")
    assert roles(llm.requests[1]["messages"]) == ["system", "user", "assistant", "user"]
    assert llm.requests[1]["messages"][-1].content == "Hi Hello?"


async def test_tool_calls_survive_an_interruption(db):
    bot, _, _ = make_bot(
        db,
        uses(call("unlock_mailbox", address="evelyn@hartfamily.example"), text="One moment."),
        "All unlocked. Is there anything else?",
    )
    await turn(bot, "Please unlock it")
    bot.on_interrupted("One moment.")
    assert roles(bot.messages)[-4:] == ["user", "assistant", "tool", "assistant"]
    assert bot.messages[-3].tool_calls
    assert bot.messages[-3].content == ""
    assert bot.messages[-1].content == "One moment."


async def test_interruption_leaves_earlier_turns_alone(db):
    bot, _, _ = make_bot(db, "First reply.", "Second reply is long.")
    await turn(bot, "One")
    await turn(bot, "Two")
    bot.on_interrupted("Second")
    assert [m.content for m in bot.messages if m.role == "assistant"] == [
        GREETING,
        "First reply.",
        "Second",
    ]


async def test_interrupting_the_greeting(db):
    bot, _, _ = make_bot(db)
    bot.on_interrupted("Thanks for calling")
    assert bot.messages[-1].content == "Thanks for calling"


async def test_abandoned_stream_keeps_what_was_yielded_and_closes_the_llm(db):
    bot, llm, _ = make_bot(db, "Spoken part. Tail never generated by the engine. More text here.")
    stream = bot.respond("Hi")
    first = await anext(stream)
    await stream.aclose()
    assert llm.closed_streams == 1
    assert bot.messages[-1].content == first.display
    bot.on_interrupted("Spoken")
    assert bot.messages[-1].content == "Spoken"


async def test_abandoned_turn_can_be_followed_by_another(db):
    bot, _, _ = make_bot(db, "First. Second. Third.", "Okay then.")
    stream = bot.respond("Hi")
    await anext(stream)
    await stream.aclose()
    bot.on_interrupted("First.")
    assert [s.display for s in await turn(bot, "Stop")] == ["Okay then."]
    assert roles(bot.messages)[-4:] == ["user", "assistant", "user", "assistant"]


async def test_interruption_during_a_second_tool_round_keeps_both_tool_results(db):
    bot, _, events = make_bot(
        db,
        uses(call("list_plans"), text="Checking."),
        uses(call("get_invoice", invoice_id="FH-4419")),
        "Final answer goes here. And a second sentence.",
    )
    stream = bot.respond("Hi")
    await anext(stream)
    await anext(stream)
    await stream.aclose()
    bot.on_interrupted("Checking. Final")
    assert len(tool_events(events)) == 2
    assert roles(bot.messages)[-6:] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert bot.messages[-1].content == "Checking. Final"


# Ending the call ------------------------------------------------------------------------------


async def test_end_call_marks_the_bot_ended(db):
    bot, llm, events = make_bot(
        db, uses(call("end_call", reason="caller finished"), text="Goodbye now.")
    )
    segments = await turn(bot)
    assert [s.display for s in segments] == ["Goodbye now."]
    assert bot.ended is True
    assert bot.end_reason == "caller finished"
    assert len(llm.requests) == 1
    assert events[0].name == "end_call"


async def test_silent_end_call_still_says_goodbye(db):
    bot, _, _ = make_bot(db, uses(call("end_call")))
    assert [s.display for s in await turn(bot)] == [GOODBYE]
    assert bot.ended is True


async def test_goodbye_said_before_end_call_is_not_repeated(db):
    bot, _, _ = make_bot(db, uses(call("list_plans"), text="Bye."), uses(call("end_call")))
    assert [s.display for s in await turn(bot)] == ["Bye."]


async def test_end_call_runs_after_other_tools_in_the_same_reply(db):
    bot, _, events = make_bot(
        db,
        uses(
            call("create_ticket", subject="s", summary="x"),
            call("end_call"),
            text="A person will follow up.",
        ),
    )
    await turn(bot)
    assert [event.name for event in events] == ["create_ticket", "end_call"]
    assert bot.ended


async def test_nothing_happens_after_the_call_ended(db):
    bot, llm, _ = make_bot(db, uses(call("end_call")))
    await turn(bot)
    assert await turn(bot, "Hello?") == []
    assert len(llm.requests) == 1


def test_instructions_match_what_the_bot_uses(db):
    bot, _, _ = make_bot(db)
    assert support_instructions(db.today) == bot.messages[0].content


async def test_a_new_bot_has_not_ended(db):
    bot, _, _ = make_bot(db)
    assert bot.ended is False
