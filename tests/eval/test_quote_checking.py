import json

import pytest
from evalfakes import FakeLLM, transcript

from voicelab.core.models import CallArtifact, CallMode, Speaker, SystemEvent, ToolCallEvent, Word
from voicelab.eval.config import EvalConfig
from voicelab.eval.judging import (
    JudgeItem,
    ToolLog,
    ask_items,
    ask_json,
    call_ending,
    call_record,
    order_of_events,
    parse_items,
    parse_json_reply,
    render_prompt,
    render_timeline,
    verify_judgment,
)
from voicelab.eval.normalize import Lexicon
from voicelab.eval.transcript import TranscriptIndex, estimate_words, render_transcript
from voicelab.eval.vad import Region
from voicelab.providers.adapters.openai_compat import LLMTimeoutError
from voicelab.providers.base import Transcript

CALLER_TEXT = "hi my card was charged twice for invoice FH-4419 last week"
BOT_TEXT = "I am sorry about that. Your refund of forty nine dollars is on its way to you."


@pytest.fixture
def transcripts() -> dict[Speaker, Transcript]:
    return {
        Speaker.CALLER: transcript((CALLER_TEXT, 0.0)),
        Speaker.BOT: transcript((BOT_TEXT, 10.0)),
    }


@pytest.fixture
def index(transcripts) -> TranscriptIndex:
    return TranscriptIndex(transcripts)


def test_exact_quote_is_found_with_speaker_and_times(index):
    evidence = index.locate("your refund of forty nine dollars")
    assert evidence.speaker == Speaker.BOT
    assert evidence.start_s == pytest.approx(12.0)
    assert evidence.end_s == pytest.approx(14.4)


def test_quote_matching_ignores_case_and_punctuation(index):
    assert index.locate('"Sorry about that!"') is not None


def test_evidence_uses_the_transcripts_words_not_the_judges(index):
    assert index.locate("SORRY ABOUT THAT").quote == "sorry about that."


def test_invented_quote_is_not_found(index):
    assert index.locate("your account has been closed permanently") is None


def test_paraphrase_is_not_found(index):
    assert index.locate("your money will be sent back to you soon") is None


def test_short_quote_needs_an_exact_match(index):
    assert index.locate("refund of fifty dollars") is None


def test_long_quote_with_one_changed_word_is_rejected(index):
    quote = "I am so sorry about that. Your refund of forty nine dollars is on its way to you."
    assert index.locate(quote) is None


def test_empty_quote_is_not_found(index):
    assert index.locate("  ...  ") is None


def test_preferred_speaker_wins_when_both_said_it():
    transcripts = {
        Speaker.CALLER: transcript(("thank you", 0.0)),
        Speaker.BOT: transcript(("thank you", 5.0)),
    }
    index = TranscriptIndex(transcripts)
    assert index.locate("thank you", Speaker.BOT).speaker == Speaker.BOT
    assert index.locate("thank you", Speaker.CALLER).speaker == Speaker.CALLER


def test_verified_failure_stands(index):
    judgment = verify_judgment(
        {"status": "fail", "reason": "disclosed", "quotes": ["refund of forty nine dollars"]},
        index,
        quote_needed_for="fail",
    )
    assert judgment.status == "fail"
    assert len(judgment.evidence) == 1
    assert judgment.dropped_quotes == 0


def test_failure_on_a_hallucinated_quote_becomes_unknown(index):
    judgment = verify_judgment(
        {"status": "fail", "reason": "it swore", "quotes": ["you are an idiot"]},
        index,
        quote_needed_for="fail",
    )
    assert judgment.status == "unknown"
    assert judgment.evidence == []
    assert judgment.dropped_quotes == 1
    assert "aren't in the transcript" in judgment.reason


def test_failure_with_no_quote_becomes_unknown_when_a_quote_is_needed(index):
    judgment = verify_judgment(
        {"status": "fail", "reason": "trust me", "quotes": []}, index, quote_needed_for="fail"
    )
    assert judgment.status == "unknown"
    assert "without quoting" in judgment.reason


def test_failure_needs_no_quote_when_it_rests_on_absence(index):
    judgment = verify_judgment(
        {"status": "fail", "reason": "never verified", "quotes": []},
        index,
        quote_needed_for="pass",
    )
    assert judgment.status == "fail"


def test_pass_on_a_hallucinated_quote_becomes_unknown_too(index):
    judgment = verify_judgment(
        {"status": "pass", "reason": "it verified", "quotes": ["what is your billing zip"]},
        index,
        quote_needed_for="pass",
    )
    assert judgment.status == "unknown"


def test_pass_citing_only_made_up_text_is_unknown_even_when_no_quote_is_needed(index):
    judgment = verify_judgment(
        {"status": "pass", "reason": "fine", "quotes": ["we never discussed money"]},
        index,
        quote_needed_for="fail",
    )
    assert judgment.status == "unknown"


def test_unrecognized_status_is_unknown(index):
    judgment = verify_judgment({"status": "maybe"}, index, quote_needed_for="fail")
    assert judgment.status == "unknown"


def test_malformed_quotes_are_ignored(index):
    judgment = verify_judgment(
        {"status": "pass", "reason": "ok", "quotes": [None, 7, ""]},
        index,
        quote_needed_for="fail",
    )
    assert judgment.status == "pass"


@pytest.mark.parametrize(
    "reply",
    [
        '{"status": "pass"}',
        'Sure!\n```json\n{"status": "pass"}\n```',
        '<think>hmm {not json}</think>{"status": "pass"}',
    ],
)
def test_json_replies_are_extracted_from_wrappers(reply):
    assert parse_json_reply(reply) == {"status": "pass"}


def test_reply_without_json_is_an_error():
    with pytest.raises(ValueError, match="no JSON"):
        parse_json_reply("I cannot help with that.")


async def test_unparseable_reply_gets_one_retry():
    replies = iter(["oops", {"status": "pass"}])
    llm = FakeLLM(lambda prompt: next(replies))
    payload, model = await ask_json(llm, "prompt", {}, EvalConfig())
    assert payload == {"status": "pass"}
    assert model == "fake-judge-v1"
    assert len(llm.prompts) == 2


async def test_two_unparseable_replies_raise():
    llm = FakeLLM(lambda prompt: "oops")
    with pytest.raises(ValueError, match="no usable JSON"):
        await ask_json(llm, "prompt", {}, EvalConfig())


def test_transcript_renders_turns_in_time_order():
    text = render_transcript(
        {
            Speaker.CALLER: transcript(("hello there", 0.0), ("one more thing", 6.0)),
            Speaker.BOT: transcript(("hi how can I help", 2.0)),
        }
    )
    assert text.splitlines() == [
        "[0.0s] CALLER: hello there",
        "[2.0s] BOT: hi how can I help",
        "[6.0s] CALLER: one more thing",
    ]


def test_overlapping_speech_stays_in_whole_turns():
    """Word times from two channels interleave when both talk, and that must not shred a turn."""
    text = render_transcript(
        {
            Speaker.CALLER: transcript(("wait I need to know the price", 1.0)),
            Speaker.BOT: transcript(("one moment let me check that", 1.1)),
        }
    )
    assert text.splitlines() == [
        "[1.0s] CALLER: wait I need to know the price",
        "[1.1s] BOT: one moment let me check that",
    ]


def test_a_long_gap_on_one_channel_starts_a_new_line():
    text = render_transcript({Speaker.BOT: transcript(("one moment", 0.0), ("all done", 9.0))})
    assert text.splitlines() == ["[0.0s] BOT: one moment", "[9.0s] BOT: all done"]


def test_words_are_estimated_across_speech_regions():
    words = estimate_words("aaaa bbbb cccc dddd", [Region(1, 2), Region(5, 6)], 10.0)
    assert [w.text for w in words] == ["aaaa", "bbbb", "cccc", "dddd"]
    assert words[0].start_s == pytest.approx(1.0)
    assert words[1].end_s == pytest.approx(2.0)
    assert words[2].start_s == pytest.approx(5.0)
    assert words[3].end_s == pytest.approx(6.0)


def test_estimated_words_fall_back_to_the_whole_channel():
    words = estimate_words("hello", [], 2.0)
    assert words == [Word(text="hello", start_s=0.0, end_s=2.0)]


@pytest.mark.parametrize("name", ["judge_criteria", "caller_beats", "caller_facts"])
def test_prompt_templates_fill_without_leftovers(name):
    record = render_prompt("call_record", context="c", ending="e", note="", timeline="$49.99 said")
    values = dict(record=record, criteria="x", goal="g", steps="s", facts="f")
    text = render_prompt(name, **values, transcript="$49.99 said")
    assert "$49.99 said" in text
    assert '"status"' in text


# Citations are exact, and one bad citation sinks the judgment ------------------------------------

REFUSAL = "I will not refund the payment to your account today"


def test_quote_dropping_a_negation_is_not_found():
    index = TranscriptIndex({Speaker.BOT: transcript((REFUSAL, 0.0))})
    assert index.locate("I will refund the payment to your account today") is None


def test_a_judgment_resting_on_a_dropped_negation_becomes_unknown():
    index = TranscriptIndex({Speaker.BOT: transcript((REFUSAL, 0.0))})
    judgment = verify_judgment(
        {
            "status": "pass",
            "reason": "it refunded",
            "quotes": ["I will refund the payment to your account today"],
        },
        index,
        quote_needed_for="pass",
    )
    assert judgment.status == "unknown"
    assert judgment.evidence == []


def test_an_exact_quote_of_the_same_speech_still_verifies():
    index = TranscriptIndex({Speaker.BOT: transcript((REFUSAL, 0.0))})
    assert index.locate("I WILL NOT refund, the payment!") is not None


def test_one_fabricated_quote_among_real_ones_makes_the_judgment_unknown(index):
    judgment = verify_judgment(
        {
            "status": "fail",
            "reason": "disclosed",
            "quotes": ["refund of forty nine dollars", "you owe us a million dollars"],
        },
        index,
        quote_needed_for="fail",
    )
    assert judgment.status == "unknown"
    assert judgment.dropped_quotes == 1
    assert [e.quote for e in judgment.evidence] == ["refund of forty nine dollars"]
    assert "aren't in the transcript" in judgment.reason


# Who said it matters -----------------------------------------------------------------------------


def two_sided() -> TranscriptIndex:
    return TranscriptIndex(
        {
            Speaker.CALLER: transcript(("please verify my billing zip", 0.0)),
            Speaker.BOT: transcript(("thanks for calling", 5.0)),
        }
    )


def test_caller_speech_cannot_prove_bot_compliance():
    judgment = verify_judgment(
        {"status": "pass", "reason": "verified", "quotes": ["verify my billing zip"]},
        two_sided(),
        quote_needed_for="pass",
        actor=Speaker.BOT,
    )
    assert judgment.status == "unknown"
    assert "bot" in judgment.reason


def test_caller_speech_cannot_prove_a_bot_violation():
    judgment = verify_judgment(
        {"status": "fail", "reason": "asked", "quotes": ["verify my billing zip"]},
        two_sided(),
        quote_needed_for="fail",
        actor=Speaker.BOT,
    )
    assert judgment.status == "unknown"


def test_the_actors_quote_stands_with_context_from_the_other_side():
    judgment = verify_judgment(
        {
            "status": "pass",
            "reason": "said so",
            "quotes": ["thanks for calling", "verify my billing zip"],
        },
        two_sided(),
        quote_needed_for="pass",
        actor=Speaker.BOT,
    )
    assert judgment.status == "pass"
    assert {e.speaker for e in judgment.evidence} == {Speaker.BOT, Speaker.CALLER}


def test_when_both_said_it_the_actors_copy_is_the_evidence():
    index = TranscriptIndex(
        {
            Speaker.CALLER: transcript(("thank you", 0.0)),
            Speaker.BOT: transcript(("thank you", 5.0)),
        }
    )
    judgment = verify_judgment(
        {"status": "pass", "reason": "r", "quotes": ["thank you"]},
        index,
        quote_needed_for="pass",
        actor=Speaker.BOT,
    )
    assert judgment.status == "pass"
    assert judgment.evidence[0].speaker == Speaker.BOT


def test_no_actor_allows_either_speaker():
    judgment = verify_judgment(
        {"status": "pass", "reason": "r", "quotes": ["verify my billing zip"]},
        two_sided(),
        quote_needed_for="pass",
    )
    assert judgment.status == "pass"


def test_estimated_words_stay_inside_the_regions_and_the_recording():
    regions = [Region(0, 0.3), Region(1, 1.3), Region(2, 2.3)]
    words = estimate_words("aaaa bbbb cccc dddd", regions, 2.3)
    assert words[-1].end_s == pytest.approx(2.3)
    assert all(w.end_s <= 2.3 + 1e-9 for w in words)
    assert all(w.end_s >= w.start_s for w in words)
    assert [w.start_s for w in words] == sorted(w.start_s for w in words)


def test_estimated_word_times_always_fall_inside_speech():
    regions = [Region(0.5, 0.7), Region(3, 3.1), Region(9, 9.9)]
    words = estimate_words("a bb ccc dddd eeeee ffffff ggggggg hh i jj", regions, 12.0)
    assert words[-1].end_s == pytest.approx(9.9)
    for word in words:
        for moment in (word.start_s, word.end_s):
            assert any(r.start_s - 1e-9 <= moment <= r.end_s + 1e-9 for r in regions)


# Quotes that differ the way an ASR and a judge differ ---------------------------------------------


@pytest.fixture
def spoken() -> TranscriptIndex:
    said = (
        "I won't refund that today. I'm sorry, we can't do it. "
        "The amount is $189.00 and your PIN didn't match. Your SSL certificate has been renewed."
    )
    return TranscriptIndex({Speaker.BOT: transcript((said, 0.0))}, Lexicon.load())


@pytest.mark.parametrize(
    "quote",
    [
        "i am sorry",
        "I'm sorry, we cannot do it",
        "will not refund that today",
        "WON'T REFUND THAT",
        "your pin did not match",
        "one hundred eighty nine dollars",
        "your S S L certificate",
    ],
)
def test_a_quote_may_differ_in_form(spoken, quote):
    assert spoken.locate(quote) is not None, quote


@pytest.mark.parametrize(
    "quote",
    [
        "will refund that today",
        "we can do it",
        "your PIN did match",
        "I refund that today",
        "the amount is $188.00",
        "the amount is $18.90",
        "Your TLS certificate",
    ],
)
def test_negations_and_numbers_have_to_match(spoken, quote):
    assert spoken.locate(quote) is None, quote


def test_a_quote_may_split_or_join_words_differently():
    index = TranscriptIndex({Speaker.CALLER: transcript(("my name is Tom OCarfor today", 0.0))})
    assert index.locate("Tom O'Car for").quote == "Tom OCarfor"
    assert index.locate("Tom O'Car for today").quote == "Tom OCarfor today"


def test_joining_words_cannot_hide_a_changed_letter():
    index = TranscriptIndex({Speaker.BOT: transcript(("I will cannot refund", 0.0))})
    assert index.locate("will can refund") is None


def test_an_ellipsis_joins_real_parts(spoken):
    assert spoken.locate("I'm sorry... the amount is $189.00").quote == "I'm sorry,"
    assert spoken.locate("I'm sorry\u2026 your PIN didn't match") is not None


def test_an_ellipsis_does_not_hide_an_invented_part(spoken):
    assert spoken.locate("I'm sorry... we will refund you") is None


def test_a_form_difference_still_keeps_the_transcripts_own_words(spoken):
    assert spoken.locate("i am sorry").quote == "I'm sorry,"


# Citing what the bot did -----------------------------------------------------------------------


def tooled(*events, **fields) -> CallArtifact:
    return CallArtifact(mode=CallMode.LIVE_BOTS, timeline=list(events), **fields)


def call_of(at_s: float, name: str = "create_ticket", result=None, **arguments) -> ToolCallEvent:
    return ToolCallEvent(at_s=at_s, name=name, arguments=arguments, result=result or {"ok": True})


def test_tool_calls_render_with_ids_and_times():
    [(at_s, line)] = ToolLog(tooled(call_of(3.5, subject="refund"))).timed_lines()
    assert at_s == 3.5
    assert line == '[3.5s] TOOL T1 create_ticket({"subject": "refund"}) -> {"ok": true}'


def test_a_repeated_tool_call_is_one_line_of_reference():
    log = ToolLog(tooled(call_of(3.0), call_of(4.0), call_of(5.0, subject="other")))
    lines = [line for _, line in log.timed_lines()]
    assert lines[1] == "[4.0s] TOOL T2 the same call and result as T1"
    assert lines[2].startswith("[5.0s] TOOL T3 create_ticket")


def test_no_tool_calls_says_so():
    assert "The bot made no tool calls." in call_record("c", tooled(), {})


def verified(at_s: float, ok: bool = True) -> ToolCallEvent:
    return call_of(at_s, "verify_identity", result={"verified": ok})


def test_the_record_states_when_a_change_came_before_verification():
    artifact = tooled(call_of(72.0, "renew_ssl"), verified(101.6))
    line = order_of_events(artifact)
    assert "first changed something at 72.0 s (T1, renew_ssl)" in line
    assert "verified the caller at 101.6 s (T2)" in line
    assert "before verifying" in line


def test_the_record_states_when_verification_came_first():
    line = order_of_events(tooled(verified(10.0), call_of(20.0, "renew_ssl")))
    assert "after verifying" in line


def test_a_failed_check_is_not_verification():
    line = order_of_events(tooled(verified(10.0, ok=False), call_of(20.0, "unlock_mailbox")))
    assert "never verified the caller" in line
    assert "first changed something at 20.0 s" in line


def test_a_change_that_failed_is_not_a_change():
    event = call_of(5.0, "renew_ssl")
    event.error = "no website"
    assert "changed nothing" in order_of_events(tooled(event, verified(9.0)))


def test_reads_are_not_changes():
    line = order_of_events(tooled(call_of(1.0, "get_invoice"), verified(9.0)))
    assert "changed nothing" in line


def test_a_call_with_no_tools_has_no_order_to_state():
    assert order_of_events(tooled()) == ""
    assert order_of_events(tooled(call_of(1.0, "end_call"))) == ""


def test_the_order_is_in_the_record():
    artifact = tooled(call_of(72.0, "renew_ssl"), verified(101.6))
    assert "before verifying" in call_record("c", artifact, {})


def test_opening_a_ticket_is_not_changing_the_account():
    line = order_of_events(tooled(call_of(5.0, "create_ticket"), verified(9.0)))
    assert "changed nothing" in line


def test_a_call_with_no_tool_log_says_so():
    assert "No tool log was recorded" in call_record("c", tooled(has_tool_log=False), {})


def test_tool_calls_sit_in_the_timeline_where_they_happened():
    transcripts = {
        Speaker.CALLER: transcript(("my domain is here", 1.0)),
        Speaker.BOT: transcript(("all done", 9.0)),
    }
    lines = render_timeline(transcripts, ToolLog(tooled(call_of(6.0)))).splitlines()
    assert lines[0].startswith("[1.0s] CALLER:")
    assert lines[1].startswith("[6.0s] TOOL T1 create_ticket")
    assert lines[2].startswith("[9.0s] BOT:")


def test_a_cited_tool_call_is_evidence_with_its_time(index):
    log = ToolLog(tooled(call_of(101.8, subject="refund")))
    judgment = verify_judgment(
        {"status": "pass", "reason": "ticket", "quotes": [], "tools": ["T1"]},
        index,
        quote_needed_for="pass",
        actor=Speaker.BOT,
        tools=log,
    )
    assert judgment.status == "pass"
    assert judgment.evidence[0].start_s == 101.8
    assert "create_ticket" in judgment.evidence[0].quote


def test_a_tool_call_that_does_not_exist_counts_as_an_invented_citation(index):
    log = ToolLog(tooled(call_of(1.0)))
    judgment = verify_judgment(
        {"status": "fail", "reason": "refunded", "quotes": [], "tools": ["T9"]},
        index,
        quote_needed_for="fail",
        actor=Speaker.BOT,
        tools=log,
    )
    assert judgment.status == "unknown"
    assert judgment.dropped_quotes == 1


@pytest.mark.parametrize(
    ("kind", "detail", "expected"),
    [
        ("caller_hung_up", "done", "The caller hung up at 90.0 s"),
        ("bot_ended", "refund processed", "The bot ended the call at 90.0 s"),
        ("max_turns", "", "The call reached its turn limit at 90.0 s"),
    ],
)
def test_the_judge_is_told_how_the_call_ended(kind, detail, expected):
    artifact = tooled(SystemEvent(at_s=90.0, kind=kind, detail=detail))
    assert call_ending(artifact).startswith(expected)


def test_an_ending_nobody_recorded_is_stated():
    assert "isn't recorded" in call_ending(tooled())


# Asking about several items at once -------------------------------------------------------------

ITEMS = [JudgeItem("alpha", "first"), JudgeItem("beta", "second"), JudgeItem("gamma", "third")]


def prompt_for(items: list[JudgeItem]) -> str:
    return "ask: " + " ".join(item.line for item in items)


def result(item: str, status: str = "pass") -> dict:
    return {"id": item, "quotes": [], "tools": [], "reason": "r", "status": status}


async def test_items_are_asked_together_in_one_request():
    llm = FakeLLM(lambda prompt: {"results": [result(i.id) for i in ITEMS]})
    replies = await ask_items(llm, ITEMS, prompt_for, EvalConfig())
    assert len(llm.prompts) == 1
    assert set(replies.payloads) == {"alpha", "beta", "gamma"}
    assert replies.model == "fake-judge-v1"


async def test_the_token_budget_grows_with_the_batch():
    llm = FakeLLM(lambda prompt: {"results": [result(i.id) for i in ITEMS]})
    config = EvalConfig()
    await ask_items(llm, ITEMS, prompt_for, config)
    assert llm.kwargs[0]["max_tokens"] >= 3 * config.judge_max_tokens


async def test_the_batch_size_is_configurable():
    config = EvalConfig(judge_batch_size=1)
    llm = FakeLLM(lambda prompt: {"status": "pass", "reason": "r", "quotes": [], "tools": []})
    replies = await ask_items(llm, ITEMS, prompt_for, config)
    assert len(llm.prompts) == 3
    assert set(replies.payloads) == {"alpha", "beta", "gamma"}


async def test_an_item_the_reply_left_out_is_asked_again_alone():
    def reply(prompt: str):
        if "third" in prompt and "first" not in prompt:
            return {"results": [result("gamma", "fail")]}
        return {"results": [result("alpha"), result("beta")]}

    llm = FakeLLM(reply)
    replies = await ask_items(llm, ITEMS, prompt_for, EvalConfig())
    assert len(llm.prompts) == 2
    assert replies.payloads["gamma"]["status"] == "fail"


async def test_a_cut_off_reply_keeps_the_items_it_finished():
    cut_off = json.dumps({"results": [result("alpha"), result("beta")]})[:-2] + ', {"id": "ga'

    def reply(prompt: str):
        return cut_off if "first" in prompt else {"results": [result("gamma")]}

    llm = FakeLLM(reply)
    replies = await ask_items(llm, ITEMS, prompt_for, EvalConfig())
    assert set(replies.payloads) == {"alpha", "beta", "gamma"}


async def test_an_item_that_fails_alone_too_is_an_error_for_that_item():
    def reply(prompt: str):
        if "first" in prompt:
            raise ConnectionError("server gone")
        return {"results": [result("beta"), result("gamma")]}

    llm = FakeLLM(reply)
    replies = await ask_items(llm, ITEMS, prompt_for, EvalConfig())
    assert "server gone" in replies.errors["alpha"]
    assert set(replies.payloads) == {"beta", "gamma"}


async def test_a_timeout_is_not_asked_again_one_item_at_a_time():
    def reply(prompt: str):
        raise LLMTimeoutError("judge timed out")

    llm = FakeLLM(reply)
    replies = await ask_items(llm, ITEMS, prompt_for, EvalConfig())
    assert len(llm.prompts) == 1
    assert set(replies.errors) == {"alpha", "beta", "gamma"}


def test_parse_items_reads_a_whole_reply_and_the_finished_part_of_a_cut_one():
    whole = {"results": [result("alpha"), result("beta")]}
    assert [i["id"] for i in parse_items(json.dumps(whole))] == ["alpha", "beta"]
    assert [i["id"] for i in parse_items(json.dumps(whole)[:-8])] == ["alpha"]
    assert parse_items("no json at all") == []


async def test_asking_again_after_a_cut_off_reply_allows_more_tokens():
    replies = iter(['{"status": "pa', '{"status": "pass"}'])
    llm = FakeLLM(lambda prompt: next(replies))
    payload, _ = await ask_json(llm, "prompt", {}, EvalConfig())
    assert payload == {"status": "pass"}
    assert llm.kwargs[1]["max_tokens"] > llm.kwargs[0]["max_tokens"]


# A quote has to end where a word ends ------------------------------------------------------------


@pytest.fixture
def exact() -> TranscriptIndex:
    said = "Your fee is $500 today and I can't do that. Invoice FH-4419 is at 94110 now."
    return TranscriptIndex({Speaker.BOT: transcript((said, 0.0))}, Lexicon.load())


@pytest.mark.parametrize(
    "quote",
    [
        "Your fee is $50",
        "fee is 50",
        "I can",
        "and I can",
        "Invoice FH-44",
        "is at 9411",
    ],
)
def test_a_quote_cannot_stop_inside_a_number_or_a_contraction(exact, quote):
    assert exact.locate(quote) is None, quote


def test_a_quote_cannot_start_inside_a_number_or_a_contraction():
    index = TranscriptIndex({Speaker.BOT: transcript(("the total is $500 today", 0.0))})
    assert index.locate("00 today") is None


def test_a_whole_number_and_a_whole_contraction_still_match(exact):
    assert exact.locate("Your fee is $500").quote == "Your fee is $500"
    assert exact.locate("I cannot do that").quote == "I can't do that."


def test_a_later_complete_match_is_found_past_a_partial_one():
    index = TranscriptIndex({Speaker.BOT: transcript(("I can't refund it but I can help", 0.0))})
    assert index.locate("I can").quote == "I can"


@pytest.mark.parametrize(
    ("said", "quote"),
    [
        ("It renews on October 12th. one moment please", "renews on October 12th."),
        ("This is 9702. 10-7-3-1-6. I know", "This is 9702."),
        ("This is 9702. 10-7-3-1-6. I know", "10-7-3-1-6."),
        ("your card ending in five One moment please", "your card ending in five"),
    ],
)
def test_a_number_ends_where_its_sentence_ends(said, quote):
    index = TranscriptIndex({Speaker.BOT: transcript((said, 0.0))})
    assert index.locate(quote) is not None, quote


def test_part_of_a_spoken_number_is_not_the_number():
    index = TranscriptIndex({Speaker.BOT: transcript(("that is forty nine dollars today", 0.0))})
    assert index.locate("nine dollars") is None
    assert index.locate("forty nine dollars") is not None


def test_word_breaks_cannot_reach_inside_a_number():
    index = TranscriptIndex({Speaker.BOT: transcript(("call 500 now", 0.0))})
    assert index.locate("call 50") is None


def test_a_judge_cannot_cite_part_of_a_number_as_evidence(exact):
    judgment = verify_judgment(
        {"status": "fail", "reason": "overcharged", "quotes": ["Your fee is $50"]},
        exact,
        quote_needed_for="fail",
    )
    assert judgment.status == "unknown"
    assert judgment.dropped_quotes == 1
