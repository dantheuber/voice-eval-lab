import pytest
from evalfakes import FakeLLM, transcript

from voicelab.core.models import Speaker, Word
from voicelab.eval.config import EvalConfig
from voicelab.eval.judging import ask_json, parse_json_reply, render_prompt, verify_judgment
from voicelab.eval.transcript import TranscriptIndex, estimate_words, render_transcript
from voicelab.eval.vad import Region
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


@pytest.mark.parametrize("name", ["judge_criterion", "caller_beat", "caller_facts"])
def test_prompt_templates_fill_without_leftovers(name):
    values = dict(
        context="c", tool_log="t", transcript="$49.99 said", criterion="x", goal="g",
        instruction="i", checkpoint="k", facts="f",
    )  # fmt: skip
    text = render_prompt(name, **values)
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
