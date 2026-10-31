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


def test_long_quote_with_one_stumbled_word_still_matches(index):
    quote = "I am so sorry about that. Your refund of forty nine dollars is on its way to you."
    assert index.locate(quote) is not None


def test_long_quote_with_a_changed_fact_near_the_threshold_is_rejected():
    words = "please confirm the account number and the zip code"
    transcripts = {Speaker.BOT: transcript((words, 0.0))}
    assert TranscriptIndex(transcripts).locate("please confirm the card number and the pin") is None


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


def test_one_real_quote_keeps_the_verdict_and_drops_the_fake(index):
    judgment = verify_judgment(
        {
            "status": "fail",
            "reason": "disclosed",
            "quotes": ["refund of forty nine dollars", "you owe us a million dollars"],
        },
        index,
        quote_needed_for="fail",
    )
    assert judgment.status == "fail"
    assert len(judgment.evidence) == 1
    assert judgment.dropped_quotes == 1
    assert "dropped" in judgment.reason


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
