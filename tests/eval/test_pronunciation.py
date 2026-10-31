import pytest
from evalfakes import timed_words, write_call

from voicelab.core.models import (
    CallArtifact,
    CallMode,
    Persona,
    Scenario,
    Speaker,
    ToolCallEvent,
    Utterance,
    VoiceRef,
)
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput, Skip
from voicelab.eval.evaluators.pronunciation import LIMITATION, PronunciationEvaluator
from voicelab.eval.normalize import Lexicon, align, sounds_alike, tokenize
from voicelab.providers.base import Transcript


def norms(text: str, **kwargs) -> list[str]:
    return [t.norm for t in tokenize(text.split(), **kwargs)]


def critical(text: str, **kwargs) -> dict[str, str]:
    return {t.norm: t.critical for t in tokenize(text.split(), **kwargs) if t.critical}


def matches(ref: str, hyp: str, **kwargs) -> bool:
    aligned = align(tokenize(ref.split(), **kwargs), tokenize(hyp.split(), **kwargs))
    return all(a.matched for a in aligned)


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("$49.99", "forty nine dollars and ninety nine cents"),
        ("$49.99", "forty nine dollars ninety nine"),
        ("$100.00", "one hundred dollars"),
        ("$100", "a hundred dollars"),
        ("$1,200.00", "one thousand two hundred dollars"),
        ("$1,200.00", "twelve hundred dollars"),
        ("2024", "twenty twenty four"),
        ("2024", "two thousand twenty four"),
        ("1999", "nineteen ninety nine"),
        ("555-0101", "five five five oh one oh one"),
        ("(503) 555-0101", "five oh three five five five oh one oh one"),
        ("4821", "four eight two one"),
        ("97214", "nine seven two one four"),
        ("24", "twenty four"),
        ("105", "one hundred and five"),
        ("993", "nine nine three"),
        ("1st", "first"),
        ("FH-4419", "FH 4419"),
        ("FH-4419", "F H four four one nine"),
        ("FH-4419", "fh-4419"),
        ("SSL", "S S L"),
        ("SSL", "s.s.l."),
        ("fernhill.example", "fernhill dot example"),
        ("billing.fernhill.example", "billing dot fern hill dot example"),
        ("dana@whitfieldlandscaping.example", "dana at whitfield landscaping dot example"),
        ("https://fernhill.example/help", "fernhill dot example slash help"),
        ("can't", "cannot"),
        ("you're", "you are"),
        ("email", "e mail"),
        ("web site", "website"),
        ("Hello, there!", "hello there"),
    ],
)
def test_written_and_spoken_forms_match(written, spoken):
    assert matches(written, spoken), (norms(written), norms(spoken))


@pytest.mark.parametrize(
    ("written", "heard"),
    [
        ("$49.99", "forty nine dollars and ninety eight cents"),
        ("$49.99", "fifty nine dollars and ninety nine cents"),
        ("$49.99", "forty nine ninety nine"),
        ("4821", "four eight two two"),
        ("FH-4419", "FH 4491"),
        ("fernhill.example", "fern hills dot example"),
    ],
)
def test_real_differences_do_not_match(written, heard):
    assert not matches(written, heard)


@pytest.mark.parametrize(
    ("text", "norm", "kind"),
    [
        ("It costs $49.99 today", "4", "money"),
        ("Your PIN is 4821", "4", "number"),
        ("about forty nine people", "4", "number"),
        ("Visit fernhill.example today", "fernhill", "url"),
        ("Email help@fernhill.example now", "help", "url"),
        ("Invoice FH-4419 is paid", "fh", "id"),
        ("We use SSL here", "ssl", "acronym"),
    ],
)
def test_critical_tokens_are_classified(text, norm, kind):
    assert critical(text)[norm] == kind


def test_ordinary_words_are_not_critical():
    assert critical("please hold while I check that for you") == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("on October 12th. one moment", ["on", "october", "1", "2", "1", "moment"]),
        ("ending in five One moment", ["ending", "in", "5", "1", "moment"]),
    ],
)
def test_a_number_stops_at_the_end_of_its_sentence(text, expected):
    tokens = tokenize(text.split())
    assert [t.norm for t in tokens] == expected
    assert len({(t.start, t.end) for t in tokens if t.norm.isdigit()}) == 2


def test_commas_inside_a_spoken_number_do_not_end_it():
    tokens = tokenize("nine, seven, seven, zero, one.".split())
    assert "".join(t.norm for t in tokens) == "97701"
    assert len({(t.start, t.end) for t in tokens}) == 1


def test_incidental_number_words_are_not_critical():
    assert critical("one moment please, just a second") == {}


def test_sentence_start_capitals_are_not_names_unless_known():
    assert critical("Thanks for calling.") == {}
    assert critical("Dana, thanks for calling.", known_names=["Dana"])["dana"] == "name"


def test_watch_phrases_are_critical():
    assert critical("the Whitfield Landscaping site", watch=["whitfield landscaping"]) == {
        "whitfield": "watch",
        "landscaping": "watch",
    }


def test_lexicon_folds_spoken_forms_on_both_sides():
    lexicon = Lexicon({"IMAP": ["eye map"], "Fernhill": ["fern hill", "fern hil"]})
    assert matches("IMAP on Fernhill", "eye map on fern hill", lexicon=lexicon)


def test_distinctive_spoken_form_is_critical_but_the_plain_word_is_not():
    lexicon = Lexicon({"IMAP": ["eye map"]})
    assert critical("the eye map server", lexicon=lexicon) == {"imap": "lexicon"}
    assert critical("the imap server", lexicon=lexicon) == {}


@pytest.mark.parametrize("text", ["who is calling", "we can text you", "you win this one"])
def test_lexicon_forms_that_are_everyday_english_are_not_critical(text):
    assert critical(text, lexicon=Lexicon.load()) == {}


def test_missing_lexicon_file_is_empty(tmp_path):
    lexicon = Lexicon.load(tmp_path / "absent.yaml")
    assert norms("hello there", lexicon=lexicon) == ["hello", "there"]


def test_real_lexicon_file_loads_terms_units_and_endings():
    lexicon = Lexicon.load()
    assert "example" in lexicon.tlds
    assert matches("STARTTLS over IMAP", "start T L S over eye map", lexicon=lexicon)
    assert matches("5 GB", "five gigabytes", lexicon=lexicon)
    assert matches("5GB", "five gigabytes", lexicon=lexicon)
    assert matches("1 KB", "one kilobyte", lexicon=lexicon)
    assert matches("Okafor", "oh kah for", lexicon=lexicon)
    assert matches("nguyenauto.example", "win auto dot example", lexicon=lexicon)


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("two thousand twenty four", "2024"),
        ("twenty twenty four", "2024"),
        ("two thousand five", "2005"),
        ("twenty oh five", "2005"),
        ("one thousand two hundred thirty four", "1234"),
        ("twelve hundred", "1200"),
        ("two thousand", "2000"),
        ("nineteen hundred", "1900"),
        ("one hundred five", "105"),
        ("two hundred fifteen", "215"),
        ("five oh three", "503"),
        ("forty nine", "49"),
        ("forty nine ninety nine", "4999"),
        ("five dollars and five cents", "5.05"),
        ("one dollar and five cents", "1.05"),
        ("ninety nine cents", "0.99"),
        ("five cents", "0.05"),
        ("one point five", "1.50"),
        ("twenty four seven", "247"),
        ("thirty first", "31"),
        ("twentieth", "20"),
    ],
)
def test_spoken_numbers_read_as_the_digits_people_mean(words, digits):
    assert "".join(norms(words)) == digits


@pytest.mark.parametrize(
    ("written", "digits"),
    [("$5.05", "5.05"), ("$0.99", "0.99"), ("$0.05", "0.05"), ("$100.00", "100"), ("3:00", "3")],
)
def test_written_numbers_drop_what_speech_leaves_out(written, digits):
    assert "".join(norms(written)) == digits


@pytest.mark.parametrize(
    ("ref", "heard"),
    [("$49.99", "$449.99"), ("4821", "48211"), ("FH-4419", "FH-44199"), ("$49.99", "$499.99")],
)
def test_extra_digits_inside_a_number_are_a_mismatch(ref, heard):
    assert not matches(ref, heard)


def test_extra_words_next_to_a_critical_token_are_ignored():
    assert matches("costs $49.99 today", "it costs $49.99 right today")


def test_alignment_reports_substitutions_and_deletions():
    aligned = align(tokenize("the quick brown fox".split()), tokenize("the slow fox".split()))
    assert [a.matched for a in aligned] == [True, False, False, True]
    assert aligned[1].hyp[0].norm == "slow"
    assert aligned[2].hyp == ()


def test_alignment_ignores_extra_heard_words():
    aligned = align(tokenize("hello there".split()), tokenize("um hello there friend".split()))
    assert all(a.matched for a in aligned)


def bot_utterance(text: str, start: float, end: float, **kwargs) -> Utterance:
    return Utterance(speaker=Speaker.BOT, start_s=start, end_s=end, intended_text=text, **kwargs)


async def evaluate(
    store, artifact, heard: str, *, confidence=0.95, lexicon=None
) -> EvaluatorOutput:
    artifact, paths = write_call(store, artifact, duration_s=30.0)
    words = timed_words(heard, 1.0, confidence=confidence)
    ctx = EvalContext(artifact, paths, EvalDeps(lexicon_path=lexicon))
    ctx.transcripts[Speaker.BOT] = Transcript(text=heard, words=words)
    out = EvaluatorOutput()
    await PronunciationEvaluator().run(ctx, out)
    return out


def call(*utterances: Utterance, mode=CallMode.LIVE_BOTS, **kwargs) -> CallArtifact:
    return CallArtifact(mode=mode, timeline=list(utterances), **kwargs)


async def test_clean_call_has_no_findings_and_notes_the_asr_limitation(store):
    artifact = call(bot_utterance("Your total is $49.99 today.", 1.0, 5.0))
    out = await evaluate(
        store, artifact, "your total is forty nine dollars and ninety nine cents today"
    )
    assert out.findings == []
    assert out.metrics["token_accuracy"] == 1.0
    assert out.metrics["critical_mismatches"] == 0
    assert out.metrics["limitation"] == LIMITATION


async def test_wrong_amount_is_a_major_finding_with_timing(store):
    artifact = call(bot_utterance("Your total is $49.99 today.", 1.0, 5.0))
    out = await evaluate(
        store, artifact, "your total is forty nine dollars and fifty nine cents today"
    )
    assert len(out.findings) == 1
    finding = out.findings[0]
    assert finding.category == "critical_token_asr_only"
    assert finding.severity == "major"
    assert finding.advisory is True
    assert "$49.99" in finding.title
    assert "forty nine dollars and fifty nine cents" in finding.detail
    assert finding.evidence[0].start_s == pytest.approx(2.2)
    assert finding.evidence[0].speaker == Speaker.BOT


async def test_each_kind_of_critical_mismatch_is_major(store):
    scenario = Scenario(title="t", goal="g", facts={"name": "Dana Whitfield"})
    artifact = call(
        bot_utterance("Invoice FH-4419 for Dana on fernhill.example uses SSL.", 1.0, 9.0),
        scenario_snapshot=scenario,
    )
    out = await evaluate(
        store, artifact, "invoice FH 4491 for Dina on fernhill dot sample uses TLS"
    )
    kinds = out.metrics["critical_by_kind"]
    assert kinds["id"]["mismatched"] == 1
    assert kinds["name"]["mismatched"] == 1
    assert kinds["url"]["mismatched"] == 1
    assert kinds["acronym"]["mismatched"] == 1
    critical = [f for f in out.findings if f.category.startswith("critical_token")]
    assert len(critical) == 4
    assert all(f.severity == "major" for f in critical)


async def test_watch_terms_are_held_to_the_critical_standard(store):
    scenario = Scenario(title="t", goal="g", pronunciation_watch=["okafor"])
    artifact = call(bot_utterance("thanks mister okafor", 1.0, 3.0), scenario_snapshot=scenario)
    out = await evaluate(store, artifact, "thanks mister oak affair")
    assert [f.severity for f in out.findings] == ["major"]
    assert out.metrics["critical_by_kind"]["watch"]["mismatched"] == 1


async def test_persona_name_is_critical_even_at_sentence_start(store):
    persona = Persona(name="Dana Whitfield", voice=VoiceRef(voice_id="v"))
    artifact = call(
        bot_utterance("Dana, I can help with that.", 1.0, 4.0), persona_snapshot=persona
    )
    out = await evaluate(store, artifact, "dina I can help with that")
    assert [f.category for f in out.findings] == ["critical_token_asr_only"]


async def test_ordinary_word_mismatch_is_advisory(store):
    artifact = call(bot_utterance("please hold while I check that for you now", 1.0, 5.0))
    out = await evaluate(store, artifact, "please hold wile I check that for you now")
    assert len(out.findings) == 1
    assert out.findings[0].category == "word_mismatch"
    assert out.findings[0].advisory is True
    assert out.findings[0].severity == "info"
    assert 'heard as "wile"' in out.findings[0].detail


async def test_many_ordinary_mismatches_rise_to_minor_but_stay_advisory(store):
    artifact = call(bot_utterance("please hold while I check that", 1.0, 3.0))
    out = await evaluate(store, artifact, "fleas fold wile I check that")
    assert out.findings[0].severity == "minor"
    assert out.findings[0].advisory is True


async def test_low_confidence_on_a_critical_word_is_flagged_as_a_clip(store):
    artifact = call(bot_utterance("Your PIN is 4821 today", 1.0, 4.0))
    out = await evaluate(store, artifact, "your pin is 4821 today", confidence=0.3)
    [flagged] = [f for f in out.findings if f.category == "low_confidence_critical"]
    assert [e.quote for e in flagged.evidence] == ["pin", "4821"]
    assert flagged.advisory


async def test_cut_off_utterance_compares_against_what_was_spoken(store):
    artifact = call(
        bot_utterance(
            "Your balance is $49.99 and your renewal is $14.99 next month.",
            1.0,
            3.0,
            interrupted=True,
            spoken_text="Your balance is $49.99 and",
        )
    )
    out = await evaluate(
        store, artifact, "your balance is forty nine dollars and ninety nine cents and"
    )
    assert out.findings == []


async def test_cut_off_without_spoken_text_ignores_the_unspoken_tail(store):
    artifact = call(
        bot_utterance(
            "Your balance is $49.99 and your renewal is $14.99 next.", 1.0, 3.0, interrupted=True
        )
    )
    out = await evaluate(
        store, artifact, "your balance is forty nine dollars and ninety nine cents"
    )
    assert out.findings == []
    assert out.metrics["critical_tokens"] == 1


async def test_word_missing_from_the_recording_is_a_mismatch(store):
    artifact = call(bot_utterance("Your total is $49.99 today", 1.0, 4.0))
    out = await evaluate(store, artifact, "your total is today")
    finding = out.findings[0]
    assert finding.category == "critical_token_asr_only"
    assert "nothing" in finding.detail
    assert finding.evidence[0].start_s == pytest.approx(2.2)
    assert finding.evidence[0].end_s == pytest.approx(2.2)


async def test_utterances_compare_across_timestamp_drift(store):
    # ASR times here run well past the utterance bounds, which a per-utterance window would fail.
    artifact = call(
        bot_utterance("Your total is $49.99.", 0.0, 1.0),
        bot_utterance("Your pin is 4821.", 1.0, 2.0),
    )
    out = await evaluate(store, artifact, "your total is 49.99 your pin is 4821")
    assert out.findings == []


async def test_upload_gets_only_the_confidence_pass(store):
    artifact = call(
        Utterance(speaker=Speaker.BOT, start_s=1.0, end_s=4.0),
        mode=CallMode.UPLOAD,
        has_intended_text=False,
    )
    out = await evaluate(store, artifact, "your invoice is FH-4419", confidence=0.2)
    assert out.metrics["compared"] is False
    assert out.metrics["low_confidence_flags"] == 1
    finding = out.findings[0]
    assert finding.category == "low_confidence"
    assert finding.advisory is True
    assert finding.severity == "minor"
    assert out.metrics["limitation"] == LIMITATION


async def test_confidence_pass_separates_doubtful_stretches_and_ranks_critical_first(store):
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_intended_text=False)
    artifact, paths = write_call(store, artifact, duration_s=10.0)
    words = timed_words("please hold FH-4419 and then thanks", confidence=0.95)
    words[0].confidence = 0.2
    words[2].confidence = 0.4
    words[5].confidence = 0.1
    ctx = EvalContext(artifact, paths, EvalDeps())
    ctx.transcripts[Speaker.BOT] = Transcript(text="", words=words)
    out = EvaluatorOutput()
    await PronunciationEvaluator().run(ctx, out)
    assert [f.evidence[0].quote for f in out.findings] == ["FH-4419", "thanks", "please"]
    assert [f.severity for f in out.findings] == ["minor", "info", "info"]


async def test_human_call_runs_comparison_and_confidence_pass(store):
    artifact = call(bot_utterance("please hold", 1.0, 2.0), mode=CallMode.HUMAN)
    out = await evaluate(store, artifact, "please hold", confidence=0.2)
    assert out.metrics["compared"] is True
    assert out.metrics["low_confidence_flags"] == 1


async def test_asr_without_confidence_says_so(store):
    artifact = call(mode=CallMode.UPLOAD)
    out = await evaluate(store, artifact, "please hold", confidence=None)
    assert "no word confidence" in out.metrics["low_confidence_pass"]


async def test_lexicon_file_folds_spoken_forms_during_comparison(store, tmp_path):
    path = tmp_path / "lexicon.yaml"
    path.write_text("terms:\n  STARTTLS: start t l s\n")
    artifact = call(bot_utterance("We use STARTTLS on that port", 1.0, 4.0))
    out = await evaluate(store, artifact, "we use start t l s on that port", lexicon=path)
    assert out.findings == []


async def test_skips_without_a_bot_transcript(store):
    artifact, paths = write_call(store, call(), duration_s=1.0)
    with pytest.raises(Skip, match="independent transcript"):
        await PronunciationEvaluator().run(
            EvalContext(artifact, paths, EvalDeps()), EvaluatorOutput()
        )


# An ASR's word against the script is suspicion. Two engines agreeing is evidence -----------------


async def compare_with_live(store, heard_eval, heard_live, *, providers=None, model="eval-stt"):
    artifact = call(
        bot_utterance("Your total is $49.99 today", 1.0, 5.0, heard_text=heard_live),
        providers=providers or {},
    )
    artifact, paths = write_call(store, artifact, duration_s=10.0)
    ctx = EvalContext(artifact, paths, EvalDeps())
    words = timed_words(heard_eval, 1.0)
    ctx.transcripts[Speaker.BOT] = Transcript(text=heard_eval, words=words, model=model)
    out = EvaluatorOutput()
    await PronunciationEvaluator().run(ctx, out)
    return out


def critical_findings(out: EvaluatorOutput):
    return [f for f in out.findings if f.category.startswith("critical_token")]


async def test_asr_mismatch_alone_is_advisory_even_at_low_confidence(store):
    artifact = call(bot_utterance("Your total is $49.99 today", 1.0, 5.0))
    out = await evaluate(store, artifact, "your total is $49.98 today", confidence=0.2)
    [finding] = critical_findings(out)
    assert finding.category == "critical_token_asr_only"
    assert finding.severity == "major"
    assert finding.advisory is True
    assert "no second engine" in finding.detail


@pytest.mark.parametrize(
    "live",
    ["your total is $49.98 today", "your total is forty nine dollars and ninety eight cents today"],
)
async def test_two_engines_hearing_the_same_wrong_token_is_corroborated(store, live):
    out = await compare_with_live(store, "your total is $49.98 today", live)
    [finding] = critical_findings(out)
    assert finding.category == "critical_token_corroborated"
    assert finding.severity == "major"
    assert finding.advisory is False
    assert "both" in finding.detail
    assert out.metrics["critical_corroborated"] == 1


async def test_live_engine_hearing_the_script_does_not_corroborate(store):
    out = await compare_with_live(store, "your total is $49.98 today", "your total is $49.99 today")
    [finding] = critical_findings(out)
    assert finding.category == "critical_token_asr_only"
    assert finding.advisory is True
    assert "live engine heard it as scripted" in finding.detail


async def test_engines_hearing_different_wrong_tokens_do_not_corroborate(store):
    out = await compare_with_live(store, "your total is $49.98 today", "your total is $49.97 today")
    [finding] = critical_findings(out)
    assert finding.category == "critical_token_asr_only"
    assert finding.advisory is True


async def test_both_engines_hearing_nothing_is_not_a_wrong_token(store):
    out = await compare_with_live(store, "your total is today", "your total is today")
    assert critical_findings(out)[0].advisory is True


async def test_the_same_engine_twice_is_not_independent(store):
    out = await compare_with_live(
        store,
        "your total is $49.98 today",
        "your total is $49.98 today",
        providers={"stt_live": "local/whisper-large"},
        model="whisper-large",
    )
    [finding] = critical_findings(out)
    assert finding.category == "critical_token_asr_only"
    assert finding.advisory is True
    assert "same engine" in finding.detail


async def test_different_recorded_engines_can_corroborate(store):
    out = await compare_with_live(
        store,
        "your total is $49.98 today",
        "your total is $49.98 today",
        providers={"stt_live": "local/parakeet"},
        model="whisper-large",
    )
    assert critical_findings(out)[0].category == "critical_token_corroborated"


# Money reads the same however it's written, and different amounts stay different -----------------


@pytest.mark.parametrize(
    ("written", "heard"),
    [
        ("$5.00", "5.00"),
        ("$5.00", "$5"),
        ("$5", "5.00"),
        ("$5.00", "five dollars"),
        ("$5.00", "5 dollars"),
        ("5.00", "five dollars"),
        ("$49.90", "49.90"),
        ("$49.90", "49.9"),
        ("$49.90", "forty nine dollars and ninety cents"),
        ("$49.90", "49 dollars and 90 cents"),
        ("$0.99", "99 cents"),
        ("$0.99", "ninety nine cents"),
        ("$0.99", "0.99"),
        ("$0.05", "five cents"),
        ("$1,200.00", "1200.00"),
        ("$1,200.00", "1,200 dollars"),
        ("$1,200.50", "one thousand two hundred dollars and fifty cents"),
        ("1.5", "one point five"),
        ("5.5%", "five point five percent"),
    ],
)
def test_the_same_amount_matches_in_every_format(written, heard):
    assert matches(written, heard), (norms(written), norms(heard))


@pytest.mark.parametrize(
    ("written", "heard"),
    [
        ("$5.00", "$500"),
        ("$5.50", "$550"),
        ("$550", "$5.50"),
        ("$5.50", "five hundred fifty dollars"),
        ("$5.05", "$5.50"),
        ("$49.90", "$49.09"),
        ("$49.99", "4999"),
        ("$5", "$50"),
        ("$0.50", "$50"),
        ("5.05", "5.5"),
    ],
)
def test_genuinely_different_amounts_stay_different(written, heard):
    assert not matches(written, heard), (norms(written), norms(heard))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.5 2", ["1", ".", "5", "0", "2"]),
        ("$5.50 and 20", ["5", ".", "5", "0", "and", "2", "0"]),
        ("$5 and $20", ["5", "and", "2", "0"]),
    ],
)
def test_a_finished_amount_does_not_swallow_the_next_number(text, expected):
    assert norms(text) == expected


@pytest.mark.parametrize(
    "said",
    [
        "F as in Frank H as in house 4 4 1 9",
        "F, as in frank. H, as in house. 4 4 1 9",
        "F as in Frank H 4 4 1 9",
    ],
)
def test_spelling_that_agrees_with_itself_is_the_same_id(said):
    assert norms(said) == norms("F H 4 4 1 9")


@pytest.mark.parametrize(
    "said",
    ["F as in Sam H as in house 4 4 1 9", "S as in Frank H as in house 4 4 1 9"],
)
def test_a_letter_and_anchor_that_disagree_match_neither_reading(said):
    assert norms(said) != norms("F H 4 4 1 9")
    assert norms(said) != norms("S H 4 4 1 9")


def test_a_contradicted_letter_is_still_part_of_its_id():
    tokens = tokenize("F as in Sam H as in house 4 4 1 9".split())

    assert [t.critical for t in tokens] == ["id"] * 6
    assert (tokens[0].start, tokens[0].end) == (0, 12)  # the id spans every word said for it


@pytest.mark.parametrize("anchor", ["x-ray", "X-ray.", "xray"])
def test_a_hyphenated_anchor_is_one_word(anchor):
    assert norms(f"X as in {anchor} Y as in yellow 9 0 7 7") == norms("X Y 9 0 7 7")


# What counts as a name ---------------------------------------------------------------------------


def test_a_known_name_is_critical():
    assert critical("Thanks for calling, Dana.", known_names=["Dana"]) == {"dana": "name"}


@pytest.mark.parametrize(
    "text",
    [
        "You are on the Website Business plan.",
        "The next plan up is Email Plus at the same rate.",
        "Well, I'm sorry about that.",
        "Yes, and I've checked it.",
    ],
)
def test_capitals_alone_do_not_make_a_name(text):
    assert critical(text) == {}


async def test_a_name_from_a_tool_result_is_critical(store):
    found = ToolCallEvent(
        at_s=1.0, name="find_customer", result={"customer_id": "c1", "name": "Marcus Bell"}
    )
    artifact = call(bot_utterance("Thanks Marcus, one moment.", 2.0, 4.0))
    artifact.timeline.insert(0, found)
    out = await evaluate(store, artifact, "thanks markets one moment")
    assert out.metrics["critical_by_kind"]["name"]["mismatched"] == 1


async def test_a_plan_name_in_a_tool_result_is_not_a_person(store):
    plans = ToolCallEvent(
        at_s=1.0, name="list_plans", result={"plans": [{"plan_id": "p1", "name": "Email Plus"}]}
    )
    artifact = call(bot_utterance("The plan is Email Plus today.", 2.0, 4.0))
    artifact.timeline.insert(0, plans)
    out = await evaluate(store, artifact, "the plan is email plus today")
    assert "name" not in out.metrics["critical_by_kind"]


async def test_only_name_facts_are_names(store):
    scenario = Scenario(
        title="t", goal="g", facts={"name": "Tom Okafor", "reason": "Moving the business"}
    )
    artifact = call(bot_utterance("Moving is hard, Tom.", 2.0, 4.0), scenario_snapshot=scenario)
    out = await evaluate(store, artifact, "moving is hard tom")
    assert out.metrics["critical_by_kind"]["name"]["total"] == 1


# Sound alikes ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("said", "heard"),
    [
        ("Sofia", "Sophia"),
        ("Aisha", "Ayesha"),
        ("Rahman", "Raman"),
        ("okafor", "ocawfor"),
        ("rahmanlawdotexample", "ramenlawdotexample"),
        ("marinocafe", "merinocafe"),
        ("Marino", "Marino"),
    ],
)
def test_spellings_of_one_sound_are_alike(said, heard):
    assert sounds_alike(said, heard)


@pytest.mark.parametrize(
    ("said", "heard"),
    [
        ("Dana", "Dina"),
        ("okafor", "oakaffair"),
        ("hill", "hall"),
        ("ssl", "ss"),
        ("mx", "ms"),
        ("fh4419", "fh4491"),
        ("mail2example", "mail3example"),
        ("11", "1"),
        ("$11", "$1"),
        ("100", "10"),
        ("room 22", "room 2"),
        ("marcus", "markets"),
        ("Marino", "Marina"),
        ("", "Sophia"),
    ],
)
def test_real_differences_are_not_alike(said, heard):
    assert not sounds_alike(said, heard)


async def test_both_engines_spelling_a_name_another_way_is_no_finding(store):
    scenario = Scenario(title="t", goal="g", facts={"name": "Sofia Marino"})
    artifact = call(
        bot_utterance("I found your account, Sofia.", 1.0, 4.0, heard_text="found account Sophia."),
        scenario_snapshot=scenario,
    )
    out = await evaluate(store, artifact, "I found your account, Sophia.")
    assert critical_findings(out) == []
    assert out.metrics["critical_mismatches"] == 0
    assert out.metrics["sound_alike"] == 1


async def test_a_misheard_name_that_sounds_different_is_still_caught(store):
    scenario = Scenario(title="t", goal="g", facts={"name": "Dana Whitfield"})
    artifact = call(
        bot_utterance("Thanks Dana, one moment.", 1.0, 4.0, heard_text="Thanks Dina, one moment."),
        scenario_snapshot=scenario,
    )
    out = await evaluate(store, artifact, "thanks Dina one moment")
    [finding] = critical_findings(out)
    assert finding.category == "critical_token_corroborated"


async def test_a_respelled_term_heard_as_one_word_matches(store):
    """The TTS text says "win auto", and an ASR writes what it heard as one word."""
    scenario = Scenario(title="t", goal="g", pronunciation_watch=["nguyenauto.example"])
    artifact = call(
        bot_utterance("The domain is win auto dot example.", 1.0, 4.0),
        scenario_snapshot=scenario,
    )
    out = await evaluate(store, artifact, "The domain is winauto.example.")
    assert critical_findings(out) == []


async def test_joined_and_spaced_letters_with_punctuation_match(store):
    scenario = Scenario(title="t", goal="g", pronunciation_watch=["SSL"])
    artifact = call(
        bot_utterance("Your S S L certificate.", 1.0, 3.0, heard_text="Your SSL certificate."),
        scenario_snapshot=scenario,
    )
    out = await evaluate(store, artifact, "Your SSL. certificate")
    assert critical_findings(out) == []


async def test_a_dropped_letter_both_engines_heard_stays_corroborated(store):
    """Seen on a real call: the TTS said "S S certificate", and both engines wrote SS."""
    scenario = Scenario(title="t", goal="g", pronunciation_watch=["SSL"])
    artifact = call(
        bot_utterance("Your S S L certificate.", 1.0, 3.0, heard_text="Your SS certificate."),
        scenario_snapshot=scenario,
    )
    out = await evaluate(store, artifact, "Your SS certificate")
    [finding] = critical_findings(out)
    assert finding.category == "critical_token_corroborated"


async def test_a_wrong_digit_is_never_excused_as_a_sound_alike(store):
    artifact = call(bot_utterance("Invoice FH-4419 is paid.", 1.0, 4.0))
    out = await evaluate(store, artifact, "invoice FH-4491 is paid")
    assert len(critical_findings(out)) == 1


async def test_the_near_word_after_a_short_heard_form_still_counts(store):
    """A two-word term the ASR split: only the first word lines up with the folded token."""
    artifact = call(bot_utterance("Thanks for calling Marino Cafe today.", 1.0, 4.0))
    out = await evaluate(store, artifact, "thanks for calling Merino Cafe today")
    assert critical_findings(out) == []


# Noise -------------------------------------------------------------------------------------------


async def test_ordinary_mismatches_roll_up_into_one_finding_with_every_occurrence(store):
    artifact = call(
        bot_utterance("please hold while I check that for you now today", 1.0, 5.0),
        bot_utterance("thanks for waiting while I look that up for you", 10.0, 14.0),
    )
    out = await evaluate(
        store,
        artifact,
        "please hold while I chuck that for you now today "
        "thanks for wading while I look that up for you",
    )
    [finding] = [f for f in out.findings if f.category == "word_mismatch"]
    assert len(finding.evidence) == 2
    assert finding.severity == "info"
    assert "2 utterances" in finding.title


async def test_low_confidence_on_matched_words_is_one_detail_not_a_review_item(store):
    artifact = call(bot_utterance("Your PIN is 4821 and FH-4419 today", 1.0, 4.0))
    out = await evaluate(store, artifact, "your pin is 4821 and FH-4419 today", confidence=0.3)
    [finding] = [f for f in out.findings if f.category == "low_confidence_critical"]
    assert finding.severity == "info"
    assert finding.advisory is True
    assert len(finding.evidence) == 3
    assert "pin" in finding.detail and "4821" in finding.detail


async def test_a_repeated_phrase_is_not_lost_to_the_whole_call_alignment(store):
    """Two near-identical utterances, the first cut off. Each must find its own words."""
    first = bot_utterance(
        "One moment. The system shows two accounts. Let me verify the billing details",
        1.0,
        6.0,
        interrupted=True,
    )
    second = bot_utterance(
        "One moment. The system shows two accounts. Let me verify the billing address",
        10.0,
        15.0,
    )
    artifact, paths = write_call(store, call(first, second), duration_s=30.0)
    heard_first = timed_words("one moment the system shows two accounts let me", 1.0)
    heard_second = timed_words(
        "one moment the system shows two accounts let me verify the billing address", 10.0
    )
    words = heard_first + heard_second
    ctx = EvalContext(artifact, paths, EvalDeps())
    ctx.transcripts[Speaker.BOT] = Transcript(text="", words=words)
    out = EvaluatorOutput()
    await PronunciationEvaluator().run(ctx, out)
    assert [f for f in out.findings if f.category == "word_mismatch"] == []


async def test_a_name_inside_an_address_compares_against_its_own_word(store):
    scenario = Scenario(title="t", goal="g", facts={"name": "Aisha Rahman"})
    artifact = call(
        bot_utterance(
            "The account is for aisha at rahman law dot example.",
            1.0,
            5.0,
            heard_text="The account is for aisha@ramenlaw.example.",
        ),
        scenario_snapshot=scenario,
    )
    out = await evaluate(store, artifact, "The account is for ayesha at ramanlaw.example.")
    assert critical_findings(out) == []


async def test_a_persona_label_is_not_a_person(store):
    persona = Persona(name="Polite and overly chatty caller", voice=VoiceRef(voice_id="v"))
    artifact = call(bot_utterance("Yes, and overly so.", 1.0, 3.0), persona_snapshot=persona)
    out = await evaluate(store, artifact, "yes and overly so")
    assert "name" not in out.metrics["critical_by_kind"]


async def test_a_sound_alike_counts_every_repeated_letter_of_its_token(store):
    """Two aligned entries here compare equal, so finding one by value finds the first twice."""
    scenario = Scenario(title="t", goal="g", pronunciation_watch=["anna anna"])
    artifact = call(bot_utterance("Visit anna anna today.", 1.0, 4.0), scenario_snapshot=scenario)
    out = await evaluate(store, artifact, "visit ana-ana today")
    assert out.metrics["sound_alike"] == 1
    assert out.metrics["token_accuracy"] == 1.0


async def test_both_engines_hearing_a_wrong_repeated_digit_is_corroborated(store):
    """$11 and $1 squeeze to one sound, and that read as the live engine hearing the script."""
    artifact = call(
        bot_utterance("Your total is $11 today.", 1.0, 4.0, heard_text="Your total is $1 today.")
    )
    out = await evaluate(store, artifact, "your total is $1 today")
    [finding] = critical_findings(out)
    assert finding.category == "critical_token_corroborated"
    assert finding.advisory is False
