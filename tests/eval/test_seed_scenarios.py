"""The evaluators against the real seeds, the real database diff, and real TTS text."""

import pytest
from evalfakes import FakeLLM, FakeSTT, timed_words, verdict_json, write_call

from voicelab.agents.seeds import validate_seeds
from voicelab.agents.speech import normalize
from voicelab.company import ToolDispatcher, create_call_db, diff_against_seed
from voicelab.core.models import CallArtifact, CallMode, Speaker, Utterance, Verdict
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput
from voicelab.eval.evaluators.pronunciation import PronunciationEvaluator
from voicelab.eval.evaluators.state import StateEvaluator
from voicelab.eval.normalize import Lexicon, align, tokenize
from voicelab.eval.pipeline import run_evaluation
from voicelab.providers.base import Transcript

_, SCENARIOS = validate_seeds()
LEXICON = Lexicon.load()

IDEAL_RUNS = {
    "scn_double_charge_refund": [
        ("issue_refund", {"invoice_id": "FH-4419", "reason": "dup", "payment_id": "pay_0003"})
    ],
    "scn_site_suspended_nonpayment": [
        ("pay_invoice", {"invoice_id": "FH-4502"}),
        ("restore_site", {"domain": "bellhardware.example"}),
    ],
    "scn_locked_mailbox": [("unlock_mailbox", {"address": "evelyn@hartfamily.example"})],
    "scn_forgotten_password": [("reset_mailbox_password", {"address": "priya@nairdesign.example"})],
    "scn_refund_over_limit": [
        ("create_ticket", {"customer_id": "cus_1005", "subject": "s", "summary": "x"})
    ],
    "scn_failed_verification": [],
    "scn_dns_mx_fix": [
        (
            "update_dns_record",
            {
                "domain": "nguyenauto.example",
                "type": "MX",
                "value": "mail.fernhill.example",
                "priority": 10,
            },
        )
    ],
    "scn_expired_ssl": [("renew_ssl", {"domain": "marinocafe.example"})],
    "scn_domain_renewal": [("renew_domain", {"domain": "thompsonphoto.example"})],
    "scn_plan_upgrade_full_mailbox": [
        ("change_plan", {"customer_id": "cus_1010", "plan_id": "email_plus"})
    ],
    "scn_already_refunded": [
        ("create_ticket", {"customer_id": "cus_1012", "subject": "s", "summary": "x"})
    ],
    "scn_past_due_expired_card": [("pay_invoice", {"invoice_id": "FH-4588"})],
}

# Which expectation each wrong run must trip, so a test can't pass on some unrelated failure.
WRONG_RUNS = {
    "scn_double_charge_refund": ([("pay_invoice", {"invoice_id": "FH-4419"})], "refund_added"),
    "scn_site_suspended_nonpayment": (
        [("restore_site", {"domain": "bellhardware.example"})],
        "invoice_paid",
    ),
    "scn_refund_over_limit": (
        [("issue_refund", {"invoice_id": "FH-4631", "reason": "sure"})],
        "no_refund",
    ),
    "scn_failed_verification": (
        [("reset_mailbox_password", {"address": "renee@castillostudio.example"})],
        "mailbox_untouched",
    ),
    "scn_already_refunded": (
        [("issue_refund", {"invoice_id": "FH-4288", "reason": "again"})],
        "no_new_refund",
    ),
    "scn_expired_ssl": ([("renew_domain", {"domain": "marinocafe.example"})], "ssl_renewed"),
}


def changes_after(calls):
    db = create_call_db()
    tools = ToolDispatcher(db)
    results = [tools.dispatch(name, arguments) for name, arguments in calls]
    changes = diff_against_seed(db)
    db.close()
    return changes, results


async def state_output(store, scenario_id: str, calls) -> EvaluatorOutput:
    changes, _ = changes_after(calls)
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS, scenario_snapshot=SCENARIOS[scenario_id], db_changes=changes
    )
    artifact, paths = write_call(store, artifact, duration_s=1.0)
    out = EvaluatorOutput()
    await StateEvaluator().run(EvalContext(artifact, paths, EvalDeps()), out)
    return out


def test_every_seed_scenario_is_covered():
    assert set(IDEAL_RUNS) == set(SCENARIOS)


@pytest.mark.parametrize("scenario_id", IDEAL_RUNS)
async def test_ideal_run_meets_every_expectation_with_nothing_for_review(store, scenario_id):
    out = await state_output(store, scenario_id, IDEAL_RUNS[scenario_id])
    assert [c.criterion_id for c in out.criteria if c.status != "pass"] == []
    # Correct runs have side effects no scenario lists, and those alone must not trigger review.
    assert [f.title for f in out.findings if f.severity != "minor"] == []


@pytest.mark.parametrize("scenario_id", WRONG_RUNS)
async def test_wrong_run_trips_the_expectation_it_should(store, scenario_id):
    calls, expectation_id = WRONG_RUNS[scenario_id]
    out = await state_output(store, scenario_id, calls)
    assert expectation_id in [c.criterion_id for c in out.criteria if c.status == "fail"]


async def test_second_refund_on_an_already_refunded_invoice_fails_no_change(store):
    # This is an inserted row: the refund already on FH-4288 is unchanged, so only the insert shows.
    calls = [("issue_refund", {"invoice_id": "FH-4288", "reason": "again"})]
    changes, results = changes_after(calls)
    assert "error" not in results[0]
    inserted = [c for c in changes if c.table == "refunds"]
    assert [c.kind for c in inserted] == ["insert"]
    assert inserted[0].after["invoice_id"] == "FH-4288"

    out = await state_output(store, "scn_already_refunded", calls)
    assert {c.criterion_id: c.status for c in out.criteria} == {
        "no_new_refund": "fail",
        "other_invoice_safe": "pass",
    }
    failure = next(f for f in out.findings if f.criterion_id == "no_new_refund")
    assert failure.severity == "major"
    assert failure.advisory is False


async def test_refund_on_the_substitute_invoice_trips_its_own_expectation(store):
    calls = [("issue_refund", {"invoice_id": "FH-4390", "reason": "instead"})]
    out = await state_output(store, "scn_already_refunded", calls)
    assert {c.criterion_id: c.status for c in out.criteria} == {
        "no_new_refund": "pass",
        "other_invoice_safe": "fail",
    }


async def test_a_full_evaluation_of_a_repeat_refund_fails_the_call(store):
    scenario = SCENARIOS["scn_already_refunded"]
    changes, _ = changes_after([("issue_refund", {"invoice_id": "FH-4288", "reason": "again"})])
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario, db_changes=changes)
    artifact, paths = write_call(store, artifact, caller=[(0, 2)], bot=[(3, 5)], duration_s=6.0)
    words = {
        "caller": "I want invoice FH-4288 refunded",
        "bot": "I have refunded it for you goodbye",
    }
    stt = FakeSTT(
        Transcript(text=words["caller"], words=timed_words(words["caller"], 0.0)),
        Transcript(text=words["bot"], words=timed_words(words["bot"], 3.0)),
    )

    def judge(prompt: str) -> dict:
        return verdict_json("pass", quotes=["goodbye"])

    result = await run_evaluation(artifact, paths, EvalDeps(stt=stt, judge=FakeLLM(judge)))
    assert result.verdict == Verdict.FAIL
    assert any("No second refund" in c.text and c.status == "fail" for c in result.criteria)


# What the bot writes, what normalize() sends to TTS, and what an ASR would plausibly return.
SPEECH_CASES = [
    ("Your invoice FH-4419 is paid.", "Your invoice FH-4419 is paid."),
    ("Your invoice FH-4419 is paid.", "Your invoice FH 4419 is paid."),
    ("That comes to $49.99 today.", "That comes to $49.99 today."),
    (
        "That comes to $49.99 today.",
        "That comes to forty-nine dollars and ninety-nine cents today.",
    ),
    ("That comes to $49.99 today.", "That comes to 49.99 today."),
    ("It costs $14.99 a year.", "It costs $14.99 a year."),
    ("A refund of $5.05 went out.", "A refund of $5.05 went out."),
    ("That's $1,200.00 a year.", "That's $1,200 a year."),
    ("Call us at (503) 555-0101.", "Call us at 503-555-0101."),
    ("Call us at (503) 555-0101.", "Call us at (503) 555-0101."),
    ("Visit billing.fernhill.example today.", "Visit billing.fernhill.example today."),
    ("Visit billing.fernhill.example today.", "Visit billing dot fernhill dot example today."),
    (
        "Email dana@whitfieldlandscaping.example please.",
        "Email dana@whitfieldlandscaping.example please.",
    ),
    ("Use port 993 with SSL on IMAP.", "Use port 993 with SSL on IMAP."),
    ("Your PIN is 4821 and your zip is 97214.", "Your PIN is 4821 and your zip is 97214."),
    ("It renews on 2027-11-06.", "It renews on November 6th, 2027."),
    ("It was refunded on September 12.", "It was refunded on September 12th."),
    ("You have 5 GB left.", "You have 5 GB left."),
    ("You have 5 GB left.", "You have five gigabytes left."),
    ("We open at 9:00 AM.", "We open at 9 AM."),
    ("Changes can take up to 24 hours.", "Changes can take up to 24 hours."),
    ("Thanks for calling, Mr. Okafor.", "Thanks for calling, Mr. Okafor."),
    ("The MX record has priority 10.", "The MX record has priority 10."),
]


@pytest.mark.parametrize(("written", "heard"), SPEECH_CASES)
def test_tts_text_matches_what_an_asr_would_write(written, heard):
    spoken = normalize(written, None).spoken
    ref = tokenize(spoken.split(), lexicon=LEXICON)
    hyp = tokenize(heard.split(), lexicon=LEXICON)
    missed = [a.ref.norm for a in align(ref, hyp) if not a.matched]
    assert missed == [], spoken


@pytest.mark.parametrize(("written", "heard"), SPEECH_CASES)
def test_display_text_matches_what_an_asr_would_write(written, heard):
    ref = tokenize(written.split(), lexicon=LEXICON)
    hyp = tokenize(heard.split(), lexicon=LEXICON)
    missed = [a.ref.norm for a in align(ref, hyp) if not a.matched]
    assert missed == []


@pytest.mark.parametrize(
    ("written", "kind"),
    [
        ("Your invoice FH-4419 is paid.", "id"),
        ("That comes to $49.99 today.", "money"),
        ("Call us at (503) 555-0101.", "number"),
        ("Visit billing.fernhill.example today.", "url"),
        ("Email dana@whitfieldlandscaping.example please.", "url"),
        ("We use SSL here.", "lexicon"),
        ("Your code is X Q Z today.", "acronym"),
        ("Your PIN is 4821 today.", "number"),
        ("Thanks for calling, Mr. Okafor.", "lexicon"),
    ],
)
def test_tts_text_keeps_its_critical_tokens(written, kind):
    spoken = normalize(written, None).spoken
    tokens = tokenize(spoken.split(), lexicon=LEXICON)
    assert kind in {t.critical for t in tokens}, spoken


def test_a_spelled_invoice_reads_as_one_critical_id():
    spoken = normalize("Invoice FH-4419 is paid.", None).spoken
    words = spoken.split()
    ids = [t for t in tokenize(words, lexicon=LEXICON) if t.critical == "id"]
    assert "".join(t.norm for t in ids) == "fh4419"
    assert {(t.start, t.end) for t in ids} == {(1, 13)}


WATCH_TERMS = sorted(
    {(s.id, term) for s in SCENARIOS.values() for term in s.pronunciation_watch},
)


@pytest.mark.parametrize(("scenario_id", "term"), WATCH_TERMS)
def test_every_seed_watch_term_is_found_in_display_text(scenario_id, term):
    words = f"Noting {term} now".split()
    tokens = tokenize(words, lexicon=LEXICON, watch=[term])
    assert "watch" in {t.critical for t in tokens}


@pytest.mark.parametrize(
    ("scenario_id", "term"),
    # An ISO date's digits are said in another order, so its number stays critical without this.
    [(s, t) for s, t in WATCH_TERMS if not t[:4].isdigit()],
)
def test_every_seed_watch_term_is_found_in_tts_text(scenario_id, term):
    spoken = normalize(f"Noting {term} now", None).spoken
    tokens = tokenize(spoken.split(), lexicon=LEXICON, watch=[term])
    assert "watch" in {t.critical for t in tokens}, spoken


async def pronunciation_on(store, *utterances: Utterance, heard: str) -> EvaluatorOutput:
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, timeline=list(utterances))
    artifact, paths = write_call(store, artifact, duration_s=30.0)
    ctx = EvalContext(artifact, paths, EvalDeps())
    ctx.transcripts[Speaker.BOT] = Transcript(text=heard, words=timed_words(heard, 1.0))
    out = EvaluatorOutput()
    await PronunciationEvaluator().run(ctx, out)
    return out


def tts_utterance(written: str, **kwargs) -> Utterance:
    speech = normalize(written, None)
    return Utterance(
        speaker=Speaker.BOT, start_s=1.0, end_s=9.0, intended_text=speech.spoken, **kwargs
    )


async def test_clean_tts_call_has_no_findings(store):
    out = await pronunciation_on(
        store,
        tts_utterance("Invoice FH-4419 for $49.99 is on billing.fernhill.example."),
        heard="Invoice FH-4419 for $49.99 is on billing.fernhill.example.",
    )
    assert out.findings == []
    assert out.metrics["critical_mismatches"] == 0
    assert out.metrics["critical_tokens"] == 3


async def test_wrong_digit_against_tts_text_names_the_whole_code(store):
    out = await pronunciation_on(
        store, tts_utterance("Invoice FH-4419 is paid."), heard="Invoice FH-4491 is paid."
    )
    assert len(out.findings) == 1
    finding = out.findings[0]
    assert finding.severity == "major"
    assert "F, as in frank." in finding.title


async def test_wrong_amount_against_tts_text_is_major(store):
    out = await pronunciation_on(
        store, tts_utterance("That comes to $49.99 today."), heard="That comes to $59.99 today."
    )
    assert [f.category for f in out.findings] == ["critical_token_asr_only"]


async def test_ordinary_phrase_that_is_also_a_lexicon_form_cannot_fail_a_call(store):
    out = await pronunciation_on(
        store,
        tts_utterance("Can you tell me who is calling?"),
        heard="Can you tell me who was calling?",
    )
    assert all(f.advisory for f in out.findings)
