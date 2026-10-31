import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from enginefakes import (
    CLEAN,
    DeafSTT,
    FailingTTS,
    HangingLLM,
    Messages,
    Rewriting,
    Scripted,
    SilentTTS,
    fake_profile,
    persona,
    request,
    says,
    scenario,
    stack,
    unguarded,
    uses,
)

from voicelab.agents import load_personas, load_scenarios
from voicelab.agents.caller import MAX_TURNS_LINE
from voicelab.agents.support import GREETING
from voicelab.company import db as company_db
from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    BargeInEvent,
    CallArtifact,
    CallMode,
    CallStatus,
    LineConditions,
    Speaker,
    SystemEvent,
    ToolCallEvent,
    Utterance,
    Verdict,
)
from voicelab.core.profiles import build_stack
from voicelab.engine.service import run_bot_call
from voicelab.engine.speech_ledger import words_played
from voicelab.engine.turn_based import TurnConfig, run_turn_based_call
from voicelab.eval.context import EvalDeps
from voicelab.eval.pipeline import run_evaluation
from voicelab.providers.adapters.fake_speech import decode_speech
from voicelab.store.artifacts import CALL_SAMPLE_RATE, ArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path)


def utterances(artifact: CallArtifact, speaker: Speaker | None = None) -> list[Utterance]:
    found = [e for e in artifact.timeline if isinstance(e, Utterance)]
    return [u for u in found if speaker is None or u.speaker == speaker]


def events(artifact: CallArtifact, kind: type) -> list:
    return [e for e in artifact.timeline if isinstance(e, kind)]


def kinds(artifact: CallArtifact) -> list[str]:
    return [e.kind for e in events(artifact, SystemEvent)]


async def call(store: ArtifactStore, *, emit=None, config=None, req=None, **adapters):
    return await run_turn_based_call(
        req or request(),
        stack=stack(**adapters),
        store=store,
        emit=emit or Messages(),
        config=config,
    )


def real_seed(pid: str = "per_chatty_polite", sid: str = "scn_double_charge_refund"):
    return load_personas()[pid], load_scenarios()[sid]


LONG_REPLY = (
    "I can certainly look at that for you right now while we are on the phone together. "
    "It should only take me a moment to find everything on the account and read it back."
)


# Whole calls -------------------------------------------------------------------------------------


async def test_a_fake_call_runs_end_to_end_and_writes_its_artifact(store):
    chatty, double_charge = real_seed()
    artifact = await run_bot_call(
        request(persona=chatty.model_copy(update={"line": CLEAN}), scenario=double_charge),
        stack=stack(),
        store=store,
        emit=Messages(),
    )

    assert artifact.status == CallStatus.EVALUATING and artifact.error is None
    assert artifact.mode == CallMode.TURN_BASED and artifact.timing_is_real is False
    assert artifact.persona_snapshot.id == chatty.id
    assert artifact.scenario_snapshot.id == double_charge.id
    assert artifact.profile == "fake"
    assert set(artifact.providers) == {
        "stt_live", "tts_bot", "tts_caller", "llm_support", "llm_caller",
    }  # fmt: skip
    assert {"caller_hung_up", "bot_ended"} & set(kinds(artifact))
    assert kinds(artifact)[-1] == "call_ended"
    assert artifact.duration_s > 10
    paths = store.paths(artifact.id)
    for path in (paths.recording, paths.caller, paths.bot, paths.peaks, paths.call_json):
        assert path.exists()
    assert store.read_artifact(artifact.id).status == CallStatus.EVALUATING


async def test_a_full_call_then_evaluation_clears_every_criterion_on_a_clean_line(store):
    chatty, double_charge = real_seed()
    built = stack()
    artifact = await run_bot_call(
        request(persona=chatty.model_copy(update={"line": CLEAN}), scenario=double_charge),
        stack=built,
        store=store,
        emit=Messages(),
    )

    result = await run_evaluation(
        artifact,
        store.paths(artifact.id),
        EvalDeps(stt=built.stt_eval, judge=built.llm_judge),
    )

    # Tones aren't voices, so delivery checks may flag them for review. They can't fail a call.
    assert result.verdict in (Verdict.PASS, Verdict.NEEDS_REVIEW), result.summary
    assert all(finding.advisory for finding in result.findings)
    assert {run.status for run in result.evaluators} <= {"ok", "skipped"}
    assert result.criteria and all(c.status == "pass" for c in result.criteria)
    assert any(c.kind == "state" for c in result.criteria)
    assert any(c.kind == "beat" for c in result.criteria)


async def test_every_utterance_says_what_was_meant_played_and_heard(store):
    chatty, double_charge = real_seed()
    artifact = await call(
        store,
        req=request(persona=chatty.model_copy(update={"line": CLEAN}), scenario=double_charge),
    )

    for utterance in utterances(artifact):
        assert utterance.intended_text and utterance.spoken_text and utterance.heard_text
        assert utterance.end_s > utterance.start_s
        assert utterance.heard_text == utterance.intended_text


async def test_a_repeat_of_the_same_seed_is_the_same_call(store):
    noisy = persona(
        interruption_rate=0.5, line=LineConditions(noise="office", snr_db=18, packet_loss_pct=1)
    )
    first = await call(store, req=request("call_one", persona=noisy))
    second = await call(store, req=request("call_two", persona=noisy))

    def shape(artifact: CallArtifact) -> list[tuple]:
        return [
            (e.speaker, e.start_s, e.end_s, e.heard_text)
            for e in artifact.timeline
            if isinstance(e, Utterance)
        ]

    assert shape(first) == shape(second)
    assert (
        store.paths("call_one").recording.read_bytes()
        == store.paths("call_two").recording.read_bytes()
    )


async def test_compute_times_are_real_and_kept_off_the_timeline(store):
    artifact = await call(store)

    compute = artifact.metadata["compute"]
    assert {"tts_bot", "stt_live", "phone_line", "llm_caller"} <= set(compute["by_step_ms"])
    assert compute["total_ms"] >= 0 and compute["steps"]
    assert all(u.ttfb_ms is None for u in utterances(artifact))


async def test_request_metadata_is_carried_onto_the_artifact(store):
    artifact = await call(store, req=request(metadata={"run_label": "nightly"}))

    assert artifact.metadata["run_label"] == "nightly"


# The argument guard ------------------------------------------------------------------------------


GUESS = {"customer_id": "cus_1001", "billing_zip": "90210", "pin": "1234"}


async def test_a_blocked_attempt_is_on_the_timeline_of_a_call(store):
    guess = uses("verify_identity", **GUESS)
    artifact = await call(store, llm_support=Scripted(guess, default="What's your zip?"))
    attempt = events(artifact, ToolCallEvent)[0]
    assert attempt.arguments == GUESS
    assert attempt.guard.stopped
    assert "argument_guard rule not_heard" in attempt.error
    assert "bad_arguments" not in kinds(artifact)
    assert artifact.metadata["argument_guard"] is True


async def test_a_blocked_attempt_leaves_heard_text_as_stt_gave_it(store):
    guess = uses("verify_identity", **GUESS)
    artifact = await call(store, llm_support=Scripted(guess, default="What's your zip?"))
    heard = " ".join(u.heard_text or "" for u in utterances(artifact, Speaker.CALLER))
    assert "90210" not in heard
    assert "1234" not in heard


async def test_a_profile_can_run_the_raw_model_with_no_guard(store):
    guess = uses("verify_identity", **GUESS)
    built = unguarded(stack(llm_support=Scripted(guess, default="Sorry about that.")))
    artifact = await run_turn_based_call(request(), stack=built, store=store, emit=Messages())
    attempt = events(artifact, ToolCallEvent)[0]
    assert attempt.result["verified"] is False
    assert not attempt.guard.stopped
    assert artifact.metadata["argument_guard"] is False


# Turn order --------------------------------------------------------------------------------------


class Traced(Scripted):
    """Logs entry and exit, and yields while inside, so overlapping calls would interleave."""

    def __init__(self, tag: str, log: list[str], *script, default=None) -> None:
        super().__init__(*script, default=default)
        self.tag = tag
        self.log = log

    async def chat(self, messages, **options):
        self.log.append(f"{self.tag}:in")
        await asyncio.sleep(0.001)
        response = await super().chat(messages, **options)
        self.log.append(f"{self.tag}:out")
        return response

    async def stream_chat(self, messages, **options):
        self.log.append(f"{self.tag}:in")
        await asyncio.sleep(0.001)
        async for delta in super().stream_chat(messages, **options):
            yield delta
        self.log.append(f"{self.tag}:out")


async def test_the_two_brains_never_run_at_the_same_time(store):
    log: list[str] = []
    caller = Traced(
        "caller",
        log,
        says("Hello there."),
        says("And then this."),
        says("Bye.", beat="only", hang_up=True),
    )
    support = Traced("support", log, "First answer.", "Second answer.")

    await call(store, llm_caller=caller, llm_support=support)

    assert log == [
        "caller:in", "caller:out",
        "support:in", "support:out",
        "caller:in", "caller:out",
        "support:in", "support:out",
        "caller:in", "caller:out",
    ]  # fmt: skip


async def test_the_bot_greets_first_and_turns_alternate(store):
    caller = Scripted(says("One."), says("Two."), says("Bye.", beat="only", hang_up=True))
    artifact = await call(store, llm_caller=caller, llm_support=Scripted(default="Noted."))

    order = [u.speaker for u in utterances(artifact)]
    assert order == [
        Speaker.BOT, Speaker.CALLER, Speaker.BOT, Speaker.CALLER, Speaker.BOT, Speaker.CALLER,
    ]  # fmt: skip
    starts = [u.start_s for u in utterances(artifact)]
    assert starts == sorted(starts)
    assert all(
        a.end_s < b.start_s for a, b in zip(*(iter(utterances(artifact)),) * 2, strict=False)
    )


# Each side knows only what it heard --------------------------------------------------------------


async def test_each_brain_gets_the_heard_text_and_never_the_intended_text(store):
    caller = Scripted(says("My name is Dana."), says("Bye.", beat="only", hang_up=True))
    support = Scripted("Your balance is fine.")
    hearing = Rewriting({"Dana": "Diana", "fine": "nine", "Fernhill": "Fern Hill"})

    artifact = await call(store, llm_caller=caller, llm_support=support, stt_live=hearing)

    to_support = support.requests[0]
    assert to_support[-1].role == "user" and to_support[-1].content == "My name is Diana."
    prompt_after_greeting = caller.requests[0][-1].content
    assert "Fern Hill" in prompt_after_greeting and "Fernhill" not in prompt_after_greeting
    prompt_after_reply = caller.requests[1][-1].content
    assert "Your balance is nine." in prompt_after_reply and "is fine" not in prompt_after_reply

    bot_turn = utterances(artifact, Speaker.BOT)[1]
    assert bot_turn.intended_text == "Your balance is fine."
    assert bot_turn.heard_text == "Your balance is nine."
    caller_turn = utterances(artifact, Speaker.CALLER)[0]
    assert caller_turn.intended_text == "My name is Dana."
    assert caller_turn.heard_text == "My name is Diana."


async def test_spelled_ids_are_repaired_for_the_brain_but_stay_raw_in_the_artifact(store):
    raw = "My account is a C nine zero seven seven."
    caller = Scripted(says("My account is AC9077."), says("Bye.", beat="only", hang_up=True))
    support = Scripted("Got it.")
    hearing = Rewriting(
        {"A, as in apple. C, as in cat. Nine, zero, seven, seven.": "a C nine zero seven seven."}
    )

    artifact = await call(
        store,
        req=request(scenario=scenario(facts={"account": "AC9077"})),
        llm_caller=caller,
        llm_support=support,
        stt_live=hearing,
    )

    assert support.requests[0][-1].content == "My account is AC9077."
    assert utterances(artifact, Speaker.CALLER)[0].heard_text == raw


async def test_a_caller_line_is_spoken_in_its_tts_form(store):
    caller = Scripted(says("It's invoice FH-4419."), says("Bye.", beat="only", hang_up=True))
    support = Scripted("Thanks.")

    artifact = await call(
        store,
        req=request(scenario=scenario(facts={"invoice": "FH-4419"})),
        llm_caller=caller,
        llm_support=support,
    )

    turn = utterances(artifact, Speaker.CALLER)[0]
    assert "as in frank" in turn.intended_text and "as in frank" in turn.heard_text
    assert support.requests[0][-1].content == "It's invoice FH4419."


async def test_real_audio_crosses_the_line_in_both_directions(store):
    hearing = Rewriting({})
    line = LineConditions(codec="mulaw_8k", noise="none", one_way_latency_ms=120)

    artifact = await call(store, req=request(persona=persona(line=line)), stt_live=hearing)

    assert hearing.clips and all(clip.sample_rate == 8000 for clip in hearing.clips)
    assert utterances(artifact)[0].heard_text == utterances(artifact)[0].intended_text


async def test_request_line_overrides_the_persona_line(store):
    hearing = Rewriting({})

    await call(store, req=request(line=LineConditions(codec="mulaw_8k")), stt_live=hearing)

    assert all(clip.sample_rate == 8000 for clip in hearing.clips)


async def test_each_turn_arrives_one_line_delay_after_the_sender_acted(store):
    line = CLEAN.model_copy(update={"one_way_latency_ms": 500})
    caller = Scripted(says("Hello."), says("Bye.", beat="only", hang_up=True))
    config = TurnConfig(think_s=(0.3, 0.3), react_s=(0.7, 0.7), answer_s=0.2)

    artifact = await call(
        store,
        req=request(persona=persona(line=line)),
        llm_caller=caller,
        llm_support=Scripted("Noted."),
        config=config,
    )

    greeting, hello, noted = utterances(artifact)[:3]
    assert greeting.start_s == pytest.approx(0.7)
    assert hello.start_s == pytest.approx(greeting.end_s + 0.7 + 0.5)
    assert noted.start_s == pytest.approx(hello.end_s + 0.3 + 0.5)


# Interruptions -----------------------------------------------------------------------------------


def cutting_caller(*lines: str) -> Scripted:
    return Scripted(*lines, says("Bye.", beat="only", hang_up=True))


async def test_a_cut_turn_is_cut_on_disk_at_the_words_recorded_as_spoken(store):
    support = Scripted(default=LONG_REPLY)
    artifact = await call(
        store,
        req=request(persona=persona(interruption_rate=1.0)),
        llm_caller=cutting_caller(says("Wait."), says("Hold on."), says("Sorry.")),
        llm_support=support,
    )

    channel = AudioClip.from_file(store.paths(artifact.id).bot)
    cut = [u for u in utterances(artifact, Speaker.BOT) if u.interrupted]
    assert len(cut) >= 3
    for utterance in cut:
        assert utterance.spoken_text != utterance.intended_text
        assert utterance.intended_text.startswith(utterance.spoken_text)
        on_disk = decode_speech(channel.slice(utterance.start_s, utterance.end_s + 0.2))
        assert " ".join(w.text for w in on_disk) == utterance.spoken_text
        assert utterance.heard_text == utterance.spoken_text

    played_s = sum(u.end_s - u.start_s for u in utterances(artifact, Speaker.BOT))
    active_s = np.count_nonzero(channel.samples()) / CALL_SAMPLE_RATE
    assert active_s <= played_s


async def test_a_cut_records_a_barge_in_and_the_caller_overlaps_the_bot(store):
    artifact = await call(
        store,
        req=request(persona=persona(interruption_rate=1.0)),
        llm_caller=cutting_caller(says("Wait a second please.")),
        llm_support=Scripted(default=LONG_REPLY),
    )

    barge = events(artifact, BargeInEvent)[0]
    bot_turn = next(
        u for u in utterances(artifact, Speaker.BOT) if u.id == barge.interrupted_utterance_id
    )
    caller_turn = next(
        u for u in utterances(artifact, Speaker.CALLER) if u.start_s >= bot_turn.start_s
    )

    assert caller_turn.start_s == pytest.approx(barge.at_s)
    assert barge.bot_stopped_at_s == pytest.approx(bot_turn.end_s)
    assert bot_turn.start_s < caller_turn.start_s < bot_turn.end_s

    caller_audio = AudioClip.from_file(store.paths(artifact.id).caller)
    bot_audio = AudioClip.from_file(store.paths(artifact.id).bot)
    both = (barge.at_s, barge.bot_stopped_at_s)
    assert np.abs(caller_audio.slice(*both).samples()).max() > 0.02
    assert np.abs(bot_audio.slice(*both).samples()).max() > 0.02


async def test_the_bots_memory_holds_only_what_was_played_before_the_cut(store):
    support = Scripted(LONG_REPLY, default="All right.")
    artifact = await call(
        store,
        req=request(persona=persona(interruption_rate=1.0)),
        llm_caller=cutting_caller(says("Hello."), says("Wait.")),
        llm_support=support,
    )

    cut_turn = next(
        u
        for u in utterances(artifact, Speaker.BOT)
        if u.intended_text.startswith("I can certainly")
    )
    assert cut_turn.interrupted
    next_request = support.requests[1]
    assistant = [m.content for m in next_request if m.role == "assistant"]
    spoken_to_caller = " ".join([*assistant[-1:], next_request[-1].content])
    assert all(word in spoken_to_caller for word in cut_turn.spoken_text.split())
    assert "right now while" not in " ".join(assistant) or (
        "right now while" in cut_turn.spoken_text
    )


async def test_a_caller_cutting_in_cant_stop_a_tool_that_ran_before_any_words(store):
    speech_then_tool = uses(
        "create_ticket", text=f"{LONG_REPLY} {LONG_REPLY}", subject="Help", summary="Help"
    )
    support = Scripted(speech_then_tool, default="Done.")

    artifact = await call(
        store,
        req=request(persona=persona(interruption_rate=1.0)),
        llm_caller=cutting_caller(says("Hello."), says("Wait.")),
        llm_support=support,
    )

    assert [e.name for e in events(artifact, ToolCallEvent)] == ["create_ticket"]
    assert all(LONG_REPLY not in u.intended_text for u in utterances(artifact, Speaker.BOT))


async def test_a_short_reply_is_still_cut(store):
    artifact = await call(
        store,
        req=request(persona=persona(interruption_rate=1.0)),
        llm_caller=cutting_caller(says("Hello.")),
        llm_support=Scripted(default="Yes."),
    )

    reply = next(u for u in utterances(artifact, Speaker.BOT) if u.intended_text == "Yes.")
    assert reply.interrupted
    assert reply.end_s - reply.start_s < 0.43


async def test_a_goodbye_that_ends_the_call_is_not_cut(store):
    goodbye = uses("end_call", text="Thanks for calling. Goodbye.", reason="done")
    artifact = await call(
        store,
        req=request(persona=persona(interruption_rate=1.0)),
        llm_caller=Scripted(says("I'm done.")),
        llm_support=Scripted(goodbye),
    )

    assert not utterances(artifact, Speaker.BOT)[-1].interrupted
    assert "bot_ended" in kinds(artifact)


async def test_a_quiet_persona_never_cuts_in(store):
    artifact = await call(store, llm_support=Scripted(default=LONG_REPLY))

    assert not events(artifact, BargeInEvent)


# Tool calls and the database ---------------------------------------------------------------------


async def test_a_tool_before_any_speech_lands_between_the_caller_and_the_reply(store):
    support = Scripted(uses("list_plans"), "We have plans.")
    config = TurnConfig(think_s=(0.5, 0.5))

    artifact = await call(
        store,
        llm_caller=Scripted(says("What plans?"), says("Bye.", beat="only", hang_up=True)),
        llm_support=support,
        config=config,
    )

    question = utterances(artifact, Speaker.CALLER)[0]
    reply = utterances(artifact, Speaker.BOT)[1]
    tool = events(artifact, ToolCallEvent)[0]
    assert tool.at_s == pytest.approx(question.end_s + 0.5)
    assert question.end_s < tool.at_s < reply.start_s
    assert reply.start_s == pytest.approx(tool.at_s + 0.5 + 0.08)


async def test_a_preamble_beside_a_tool_is_never_spoken(store):
    preamble = uses("list_plans", text="Let me pull those up for you right now.")
    support = Scripted(preamble, "We have several plans.")

    artifact = await call(
        store,
        llm_caller=Scripted(says("What plans?"), says("Bye.", beat="only", hang_up=True)),
        llm_support=support,
    )

    assert utterances(artifact, Speaker.BOT)[1].intended_text == "We have several plans."


async def test_the_timeline_comes_out_in_time_order(store):
    chatty, double_charge = real_seed()
    artifact = await call(
        store,
        req=request(persona=chatty.model_copy(update={"line": CLEAN}), scenario=double_charge),
    )

    times = [e.start_s if isinstance(e, Utterance) else e.at_s for e in artifact.timeline]
    assert times == sorted(times)
    assert all(isinstance(e, ToolCallEvent) for e in events(artifact, ToolCallEvent))


async def test_changes_land_in_db_changes_and_each_call_gets_its_own_database(store):
    chatty, double_charge = real_seed()
    quiet = chatty.model_copy(update={"line": CLEAN})
    refunding = request("call_refund", persona=quiet, scenario=double_charge)

    idle_caller = Scripted(says("Never mind."), says("Bye.", beat="only", hang_up=True))
    idle = request("call_idle", persona=quiet)
    refunded, untouched = await asyncio.gather(
        call(store, req=refunding),
        call(store, req=idle, llm_caller=idle_caller, llm_support=Scripted(default="Okay.")),
    )

    assert [c.table for c in refunded.db_changes] == ["refunds"]
    assert refunded.db_changes[0].after["invoice_id"] == "FH-4419"
    assert untouched.db_changes == []

    fresh = company_db.create_call_db()
    assert fresh.row("SELECT COUNT(*) AS n FROM refunds WHERE invoice_id = 'FH-4419'")["n"] == 0


# Ending a call -----------------------------------------------------------------------------------


async def test_the_call_ends_when_the_caller_hangs_up(store):
    caller = Scripted(says("I'm done here.", beat="only", hang_up=True, reason="finished"))
    artifact = await call(store, llm_caller=caller, llm_support=Scripted())

    assert kinds(artifact)[-2:] == ["caller_hung_up", "call_ended"]
    assert next(e for e in events(artifact, SystemEvent) if e.kind == "caller_hung_up").detail == (
        "finished"
    )


async def test_the_call_ends_when_the_bot_calls_end_call(store):
    goodbye = uses("end_call", text="Goodbye.", reason="resolved")
    artifact = await call(
        store, llm_caller=Scripted(says("Thanks.")), llm_support=Scripted(goodbye)
    )

    assert kinds(artifact)[-2:] == ["bot_ended", "call_ended"]
    assert utterances(artifact)[-1].intended_text == "Goodbye."


async def test_the_call_ends_at_max_turns(store):
    artifact = await call(
        store,
        req=request(scenario=scenario(max_turns=2)),
        llm_caller=Scripted(default=says("Still here.")),
        llm_support=Scripted(default="Okay."),
    )

    assert "max_turns" in kinds(artifact)
    assert utterances(artifact, Speaker.CALLER)[-1].intended_text == MAX_TURNS_LINE
    assert len(utterances(artifact, Speaker.CALLER)) == 3


async def test_a_hang_up_with_no_words_leaves_no_empty_utterance(store):
    silent = json.dumps(
        {"say": "", "completed_beat": "only", "hang_up": True, "hang_up_reason": "gave up"}
    )
    artifact = await call(store, llm_caller=Scripted(silent), llm_support=Scripted())

    assert len(utterances(artifact, Speaker.CALLER)) == 0
    assert "caller_hung_up" in kinds(artifact)


async def test_beats_reached_are_on_the_timeline(store):
    caller = Scripted(says("Hello.", beat="only"), says("Bye.", hang_up=True))
    artifact = await call(store, llm_caller=caller, llm_support=Scripted("Noted."))

    beat = next(e for e in events(artifact, SystemEvent) if e.kind == "beat_completed")
    assert beat.detail == "only"


# Failure paths -----------------------------------------------------------------------------------


async def test_the_bot_gets_no_turn_when_it_hears_nothing(store):
    support = Scripted(default="Should never run.")
    artifact = await call(
        store,
        req=request(scenario=scenario(max_turns=2)),
        llm_caller=Scripted(default=says("Hello?")),
        llm_support=support,
        stt_live=DeafSTT(deaf_from=1),
    )

    assert artifact.status == CallStatus.EVALUATING
    assert support.requests == []
    assert "nothing_heard" in kinds(artifact)
    assert all(u.heard_text == "" for u in utterances(artifact, Speaker.CALLER))
    # The silence shows up as a pause on the timeline, not as a crash.
    turns = utterances(artifact, Speaker.CALLER)
    assert turns[1].start_s - turns[0].end_s >= 2.0


async def test_a_caller_who_hears_nothing_carries_on(store):
    artifact = await call(
        store,
        llm_caller=Scripted(says("Hello?"), says("Bye.", beat="only", hang_up=True)),
        llm_support=Scripted(default="Fine."),
        stt_live=DeafSTT(deaf_from=0),
    )

    assert artifact.status == CallStatus.EVALUATING
    assert "nothing_heard" in kinds(artifact)


async def test_a_tts_failure_fails_the_call_and_keeps_what_was_recorded(store):
    artifact = await call(
        store,
        llm_caller=Scripted(default=says("Hello.")),
        llm_support=Scripted(default="Fine."),
        tts_bot=FailingTTS(ok_for=1),
    )

    assert artifact.status == CallStatus.FAILED
    assert "tts_bot" in artifact.error and "went away" in artifact.error
    assert "call_failed" in kinds(artifact) and kinds(artifact)[-1] == "call_ended"
    saved = store.read_artifact(artifact.id)
    assert saved.status == CallStatus.FAILED and len(utterances(saved)) == 2
    assert AudioClip.from_file(store.paths(artifact.id).bot).duration_s > 1


async def test_tts_that_returns_no_audio_is_a_failure(store):
    artifact = await call(store, tts_bot=SilentTTS())

    assert artifact.status == CallStatus.FAILED and "no audio" in artifact.error


async def test_a_failure_before_anything_was_said_still_writes_an_artifact(store):
    artifact = await call(store, tts_bot=FailingTTS(ok_for=0))

    assert artifact.status == CallStatus.FAILED
    assert store.read_artifact(artifact.id).status == CallStatus.FAILED
    assert store.paths(artifact.id).recording.exists()


async def test_a_support_llm_that_hangs_times_out_and_fails_the_call(store):
    hanging = HangingLLM()
    artifact = await call(
        store,
        llm_caller=Scripted(default=says("Hello.")),
        llm_support=hanging,
        config=TurnConfig(step_timeout_s=0.05),
    )

    assert artifact.status == CallStatus.FAILED
    assert "llm_support timed out" in artifact.error
    assert hanging.closed
    assert len(utterances(artifact)) == 2


async def test_a_caller_llm_that_hangs_times_out_and_fails_the_call(store):
    artifact = await call(store, llm_caller=HangingLLM(), config=TurnConfig(step_timeout_s=0.05))

    assert artifact.status == CallStatus.FAILED and "llm_caller timed out" in artifact.error


async def test_an_llm_error_fails_the_call_with_the_reason(store):
    class Broken(Scripted):
        async def chat(self, messages, **options):
            raise ConnectionError("no route to model server")

    artifact = await call(store, llm_caller=Broken())

    assert artifact.status == CallStatus.FAILED and "no route to model server" in artifact.error


async def test_changes_made_before_a_failure_are_still_diffed(store):
    refund = uses("create_ticket", subject="s", summary="t")
    artifact = await call(
        store,
        llm_caller=Scripted(default=says("Hello.")),
        llm_support=Scripted(refund, "Done."),
        tts_bot=FailingTTS(ok_for=1),
    )

    assert artifact.status == CallStatus.FAILED
    assert [c.table for c in artifact.db_changes] == ["tickets"]


async def test_an_unexpected_setup_error_ends_as_a_failed_artifact(store, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("voicelab.engine.turn_based.create_call_db", broken)

    artifact = await call(store)

    assert artifact.status == CallStatus.FAILED and "disk full" in artifact.error


async def test_cancelling_propagates_and_still_writes_the_recording(store):
    release = asyncio.Event()

    class Blocks(Scripted):
        async def stream_chat(self, messages, **options):
            release.set()
            await asyncio.Event().wait()
            yield

    messages = Messages()
    run = asyncio.create_task(
        call(
            store,
            emit=messages,
            llm_caller=Scripted(default=says("Hello.")),
            llm_support=Blocks(),
        )
    )
    await asyncio.wait_for(release.wait(), 5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    saved = store.read_artifact("call_test")
    assert saved.status == CallStatus.FAILED and saved.error == "stopped"
    assert len(utterances(saved)) == 2
    assert AudioClip.from_file(store.paths("call_test").caller).duration_s > 0.5
    assert messages.of("callDone")[0]["call"]["status"] == "failed"


async def test_a_second_cancel_during_the_last_message_cannot_lose_the_artifact(store):
    entered = asyncio.Event()

    async def stuck_emit(message):
        if message["type"] == "callDone":
            entered.set()
            await asyncio.Event().wait()

    class Blocks(Scripted):
        async def stream_chat(self, messages, **options):
            await asyncio.Event().wait()
            yield

    run = asyncio.create_task(
        call(store, emit=stuck_emit, llm_caller=Scripted(default=says("Hi.")), llm_support=Blocks())
    )
    await asyncio.sleep(0.05)
    run.cancel()
    await asyncio.wait_for(entered.wait(), 5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert store.read_artifact("call_test").error == "stopped"


# Progress messages -------------------------------------------------------------------------------


async def test_messages_follow_the_api_shapes_in_order(store):
    messages = Messages()
    caller = Scripted(says("Hello.", beat="only"), says("Bye.", hang_up=True))

    artifact = await call(
        store, emit=messages, llm_caller=caller, llm_support=Scripted(uses("list_plans"), "Noted.")
    )

    assert messages.items[0]["type"] == "callStarted"
    assert messages.items[-1]["type"] == "callDone"
    assert messages.items[-1]["call"]["status"] == "evaluating"
    assert messages.items[-1]["call"]["durationS"] == artifact.duration_s
    summary = messages.items[0]["call"]
    assert summary["id"] == "call_test" and summary["status"] == "running"
    assert summary["timingIsReal"] is False and summary["scenarioTitle"] == "Test"

    turns = messages.of("turn")
    assert len(turns) == len(utterances(artifact))
    first = turns[0]
    assert first["callId"] == "call_test"
    assert first["utterance"]["intendedText"] == GREETING.replace("this is Fern", "this is Fern")
    assert {"startS", "endS", "heardText", "spokenText"} <= set(first["utterance"])

    partial = messages.of("partial")[0]
    assert partial["speaker"] == "bot" and partial["text"] and "atS" in partial
    assert messages.items.index(partial) < messages.items.index(turns[0])

    sent = {m["event"].get("kind") or m["event"]["type"] for m in messages.of("event")}
    assert {"call_started", "beat_completed", "tool_call", "caller_hung_up", "call_ended"} <= sent


async def test_tool_arguments_keep_their_own_keys_in_messages(store):
    messages = Messages()
    plan = uses("get_dns_records", domain="nguyenauto.example")

    await call(
        store,
        emit=messages,
        llm_caller=Scripted(says("Hello."), says("Bye.", beat="only", hang_up=True)),
        llm_support=Scripted(plan, "Noted."),
    )

    tool = next(m["event"] for m in messages.of("event") if m["event"]["type"] == "tool_call")
    assert tool["atS"] > 0 and tool["durationMs"] is not None
    assert tool["arguments"] == {"domain": "nguyenauto.example"}


async def test_a_broken_watcher_does_not_affect_the_call(store):
    async def explode(message):
        raise RuntimeError("socket closed")

    artifact = await call(store, emit=explode)

    assert artifact.status == CallStatus.EVALUATING


# Small pieces ------------------------------------------------------------------------------------


def test_words_played_counts_only_finished_words():
    parts = [("one two", 1.0), ("three four", 1.0)]

    assert words_played(parts, 0.1) == ""
    assert words_played(parts, 0.5) == "one"
    assert words_played(parts, 1.0) == "one two"
    assert words_played(parts, 1.7) == "one two three"
    assert words_played(parts, 2.0) == "one two three four"


async def test_run_bot_call_rejects_modes_that_are_not_bot_to_bot(store):
    with pytest.raises(ValueError, match="not a bot-to-bot mode"):
        await run_bot_call(
            request(mode=CallMode.HUMAN), stack=stack(), store=store, emit=Messages()
        )


def test_the_shipped_fake_profile_builds():
    from voicelab.core.profiles import load_profiles

    built = build_stack(load_profiles()["fake"])

    assert built.audio_analyst is not None and built.realtime is None
    assert fake_profile().id == built.profile.id
