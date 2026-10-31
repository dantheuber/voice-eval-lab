import json

import pytest
from evalfakes import FakeAnalyst, FakeLLM, FakeSTT, verdict_json, write_call
from evalfakes import transcript as make_transcript

from voicelab.core.models import (
    Beat,
    CallArtifact,
    CallMode,
    DbChange,
    Evidence,
    Finding,
    Rule,
    Scenario,
    Speaker,
    StateExpectation,
    Utterance,
    Verdict,
)
from voicelab.eval.config import EvalConfig
from voicelab.eval.context import EvalDeps, Skip
from voicelab.eval.evaluators.transcribe import TranscribeEvaluator
from voicelab.eval.pipeline import (
    EvalProgress,
    default_evaluators,
    run_evaluation,
    set_clip_bounds,
)
from voicelab.providers.base import Transcript

CALLER_SAID = "hi I was charged twice for invoice FH-4419"
BOT_SAID = "your refund of $49.99 is done goodbye"


class Named:
    def __init__(self, name: str, action=None) -> None:
        self.name = name
        self.action = action
        self.ran = False

    async def run(self, ctx, out) -> None:
        self.ran = True
        if self.action is not None:
            self.action(ctx, out)


def always_raises(message: str):
    def action(ctx, out):
        raise RuntimeError(message)

    return action


def scenario() -> Scenario:
    return Scenario(
        title="Double charge",
        goal="get the duplicate charge refunded",
        beats=[Beat(instruction="state the problem", checkpoint="Caller mentions the charge")],
        must=[Rule(text="say goodbye")],
        must_not=[Rule(text="ask for the full card number")],
        expected_state=[
            StateExpectation(table="refunds", where={"invoice_id": "FH-4419"}, op="row_added")
        ],
        pronunciation_watch=["refund"],
    )


def scripted_call() -> CallArtifact:
    return CallArtifact(
        mode=CallMode.LIVE_BOTS,
        scenario_snapshot=scenario(),
        timeline=[
            Utterance(speaker=Speaker.CALLER, start_s=0.0, end_s=4.0),
            Utterance(speaker=Speaker.BOT, start_s=4.5, end_s=7.5, intended_text=BOT_SAID),
        ],
        db_changes=[
            DbChange(
                table="refunds",
                key={"id": "ref_1"},
                kind="insert",
                after={"id": "ref_1", "invoice_id": "FH-4419"},
            )
        ],
    )


def stt() -> FakeSTT:
    return FakeSTT(make_transcript((CALLER_SAID, 0.0)), make_transcript((BOT_SAID, 4.5)))


def friendly_judge() -> FakeLLM:
    def reply(prompt: str) -> dict:
        if "Step of the script:" in prompt:
            return verdict_json("pass", quotes=["charged twice for invoice"])
        if "say goodbye" in prompt.split("Criterion to check:")[-1]:
            return verdict_json("pass", quotes=["goodbye"])
        return verdict_json("pass")

    return FakeLLM(reply)


SPANS = {"caller": [(0.0, 4.0)], "bot": [(4.5, 7.5)], "duration_s": 9.0}


async def test_full_pipeline_on_a_synthetic_call_passes(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    deps = EvalDeps(stt=stt(), judge=friendly_judge(), analyst=FakeAnalyst())
    result = await run_evaluation(artifact, paths, deps)
    assert result.call_id == artifact.id
    assert result.verdict == Verdict.PASS, result.summary
    assert [r.name for r in result.evaluators] == [e.name for e in default_evaluators()]
    assert [r.status for r in result.evaluators] == ["ok"] * 8
    assert {c.kind for c in result.criteria} == {"must", "must_not", "general", "beat", "state"}
    assert result.summary.startswith("Verdict: pass.")
    assert all(r.duration_s >= 0 for r in result.evaluators)


async def test_the_result_round_trips_through_the_store(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    deps = EvalDeps(stt=stt(), judge=friendly_judge(), analyst=FakeAnalyst())
    result = await run_evaluation(artifact, paths, deps)
    store.write_eval(result)
    loaded = store.read_eval(artifact.id)
    assert loaded.model_dump() == result.model_dump()
    json.loads(paths.eval_json.read_text())


async def test_independent_transcript_is_stored_in_metrics(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    fake = stt()
    result = await run_evaluation(artifact, paths, EvalDeps(stt=fake))
    transcribe = result.evaluators[0]
    assert transcribe.name == "transcribe"
    assert transcribe.model == "fake-stt"
    stored = transcribe.metrics["transcript"]
    assert stored["bot"]["text"] == BOT_SAID
    assert stored["caller"]["words"][0]["text"] == "hi"
    assert transcribe.metrics["word_counts"]["bot"] == len(BOT_SAID.split())
    assert all(call["hints"] is None for call in fake.calls)


async def test_silent_channel_is_not_sent_to_the_stt(store):
    artifact, paths = write_call(store, bot=[(1.0, 2.0)], duration_s=3.0)
    fake = FakeSTT(make_transcript(("hello", 1.0)))
    await run_evaluation(artifact, paths, EvalDeps(stt=fake))
    assert len(fake.calls) == 1


async def test_stt_without_word_timing_gets_estimated_words(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    fake = FakeSTT(Transcript(text=CALLER_SAID), Transcript(text=BOT_SAID))
    result = await run_evaluation(artifact, paths, EvalDeps(stt=fake))
    metrics = result.evaluators[0].metrics
    assert metrics["estimated_word_timing"] == ["caller", "bot"]
    bot_words = metrics["transcript"]["bot"]["words"]
    assert bot_words[0]["start_s"] == pytest.approx(4.5, abs=0.03)
    assert bot_words[-1]["end_s"] == pytest.approx(7.5, abs=0.03)


async def test_one_stt_channel_failing_keeps_the_other(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    fake = FakeSTT(ConnectionError("caller channel lost"), make_transcript((BOT_SAID, 4.5)))
    result = await run_evaluation(artifact, paths, EvalDeps(stt=fake, judge=friendly_judge()))
    transcribe = result.evaluators[0]
    assert transcribe.status == "error"
    assert "caller channel lost" in transcribe.reason
    assert "bot" in transcribe.metrics["transcript"]
    pronunciation = next(r for r in result.evaluators if r.name == "pronunciation")
    assert pronunciation.status == "ok"
    assert result.verdict == Verdict.NEEDS_REVIEW


async def test_a_failing_evaluator_never_stops_the_others(store):
    artifact, paths = write_call(store, duration_s=1.0)
    first = Named("first", always_raises("boom"))
    second = Named("second")
    third = Named("third", always_raises("again"))
    fourth = Named("fourth")
    result = await run_evaluation(
        artifact, paths, EvalDeps(), evaluators=[first, second, third, fourth]
    )
    assert second.ran and fourth.ran
    assert [r.status for r in result.evaluators] == ["error", "ok", "error", "ok"]
    assert result.evaluators[0].reason == "RuntimeError: boom"
    assert result.verdict == Verdict.NEEDS_REVIEW


async def test_a_skip_is_recorded_with_its_reason(store):
    def skip(ctx, out):
        raise Skip("nothing to do")

    artifact, paths = write_call(store, duration_s=1.0)
    result = await run_evaluation(artifact, paths, EvalDeps(), evaluators=[Named("a", skip)])
    assert result.evaluators[0].status == "skipped"
    assert result.evaluators[0].reason == "nothing to do"


async def test_a_failing_evaluator_keeps_what_it_found_first(store):
    def partial(ctx, out):
        out.findings.append(Finding(evaluator="a", category="c", title="found first"))
        out.metrics["seen"] = 1
        raise RuntimeError("then broke")

    artifact, paths = write_call(store, duration_s=1.0)
    result = await run_evaluation(artifact, paths, EvalDeps(), evaluators=[Named("a", partial)])
    assert result.evaluators[0].metrics == {"seen": 1}
    assert result.findings[0].title == "found first"


async def test_missing_audio_fails_audio_evaluators_but_not_state(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    paths.caller.unlink()
    paths.bot.unlink()
    result = await run_evaluation(artifact, paths, EvalDeps(stt=stt()))
    statuses = {r.name: r.status for r in result.evaluators}
    assert statuses["timing"] == "error"
    assert statuses["audio_quality"] == "error"
    assert statuses["state"] == "ok"
    assert result.verdict == Verdict.NEEDS_REVIEW


async def test_no_models_at_all_still_gives_a_result(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    result = await run_evaluation(artifact, paths, EvalDeps())
    statuses = {r.name: r.status for r in result.evaluators}
    assert statuses == {
        "transcribe": "skipped",
        "timing": "ok",
        "state": "ok",
        "transcript_judge": "skipped",
        "caller_validity": "skipped",
        "pronunciation": "skipped",
        "audio_quality": "ok",
        "audio_listening": "skipped",
    }
    assert result.verdict == Verdict.NEEDS_REVIEW
    assert "Not checked:" in result.summary


async def test_failed_state_fails_the_call(store):
    artifact = scripted_call()
    artifact.db_changes = []
    artifact, paths = write_call(store, artifact, **SPANS)
    deps = EvalDeps(stt=stt(), judge=friendly_judge())
    result = await run_evaluation(artifact, paths, deps)
    assert result.verdict == Verdict.FAIL


async def test_missed_required_beat_invalidates_the_call(store):
    def judge(prompt: str) -> dict:
        if "Step of the script:" in prompt:
            return verdict_json("fail")
        return verdict_json("pass", quotes=["goodbye"])

    artifact, paths = write_call(store, scripted_call(), **SPANS)
    result = await run_evaluation(artifact, paths, EvalDeps(stt=stt(), judge=FakeLLM(judge)))
    assert result.verdict == Verdict.INVALID


async def test_wrong_amount_in_the_audio_fails_the_call(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    wrong = FakeSTT(
        make_transcript((CALLER_SAID, 0.0)),
        make_transcript(("your refund of $59.99 is done goodbye", 4.5)),
    )
    result = await run_evaluation(artifact, paths, EvalDeps(stt=wrong, judge=friendly_judge()))
    assert result.verdict == Verdict.FAIL
    mismatch = next(f for f in result.findings if f.category == "critical_token_mismatch")
    assert mismatch.clip_start_s is not None


async def test_advisory_listening_alone_cannot_fail_a_call(store):
    analyst = FakeAnalyst(
        lambda audio, prompt: {
            "tone": "angry",
            "delivery": "Terrible.",
            "odd_pronunciations": [],
            "issues": [{"title": "Sounds furious", "severity": "major"}],
        }
    )
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    deps = EvalDeps(stt=stt(), judge=friendly_judge(), analyst=analyst)
    result = await run_evaluation(artifact, paths, deps)
    assert result.verdict == Verdict.NEEDS_REVIEW


async def test_upload_with_no_scenario_reports_what_it_could_not_check(store):
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_tool_log=False, has_intended_text=False)
    artifact, paths = write_call(store, artifact, **SPANS)
    deps = EvalDeps(stt=stt(), judge=FakeLLM(), analyst=FakeAnalyst())
    result = await run_evaluation(artifact, paths, deps)
    assert result.verdict == Verdict.PASS
    assert {c.kind for c in result.criteria} == {"general"}
    assert "Not checked: scenario rules" in result.summary
    assert "tool result" in result.summary
    statuses = {r.name: r.status for r in result.evaluators}
    assert statuses["state"] == "skipped"
    assert statuses["caller_validity"] == "skipped"


async def test_progress_reports_each_evaluator_starting_and_finishing(store):
    artifact, paths = write_call(store, duration_s=1.0)
    events: list[EvalProgress] = []

    def skip(ctx, out):
        raise Skip("no")

    await run_evaluation(
        artifact,
        paths,
        EvalDeps(),
        on_progress=events.append,
        evaluators=[Named("a"), Named("b", skip), Named("c", always_raises("x"))],
    )
    assert [(e.evaluator, e.index, e.total, e.status) for e in events] == [
        ("a", 0, 3, "running"),
        ("a", 0, 3, "ok"),
        ("b", 1, 3, "running"),
        ("b", 1, 3, "skipped"),
        ("c", 2, 3, "running"),
        ("c", 2, 3, "error"),
    ]
    assert events[3].detail == "no"


async def test_async_progress_callback_is_awaited(store):
    artifact, paths = write_call(store, duration_s=1.0)
    seen = []

    async def on_progress(event: EvalProgress) -> None:
        seen.append(event.status)

    await run_evaluation(
        artifact, paths, EvalDeps(), on_progress=on_progress, evaluators=[Named("a")]
    )
    assert seen == ["running", "ok"]


async def test_broken_progress_callback_cannot_hurt_evaluation(store):
    artifact, paths = write_call(store, duration_s=1.0)

    def on_progress(event: EvalProgress) -> None:
        raise ConnectionError("browser left")

    result = await run_evaluation(
        artifact, paths, EvalDeps(), on_progress=on_progress, evaluators=[Named("a")]
    )
    assert result.evaluators[0].status == "ok"


async def test_slow_evaluators_narrate_steps_through_progress(store):
    artifact, paths = write_call(store, scripted_call(), **SPANS)
    events: list[EvalProgress] = []
    deps = EvalDeps(stt=stt(), judge=friendly_judge())
    await run_evaluation(artifact, paths, deps, on_progress=events.append)
    details = [e.detail for e in events if e.evaluator == "transcript_judge"]
    assert "criterion 1 of 6" in details


def test_clip_bounds_add_context_around_evidence():
    finding = Finding(
        evaluator="e", category="c", title="t",
        evidence=[Evidence(start_s=10.0, end_s=12.0), Evidence(start_s=14.0, end_s=15.0)],
    )  # fmt: skip
    set_clip_bounds(finding, 60.0, 3.0)
    assert (finding.clip_start_s, finding.clip_end_s) == (7.0, 18.0)


def test_clip_bounds_clamp_to_the_call():
    finding = Finding(
        evaluator="e", category="c", title="t", evidence=[Evidence(start_s=1.0, end_s=19.0)]
    )
    set_clip_bounds(finding, 20.0, 3.0)
    assert (finding.clip_start_s, finding.clip_end_s) == (0.0, 20.0)


def test_clip_bounds_work_from_a_start_alone():
    finding = Finding(evaluator="e", category="c", title="t", evidence=[Evidence(start_s=5.0)])
    set_clip_bounds(finding, 20.0, 2.0)
    assert (finding.clip_start_s, finding.clip_end_s) == (3.0, 7.0)


def test_findings_without_timing_get_no_clip():
    finding = Finding(evaluator="e", category="c", title="t")
    set_clip_bounds(finding, 20.0, 3.0)
    assert finding.clip_start_s is None


async def test_every_finding_in_a_real_run_has_clamped_bounds(store):
    artifact = scripted_call()
    artifact, paths = write_call(store, artifact, caller=[(0, 2)], bot=[(9.5, 11)], duration_s=11.0)
    config = EvalConfig()
    config.timing.slow_response_s = 1.0
    result = await run_evaluation(artifact, paths, EvalDeps(config=config))
    timed = [f for f in result.findings if f.evidence]
    assert timed
    for finding in timed:
        assert 0 <= finding.clip_start_s <= finding.clip_end_s <= 11.0


def test_transcribe_is_first_in_the_default_order():
    assert isinstance(default_evaluators()[0], TranscribeEvaluator)
