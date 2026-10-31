"""The real eval pipeline around injected models, and the real subprocess worker."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    CallStatus,
    CriterionResult,
    EvalResult,
    EvaluatorRun,
    Finding,
    Speaker,
    Utterance,
    Verdict,
)
from voicelab.eval.config import EvalConfig
from voicelab.eval.context import EvalDeps, Skip
from voicelab.services.errors import Invalid
from voicelab.services.eval_worker import _load_profile
from voicelab.services.evaluation import (
    EvalJob,
    EvalService,
    SubprocessEvalRunner,
    ThreadEvalRunner,
    cap_reduced_evidence,
    check_evaluators,
    evaluate_call,
    evaluator_names,
    job_from_json,
    merge_partial,
)
from voicelab.services.events import EventHub, LiveGate
from voicelab.services.settings import Settings
from voicelab.store.artifacts import ArtifactStore
from voicelab.store.calls import CallIndex
from voicelab.store.sqlite import Database

RATE = 16000


def clip(spans, duration_s=3.0) -> AudioClip:
    samples = np.zeros(round(duration_s * RATE), dtype=np.float32)
    for start, end in spans:
        first, last = round(start * RATE), round(end * RATE)
        t = np.arange(last - first) / RATE
        samples[first:last] = 0.3 * np.sin(2 * np.pi * 220 * t)
    return AudioClip.from_samples(samples, RATE)


def stored_call(store: ArtifactStore, *, mode=CallMode.LIVE_BOTS, **fields) -> CallArtifact:
    artifact = CallArtifact(
        mode=mode,
        status=CallStatus.EVALUATING,
        duration_s=3.0,
        timeline=[
            Utterance(speaker=Speaker.BOT, start_s=0.0, end_s=1.0, intended_text="Hello."),
            Utterance(speaker=Speaker.CALLER, start_s=1.5, end_s=2.5),
        ],
        **fields,
    )
    store.write_call(artifact, clip([(1.5, 2.5)]), clip([(0.0, 1.0)]))
    return artifact


class Named:
    """An evaluator that reports one finding, or raises."""

    def __init__(self, name, *, raises=None, criteria=()):
        self.name = name
        self.raises = raises
        self.criteria = list(criteria)

    async def run(self, ctx, out):
        out.model = f"{self.name}-model"
        out.metrics["seen"] = True
        out.findings.append(Finding(evaluator=self.name, category="c", title=f"{self.name} note"))
        out.criteria.extend(self.criteria)
        if self.raises:
            raise self.raises


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


async def test_progress_streams_each_finding_and_evaluator(store):
    artifact = stored_call(store)
    messages = []
    result = await evaluate_call(
        store,
        EvalJob(artifact.id),
        EvalDeps(),
        messages.append,
        [Named("first"), Named("second", raises=Skip("nothing to do"))],
    )
    kinds = [next(iter(m)) for m in messages]
    assert kinds == ["finding", "evaluator", "finding", "evaluator"]
    report = messages[1]["evaluator"]
    assert report["name"] == "first"
    assert report["status"] == "ok"
    assert report["model"] == "first-model"
    assert report["metrics"] == {"seen": True}
    skipped = messages[3]["evaluator"]
    assert (skipped["status"], skipped["reason"]) == ("skipped", "nothing to do")
    assert store.read_eval(artifact.id) == result


async def test_an_evaluator_failure_does_not_stop_the_stream(store):
    artifact = stored_call(store)
    messages = []
    await evaluate_call(
        store,
        EvalJob(artifact.id),
        EvalDeps(),
        messages.append,
        [Named("broken", raises=RuntimeError("boom")), Named("after")],
    )
    reports = [m["evaluator"] for m in messages if "evaluator" in m]
    assert [r["status"] for r in reports] == ["error", "ok"]


def result_with(evaluators, findings=(), criteria=()):
    return EvalResult(
        call_id="call_1",
        verdict=Verdict.PASS,
        evaluators=[EvaluatorRun(name=name, status="ok") for name in evaluators],
        findings=list(findings),
        criteria=list(criteria),
    )


def test_merge_keeps_what_was_not_rerun_and_decides_again():
    previous = result_with(
        ["state", "timing"],
        findings=[
            Finding(id="old_state", evaluator="state", category="c", title="kept"),
            Finding(id="old_timing", evaluator="timing", category="c", title="replaced"),
        ],
        criteria=[
            CriterionResult(criterion_id="s", kind="state", text="t", status="pass"),
            CriterionResult(criterion_id="m", kind="must", text="t", status="pass"),
        ],
    )
    partial = result_with(
        ["transcript_judge"],
        criteria=[CriterionResult(criterion_id="m2", kind="must", text="t", status="fail")],
    )
    artifact = CallArtifact(id="call_1", mode=CallMode.LIVE_BOTS)
    merged = merge_partial(previous, partial, artifact, EvalConfig())
    assert {r.name for r in merged.evaluators} == {"state", "timing", "transcript_judge"}
    assert {c.criterion_id for c in merged.criteria} == {"s", "m2"}
    assert {f.id for f in merged.findings} == {"old_state", "old_timing"}
    assert merged.verdict == Verdict.FAIL


async def test_partial_run_only_runs_what_was_asked(store):
    artifact = stored_call(store)
    all_of_them = [Named("transcribe"), Named("timing"), Named("state")]
    await evaluate_call(store, EvalJob(artifact.id), EvalDeps(), lambda _: None, all_of_them)
    job = EvalJob(artifact.id, evaluators=["timing"])
    messages = []
    fresh = [Named("transcribe"), Named("timing"), Named("state")]
    result = await evaluate_call(store, job, EvalDeps(), messages.append, fresh)
    assert [m["evaluator"]["name"] for m in messages if "evaluator" in m] == ["timing"]
    assert {r.name for r in result.evaluators} == {"transcribe", "timing", "state"}


async def test_partial_run_of_an_unlisted_evaluator_gets_a_fresh_transcript(store):
    """A new evaluator, such as delivery, shouldn't need adding to a list to see the transcript."""
    artifact = stored_call(store)
    chosen = [Named("transcribe"), Named("delivery")]
    await evaluate_call(store, EvalJob(artifact.id), EvalDeps(), lambda _: None, chosen)
    messages = []
    fresh = [Named("transcribe"), Named("delivery")]
    job = EvalJob(artifact.id, evaluators=["delivery"])
    await evaluate_call(store, job, EvalDeps(), messages.append, fresh)
    assert [m["evaluator"]["name"] for m in messages if "evaluator" in m] == [
        "transcribe",
        "delivery",
    ]


async def test_partial_request_with_no_earlier_eval_runs_everything(store):
    artifact = stored_call(store)
    job = EvalJob(artifact.id, evaluators=["timing"])
    result = await evaluate_call(
        store, job, EvalDeps(), lambda _: None, [Named("timing"), Named("state")]
    )
    assert {r.name for r in result.evaluators} == {"timing", "state"}


@pytest.mark.parametrize("verdict", [Verdict.PASS, Verdict.FAIL])
def test_uploads_are_never_passed_or_failed_by_a_machine(verdict):
    artifact = CallArtifact(mode=CallMode.UPLOAD)
    result = EvalResult(call_id=artifact.id, verdict=verdict, summary="Verdict: x.")
    capped = cap_reduced_evidence(result, artifact)
    assert capped.verdict == Verdict.NEEDS_REVIEW
    assert verdict.value in capped.summary


def test_invalid_upload_stays_invalid():
    artifact = CallArtifact(mode=CallMode.UPLOAD)
    result = EvalResult(call_id=artifact.id, verdict=Verdict.INVALID)
    assert cap_reduced_evidence(result, artifact).verdict == Verdict.INVALID


def test_recorded_calls_keep_their_verdict():
    artifact = CallArtifact(mode=CallMode.HUMAN)
    result = EvalResult(call_id=artifact.id, verdict=Verdict.FAIL)
    assert cap_reduced_evidence(result, artifact).verdict == Verdict.FAIL


async def test_evaluate_call_applies_the_cap(store):
    artifact = stored_call(store, mode=CallMode.UPLOAD)
    passed = CriterionResult(criterion_id="c", kind="general", text="t", status="pass")
    result = await evaluate_call(
        store, EvalJob(artifact.id), EvalDeps(), lambda _: None, [Named("x", criteria=[passed])]
    )
    assert result.verdict == Verdict.NEEDS_REVIEW
    assert store.read_eval(artifact.id).verdict == Verdict.NEEDS_REVIEW


def test_known_evaluators_are_checked():
    assert "timing" in evaluator_names()
    assert check_evaluators(None) is None
    assert check_evaluators(["timing"]) == ["timing"]
    with pytest.raises(Invalid):
        check_evaluators(["timing", "nope"])


def test_job_round_trips_through_json():
    from voicelab.services.evaluation import _job_json

    job = EvalJob("call_1", profile="p", evaluators=["timing"], config=EvalConfig())
    job.config.timing.dead_air_s = 2.5
    again = job_from_json(_job_json(job))
    assert again.config.timing.dead_air_s == 2.5
    assert (again.call_id, again.profile, again.evaluators) == ("call_1", "p", ["timing"])


async def test_subprocess_worker_runs_the_real_pipeline(store, tmp_path):
    artifact = stored_call(store)
    settings = Settings(data_dir=tmp_path, profiles_dir=tmp_path / "profiles")
    (tmp_path / "profiles").mkdir()
    messages = []
    runner = SubprocessEvalRunner(settings, store)
    result = await runner(EvalJob(artifact.id), messages.append)
    assert store.read_eval(artifact.id) == result
    statuses = {r.name: r.status for r in result.evaluators}
    assert statuses["timing"] == "ok"
    assert statuses["transcribe"] == "skipped"  # no profile, so no STT
    assert len([m for m in messages if "evaluator" in m]) == len(result.evaluators)


def test_worker_finds_a_profile_past_a_broken_file(tmp_path):
    (tmp_path / "a.yaml").write_text("label: Local: not valid yaml\n")
    (tmp_path / "b.yaml").write_text("id: wanted\nlabel: Wanted\nroles: {}\n")
    assert _load_profile(tmp_path, "wanted").id == "wanted"
    assert _load_profile(tmp_path, None) is None
    with pytest.raises(ValueError, match="no profile"):
        _load_profile(tmp_path, "missing")


async def test_subprocess_failure_is_reported_with_where_to_look(store, tmp_path):
    settings = Settings(data_dir=tmp_path, profiles_dir=tmp_path / "profiles")
    runner = SubprocessEvalRunner(settings, store)
    with pytest.raises(RuntimeError, match="eval.log"):
        await runner(EvalJob("call_missing"), lambda _: None)
    assert (tmp_path / "logs" / "eval.log").exists()


def service_rig(tmp_path: Path, runner):
    database = Database(tmp_path / "x.sqlite3")
    store = ArtifactStore(tmp_path)
    calls = CallIndex(database)
    hub = EventHub()
    gate = LiveGate()
    service = EvalService(
        store=store,
        calls=calls,
        hub=hub,
        gate=gate,
        runner=runner(store),
        config=EvalConfig,
        default_profile=lambda: None,
    )
    return service, store, calls, hub, gate


def index(calls, artifact):
    calls.add(id=artifact.id, mode=artifact.mode.value, status="evaluating", timing_is_real=1)


async def test_thread_runner_with_real_evaluators_and_fake_models(tmp_path):
    service, store, calls, hub, _ = service_rig(
        tmp_path, lambda s: ThreadEvalRunner(s, lambda job: EvalDeps())
    )
    artifact = stored_call(store)
    index(calls, artifact)
    service.start()
    with hub.subscribe(f"call:{artifact.id}") as subscription:
        result = await asyncio.wait_for(service.submit(artifact.id), 30)
        seen = []
        while not subscription.queue.empty():
            seen.append(subscription.queue.get_nowait()["type"])
    await service.stop()
    assert result is not None
    assert seen[0] == "status"
    assert "evaluator" in seen
    assert seen[-1] == "eval"
    assert calls.get(artifact.id)["status"] == "done"
    assert store.read_artifact(artifact.id).status == CallStatus.DONE


async def test_only_one_job_runs_at_a_time(tmp_path):
    running = []
    peak = []

    def make(store):
        async def runner(job, sink):
            running.append(job.call_id)
            peak.append(len(running))
            await asyncio.sleep(0.01)
            running.remove(job.call_id)
            result = EvalResult(call_id=job.call_id, verdict=Verdict.PASS)
            store.write_eval(result)
            return result

        return runner

    service, store, calls, _, _ = service_rig(tmp_path, make)
    artifacts = [stored_call(store) for _ in range(3)]
    for artifact in artifacts:
        index(calls, artifact)
    service.start()
    await asyncio.gather(*[service.submit(a.id) for a in artifacts])
    await service.stop()
    assert max(peak) == 1


async def test_gate_holds_jobs_until_it_clears(tmp_path):
    started = asyncio.Event()

    def make(store):
        async def runner(job, sink):
            started.set()
            result = EvalResult(call_id=job.call_id, verdict=Verdict.PASS)
            store.write_eval(result)
            return result

        return runner

    service, store, calls, _, gate = service_rig(tmp_path, make)
    artifact = stored_call(store)
    index(calls, artifact)
    gate.hold("talk")
    service.start()
    done = service.submit(artifact.id)
    await asyncio.sleep(0.05)
    assert not started.is_set()
    gate.release("talk")
    await asyncio.wait_for(done, 5)
    await service.stop()
    assert started.is_set()


async def test_gate_stays_shut_until_every_holder_leaves():
    gate = LiveGate()
    gate.hold("a")
    gate.hold("b")
    gate.release("a")
    assert gate.active
    gate.release("b")
    assert not gate.active
    await asyncio.wait_for(gate.wait_clear(), 1)


async def test_a_stalled_subscriber_loses_old_messages_only():
    hub = EventHub()
    with hub.subscribe("t") as subscription:
        for number in range(2000):
            hub.publish("t", {"n": number})
        assert subscription.queue.qsize() == 1024
        assert subscription.queue.get_nowait() == {"n": 976}
    hub.publish("t", {"n": -1})
    assert subscription.queue.qsize() == 1023
