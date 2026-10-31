"""Queues post-call evaluation and keeps it off the event loop.

Evaluation is CPU-heavy, and the event loop may be carrying a live mic call, so the default runner
is a niced subprocess (see `eval_worker`). Jobs run one at a time, since they share one GPU with
everything else, and a job doesn't start while a mic call is live.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from voicelab.api.mapping import apply_decisions, to_api
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    CallStatus,
    CriterionResult,
    EvalResult,
    EvaluatorRun,
    Finding,
    Verdict,
)
from voicelab.core.profiles import Profile, build_stack
from voicelab.eval.config import EvalConfig
from voicelab.eval.context import EvalContext, EvalDeps, Evaluator, EvaluatorOutput
from voicelab.eval.pipeline import EvalProgress, default_evaluators, run_evaluation
from voicelab.eval.verdict import decide_verdict, write_summary
from voicelab.services.errors import Invalid
from voicelab.services.events import EventHub, LiveGate
from voicelab.services.settings import Settings
from voicelab.store.artifacts import ArtifactStore
from voicelab.store.calls import CallIndex
from voicelab.store.sqlite import timestamp

logger = logging.getLogger(__name__)

# Evaluators that read the transcript the first one writes.
_NEEDS_TRANSCRIPT = {"transcript_judge", "caller_validity", "pronunciation", "audio_listening"}
_CRITERION_OWNER = {
    "must": "transcript_judge",
    "must_not": "transcript_judge",
    "general": "transcript_judge",
    "state": "state",
    "beat": "caller_validity",
}
_LINE_LIMIT = 1 << 26  # one JSON line carries a whole transcript's words

Sink = Callable[[dict[str, Any]], Awaitable[None] | None]


def evaluator_names() -> list[str]:
    return [evaluator.name for evaluator in default_evaluators()]


def check_evaluators(names: Sequence[str] | None) -> list[str] | None:
    if names is None:
        return None
    unknown = sorted(set(names) - set(evaluator_names()))
    if unknown:
        raise Invalid(f"unknown evaluators: {', '.join(unknown)}", {"known": evaluator_names()})
    return list(names)


@dataclass
class EvalJob:
    call_id: str
    profile: str | None = None
    evaluators: list[str] | None = None  # None means all
    config: EvalConfig = field(default_factory=EvalConfig)


class EvalRunner(Protocol):
    async def __call__(self, job: EvalJob, sink: Sink) -> EvalResult:
        """Evaluate one call and write `eval.json`. `sink` gets findings and evaluator reports."""
        ...


class _Observed:
    """Wraps an evaluator so its findings can stream before the pipeline finishes."""

    def __init__(self, inner: Evaluator, on_output: Callable[[EvaluatorOutput], Any]) -> None:
        self.inner = inner
        self.name = inner.name
        self.on_output = on_output

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        try:
            await self.inner.run(ctx, out)
        finally:
            result = self.on_output(out)
            if inspect.isawaitable(result):
                await result


def merge_partial(
    previous: EvalResult,
    partial: EvalResult,
    artifact: CallArtifact,
    config: EvalConfig,
) -> EvalResult:
    """Replace what the re-run evaluators produced, keep the rest, and decide again."""
    rerun = {run.name for run in partial.evaluators}
    runs = [r for r in previous.evaluators if r.name not in rerun] + partial.evaluators
    findings: list[Finding] = [
        f for f in previous.findings if f.evaluator not in rerun
    ] + partial.findings
    criteria: list[CriterionResult] = [
        c for c in previous.criteria if _CRITERION_OWNER.get(c.kind) not in rerun
    ] + partial.criteria
    decision = decide_verdict(artifact, criteria, findings, runs, config.verdict)
    return EvalResult(
        call_id=artifact.id,
        verdict=decision.verdict,
        summary=write_summary(decision, artifact, criteria, findings, runs),
        criteria=criteria,
        findings=findings,
        evaluators=runs,
    )


def cap_reduced_evidence(result: EvalResult, artifact: CallArtifact) -> EvalResult:
    """An upload never has database snapshots, so its evidence is always reduced, and it may not
    be called pass or fail by a machine. Invalid stands: that isn't a judgement of the bot."""
    if artifact.mode != CallMode.UPLOAD or result.verdict not in (Verdict.PASS, Verdict.FAIL):
        return result
    note = f"Machine verdict would be {result.verdict.value}, held for review: reduced evidence."
    return result.model_copy(
        update={"verdict": Verdict.NEEDS_REVIEW, "summary": f"{note} {result.summary}"}
    )


async def evaluate_call(
    store: ArtifactStore,
    job: EvalJob,
    deps: EvalDeps,
    sink: Sink,
    evaluators: Sequence[Evaluator] | None = None,
) -> EvalResult:
    """Evaluate a stored call, report through `sink`, and write `eval.json`. Used on both sides of
    the subprocess boundary."""
    artifact = store.read_artifact(job.call_id)
    previous = store.read_eval(job.call_id)
    chosen = list(evaluators) if evaluators is not None else default_evaluators()
    partial = job.evaluators is not None and previous is not None
    if partial:
        wanted = set(job.evaluators)
        if wanted & _NEEDS_TRANSCRIPT:
            wanted.add("transcribe")
        chosen = [evaluator for evaluator in chosen if evaluator.name in wanted]

    outputs: dict[str, EvaluatorOutput] = {}
    started: dict[str, float] = {}

    def observed(evaluator: Evaluator) -> _Observed:
        async def report(out: EvaluatorOutput) -> None:
            outputs[evaluator.name] = out
            for finding in out.findings:
                await _deliver(sink, {"finding": finding.model_dump(mode="json")})

        return _Observed(evaluator, report)

    async def on_progress(progress: EvalProgress) -> None:
        if progress.status == "running":
            started.setdefault(progress.evaluator, time.perf_counter())
            return
        out = outputs.get(progress.evaluator, EvaluatorOutput())
        run = EvaluatorRun(
            name=progress.evaluator,
            status=progress.status,
            reason=progress.detail,
            duration_s=round(time.perf_counter() - started[progress.evaluator], 3),
            model=out.model,
            metrics=out.metrics,
        )
        await _deliver(sink, {"evaluator": run.model_dump(mode="json")})

    result = await run_evaluation(
        artifact,
        store.paths(job.call_id),
        deps,
        on_progress=on_progress,
        evaluators=[observed(evaluator) for evaluator in chosen],
    )
    if partial:
        result = merge_partial(previous, result, artifact, deps.config)
    result = cap_reduced_evidence(result, artifact)
    store.write_eval(result)
    return result


async def _deliver(sink: Sink, message: dict[str, Any]) -> None:
    result = sink(message)
    if inspect.isawaitable(result):
        await result


class ThreadEvalRunner:
    """Runs evaluation on a thread with its own event loop. For injected fakes: a subprocess can't
    see them, and adapters bound to the main loop can't be used from another one."""

    def __init__(
        self,
        store: ArtifactStore,
        deps: Callable[[EvalJob], EvalDeps],
        evaluators: Callable[[], Sequence[Evaluator]] | None = None,
    ) -> None:
        self.store = store
        self.deps = deps
        self.evaluators = evaluators

    async def __call__(self, job: EvalJob, sink: Sink) -> EvalResult:
        loop = asyncio.get_running_loop()

        def forward(message: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(sink, message)

        chosen = self.evaluators() if self.evaluators else None
        coroutine = evaluate_call(self.store, job, self.deps(job), forward, chosen)
        return await asyncio.to_thread(asyncio.run, coroutine)


class SubprocessEvalRunner:
    """Runs `python -m voicelab.services.eval_worker` for each job."""

    def __init__(self, settings: Settings, store: ArtifactStore) -> None:
        self.settings = settings
        self.store = store

    async def __call__(self, job: EvalJob, sink: Sink) -> EvalResult:
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "voicelab.services.eval_worker",
            "--data-dir",
            str(self.settings.data_dir),
            "--profiles-dir",
            str(self.settings.profiles_dir),
        ]
        with (self.settings.log_dir / "eval.log").open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                limit=_LINE_LIMIT,
            )
        try:
            process.stdin.write(_job_json(job).encode())
            await process.stdin.drain()
            process.stdin.close()
            async for line in process.stdout:
                message = _parse_line(line)
                if message is not None:
                    await _deliver(sink, message)
            code = await process.wait()
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if code != 0:
            raise RuntimeError(f"evaluation worker exited with {code}, see data/logs/eval.log")
        result = self.store.read_eval(job.call_id)
        if result is None:
            raise RuntimeError("evaluation worker wrote no result")
        return result


def _job_json(job: EvalJob) -> str:
    return json.dumps(
        {
            "call_id": job.call_id,
            "profile": job.profile,
            "evaluators": job.evaluators,
            "config": job.config.model_dump(mode="json"),
        }
    )


def job_from_json(text: str) -> EvalJob:
    data = json.loads(text)
    return EvalJob(
        call_id=data["call_id"],
        profile=data["profile"],
        evaluators=data["evaluators"],
        config=EvalConfig.model_validate(data["config"]),
    )


def _parse_line(line: bytes) -> dict[str, Any] | None:
    try:
        message = json.loads(line)
    except ValueError:  # a library printed to stdout
        return None
    return message if isinstance(message, dict) else None


def deps_from_profile(profile: Profile | None, config: EvalConfig) -> EvalDeps:
    """Evaluate with what the profile offers. No profile still gets timing and audio checks."""
    if profile is None:
        return EvalDeps(config=config)
    stack = build_stack(profile)
    return EvalDeps(
        stt=stack.stt_eval, judge=stack.llm_judge, analyst=stack.audio_analyst, config=config
    )


@dataclass
class _Queued:
    job: EvalJob
    done: asyncio.Future[EvalResult | None]


class EvalService:
    def __init__(
        self,
        *,
        store: ArtifactStore,
        calls: CallIndex,
        hub: EventHub,
        gate: LiveGate,
        runner: EvalRunner,
        config: Callable[[], EvalConfig],
        default_profile: Callable[[], str | None],
    ) -> None:
        self.store = store
        self.calls = calls
        self.hub = hub
        self.gate = gate
        self.runner = runner
        self.config = config
        self.default_profile = default_profile
        self.on_settled: Callable[[str], None] | None = None  # the run queue listens here
        self._queue: asyncio.Queue[_Queued] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._pending: set[str] = set()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="eval-queue")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        while not self._queue.empty():
            self._queue.get_nowait().done.cancel()

    @property
    def queued(self) -> int:
        return len(self._pending)

    def submit(
        self, call_id: str, evaluators: list[str] | None = None
    ) -> asyncio.Future[EvalResult | None]:
        """Queue a call. Its status turns `evaluating` now, so a client that follows the accepted
        response with a WebSocket can't see the old status."""
        self._set_status(call_id, CallStatus.EVALUATING)
        self.hub.publish(f"call:{call_id}", {"type": "status", "status": "evaluating"})
        profile = self._profile_for(call_id)
        job = EvalJob(call_id, profile=profile, evaluators=evaluators, config=self.config())
        done: asyncio.Future[EvalResult | None] = asyncio.get_running_loop().create_future()
        self._pending.add(call_id)
        self._queue.put_nowait(_Queued(job, done))
        return done

    def _profile_for(self, call_id: str) -> str | None:
        row = self.calls.get(call_id)
        return (row["profile"] if row else None) or self.default_profile()

    async def _loop(self) -> None:
        while True:
            item = await self._queue.get()
            await self.gate.wait_clear()
            await self._run(item)

    async def _run(self, item: _Queued) -> None:
        call_id = item.job.call_id
        topic = f"call:{call_id}"

        def forward(message: dict[str, Any]) -> None:
            if "finding" in message:
                finding = Finding.model_validate(message["finding"])
                self.hub.publish(topic, {"type": "finding", "finding": to_api(finding)})
            elif "evaluator" in message:
                run = EvaluatorRun.model_validate(message["evaluator"])
                self.hub.publish(topic, {"type": "evaluator", "evaluator": to_api(run)})

        result: EvalResult | None = None
        try:
            result = await self.runner(item.job, forward)
            self._finish(result)
        except asyncio.CancelledError:
            item.done.cancel()
            raise
        except Exception as exc:
            logger.exception("evaluation of %s failed", call_id)
            self._fail(call_id, f"evaluation failed: {exc}")
        finally:
            self._pending.discard(call_id)
        if not item.done.done():
            item.done.set_result(result)
        if self.on_settled is not None:
            self.on_settled(call_id)

    def _finish(self, result: EvalResult) -> None:
        artifact = self._write_status(result.call_id, CallStatus.DONE, error=None)
        self.calls.record_eval(artifact, result)
        self.calls.update(result.call_id, status="done", error=None, finished_at=timestamp())
        topic = f"call:{result.call_id}"
        self.hub.publish(topic, {"type": "status", "status": "done"})
        self.hub.publish(topic, {"type": "eval", "eval": to_api(self._decided(result))})

    def _decided(self, result: EvalResult) -> EvalResult:
        return apply_decisions(result, self.calls.finding_decisions(result.call_id))

    def _fail(self, call_id: str, message: str) -> None:
        self._write_status(call_id, CallStatus.FAILED, error=message)
        self.calls.update(call_id, status="failed", error=message, finished_at=timestamp())
        topic = f"call:{call_id}"
        self.hub.publish(topic, {"type": "status", "status": "failed"})
        self.hub.publish(topic, {"type": "error", "message": message})

    def _set_status(self, call_id: str, status: CallStatus) -> None:
        self._write_status(call_id, status, error=None)
        self.calls.update(call_id, status=status.value, error=None)

    def _write_status(self, call_id: str, status: CallStatus, *, error: str | None) -> CallArtifact:
        artifact = self.store.read_artifact(call_id)
        artifact.status = status
        artifact.error = error
        self.store.write_artifact(artifact)
        return artifact
