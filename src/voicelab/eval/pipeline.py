"""Runs the evaluators over one call and produces its result.

Evaluators run in order, because later ones read the independent transcript and the clips flagged
before them. A failure is recorded against the evaluator that had it, and the rest still run.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from voicelab.core.models import (
    CallArtifact,
    CriterionResult,
    EvalResult,
    EvaluatorRun,
    Evidence,
    Finding,
)
from voicelab.eval.context import (
    EvalContext,
    EvalDeps,
    Evaluator,
    EvaluatorOutput,
    ProgressNote,
    Skip,
)
from voicelab.eval.evaluators.audio_listening import AudioListeningEvaluator
from voicelab.eval.evaluators.audio_quality import AudioQualityEvaluator
from voicelab.eval.evaluators.caller_validity import CallerValidityEvaluator
from voicelab.eval.evaluators.delivery import DeliveryEvaluator
from voicelab.eval.evaluators.pronunciation import PronunciationEvaluator
from voicelab.eval.evaluators.state import StateEvaluator
from voicelab.eval.evaluators.timing import TimingEvaluator
from voicelab.eval.evaluators.transcribe import TranscribeEvaluator
from voicelab.eval.evaluators.transcript_judge import TranscriptJudgeEvaluator
from voicelab.eval.verdict import decide_verdict, write_summary
from voicelab.store.artifacts import CallPaths

logger = logging.getLogger(__name__)

_CLIP_JOIN_S = 10.0  # evidence this near the clip plays with it


@dataclass(frozen=True)
class EvalProgress:
    """One step of a run. `index` is zero-based, and `detail` narrates a slow evaluator."""

    evaluator: str
    index: int
    total: int
    status: Literal["running", "ok", "skipped", "error"]
    detail: str = ""


ProgressCallback = Callable[[EvalProgress], Awaitable[None] | None]


def default_evaluators() -> list[Evaluator]:
    return [
        TranscribeEvaluator(),
        TimingEvaluator(),
        StateEvaluator(),
        TranscriptJudgeEvaluator(),
        CallerValidityEvaluator(),
        PronunciationEvaluator(),
        AudioQualityEvaluator(),
        DeliveryEvaluator(),  # before listening, so measurements can overrule the audio model
        AudioListeningEvaluator(),
    ]


async def _report(callback: ProgressCallback | None, progress: EvalProgress) -> None:
    if callback is None:
        return
    # A dashboard that went away must not take the evaluation down with it.
    try:
        result = callback(progress)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("eval progress callback failed")


def _progress_note(
    callback: ProgressCallback | None, name: str, index: int, total: int
) -> ProgressNote:
    async def note(detail: str) -> None:
        await _report(callback, EvalProgress(name, index, total, "running", detail))

    return note


async def _run_one(evaluator: Evaluator, ctx: EvalContext) -> tuple[EvaluatorRun, EvaluatorOutput]:
    output = EvaluatorOutput()
    status: Literal["ok", "skipped", "error"] = "ok"
    reason = ""
    started = time.perf_counter()
    try:
        await evaluator.run(ctx, output)
    except Skip as skip:
        status, reason = "skipped", str(skip)
    except Exception as exc:
        logger.exception("evaluator %s failed", evaluator.name)
        status, reason = "error", f"{type(exc).__name__}: {exc}"
    run = EvaluatorRun(
        name=evaluator.name,
        status=status,
        reason=reason,
        duration_s=round(time.perf_counter() - started, 3),
        model=output.model,
        metrics=output.metrics,
    )
    return run, output


def set_clip_bounds(finding: Finding, duration_s: float, context_s: float) -> None:
    """Give a finding playable bounds around its first evidence, clamped to the call.

    Evidence close to it joins the clip. Evidence minutes away doesn't, since a clip is one moment
    to listen to, and a finding that happened all call shouldn't play the whole call.
    """
    timed = [e for e in finding.evidence if e.start_s is not None]
    if not timed:
        return
    start, end = timed[0].start_s, _end_of(timed[0])
    for evidence in timed[1:]:
        if evidence.start_s > end + _CLIP_JOIN_S or _end_of(evidence) < start - _CLIP_JOIN_S:
            break
        start, end = min(start, evidence.start_s), max(end, _end_of(evidence))
    finding.clip_start_s = max(0.0, start - context_s)
    finding.clip_end_s = min(duration_s, end + context_s)


def _end_of(evidence: Evidence) -> float:
    return evidence.end_s if evidence.end_s is not None else evidence.start_s


async def run_evaluation(
    artifact: CallArtifact,
    paths: CallPaths,
    deps: EvalDeps,
    *,
    on_progress: ProgressCallback | None = None,
    evaluators: Sequence[Evaluator] | None = None,
) -> EvalResult:
    """Evaluate one finished call. Never raises for an evaluator failure, only records it."""
    chosen = list(evaluators) if evaluators is not None else default_evaluators()
    ctx = EvalContext(artifact, paths, deps)
    runs: list[EvaluatorRun] = []
    criteria: list[CriterionResult] = []
    total = len(chosen)

    for index, evaluator in enumerate(chosen):
        ctx.note_progress = _progress_note(on_progress, evaluator.name, index, total)
        await ctx.note_progress("")
        run, output = await _run_one(evaluator, ctx)
        runs.append(run)
        criteria.extend(output.criteria)
        ctx.findings.extend(output.findings)
        await _report(on_progress, EvalProgress(run.name, index, total, run.status, run.reason))

    for finding in ctx.findings:
        set_clip_bounds(finding, artifact.duration_s, deps.config.clip_context_s)
    decision = decide_verdict(artifact, criteria, ctx.findings, runs, deps.config.verdict)
    return EvalResult(
        call_id=artifact.id,
        verdict=decision.verdict,
        summary=write_summary(decision, artifact, criteria, ctx.findings, runs),
        criteria=criteria,
        findings=ctx.findings,
        evaluators=runs,
    )
