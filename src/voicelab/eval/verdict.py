"""Turns evaluator output into one verdict and a summary. No LLM touches this."""

from __future__ import annotations

from dataclasses import dataclass, field

from voicelab.core.models import (
    CallArtifact,
    CriterionResult,
    EvaluatorRun,
    Finding,
    Verdict,
)
from voicelab.eval.config import VerdictConfig

_BLOCKING_SEVERITIES = ("major", "critical")
_MAX_LISTED = 5


@dataclass
class VerdictDecision:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)


def decide_verdict(
    artifact: CallArtifact,
    criteria: list[CriterionResult],
    findings: list[Finding],
    evaluators: list[EvaluatorRun],
    config: VerdictConfig | None = None,
) -> VerdictDecision:
    """Apply the rules in order: invalid, fail, needs_review, pass."""
    config = config or VerdictConfig()
    scenario = artifact.scenario_snapshot
    required_beats = {b.id for b in scenario.beats if b.required} if scenario else set()
    missed = [
        c
        for c in criteria
        if c.kind == "beat" and c.status == "fail" and c.criterion_id in required_beats
    ]
    if missed:
        reasons = [f"caller missed a required checkpoint: {c.text}" for c in missed]
        return VerdictDecision(Verdict.INVALID, reasons)

    failures = _failures(criteria, findings)
    if failures:
        return VerdictDecision(Verdict.FAIL, failures)

    review = _review_reasons(criteria, findings, evaluators, config)
    if review:
        return VerdictDecision(Verdict.NEEDS_REVIEW, review)
    return VerdictDecision(Verdict.PASS)


def _failures(criteria: list[CriterionResult], findings: list[Finding]) -> list[str]:
    failed = [c for c in criteria if c.status == "fail" and c.kind in ("must", "must_not", "state")]
    failures = [f"failed {c.kind}: {c.text}" for c in failed]
    already_listed = {c.criterion_id for c in failed}
    failures += [
        f"{f.severity} finding: {f.title}"
        for f in findings
        if not f.advisory
        and f.severity in _BLOCKING_SEVERITIES
        and f.criterion_id not in already_listed
    ]
    return failures


def _review_reasons(
    criteria: list[CriterionResult],
    findings: list[Finding],
    evaluators: list[EvaluatorRun],
    config: VerdictConfig,
) -> list[str]:
    reasons = [f"could not decide: {c.text}" for c in criteria if c.status == "unknown"]
    reasons += [f"{run.name} errored: {run.reason}" for run in evaluators if run.status == "error"]
    score = sum(config.advisory_weights.get(f.severity, 0) for f in findings if f.advisory)
    if score >= config.review_score:
        reasons.append(
            f"advisory findings add up to {score}, review starts at {config.review_score}"
        )
    if not any(c.status == "pass" for c in criteria):
        # Nothing failing isn't a pass when nothing was checked.
        reasons.append("no criterion could be confirmed")
    return reasons


def unchecked_items(artifact: CallArtifact, evaluators: list[EvaluatorRun]) -> list[str]:
    """What this call's evidence couldn't support, for the summary."""
    items: list[str] = []
    if artifact.scenario_snapshot is None:
        items.append(
            "scenario rules, expected database state, and caller checkpoints (no scenario)"
        )
    for run in evaluators:
        if run.status == "skipped" and "scenario" not in run.reason:
            items.append(f"{run.name} ({run.reason})")
        items.extend(f"{run.name}: {item}" for item in run.metrics.get("not_checked") or [])
        for what, why in (run.metrics.get("skipped") or {}).items():
            items.append(f"{what} ({why})")
    return list(dict.fromkeys(items))


def write_summary(
    decision: VerdictDecision,
    artifact: CallArtifact,
    criteria: list[CriterionResult],
    findings: list[Finding],
    evaluators: list[EvaluatorRun],
) -> str:
    lines = [f"Verdict: {decision.verdict.value}."]
    if decision.reasons:
        listed = "; ".join(decision.reasons[:_MAX_LISTED])
        extra = len(decision.reasons) - _MAX_LISTED
        lines.append(f"Why: {listed}" + (f"; and {extra} more." if extra > 0 else "."))
    if criteria:
        passed = sum(c.status == "pass" for c in criteria)
        failed = sum(c.status == "fail" for c in criteria)
        unknown = sum(c.status == "unknown" for c in criteria)
        lines.append(f"Criteria: {passed} passed, {failed} failed, {unknown} unknown.")
    if findings:
        advisory = sum(f.advisory for f in findings)
        lines.append(f"Findings: {len(findings)}, of which {advisory} advisory.")
    unchecked = unchecked_items(artifact, evaluators)
    if unchecked:
        lines.append("Not checked: " + "; ".join(unchecked) + ".")
    limitation = next(
        (r.metrics["limitation"] for r in evaluators if "limitation" in r.metrics), None
    )
    if limitation:
        lines.append(limitation)
    return " ".join(lines)
