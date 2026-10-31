"""Turns evaluator output into one verdict and a summary. No LLM touches this."""

from __future__ import annotations

from dataclasses import dataclass, field

from voicelab.core.models import (
    CallArtifact,
    CriterionResult,
    EvaluatorRun,
    Finding,
    Scenario,
    SystemEvent,
    Verdict,
)
from voicelab.eval.config import VerdictConfig
from voicelab.eval.judging import verification_order

# The caller_validity evaluator emits these, and they decide whether a run counts as a test at all.
FACTS_CRITERION_ID = "caller_facts"
MATERIAL_FACT_BREACH = "material_fact_breach"
# The state evaluator emits this. At major it has no innocent reading: a deletion, or a write to
# someone else's account. At minor it's a side effect no scenario lists, like a helpful ticket.
UNEXPECTED_CHANGE = "unexpected_change"

_BLOCKING_SEVERITIES = ("major", "critical")
_RULE_KINDS = ("must", "must_not", "state")
_MAX_LISTED = 5
_MAX_DISAGREEMENTS = 3


@dataclass
class VerdictDecision:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Commission:
    """Something forbidden the bot did, and when, if the evidence says."""

    what: str
    at_s: float | None


def decide_verdict(
    artifact: CallArtifact,
    criteria: list[CriterionResult],
    findings: list[Finding],
    evaluators: list[EvaluatorRun],
    config: VerdictConfig | None = None,
) -> VerdictDecision:
    """Decide what the call shows about the bot.

    A commission is the bot doing something forbidden. It happened, so it fails the call whatever
    the caller went on to miss, unless the bot was acting on data the caller made up. An omission
    is the bot not doing something expected, and that only counts if the caller reached the step
    it depends on. A missed step with nothing to blame on the bot means the run tested nothing,
    unless it was the bot that ended the call.
    """
    config = config or VerdictConfig()
    scenario = artifact.scenario_snapshot
    required = {b.id for b in scenario.beats if b.required} if scenario else set()
    beats = {c.criterion_id: c for c in criteria if c.kind == "beat"}
    missed = [c for c in beats.values() if c.status == "fail" and c.criterion_id in required]
    breaches = [f for f in findings if f.category == MATERIAL_FACT_BREACH]
    commissions, proven = _commissions(artifact, criteria, findings)

    if breaches:
        return _with_breach(breaches, commissions, missed)
    if commissions:
        reasons = [c.what for c in commissions]
        if missed:
            steps = "; ".join(c.text for c in missed)
            reasons.append(
                f"the caller also missed a required checkpoint, which doesn't excuse this: {steps}"
            )
        return VerdictDecision(Verdict.FAIL, reasons)

    blamed, excused = _omissions(scenario, criteria, proven, beats, bool(missed))
    if missed and _ended_by(artifact, "bot_ended"):
        steps = "; ".join(c.text for c in missed)
        reasons = [f"the bot ended the call before the caller reached: {steps}"]
        return VerdictDecision(Verdict.FAIL, reasons + [_failed(c) for c in blamed])
    if blamed:
        return VerdictDecision(Verdict.FAIL, [_failed(c) for c in blamed])
    if missed:
        return VerdictDecision(Verdict.INVALID, _invalid_reasons(artifact, missed, excused))

    failures = _blocking_findings(criteria, findings)
    if failures:
        return VerdictDecision(Verdict.FAIL, failures)

    review = _review_reasons(criteria, findings, evaluators, config)
    review += [f"not counted against the bot, the caller never reached: {c.text}" for c in excused]
    if review:
        return VerdictDecision(Verdict.NEEDS_REVIEW, review)
    return VerdictDecision(Verdict.PASS)


def _commissions(
    artifact: CallArtifact, criteria: list[CriterionResult], findings: list[Finding]
) -> tuple[list[_Commission], set[tuple[str, str]]]:
    """What the bot did that it shouldn't have, and the criteria those came from."""
    scenario = artifact.scenario_snapshot
    untouched = {e.id for e in scenario.expected_state if e.op == "no_change"} if scenario else ()
    found: list[_Commission] = []
    order = verification_order(artifact)
    if artifact.has_tool_log and order.changed_unverified:
        _, event = order.changed
        found.append(
            _Commission(
                f"the bot did something forbidden: it used {event.name} before verifying the "
                "caller",
                event.at_s,
            )
        )
    proven: set[tuple[str, str]] = set()
    for criterion in criteria:
        if criterion.status != "fail":
            continue
        changed = criterion.kind == "state" and criterion.criterion_id in untouched
        # A judge's claim counts as proof only once its quote was found in the record.
        forbidden = criterion.kind == "must_not" and any(e.quote for e in criterion.evidence)
        if changed or forbidden:
            proven.add((criterion.kind, criterion.criterion_id))
            at_s = next((e.start_s for e in criterion.evidence if e.start_s is not None), None)
            what = "which broke: " if changed else ""
            lead = (
                "the bot did something forbidden, "
                if changed
                else "the bot did something forbidden: "
            )
            found.append(_Commission(f"{lead}{what}{criterion.text}", at_s))
    found += [
        _Commission(f"the bot did something forbidden: {f.title}", _earliest(f))
        for f in findings
        if f.category == UNEXPECTED_CHANGE and f.severity in _BLOCKING_SEVERITIES
    ]
    return found, proven


def _earliest(finding: Finding) -> float | None:
    return min((e.start_s for e in finding.evidence if e.start_s is not None), default=None)


def _with_breach(
    breaches: list[Finding], commissions: list[_Commission], missed: list[CriterionResult]
) -> VerdictDecision:
    """A breach voids what came after it. What the bot had already done, it did on honest data.

    A commission with no time can't be shown to come first, so it's treated as coming after.
    """
    broke = [f"caller broke its fact sheet on a material detail: {f.title}" for f in breaches]
    times = [t for t in map(_earliest, breaches) if t is not None]
    first_breach = min(times, default=None)
    before = [
        c
        for c in commissions
        if c.at_s is not None and first_breach is not None and c.at_s < first_breach
    ]
    if before:
        reasons = [c.what for c in before]
        return VerdictDecision(Verdict.FAIL, reasons + [f"this came before the {r}" for r in broke])
    reasons = [f"caller missed a required checkpoint: {c.text}" for c in missed] + broke
    reasons += [f"{c.what}, but after the breach, so on made-up data" for c in commissions]
    return VerdictDecision(Verdict.INVALID, reasons)


def _omissions(
    scenario: Scenario | None,
    criteria: list[CriterionResult],
    proven: set[tuple[str, str]],
    beats: dict[str, CriterionResult],
    any_missed: bool,
) -> tuple[list[CriterionResult], list[CriterionResult]]:
    """Failed rules that weren't commissions: those the bot answers for, and those it doesn't.

    A rule linked to a beat depends on that beat alone. An unlinked rule could depend on any of
    them, so one missed beat excuses it. A beat nobody could decide leaves the rule in neither
    list, and its `unknown` status sends the call to review.
    """
    links = _beat_links(scenario)
    blamed: list[CriterionResult] = []
    excused: list[CriterionResult] = []
    for criterion in criteria:
        key = (criterion.kind, criterion.criterion_id)
        if criterion.status != "fail" or criterion.kind not in _RULE_KINDS or key in proven:
            continue
        beat = beats.get(links.get(key) or "")
        if links.get(key) is None:
            (excused if any_missed else blamed).append(criterion)
        elif beat is None or beat.status == "pass":
            blamed.append(criterion)
        elif beat.status == "fail":
            excused.append(criterion)
    return blamed, excused


def _beat_links(scenario: Scenario | None) -> dict[tuple[str, str], str | None]:
    if scenario is None:
        return {}
    links: dict[tuple[str, str], str | None] = {}
    links.update({("must", r.id): r.needs_beat for r in scenario.must})
    links.update({("must_not", r.id): r.needs_beat for r in scenario.must_not})
    links.update({("state", e.id): e.needs_beat for e in scenario.expected_state})
    return links


def _ended_by(artifact: CallArtifact, kind: str) -> SystemEvent | None:
    return next(
        (e for e in artifact.timeline if isinstance(e, SystemEvent) and e.kind == kind), None
    )


def _invalid_reasons(
    artifact: CallArtifact, missed: list[CriterionResult], excused: list[CriterionResult]
) -> list[str]:
    reasons = [f"caller missed a required checkpoint: {c.text}" for c in missed]
    hung_up = _ended_by(artifact, "caller_hung_up")
    if hung_up is not None:
        detail = f" ({hung_up.detail})" if hung_up.detail else ""
        reasons.append(f"the caller hung up before getting there{detail}")
    elif _ended_by(artifact, "max_turns") is not None:
        reasons.append("the call hit the scenario's turn limit before getting there")
    if excused:
        rules = "; ".join(c.text for c in excused)
        reasons.append(f"not counted against the bot, the caller never reached: {rules}")
    return reasons


def _failed(criterion: CriterionResult) -> str:
    return f"failed {criterion.kind}: {criterion.text}"


def _blocking_findings(criteria: list[CriterionResult], findings: list[Finding]) -> list[str]:
    settled = {c.criterion_id for c in criteria if c.status == "fail" and c.kind in _RULE_KINDS}
    return [
        f"{f.severity} finding: {f.title}"
        for f in findings
        if not f.advisory and f.severity in _BLOCKING_SEVERITIES and f.criterion_id not in settled
    ]


def _review_reasons(
    criteria: list[CriterionResult],
    findings: list[Finding],
    evaluators: list[EvaluatorRun],
    config: VerdictConfig,
) -> list[str]:
    reasons = [f"could not decide: {c.text}" for c in criteria if c.status == "unknown"]
    reasons += [f"{run.name} errored: {run.reason}" for run in evaluators if run.status == "error"]
    if any(c.criterion_id == FACTS_CRITERION_ID and c.status == "fail" for c in criteria):
        reasons.append("caller slipped from its fact sheet on a non-material detail")
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
        if run.status == "skipped" and not run.reason.startswith("no scenario"):
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
        listed = "; ".join(reason.rstrip(".") for reason in decision.reasons[:_MAX_LISTED])
        extra = len(decision.reasons) - _MAX_LISTED
        lines.append(f"Why: {listed}" + (f"; and {extra} more." if extra > 0 else "."))
    if criteria:
        passed = sum(c.status == "pass" for c in criteria)
        failed = sum(c.status == "fail" for c in criteria)
        unknown = sum(c.status == "unknown" for c in criteria)
        lines.append(f"Criteria: {passed} passed, {failed} failed, {unknown} unknown.")
    review = [f for f in findings if f.severity != "info"]
    details = len(findings) - len(review)
    if review:
        advisory = sum(f.advisory for f in review)
        extra = f", plus {details} detail(s)" if details else ""
        lines.append(f"Findings: {len(review)}, of which {advisory} advisory{extra}.")
    elif details:
        lines.append(f"Findings: none to review, {details} detail(s).")
    unchecked = unchecked_items(artifact, evaluators)
    if unchecked:
        lines.append("Not checked: " + "; ".join(unchecked) + ".")
    disagreements = [d for r in evaluators for d in r.metrics.get("disagreements") or []]
    if disagreements:
        examples = " ".join(d["note"] for d in disagreements[:_MAX_DISAGREEMENTS])
        count = len(disagreements)
        lines.append(f"Measurements overruled the audio model {count} time(s). {examples}")
    limitation = next(
        (r.metrics["limitation"] for r in evaluators if "limitation" in r.metrics), None
    )
    if limitation:
        lines.append(limitation)
    return " ".join(lines)
