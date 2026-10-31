"""Whether the caller bot did its job. Failure here is a broken test, not a bot failure."""

from __future__ import annotations

import re
from typing import Any

from voicelab.core.models import Beat, CallMode, CriterionResult, Evidence, Finding, Speaker
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.judging import (
    JUDGE_SCHEMA,
    JudgeStatus,
    Judgment,
    ask_json,
    render_prompt,
    verify_judgment,
)
from voicelab.eval.transcript import render_transcript
from voicelab.eval.verdict import FACTS_CRITERION_ID, MATERIAL_FACT_BREACH

FACTS_TEXT = "Caller stayed within its fact sheet"
_BOT_TO_BOT = (CallMode.LIVE_BOTS, CallMode.TURN_BASED)

FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail", "unknown"]},
        "reason": {"type": "string"},
        "breaches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "material": {"type": "boolean"},
                    "why": {"type": "string"},
                },
                "required": ["quote", "material"],
            },
        },
    },
    "required": ["status", "reason", "breaches"],
}

# A checkpoint is what the caller did, so the caller's words prove it. These say what happened to
# the caller instead, and there the bot's own words are the proof.
_RECEIVED_CHECKPOINT = re.compile(r"\b(?:received|was told|were told|was given)\b", re.IGNORECASE)


def checkpoint_actor(beat: Beat) -> Speaker | None:
    """Whose words can prove this beat. None means either side's."""
    text = beat.checkpoint or beat.instruction
    return None if _RECEIVED_CHECKPOINT.search(text) else Speaker.CALLER


class CallerValidityEvaluator:
    name = "caller_validity"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        if ctx.artifact.mode not in _BOT_TO_BOT:
            raise Skip(f"only bot-to-bot calls have a scripted caller, this is {ctx.artifact.mode}")
        scenario = ctx.scenario
        if scenario is None:
            raise Skip("no scenario, so the caller had no script to follow")
        judge = ctx.deps.judge
        missing = None if judge else "no judge LLM configured"
        if not any(t.words for t in ctx.transcripts.values()):
            missing = missing or "no independent transcript"
        if missing:
            out.criteria.extend(
                CriterionResult(
                    criterion_id=b.id, kind="beat", text=b.instruction, status="unknown",
                    reason=missing,
                )
                for b in scenario.beats
            )  # fmt: skip
            raise Skip(missing)
        out.model = judge.name

        transcript = render_transcript(ctx.transcripts)
        reached: list[str] = []
        missed: list[str] = []
        undecided: list[str] = []
        errors = 0
        for position, beat in enumerate(scenario.beats, start=1):
            await ctx.note_progress(f"beat {position} of {len(scenario.beats)}")
            prompt = render_prompt(
                "caller_beat",
                goal=scenario.goal,
                transcript=transcript,
                instruction=beat.instruction,
                checkpoint=beat.checkpoint or beat.instruction,
            )
            try:
                payload, out.model = await ask_json(judge, prompt, JUDGE_SCHEMA, ctx.config)
                judgment = verify_judgment(
                    payload, ctx.index, quote_needed_for="pass", actor=checkpoint_actor(beat)
                )
            except Exception as exc:  # one unusable reply leaves one beat unknown
                errors += 1
                undecided.append(beat.id)
                out.criteria.append(
                    CriterionResult(
                        criterion_id=beat.id, kind="beat", text=beat.instruction, status="unknown",
                        reason=f"judge failed: {exc}",
                    )
                )  # fmt: skip
                continue
            out.criteria.append(
                CriterionResult(
                    criterion_id=beat.id,
                    kind="beat",
                    text=beat.instruction,
                    status=judgment.status,
                    reason=judgment.reason,
                    evidence=judgment.evidence,
                )
            )
            {"pass": reached, "fail": missed, "unknown": undecided}[judgment.status].append(beat.id)
            if judgment.status == "fail":
                out.findings.append(
                    Finding(
                        evaluator=self.name,
                        category="beat_missed",
                        severity="major" if beat.required else "minor",
                        # Verdict logic reads this as invalid. It says nothing about the bot.
                        advisory=True,
                        title=f"Caller missed checkpoint: {beat.checkpoint or beat.instruction}",
                        detail=judgment.reason,
                        evidence=judgment.evidence,
                        criterion_id=beat.id,
                    )
                )
        if scenario.facts:
            await self._check_facts(ctx, out, transcript)
        out.metrics.update({"reached": reached, "missed": missed, "unknown": undecided})
        if scenario.beats and errors == len(scenario.beats):
            raise RuntimeError(f"judge failed on all {errors} beats")

    async def _check_facts(self, ctx: EvalContext, out: EvaluatorOutput, transcript: str) -> None:
        facts = "\n".join(f"- {key}: {value}" for key, value in ctx.scenario.facts.items())
        prompt = render_prompt("caller_facts", facts=facts, transcript=transcript)
        try:
            payload, out.model = await ask_json(ctx.deps.judge, prompt, FACTS_SCHEMA, ctx.config)
        except Exception as exc:
            out.criteria.append(_facts_criterion("unknown", f"judge failed: {exc}", []))
            return
        breaches = [b for b in payload.get("breaches") or [] if isinstance(b, dict)]
        cited = {**payload, "quotes": [b.get("quote") for b in breaches]}
        judgment = verify_judgment(cited, ctx.index, quote_needed_for="fail", actor=Speaker.CALLER)
        status, reason = judgment.status, judgment.reason
        problem = _breach_problem(judgment, breaches)
        if problem:
            status, reason = "unknown", f"{problem} {reason}"
        out.criteria.append(_facts_criterion(status, reason, judgment.evidence))
        if status == "fail":
            pairs = zip(breaches, judgment.evidence, strict=True)
            out.findings.extend(_breach_finding(b, e, reason) for b, e in pairs)


def _breach_problem(judgment: Judgment, breaches: list[dict[str, Any]]) -> str | None:
    """Why a reply can't be taken at face value. Every breach has to stand up on its own."""
    if judgment.status == "pass" and breaches:
        return "The judge passed the caller but listed breaches."
    if judgment.status != "fail":
        return None
    if len(judgment.evidence) != len(breaches):
        return "A breach came without a usable quote."
    if any(e.speaker != Speaker.CALLER for e in judgment.evidence):
        return "The cited words were not the caller's."
    if not all(isinstance(b.get("material"), bool) for b in breaches):
        return "The judge didn't say whether every breach was material."
    return None


def _breach_finding(breach: dict[str, Any], evidence: Evidence, reason: str) -> Finding:
    material = breach["material"]
    label = "material detail" if material else "minor detail"
    return Finding(
        evaluator="caller_validity",
        category=MATERIAL_FACT_BREACH if material else "fact_sheet_slip",
        severity="major" if material else "minor",
        # Verdict logic reads these. They describe the test, so they never count against the bot.
        advisory=True,
        title=f"Caller went outside its fact sheet on a {label}: {evidence.quote}",
        detail=str(breach.get("why") or reason),
        evidence=[evidence],
        criterion_id=FACTS_CRITERION_ID,
    )


def _facts_criterion(status: JudgeStatus, reason: str, evidence: list[Evidence]) -> CriterionResult:
    return CriterionResult(
        criterion_id=FACTS_CRITERION_ID,
        kind="beat",
        text=FACTS_TEXT,
        status=status,
        reason=reason,
        evidence=evidence,
    )
