"""Whether the caller bot did its job. Failure here is a broken test, not a bot failure."""

from __future__ import annotations

from voicelab.core.models import CallMode, CriterionResult, Evidence, Finding, Speaker
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.judging import (
    JUDGE_SCHEMA,
    JudgeStatus,
    ask_json,
    render_prompt,
    verify_judgment,
)
from voicelab.eval.transcript import render_transcript

FACTS_CRITERION_ID = "caller_facts"
FACTS_TEXT = "Caller stayed within its fact sheet"
_BOT_TO_BOT = (CallMode.LIVE_BOTS, CallMode.TURN_BASED)


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
                judgment = verify_judgment(payload, ctx.index, quote_needed_for="pass")
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
            payload, out.model = await ask_json(ctx.deps.judge, prompt, JUDGE_SCHEMA, ctx.config)
        except Exception as exc:
            out.criteria.append(_facts_criterion("unknown", f"judge failed: {exc}", []))
            return
        judgment = verify_judgment(
            payload, ctx.index, quote_needed_for="fail", prefer=Speaker.CALLER
        )
        evidence = [e for e in judgment.evidence if e.speaker == Speaker.CALLER]
        status, reason = judgment.status, judgment.reason
        if status == "fail" and not evidence:
            status, reason = "unknown", f"The cited words were not the caller's. {reason}"
        out.criteria.append(_facts_criterion(status, reason, evidence))
        if status == "fail":
            out.findings.append(
                Finding(
                    evaluator=self.name,
                    category="fact_sheet_breach",
                    severity="major",
                    advisory=True,
                    title=FACTS_TEXT,
                    detail=reason,
                    evidence=evidence,
                    criterion_id=FACTS_CRITERION_ID,
                )
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
