import json

import pytest
from evalfakes import FakeLLM, verdict_json, write_call
from evalfakes import transcript as make_transcript

from voicelab.core.models import (
    Beat,
    CallArtifact,
    CallMode,
    Rule,
    Scenario,
    Speaker,
    ToolCallEvent,
)
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput, Skip
from voicelab.eval.evaluators.caller_validity import FACTS_CRITERION_ID, CallerValidityEvaluator
from voicelab.eval.evaluators.transcript_judge import TranscriptJudgeEvaluator

CALLER_SAID = "hi I was charged twice for invoice FH-4419"
BOT_SAID = "your card ending 4242 was refunded forty nine dollars goodbye"


def scenario(**kwargs) -> Scenario:
    return Scenario(title="Double charge", goal="get a refund", **kwargs)


def context(store, artifact: CallArtifact, judge=None, heard=True) -> EvalContext:
    artifact, paths = write_call(store, artifact, duration_s=30.0)
    ctx = EvalContext(artifact, paths, EvalDeps(judge=judge))
    if heard:
        ctx.transcripts[Speaker.CALLER] = make_transcript((CALLER_SAID, 0.0))
        ctx.transcripts[Speaker.BOT] = make_transcript((BOT_SAID, 10.0))
    return ctx


async def run(evaluator, ctx) -> EvaluatorOutput:
    out = EvaluatorOutput()
    await evaluator.run(ctx, out)
    return out


def by_id(out: EvaluatorOutput, criterion_id: str):
    return next(c for c in out.criteria if c.criterion_id == criterion_id)


def judge_for(mapping: dict[str, dict]) -> FakeLLM:
    """Replies according to which criterion text the prompt asks about."""

    def reply(prompt: str) -> dict:
        tail = prompt.split("Criterion to check:")[-1]
        for text, payload in mapping.items():
            if text in tail:
                return payload
        return verdict_json("pass")

    return FakeLLM(reply)


async def test_each_rule_and_general_criterion_gets_its_own_call(store):
    rules = scenario(
        must=[Rule(text="verify the billing zip")], must_not=[Rule(text="read a password aloud")]
    )
    judge = FakeLLM()
    ctx = context(store, CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=rules), judge)
    out = await run(TranscriptJudgeEvaluator(), ctx)
    assert len(judge.prompts) == 2 + 4
    kinds = [c.kind for c in out.criteria]
    assert kinds[:2] == ["must", "must_not"]
    assert set(kinds[2:]) == {"general"}
    assert out.model == "fake-judge-v1"
    assert all(k["json_schema"]["required"] == ["status", "reason", "quotes"] for k in judge.kwargs)


async def test_prompt_uses_the_independent_transcript_and_tool_log(store):
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS,
        timeline=[ToolCallEvent(at_s=3.0, name="lookup_invoice", result={"amount": "$49.99"})],
    )
    judge = FakeLLM()
    await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    prompt = judge.prompts[0]
    assert "BOT: your card ending 4242" in prompt
    assert "CALLER: hi I was charged" in prompt
    assert 'lookup_invoice({}) -> {"amount": "$49.99"}' in prompt


async def test_failed_must_not_with_a_real_quote_becomes_a_timed_finding(store):
    rule = Rule(text="disclose card digits", severity="critical")
    judge = judge_for(
        {"disclose card digits": verdict_json("fail", quotes=["your card ending 4242"])}
    )
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must_not=[rule]))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert by_id(out, rule.id).status == "fail"
    finding = next(f for f in out.findings if f.criterion_id == rule.id)
    assert finding.severity == "critical"
    assert finding.category == "must_not"
    assert finding.evidence[0].speaker == Speaker.BOT
    assert finding.evidence[0].start_s == pytest.approx(10.0)


async def test_hallucinated_quote_cannot_fail_a_rule(store):
    rule = Rule(text="disclose card digits")
    judge = judge_for(
        {"disclose card digits": verdict_json("fail", quotes=["your card number is 4111 1111"])}
    )
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must_not=[rule]))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert by_id(out, rule.id).status == "unknown"
    assert not [f for f in out.findings if f.criterion_id == rule.id]
    assert out.metrics["dropped_quotes"] == 1


async def test_must_pass_needs_a_quote_but_must_fail_does_not(store):
    shown = Rule(text="say goodbye")
    claimed = Rule(text="offer a ticket")
    absent = Rule(text="verify the zip")
    judge = judge_for(
        {
            "say goodbye": verdict_json("pass", quotes=["goodbye"]),
            "offer a ticket": verdict_json("pass", quotes=[]),
            "verify the zip": verdict_json("fail", quotes=[]),
        }
    )
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must=[shown, claimed, absent])
    )
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert by_id(out, shown.id).status == "pass"
    assert by_id(out, claimed.id).status == "unknown"
    assert by_id(out, absent.id).status == "fail"


async def test_a_judge_error_leaves_that_criterion_unknown_and_continues(store):
    rules = [Rule(text="alpha rule"), Rule(text="beta rule")]

    def reply(prompt: str):
        if "alpha rule" in prompt.split("Criterion to check:")[-1]:
            raise ConnectionError("server gone")
        return verdict_json("pass", quotes=["goodbye"])

    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must=rules))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, FakeLLM(reply)))
    assert by_id(out, rules[0].id).status == "unknown"
    assert "server gone" in by_id(out, rules[0].id).reason
    assert by_id(out, rules[1].id).status == "pass"


async def test_garbage_replies_become_unknown(store):
    rule = Rule(text="alpha rule")
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must=[rule]))

    def reply(prompt: str):
        return (
            "no idea"
            if "alpha rule" in prompt.split("Criterion to check:")[-1]
            else verdict_json("pass")
        )

    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, FakeLLM(reply)))
    assert by_id(out, rule.id).status == "unknown"


async def test_judge_failing_every_criterion_is_an_evaluator_error(store):
    def reply(prompt: str):
        raise ConnectionError("down")

    ctx = context(store, CallArtifact(mode=CallMode.LIVE_BOTS), FakeLLM(reply))
    out = EvaluatorOutput()
    with pytest.raises(RuntimeError, match="all 4 criteria"):
        await TranscriptJudgeEvaluator().run(ctx, out)
    assert all(c.status == "unknown" for c in out.criteria)


async def test_call_with_no_scenario_gets_general_criteria_only(store):
    judge = FakeLLM()
    ctx = context(store, CallArtifact(mode=CallMode.HUMAN), judge)
    out = await run(TranscriptJudgeEvaluator(), ctx)
    assert {c.kind for c in out.criteria} == {"general"}
    assert "No scenario" in judge.prompts[0]


async def test_call_with_no_tool_log_omits_the_fact_check_and_reports_it(store):
    judge = FakeLLM()
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_tool_log=False, has_intended_text=False)
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert len(out.criteria) == 3
    assert all(c.criterion_id != "general_no_invented_facts" for c in out.criteria)
    assert "tool result" in out.metrics["not_checked"][0]
    assert "Not available." in judge.prompts[0]


async def test_general_failure_uses_its_configured_severity(store):
    judge = judge_for({"verified who": verdict_json("fail", quotes=["your card ending 4242"])})
    ctx = context(store, CallArtifact(mode=CallMode.HUMAN), judge)
    out = await run(TranscriptJudgeEvaluator(), ctx)
    finding = out.findings[0]
    assert finding.criterion_id == "general_identity_verification"
    assert finding.severity == "major"


async def test_missing_judge_leaves_rules_unknown_and_skips(store):
    rule = Rule(text="alpha rule")
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must=[rule]))
    ctx = context(store, artifact, judge=None)
    out = EvaluatorOutput()
    with pytest.raises(Skip, match="no judge"):
        await TranscriptJudgeEvaluator().run(ctx, out)
    assert by_id(out, rule.id).status == "unknown"


async def test_missing_transcript_leaves_rules_unknown_and_skips(store):
    rule = Rule(text="alpha rule")
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must=[rule]))
    ctx = context(store, artifact, judge=FakeLLM(), heard=False)
    out = EvaluatorOutput()
    with pytest.raises(Skip, match="no independent transcript"):
        await TranscriptJudgeEvaluator().run(ctx, out)
    assert by_id(out, rule.id).status == "unknown"


def validity_call(**kwargs) -> CallArtifact:
    return CallArtifact(mode=CallMode.TURN_BASED, scenario_snapshot=scenario(**kwargs))


def beat_judge(mapping: dict[str, dict], facts: dict | None = None) -> FakeLLM:
    def reply(prompt: str) -> dict:
        if "fact sheet" in prompt:
            return facts or verdict_json("pass")
        step = prompt.split("Step of the script:")[-1]
        return next(
            (payload for text, payload in mapping.items() if text in step), verdict_json("pass")
        )

    return FakeLLM(reply)


async def test_reached_and_missed_checkpoints_are_reported(store):
    reached = Beat(instruction="state the problem", checkpoint="Caller mentions the charge")
    missed = Beat(instruction="ask for a refund", checkpoint="Caller asks for money back")
    judge = beat_judge(
        {
            "state the problem": verdict_json("pass", quotes=["charged twice for invoice"]),
            "ask for a refund": verdict_json("fail"),
        }
    )
    artifact = validity_call(beats=[reached, missed])
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert out.metrics["reached"] == [reached.id]
    assert out.metrics["missed"] == [missed.id]
    assert by_id(out, missed.id).kind == "beat"
    finding = out.findings[0]
    assert finding.category == "beat_missed"
    assert finding.severity == "major"
    assert "asks for money back" in finding.title


async def test_prompt_carries_the_checkpoint_or_falls_back_to_the_instruction(store):
    judge = beat_judge({})
    artifact = validity_call(beats=[Beat(instruction="say hello to the bot")])
    await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert "What proves the step was reached:\nsay hello to the bot" in judge.prompts[0]


async def test_a_reached_claim_needs_real_evidence(store):
    beat = Beat(instruction="state the problem")
    judge = beat_judge({"state the problem": verdict_json("pass", quotes=["I want my money"])})
    out = await run(CallerValidityEvaluator(), context(store, validity_call(beats=[beat]), judge))
    assert by_id(out, beat.id).status == "unknown"
    assert out.metrics["unknown"] == [beat.id]


async def test_optional_beat_missed_is_only_minor(store):
    beat = Beat(instruction="mention the weather", required=False)
    judge = beat_judge({"mention the weather": verdict_json("fail")})
    out = await run(CallerValidityEvaluator(), context(store, validity_call(beats=[beat]), judge))
    assert out.findings[0].severity == "minor"


async def test_fact_sheet_breach_with_a_caller_quote_is_reported(store):
    judge = beat_judge({}, facts=verdict_json("fail", quotes=["invoice FH-4419"]))
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "fail"
    finding = out.findings[0]
    assert finding.category == "fact_sheet_breach"
    assert finding.advisory is True


async def test_fact_sheet_breach_quoting_the_bot_does_not_count(store):
    judge = beat_judge({}, facts=verdict_json("fail", quotes=["refunded forty nine dollars"]))
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "unknown"
    assert out.findings == []


async def test_no_fact_sheet_means_no_fact_check(store):
    judge = beat_judge({})
    artifact = validity_call(beats=[Beat(instruction="say hi")])
    await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert len(judge.prompts) == 1


@pytest.mark.parametrize("mode", [CallMode.HUMAN, CallMode.UPLOAD])
async def test_skips_when_there_is_no_scripted_caller(store, mode):
    ctx = context(store, CallArtifact(mode=mode, scenario_snapshot=scenario()), FakeLLM())
    with pytest.raises(Skip, match="bot-to-bot"):
        await run(CallerValidityEvaluator(), ctx)


async def test_skips_without_a_scenario(store):
    ctx = context(store, CallArtifact(mode=CallMode.LIVE_BOTS), FakeLLM())
    with pytest.raises(Skip, match="no scenario"):
        await run(CallerValidityEvaluator(), ctx)


async def test_missing_judge_leaves_beats_unknown(store):
    beat = Beat(instruction="say hi")
    ctx = context(store, validity_call(beats=[beat]), judge=None)
    out = EvaluatorOutput()
    with pytest.raises(Skip, match="no judge"):
        await CallerValidityEvaluator().run(ctx, out)
    assert by_id(out, beat.id).status == "unknown"


def test_judge_replies_are_valid_json_for_the_fake():
    assert json.loads(json.dumps(verdict_json("pass")))["status"] == "pass"
