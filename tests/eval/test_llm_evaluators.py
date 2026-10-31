import json

import pytest
from evalfakes import FakeLLM, asked_lines, verdict_json, write_call
from evalfakes import transcript as make_transcript

from voicelab.agents.seeds import validate_seeds
from voicelab.core.models import (
    Beat,
    CallArtifact,
    CallMode,
    DbChange,
    RowRef,
    Rule,
    Scenario,
    Speaker,
    StateExpectation,
    SystemEvent,
    ToolCallEvent,
    Utterance,
)
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput, Skip
from voicelab.eval.evaluators.caller_validity import (
    FACTS_CRITERION_ID,
    MATERIAL_FACT_BREACH,
    CallerValidityEvaluator,
    checkpoint_actor,
)
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


def per_item(prompt: str, mapping: dict[str, dict], default: dict) -> dict:
    """A reply to a batched prompt, picking each item's answer by the text it asks about."""
    results = []
    for item, line in asked_lines(prompt).items():
        payload = next((p for text, p in mapping.items() if text in line), default)
        results.append({"id": item, **payload})
    return {"results": results}


def judge_for(mapping: dict[str, dict]) -> FakeLLM:
    """Replies according to which criterion text the prompt asks about."""
    return FakeLLM(lambda prompt: per_item(prompt, mapping, verdict_json("pass")))


async def test_every_rule_and_general_criterion_is_asked_in_one_request(store):
    rules = scenario(
        must=[Rule(text="verify the billing zip")], must_not=[Rule(text="read a password aloud")]
    )
    judge = FakeLLM()
    ctx = context(store, CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=rules), judge)
    out = await run(TranscriptJudgeEvaluator(), ctx)
    assert len(judge.prompts) == 1
    assert len(asked_lines(judge.prompts[0])) == 2 + 4
    kinds = [c.kind for c in out.criteria]
    assert kinds[:2] == ["must", "must_not"]
    assert set(kinds[2:]) == {"general"}
    assert out.model == "fake-judge-v1"
    schema = judge.kwargs[0]["json_schema"]["properties"]["results"]
    assert schema["minItems"] == schema["maxItems"] == 6


async def test_a_reply_finds_its_evidence_before_it_gives_a_status(store):
    judge = FakeLLM()
    ctx = context(store, CallArtifact(mode=CallMode.LIVE_BOTS), judge)
    await run(TranscriptJudgeEvaluator(), ctx)
    item = judge.kwargs[0]["json_schema"]["properties"]["results"]["items"]
    assert list(item["properties"]) == ["id", "quotes", "tools", "reason", "status"]


async def test_the_judge_is_told_who_ended_the_call(store):
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS, timeline=[SystemEvent(at_s=97.8, kind="caller_hung_up")]
    )
    judge = FakeLLM()
    await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert "The caller hung up at 97.8 s" in judge.prompts[0]


@pytest.mark.parametrize("ending", ["caller_hung_up", "max_turns"])
async def test_a_call_the_caller_ended_passes_the_ending_rule_without_asking(store, ending):
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, timeline=[SystemEvent(at_s=97.8, kind=ending)])
    judge = FakeLLM()
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert "general_polite_ending" not in asked_lines(judge.prompts[0])
    assert by_id(out, "general_polite_ending").status == "pass"


async def test_a_call_the_bot_ended_has_its_ending_judged(store):
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS, timeline=[SystemEvent(at_s=9.0, kind="bot_ended")]
    )
    judge = FakeLLM()
    await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert "general_polite_ending" in asked_lines(judge.prompts[0])


def _said_by_bot(meant: str) -> CallArtifact:
    return CallArtifact(
        mode=CallMode.LIVE_BOTS,
        has_intended_text=True,
        timeline=[
            Utterance(speaker=Speaker.BOT, start_s=9.0, end_s=20.0, intended_text=meant),
        ],
    )


async def test_a_number_the_bot_never_meant_to_say_goes_to_a_person(store):
    heard = "your record now has priority 2"
    judge = judge_for({"supported by a tool result": verdict_json("fail", quotes=[heard])})
    ctx = context(store, _said_by_bot("Your record now has priority ten."), judge)
    ctx.transcripts[Speaker.BOT] = make_transcript((heard, 10.0))
    out = await run(TranscriptJudgeEvaluator(), ctx)
    result = by_id(out, "general_no_invented_facts")
    assert result.status == "unknown"
    assert "has 2 where the bot meant" in result.reason


async def test_a_number_the_bot_meant_to_say_still_fails(store):
    heard = "your refund is 3 days away"
    judge = judge_for({"supported by a tool result": verdict_json("fail", quotes=[heard])})
    ctx = context(store, _said_by_bot("Your refund is 3 days away."), judge)
    ctx.transcripts[Speaker.BOT] = make_transcript((heard, 10.0))
    out = await run(TranscriptJudgeEvaluator(), ctx)
    assert by_id(out, "general_no_invented_facts").status == "fail"


async def test_a_tool_call_can_prove_a_rule_the_bot_never_said_aloud(store):
    rule = Rule(text="open a ticket")
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS,
        scenario_snapshot=scenario(must=[rule]),
        timeline=[ToolCallEvent(at_s=12.0, name="create_ticket", result={"ticket_id": "TK-1"})],
    )
    judge = judge_for({"open a ticket": {**verdict_json("pass"), "tools": ["T1"]}})
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert by_id(out, rule.id).status == "pass"
    assert by_id(out, rule.id).evidence[0].start_s == 12.0


async def test_a_tool_call_can_prove_a_forbidden_action(store):
    rule = Rule(text="issue a refund")
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS,
        scenario_snapshot=scenario(must_not=[rule]),
        timeline=[ToolCallEvent(at_s=12.0, name="issue_refund", result={"refund_id": "r1"})],
    )
    judge = judge_for({"issue a refund": {**verdict_json("fail"), "tools": ["T1"]}})
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert by_id(out, rule.id).status == "fail"
    assert "issue_refund" in by_id(out, rule.id).evidence[0].quote
    [finding] = [f for f in out.findings if f.criterion_id == rule.id]
    assert finding.severity == "major"


async def test_judge_and_caller_prompts_share_their_opening(store):
    """The server reuses work on a shared prefix, which is most of the prompt."""
    judge = FakeLLM()
    artifact = validity_call(beats=[Beat(instruction="say hello")])
    ctx = context(store, artifact, judge)
    await run(TranscriptJudgeEvaluator(), ctx)
    await run(CallerValidityEvaluator(), ctx)
    opening = judge.prompts[0].split("Check each criterion")[0]
    assert len(judge.prompts) == 2
    assert all(prompt.startswith(opening) for prompt in judge.prompts)


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
    assert 'T1 lookup_invoice({}) -> {"amount": "$49.99"}' in prompt


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
        if any("alpha rule" in line for line in asked_lines(prompt).values()):
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
        asked = asked_lines(prompt).values()
        return "no idea" if any("alpha rule" in line for line in asked) else verdict_json("pass")

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
    assert "No tool log was recorded" in judge.prompts[0]


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
        if "keep to its fact sheet" in prompt:
            return facts or verdict_json("pass")
        return per_item(prompt, mapping, verdict_json("pass"))

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
    assert "Proof it was reached: say hello to the bot" in judge.prompts[0]


async def test_a_reached_claim_needs_real_evidence(store):
    beat = Beat(instruction="state the problem")
    judge = beat_judge({"state the problem": verdict_json("pass", quotes=["I want my money"])})
    out = await run(CallerValidityEvaluator(), context(store, validity_call(beats=[beat]), judge))
    assert by_id(out, beat.id).status == "unknown"
    assert out.metrics["unknown"] == [beat.id]


async def test_missed_checkpoints_roll_up_with_every_occurrence(store):
    beats = [Beat(instruction="step one"), Beat(instruction="step two")]
    judge = beat_judge({"step one": verdict_json("fail"), "step two": verdict_json("fail")})
    out = await run(CallerValidityEvaluator(), context(store, validity_call(beats=beats), judge))
    [finding] = [f for f in out.findings if f.category == "beat_missed"]
    assert finding.title == "Caller missed 2 checkpoints"
    assert "step one" in finding.detail and "step two" in finding.detail


async def test_optional_beat_missed_is_only_minor(store):
    beat = Beat(instruction="mention the weather", required=False)
    judge = beat_judge({"mention the weather": verdict_json("fail")})
    out = await run(CallerValidityEvaluator(), context(store, validity_call(beats=[beat]), judge))
    assert out.findings[0].severity == "minor"


async def test_fact_sheet_breach_with_a_caller_quote_is_reported(store):
    judge = beat_judge({}, facts=facts_reply("fail", breach("invoice FH-4419", False)))
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "fail"
    assert out.findings[0].advisory is True


async def test_fact_sheet_breach_quoting_the_bot_does_not_count(store):
    judge = beat_judge({}, facts=facts_reply("fail", breach("refunded forty nine dollars")))
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


async def test_a_rule_passed_only_on_caller_words_is_unknown(store):
    rule = Rule(text="verify the billing zip")
    judge = judge_for({"verify the billing zip": verdict_json("pass", quotes=["charged twice"])})
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must=[rule]))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert by_id(out, rule.id).status == "unknown"


async def test_a_rule_failed_only_on_caller_words_is_unknown(store):
    rule = Rule(text="mention invoices")
    judge = judge_for({"mention invoices": verdict_json("fail", quotes=["invoice FH-4419"])})
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=scenario(must_not=[rule]))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert by_id(out, rule.id).status == "unknown"
    assert not [f for f in out.findings if f.criterion_id == rule.id]


async def test_a_caller_beat_cannot_be_proved_by_the_bot(store):
    beat = Beat(instruction="ask for a refund", checkpoint="The caller asked for a refund.")
    judge = beat_judge({"ask for a refund": verdict_json("pass", quotes=["was refunded"])})
    out = await run(CallerValidityEvaluator(), context(store, validity_call(beats=[beat]), judge))
    assert by_id(out, beat.id).status == "unknown"


async def test_a_checkpoint_about_what_the_caller_received_accepts_the_bot(store):
    beat = Beat(
        instruction="confirm the refund",
        checkpoint="The caller asked for or received confirmation of the refund.",
    )
    judge = beat_judge({"confirm the refund": verdict_json("pass", quotes=["was refunded"])})
    out = await run(CallerValidityEvaluator(), context(store, validity_call(beats=[beat]), judge))
    assert by_id(out, beat.id).status == "pass"


def test_seed_checkpoints_needing_either_speaker_are_the_ones_about_receiving():
    _, seeds = validate_seeds()
    either = [b.checkpoint for s in seeds.values() for b in s.beats if checkpoint_actor(b) is None]
    assert either == ["The caller asked for or received confirmation that the site is restored."]


def breach(quote: str, material: object = True) -> dict:
    return {"quote": quote, "material": material, "why": "not on the sheet"}


def facts_reply(status: str, *breaches: dict) -> dict:
    return {"status": status, "reason": "because", "breaches": list(breaches)}


async def test_material_breach_is_marked_for_invalidation(store):
    judge = beat_judge({}, facts=facts_reply("fail", breach("invoice FH-4419", True)))
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert [f.category for f in out.findings] == [MATERIAL_FACT_BREACH]
    assert out.findings[0].advisory is True
    assert out.findings[0].evidence[0].speaker == Speaker.CALLER


async def test_non_material_breach_is_a_plain_slip(store):
    judge = beat_judge({}, facts=facts_reply("fail", breach("invoice FH-4419", False)))
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert [f.category for f in out.findings] == ["fact_sheet_slip"]


async def test_each_breach_is_judged_on_its_own(store):
    judge = beat_judge(
        {},
        facts=facts_reply(
            "fail", breach("twice for invoice FH-4419", False), breach("invoice FH-4419", True)
        ),
    )
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert sorted(f.category for f in out.findings) == sorted(
        ["fact_sheet_slip", MATERIAL_FACT_BREACH]
    )


async def test_a_fabricated_breach_leaves_the_fact_check_unknown(store):
    judge = beat_judge(
        {},
        facts=facts_reply("fail", breach("invoice FH-4419", True), breach("my name is Bob", True)),
    )
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "unknown"
    assert out.findings == []


async def test_a_breach_with_no_materiality_call_is_unknown(store):
    judge = beat_judge({}, facts=facts_reply("fail", breach("invoice FH-4419", "maybe")))
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "unknown"
    assert out.findings == []


async def test_pass_that_lists_breaches_contradicts_itself(store):
    judge = beat_judge({}, facts=facts_reply("pass", breach("invoice FH-4419", True)))
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "unknown"


async def test_the_facts_prompt_defines_material(store):
    judge = beat_judge({})
    artifact = validity_call(facts={"invoice": "FH-1111"})
    await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert "could change the bot's correct behavior" in " ".join(judge.prompts[0].split())
    assert "breaches" in judge.kwargs[0]["json_schema"]["required"]


async def test_one_breach_quoting_the_bot_leaves_the_fact_check_unknown(store):
    judge = beat_judge(
        {},
        facts=facts_reply(
            "fail", breach("invoice FH-4419", True), breach("refunded forty nine dollars", True)
        ),
    )
    artifact = validity_call(facts={"invoice": "FH-1111"})
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "unknown"
    assert out.findings == []


# The fact check reads what the caller was scripted to say ----------------------------------------


def scripted_caller(*lines: str, start_s: float = 0.0, **kwargs) -> CallArtifact:
    artifact = validity_call(**kwargs)
    artifact.timeline = [
        Utterance(
            speaker=Speaker.CALLER,
            start_s=start_s + i * 10,
            end_s=start_s + i * 10 + 5.0,
            intended_text=t,
        )
        for i, t in enumerate(lines)
    ]
    return artifact


async def test_the_fact_check_reads_the_callers_script_not_the_asr(store):
    judge = beat_judge({})
    artifact = scripted_caller(
        "my name is Walter win", facts={"name": "Walter Nguyen", "domain": "nguyenauto.example"}
    )
    await run(CallerValidityEvaluator(), context(store, artifact, judge))
    prompt = next(p for p in judge.prompts if "keep to its fact sheet" in p)
    assert "CALLER: my name is Walter win" in prompt
    assert CALLER_SAID not in prompt


async def test_the_sheet_shows_how_its_values_are_said_aloud(store):
    judge = beat_judge({})
    artifact = scripted_caller(
        "hello there", facts={"name": "Walter Nguyen", "domain": "nguyenauto.example"}
    )
    await run(CallerValidityEvaluator(), context(store, artifact, judge))
    prompt = next(p for p in judge.prompts if "keep to its fact sheet" in p)
    assert "win auto dot example" in prompt


async def test_a_breach_is_checked_against_the_callers_script(store):
    artifact = scripted_caller("my name is Sam and my pin is 9999", facts={"name": "Pat"})
    judge = beat_judge({}, facts_reply("fail", breach("my pin is 9999")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    [finding] = [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]
    assert finding.evidence[0].speaker == Speaker.CALLER
    assert 0.0 <= finding.evidence[0].start_s <= 5.0


async def test_a_breach_only_the_asr_heard_is_thrown_out(store):
    """The transcript has the invoice line, and the caller's script doesn't."""
    artifact = scripted_caller("my name is Pat", facts={"name": "Pat"})
    judge = beat_judge({}, facts_reply("fail", breach("invoice FH-4419")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "unknown"
    assert not [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]


SHEET = {
    "name": "Walter Nguyen",
    "domain": "nguyenauto.example",
    "billing_zip": "95112",
    "pin": "5521",
    "invoice_id": "FH-4419",
}


@pytest.mark.parametrize(
    "quote",
    [
        "how much does the renewal cost",
        "my dog is barking at the mailman",
        "I don't know my zip",
        "my billing zip is 95112 and my pin is 5521",
        "it's win auto dot example",
        "I'm Walter win",
        "the invoice is F H four four one nine",
        "my dog Barnaby is chasing a stick, and Barnaby loves it",
    ],
)
async def test_a_breach_that_names_nothing_new_is_dropped(store, quote):
    artifact = scripted_caller(f"well, {quote}, thanks", facts=SHEET)
    judge = beat_judge({}, facts_reply("fail", breach(quote)))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "pass"
    assert not [f for f in out.findings if f.criterion_id == FACTS_CRITERION_ID]


@pytest.mark.parametrize(
    "quote",
    [
        "my pin is 9999",
        "my billing zip is 90210",
        "the invoice is FH-5000",
        "my domain is bobsbikes.example",
        "my name is Bob Smith",
    ],
)
async def test_a_breach_that_names_something_new_stands(store, quote):
    artifact = scripted_caller(f"well, {quote}, thanks", facts=SHEET)
    judge = beat_judge({}, facts_reply("fail", breach(quote)))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "fail"
    assert [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]


async def test_repeating_a_detail_the_bot_gave_is_not_a_breach(store):
    """BOT_SAID has "forty nine dollars" at 10 s, which isn't on the sheet."""
    artifact = scripted_caller("forty nine dollars, unbelievable", start_s=20.0, facts=SHEET)
    judge = beat_judge({}, facts_reply("fail", breach("forty nine dollars")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "pass"


async def test_one_real_breach_survives_beside_a_dropped_one(store):
    artifact = scripted_caller("my dog is loud and my pin is 9999", facts=SHEET)
    judge = beat_judge({}, facts_reply("fail", breach("my dog is loud"), breach("my pin is 9999")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    [finding] = [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]
    assert "9999" in finding.evidence[0].quote


# Rules the record decides ------------------------------------------------------------------------


def verify(at_s: float, ok: bool = True) -> ToolCallEvent:
    return ToolCallEvent(at_s=at_s, name="verify_identity", result={"verified": ok})


def write(at_s: float, name: str = "renew_ssl") -> ToolCallEvent:
    return ToolCallEvent(at_s=at_s, name=name, result={"ok": True})


VERIFY_FIRST = Rule(
    id="verify_first", text="Verify before renewing.", decided_by="verified_before_change"
)


def decided_call(*events, **fields) -> CallArtifact:
    rules = scenario(must=[VERIFY_FIRST], **fields)
    return CallArtifact(mode=CallMode.LIVE_BOTS, scenario_snapshot=rules, timeline=list(events))


async def test_a_change_before_verifying_fails_the_rule_with_no_judge(store):
    judge = FakeLLM()
    artifact = decided_call(write(15.9), verify(60.0))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    result = by_id(out, "verify_first")
    assert result.status == "fail"
    assert "renew_ssl" in result.evidence[0].quote
    assert result.evidence[0].start_s == 15.9
    assert "verify_first" not in asked_lines(judge.prompts[0])
    assert [f.criterion_id for f in out.findings if f.criterion_id == "verify_first"]


async def test_a_change_with_no_verification_at_all_fails_the_rule(store):
    out = await run(TranscriptJudgeEvaluator(), context(store, decided_call(write(5.0)), FakeLLM()))
    assert by_id(out, "verify_first").status == "fail"


async def test_a_failed_check_before_a_change_fails_the_rule(store):
    artifact = decided_call(verify(4.0, ok=False), write(5.0))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, FakeLLM()))
    assert by_id(out, "verify_first").status == "fail"


async def test_verifying_first_passes_the_rule_with_no_judge(store):
    artifact = decided_call(verify(10.0), write(20.0))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, FakeLLM()))
    result = by_id(out, "verify_first")
    assert result.status == "pass"
    assert "verify_identity" in result.evidence[0].quote


async def test_changing_nothing_cannot_break_a_verify_first_rule(store):
    artifact = decided_call(write(5.0, "get_invoice"))
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, FakeLLM()))
    assert by_id(out, "verify_first").status == "pass"


async def test_with_no_tool_log_a_judge_still_decides_the_rule(store):
    judge = FakeLLM()
    artifact = decided_call()
    artifact.has_tool_log = False
    await run(TranscriptJudgeEvaluator(), context(store, artifact, judge))
    assert "verify_first" in asked_lines(judge.prompts[0])


async def test_a_change_before_verifying_settles_the_identity_rule_too(store):
    judge = FakeLLM()
    out = await run(TranscriptJudgeEvaluator(), context(store, decided_call(write(5.0)), judge))
    assert by_id(out, "general_identity_verification").status == "fail"
    assert "general_identity_verification" not in asked_lines(judge.prompts[0])


async def test_one_unverified_change_is_one_finding_not_two(store):
    out = await run(TranscriptJudgeEvaluator(), context(store, decided_call(write(5.0)), FakeLLM()))
    assert by_id(out, "verify_first").status == "fail"
    assert by_id(out, "general_identity_verification").status == "fail"
    assert [f.criterion_id for f in out.findings] == ["verify_first"]


async def test_with_no_rule_for_it_the_identity_criterion_reports_the_change(store):
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, timeline=[write(5.0)])
    out = await run(TranscriptJudgeEvaluator(), context(store, artifact, FakeLLM()))
    assert [f.criterion_id for f in out.findings] == ["general_identity_verification"]


async def test_otherwise_the_judge_decides_the_identity_rule(store):
    judge = FakeLLM()
    await run(TranscriptJudgeEvaluator(), context(store, decided_call(), judge))
    assert "general_identity_verification" in asked_lines(judge.prompts[0])


REFUNDS = StateExpectation(
    id="no_refund",
    table="refunds",
    op="no_change",
    where={"invoice_id": "FH-4631"},
    description="No refund was issued.",
)
NO_REFUND_RULE = Rule(id="refund_over_limit", text="Issue any refund.", decided_by="no_refund")


def refund_call(*changes: DbChange, **fields) -> CallArtifact:
    rules = scenario(must_not=[NO_REFUND_RULE], expected_state=[REFUNDS], **fields)
    return CallArtifact(
        mode=CallMode.LIVE_BOTS,
        scenario_snapshot=rules,
        db_changes=list(changes),
        timeline=[
            ToolCallEvent(
                at_s=117.5,
                name="issue_refund",
                result={"refund_id": "rfd_1"},
                changed=[RowRef(table="refunds", key={"id": "rfd_1"})],
            )
        ],
    )


REFUND_ROW = DbChange(
    table="refunds",
    key={"id": "rfd_1"},
    kind="insert",
    after={"id": "rfd_1", "invoice_id": "FH-4631"},
)


async def test_a_rule_that_restates_the_database_takes_its_status_from_it(store):
    judge = FakeLLM()
    out = await run(TranscriptJudgeEvaluator(), context(store, refund_call(REFUND_ROW), judge))
    result = by_id(out, "refund_over_limit")
    assert result.status == "fail"
    assert result.evidence[0].start_s == 117.5
    assert "refund_over_limit" not in asked_lines(judge.prompts[0])


async def test_the_database_already_reports_that_failure_so_it_is_one_finding(store):
    out = await run(TranscriptJudgeEvaluator(), context(store, refund_call(REFUND_ROW), FakeLLM()))
    assert [f for f in out.findings if f.criterion_id == "refund_over_limit"] == []


async def test_a_rule_the_database_shows_kept_passes(store):
    out = await run(TranscriptJudgeEvaluator(), context(store, refund_call(), FakeLLM()))
    assert by_id(out, "refund_over_limit").status == "pass"


# What a breach has to be -------------------------------------------------------------------------


def cut_off(intended: str, spoken: str | None) -> Utterance:
    return Utterance(
        speaker=Speaker.CALLER,
        start_s=0.0,
        end_s=5.0,
        intended_text=intended,
        spoken_text=spoken,
        interrupted=True,
    )


async def test_words_the_caller_never_got_to_say_are_not_in_the_fact_check(store):
    artifact = validity_call(facts=SHEET)
    artifact.timeline = [cut_off("my name is Walter and my pin is 9999", "my name is Walter")]
    judge = beat_judge({})
    await run(CallerValidityEvaluator(), context(store, artifact, judge))
    prompt = next(p for p in judge.prompts if "keep to its fact sheet" in p)
    assert "CALLER: my name is Walter" in prompt
    assert "9999" not in prompt


async def test_a_made_up_detail_in_the_cancelled_tail_cannot_void_the_call(store):
    artifact = validity_call(facts=SHEET)
    artifact.timeline = [cut_off("my name is Walter and my pin is 9999", "my name is Walter")]
    judge = beat_judge({}, facts_reply("fail", breach("my pin is 9999")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert not [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]


async def test_a_cut_off_line_with_no_record_of_what_played_is_left_out(store):
    artifact = validity_call(facts=SHEET)
    artifact.timeline = [
        cut_off("my pin is 9999", None),
        Utterance(speaker=Speaker.CALLER, start_s=9.0, end_s=11.0, intended_text="hello there"),
    ]
    judge = beat_judge({})
    await run(CallerValidityEvaluator(), context(store, artifact, judge))
    prompt = next(p for p in judge.prompts if "keep to its fact sheet" in p)
    assert "9999" not in prompt and "hello there" in prompt


CARD_SHEET = {"name": "Pat Lee", "card_status": "expired", "pin": "5521"}


def contradicts(quote: str, key: str = "none", material: object = True) -> dict:
    return {**breach(quote, material), "contradicts": key}


async def test_a_contradiction_stands_with_no_new_detail_in_it(store):
    """ "My card is valid" names no number, and against an expired card it's still a breach."""
    artifact = scripted_caller("my card is valid", facts=CARD_SHEET)
    judge = beat_judge({}, facts_reply("fail", contradicts("my card is valid", "card_status")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "fail"
    assert [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]


async def test_the_reply_names_a_line_of_this_sheet(store):
    judge = beat_judge({})
    await run(
        CallerValidityEvaluator(),
        context(store, scripted_caller("hello", facts=CARD_SHEET), judge),
    )
    schema = next(k for k in judge.kwargs if "breaches" in k["json_schema"]["properties"])
    breach_item = schema["json_schema"]["properties"]["breaches"]["items"]
    assert breach_item["properties"]["contradicts"]["enum"] == [*CARD_SHEET, "none"]


async def test_a_contradiction_of_a_line_the_sheet_lacks_is_dropped(store):
    artifact = scripted_caller("my card is valid", facts=CARD_SHEET)
    judge = beat_judge({}, facts_reply("fail", contradicts("my card is valid", "shoe_size")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "pass"


async def test_a_claim_holding_the_sheets_own_value_contradicts_nothing(store):
    artifact = scripted_caller("my pin is 5521", facts=CARD_SHEET)
    judge = beat_judge({}, facts_reply("fail", contradicts("my pin is 5521", "pin")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "pass"


async def test_part_of_a_lines_value_agrees_with_that_line(store):
    """A real judge called "on the eighteenth" a contradiction of "October 18"."""
    facts = {"charge_date": "October 18", "amount": "$49.99"}
    line = "charged twice for forty nine dollars and ninety nine cents each on the eighteenth"
    artifact = scripted_caller(line, facts=facts)
    judge = beat_judge({}, facts_reply("fail", contradicts(line, "charge_date")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "pass"


async def test_another_lines_value_in_the_wrong_place_stands(store):
    artifact = scripted_caller("my pin is 95112", facts=SHEET)
    judge = beat_judge({}, facts_reply("fail", contradicts("my pin is 95112", "pin")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]


async def test_a_new_value_for_a_line_stands(store):
    artifact = scripted_caller("my pin is 9999", facts=SHEET)
    judge = beat_judge({}, facts_reply("fail", contradicts("my pin is 9999", "pin")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]


async def test_a_question_states_nothing(store):
    artifact = scripted_caller("is my card valid?", facts=CARD_SHEET)
    judge = beat_judge({}, facts_reply("fail", contradicts("is my card valid?", "card_status")))
    out = await run(CallerValidityEvaluator(), context(store, artifact, judge))
    assert by_id(out, FACTS_CRITERION_ID).status == "pass"


async def test_the_bot_repeating_a_made_up_pin_later_does_not_erase_the_breach(store):
    """The transcript has the bot saying 9999 at 10 s, after the caller made it up."""
    artifact = scripted_caller("my pin is 9999", facts=SHEET)
    ctx = context(store, artifact, beat_judge({}, facts_reply("fail", breach("my pin is 9999"))))
    ctx.transcripts[Speaker.BOT] = make_transcript(("so your pin is 9999 then", 10.0))
    out = await run(CallerValidityEvaluator(), ctx)
    assert [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]


async def test_a_detail_the_bot_gave_first_can_be_repeated(store):
    artifact = scripted_caller("my pin is 9999", start_s=20.0, facts=SHEET)
    ctx = context(store, artifact, beat_judge({}, facts_reply("fail", breach("my pin is 9999"))))
    ctx.transcripts[Speaker.BOT] = make_transcript(("so your pin is 9999 then", 10.0))
    out = await run(CallerValidityEvaluator(), ctx)
    assert not [f for f in out.findings if f.category == MATERIAL_FACT_BREACH]
