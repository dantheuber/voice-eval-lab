"""The judge against hand labels on real calls.

`judge_labels.yaml` has what a careful reader concluded from 18 real calls, and `calibration/` has
what a judge reads about them, with the replies a real model gave. Replaying those needs no server,
so verification, the fact check, and the verdict rules stay checked against real output.

The `models` test asks a live judge. Run a server and then:
    VOICELAB_JUDGE_URL=http://127.0.0.1:18105/v1 uv run pytest -m models -s \
        tests/eval/test_judge_calibration.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest
import yaml
from evalfakes import FakeLLM

from voicelab.core.models import CallArtifact, CriterionResult, Finding, Speaker, Word
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput
from voicelab.eval.evaluators.caller_validity import CallerValidityEvaluator
from voicelab.eval.evaluators.state import StateEvaluator
from voicelab.eval.evaluators.transcript_judge import TranscriptJudgeEvaluator
from voicelab.eval.verdict import decide_verdict
from voicelab.providers.adapters.openai_compat import OpenAICompatLLM
from voicelab.providers.base import LLM, Transcript
from voicelab.store.artifacts import CallPaths

FIXTURES = Path(__file__).parent / "fixtures"
LABELS: dict = yaml.safe_load((FIXTURES / "judge_labels.yaml").read_text())["calls"]
JUDGE_URL = os.environ.get("VOICELAB_JUDGE_URL", "http://127.0.0.1:18105/v1")

# As measured with Qwen3.8-27B on 2026-10-31: 222 of 225 and no unknowns. A little room, because the
# set was also what the prompts were tuned on, and a model swap shouldn't need a new number.
MEASURED_CORRECT = 0.98
MEASURED_UNKNOWN = 0.01
LIVE_CORRECT = 0.92
LIVE_UNKNOWN = 0.10
LIVE_VERDICTS = 16


def load_fixture(call_id: str) -> dict:
    return json.loads((FIXTURES / "calibration" / f"{call_id}.json").read_text())


def transcripts_of(fixture: dict) -> dict[Speaker, Transcript]:
    transcripts = {}
    for speaker, words in fixture["words"].items():
        timed = [Word(text=t, start_s=s, end_s=e, confidence=c) for t, s, e, c in words]
        transcripts[Speaker(speaker)] = Transcript(
            text=" ".join(w.text for w in timed), words=timed, model="whisper-large-v3-q5"
        )
    return transcripts


async def judge_call(
    call_id: str, llm: LLM, tmp_path: Path
) -> tuple[list[CriterionResult], list[Finding], CallArtifact]:
    fixture = load_fixture(call_id)
    artifact = CallArtifact.model_validate(fixture["artifact"])
    ctx = EvalContext(artifact, CallPaths(tmp_path), EvalDeps(judge=llm))
    ctx.transcripts = transcripts_of(fixture)
    criteria: list[CriterionResult] = []
    findings: list[Finding] = []
    for evaluator in (StateEvaluator(), TranscriptJudgeEvaluator(), CallerValidityEvaluator()):
        out = EvaluatorOutput()
        await evaluator.run(ctx, out)
        ctx.findings.extend(out.findings)
        criteria += out.criteria
        findings += out.findings
    return criteria, findings, artifact


def allowed(label) -> list[str]:
    return label if isinstance(label, list) else [label]


def score(call_id: str, criteria: list[CriterionResult]) -> Counter:
    labels = LABELS[call_id]["criteria"]
    tally: Counter = Counter()
    for criterion in criteria:
        if criterion.criterion_id not in labels or criterion.kind == "state":
            continue
        tally["items"] += 1
        if criterion.status in allowed(labels[criterion.criterion_id]):
            tally["correct"] += 1
        elif criterion.status == "unknown":
            tally["unknown"] += 1
        else:
            tally["wrong"] += 1
    return tally


def replay(call_id: str) -> FakeLLM:
    replies = iter(load_fixture(call_id)["replies"])
    return FakeLLM(lambda prompt: next(replies))


def test_every_labeled_criterion_exists_in_its_scenario():
    for call_id, label in LABELS.items():
        scenario = CallArtifact.model_validate(load_fixture(call_id)["artifact"]).scenario_snapshot
        known = {r.id for r in [*scenario.must, *scenario.must_not, *scenario.beats]}
        extra = {c for c in label["criteria"] if not c.startswith("general_")} - known
        assert extra <= {"caller_facts"}, call_id


async def test_recorded_replies_score_as_measured(tmp_path):
    """Counts as measured on real replies. A change here means verification logic moved."""
    tally: Counter = Counter()
    for call_id in LABELS:
        criteria, _, _ = await judge_call(call_id, replay(call_id), tmp_path)
        tally += score(call_id, criteria)
    assert tally["items"] == 225
    assert tally["correct"] / tally["items"] >= MEASURED_CORRECT
    assert tally["unknown"] / tally["items"] < MEASURED_UNKNOWN


@pytest.mark.parametrize("call_id", list(LABELS))
async def test_recorded_replies_reach_the_labeled_verdict(call_id, tmp_path):
    criteria, findings, artifact = await judge_call(call_id, replay(call_id), tmp_path)
    decision = decide_verdict(artifact, criteria, findings, [])
    assert decision.verdict.value in allowed(LABELS[call_id]["verdict"]), decision.reasons


async def test_the_refund_past_the_limit_fails_and_says_why(tmp_path):
    call_id = "call_d5c7c018f14f"
    criteria, findings, artifact = await judge_call(call_id, replay(call_id), tmp_path)
    decision = decide_verdict(artifact, criteria, findings, [])
    assert decision.verdict.value == "fail"
    assert "the bot did something forbidden" in decision.reasons[0]


async def test_the_call_the_caller_hung_up_on_is_invalid_and_says_why(tmp_path):
    call_id = "call_8d244d980f22"
    criteria, findings, artifact = await judge_call(call_id, replay(call_id), tmp_path)
    decision = decide_verdict(artifact, criteria, findings, [])
    assert decision.verdict.value == "invalid"
    assert any("the caller hung up" in reason for reason in decision.reasons)


@pytest.mark.models
async def test_a_live_judge_matches_the_labels(tmp_path):
    llm = OpenAICompatLLM(JUDGE_URL, name="judge", timeout_s=600)
    tally: Counter = Counter()
    verdicts = 0
    for call_id in LABELS:
        criteria, findings, artifact = await judge_call(call_id, llm, tmp_path)
        tally += score(call_id, criteria)
        decision = decide_verdict(artifact, criteria, findings, [])
        verdicts += decision.verdict.value in allowed(LABELS[call_id]["verdict"])
    items = tally["items"]
    print(
        f"\n{tally['correct']}/{items} correct, {tally['wrong']} wrong, {tally['unknown']} unknown"
    )
    print(f"{verdicts}/{len(LABELS)} verdicts as labeled")
    assert tally["correct"] / items >= LIVE_CORRECT
    assert tally["unknown"] / items < LIVE_UNKNOWN
    assert verdicts >= LIVE_VERDICTS
