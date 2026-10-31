"""Shared data contracts: personas, scenarios, the call artifact, and eval results.

These types cross every module boundary and are the JSON shapes the API serves, so change them
deliberately. The call artifact is the only input post-call evaluation receives.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


# Personas and scenarios ------------------------------------------------------------------------


class LineConditions(BaseModel):
    """What the simulated phone line does to audio in both directions."""

    codec: Literal["none", "mulaw_8k"] = "mulaw_8k"
    noise: Literal["none", "office", "street", "cafe", "car", "tv", "hum"] = "none"
    snr_db: float = 20.0
    packet_loss_pct: float = Field(default=0.0, ge=0, le=30)
    one_way_latency_ms: int = Field(default=80, ge=0, le=1000)


class VoiceRef(BaseModel):
    """A voice on a TTS provider. `voice_id` is meaningful only to that provider."""

    provider: str | None = None  # None means the active profile's TTS provider
    voice_id: str
    style: str | None = None  # free-text delivery hint for providers that take one


class Persona(BaseModel):
    id: str = Field(default_factory=lambda: new_id("per"))
    name: str
    summary: str = ""
    age: int | None = None
    temperament: str = "neutral"  # e.g. calm, irritated, anxious, confused, friendly
    speaking_style: str = ""  # free text fed to the caller's prompt
    patience: int = Field(default=3, ge=1, le=5)
    verbosity: int = Field(default=3, ge=1, le=5)
    tech_savviness: int = Field(default=3, ge=1, le=5)
    interruption_rate: float = Field(default=0.0, ge=0, le=1)  # chance per bot turn of cutting in
    voice: VoiceRef
    line: LineConditions = Field(default_factory=LineConditions)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Beat(BaseModel):
    """One ordered step the caller must attempt."""

    id: str = Field(default_factory=lambda: new_id("beat"))
    instruction: str  # what the caller should do or say
    checkpoint: str | None = None  # what proves this beat was reached, for validity judging
    required: bool = True


class Rule(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rule"))
    text: str
    severity: Literal["critical", "major", "minor"] = "major"


class StateExpectation(BaseModel):
    """An expected Fernhill database outcome, checked against the per-call database copy.

    `changes` empty with `op="no_change"` asserts the matched row was left alone. `op="row_added"`
    asserts a matching new row exists, with `where` holding the fields it must have.
    """

    id: str = Field(default_factory=lambda: new_id("state"))
    table: str
    where: dict[str, Any] = Field(default_factory=dict)
    op: Literal["changed", "no_change", "row_added"] = "changed"
    changes: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class Scenario(BaseModel):
    id: str = Field(default_factory=lambda: new_id("scn"))
    title: str
    description: str = ""
    goal: str  # what the caller wants, in the caller's terms
    customer_id: str | None = None  # seeded Fernhill customer this caller is, if any
    facts: dict[str, str] = Field(default_factory=dict)  # the only facts the caller may state
    beats: list[Beat] = Field(default_factory=list)
    must: list[Rule] = Field(default_factory=list)  # the bot must do these
    must_not: list[Rule] = Field(default_factory=list)  # the bot must not do these
    expected_state: list[StateExpectation] = Field(default_factory=list)
    max_turns: int = Field(default=24, ge=2, le=120)
    pronunciation_watch: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# Call artifact ---------------------------------------------------------------------------------


class CallMode(StrEnum):
    LIVE_BOTS = "live_bots"  # both bots streaming in real time
    TURN_BASED = "turn_based"  # bots alternate, slower than real time
    HUMAN = "human"  # a person on the microphone
    UPLOAD = "upload"  # an outside recording


class CallStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    EVALUATING = "evaluating"
    DONE = "done"
    FAILED = "failed"


class Speaker(StrEnum):
    CALLER = "caller"
    BOT = "bot"


class Word(BaseModel):
    text: str
    start_s: float
    end_s: float
    confidence: float | None = None


class Utterance(BaseModel):
    type: Literal["utterance"] = "utterance"
    id: str = Field(default_factory=lambda: new_id("utt"))
    speaker: Speaker
    start_s: float
    end_s: float
    intended_text: str | None = None  # text sent to TTS; None for humans and uploads
    spoken_text: str | None = None  # part actually played before any cut-off
    heard_text: str | None = None  # what the other side's STT produced
    interrupted: bool = False
    ttfb_ms: float | None = None  # speech end to first audio, when measured


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str = Field(default_factory=lambda: new_id("tc"))
    at_s: float
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None


class BargeInEvent(BaseModel):
    type: Literal["barge_in"] = "barge_in"
    id: str = Field(default_factory=lambda: new_id("bi"))
    at_s: float  # when the interrupter started speaking
    bot_stopped_at_s: float | None = None
    interrupted_utterance_id: str | None = None


class SystemEvent(BaseModel):
    type: Literal["system"] = "system"
    id: str = Field(default_factory=lambda: new_id("sys"))
    at_s: float
    kind: str  # e.g. call_started, call_ended, caller_hung_up, max_turns
    detail: str = ""


TimelineEvent = Annotated[
    Utterance | ToolCallEvent | BargeInEvent | SystemEvent, Field(discriminator="type")
]


class DbChange(BaseModel):
    table: str
    key: dict[str, Any]
    kind: Literal["insert", "update", "delete"]
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


class CallArtifact(BaseModel):
    """Everything known about one call. Stored as `call.json` beside its audio."""

    id: str = Field(default_factory=lambda: new_id("call"))
    mode: CallMode
    status: CallStatus = CallStatus.QUEUED
    timing_is_real: bool = True  # False for turn-based: pauses are synthetic
    batch_id: str | None = None
    persona_id: str | None = None
    scenario_id: str | None = None
    persona_snapshot: Persona | None = None  # copies, so later edits don't rewrite history
    scenario_snapshot: Scenario | None = None
    profile: str | None = None  # model profile name
    providers: dict[str, str] = Field(default_factory=dict)  # role -> provider/model actually used
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_s: float = 0.0
    timeline: list[TimelineEvent] = Field(default_factory=list)
    db_changes: list[DbChange] = Field(default_factory=list)
    has_tool_log: bool = True  # False for uploads
    has_intended_text: bool = True  # False for humans' side and uploads
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # e.g. real compute times, run labels


# Evaluation ------------------------------------------------------------------------------------


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"
    INVALID = "invalid"  # the caller never reached the scenario; a test problem, not a bot failure


class Evidence(BaseModel):
    quote: str = ""
    speaker: Speaker | None = None
    start_s: float | None = None
    end_s: float | None = None


class HumanVerdict(StrEnum):
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class Finding(BaseModel):
    id: str = Field(default_factory=lambda: new_id("fnd"))
    evaluator: str  # e.g. transcript_judge, timing, pronunciation
    category: str
    severity: Literal["info", "minor", "major", "critical"] = "minor"
    advisory: bool = False  # True means it can flag for review but never fail a call
    title: str
    detail: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    clip_start_s: float | None = None
    clip_end_s: float | None = None
    criterion_id: str | None = None  # the Rule, Beat, or StateExpectation this relates to
    human_verdict: HumanVerdict | None = None
    human_note: str = ""
    human_at: datetime | None = None


class CriterionResult(BaseModel):
    criterion_id: str
    kind: Literal["must", "must_not", "beat", "state", "general"]
    text: str
    status: Literal["pass", "fail", "unknown"]
    reason: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class EvaluatorRun(BaseModel):
    name: str
    status: Literal["ok", "skipped", "error"]
    reason: str = ""  # why it was skipped or failed
    duration_s: float = 0.0
    model: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    call_id: str
    verdict: Verdict
    summary: str = ""
    criteria: list[CriterionResult] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    evaluators: list[EvaluatorRun] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=utcnow)
