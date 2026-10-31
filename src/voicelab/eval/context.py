"""The interface every evaluator implements, and what the pipeline hands it."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    CallArtifact,
    CriterionResult,
    Finding,
    Scenario,
    Speaker,
    SystemEvent,
)
from voicelab.eval.config import EvalConfig
from voicelab.eval.transcript import TranscriptIndex
from voicelab.eval.vad import Region, speech_regions
from voicelab.providers.base import LLM, STT, AudioAnalyst, Transcript
from voicelab.store.artifacts import SEPARATION_EVENT, CallPaths


class Skip(Exception):
    """Raised by an evaluator that has nothing it can check. The message says why."""


class MosPredictor(Protocol):
    """A model that scores speech quality on the 1 to 5 MOS scale."""

    name: str

    def predict(self, audio: AudioClip) -> float: ...


@dataclass
class EvalDeps:
    """Everything evaluation needs from outside. An absent model skips what depends on it."""

    stt: STT | None = None
    judge: LLM | None = None
    analyst: AudioAnalyst | None = None
    mos: MosPredictor | None = None
    config: EvalConfig = field(default_factory=EvalConfig)
    lexicon_path: Path | None = None  # None means seeds/lexicon.yaml


@dataclass
class EvaluatorOutput:
    """What one evaluator produced. It stays valid if the evaluator raises partway through."""

    findings: list[Finding] = field(default_factory=list)
    criteria: list[CriterionResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


ProgressNote = Callable[[str], Awaitable[None]]


async def _no_note(_: str) -> None:
    return None


@dataclass
class EvalContext:
    """Shared state for one evaluation run. Later evaluators read what earlier ones left here."""

    artifact: CallArtifact
    paths: CallPaths
    deps: EvalDeps
    transcripts: dict[Speaker, Transcript] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    note_progress: ProgressNote = _no_note  # shows a step inside the running evaluator
    _audio: dict[Speaker, AudioClip] = field(default_factory=dict)
    _regions: dict[Speaker, list[Region]] = field(default_factory=dict)
    _index: TranscriptIndex | None = None

    @property
    def config(self) -> EvalConfig:
        return self.deps.config

    @property
    def scenario(self) -> Scenario | None:
        return self.artifact.scenario_snapshot

    @property
    def channels_inferred(self) -> bool:
        """True when channels came from splitting a mono mix, so they can't overlap."""
        return any(
            isinstance(e, SystemEvent) and e.kind == SEPARATION_EVENT
            for e in self.artifact.timeline
        )

    def audio(self, speaker: Speaker) -> AudioClip:
        """One channel. A missing file raises here and fails only the evaluator that asked."""
        if speaker not in self._audio:
            path = self.paths.caller if speaker == Speaker.CALLER else self.paths.bot
            self._audio[speaker] = AudioClip.from_file(path)
        return self._audio[speaker]

    def regions(self, speaker: Speaker) -> list[Region]:
        if speaker not in self._regions:
            timing = self.config.timing
            self._regions[speaker] = speech_regions(
                self.audio(speaker),
                min_speech_s=timing.min_speech_s,
                min_gap_s=timing.min_gap_s,
            )
        return self._regions[speaker]

    @property
    def index(self) -> TranscriptIndex:
        if self._index is None:
            self._index = TranscriptIndex(self.transcripts)
        return self._index


class Evaluator(Protocol):
    name: str

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        """Add results to `out`. Raise `Skip` if nothing can be checked."""
        ...
