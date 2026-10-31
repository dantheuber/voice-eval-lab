"""Eval thresholds. Plain models so a profile YAML can override any of them."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "minor", "major", "critical"]


class TimingConfig(BaseModel):
    min_speech_s: float = 0.1
    min_gap_s: float = 0.2  # shorter pauses stay inside one speech region
    turn_gap_s: float = 0.7  # a speaker's regions this close form one turn
    monologue_gap_s: float = 1.5
    min_overlap_s: float = 0.1  # shorter overlaps are VAD edges, not people talking over each other
    dead_air_s: float = 6.0
    dead_air_major_s: float = 20.0
    slow_response_s: float = 3.0
    slow_response_major_s: float = 8.0
    talk_over_s: float = 1.0
    talk_over_major_s: float = 3.0
    barge_in_reaction_s: float = 1.0  # bot still talking this long after the caller cut in


class PronunciationConfig(BaseModel):
    low_confidence: float = 0.6  # ASR confidence under this marks a word as doubtful
    minor_mismatch_rate: float = 0.15  # ordinary word errors per utterance, above which it's minor
    max_low_confidence_findings: int = 10


class AudioQualityConfig(BaseModel):
    clip_level: float = 0.99
    clip_fraction: float = 0.001
    clip_major_fraction: float = 0.01
    loudness_low_lufs: float = -32.0
    loudness_high_lufs: float = -12.0
    max_internal_silence_s: float = 1.5
    outlier_level_db: float = 6.0
    outlier_centroid_ratio: float = 1.4
    abrupt_end_ratio: float = 0.7  # final level against the utterance's median level
    low_mos: float = 3.0


class ListeningConfig(BaseModel):
    max_clips: int = 6
    min_samples: int = 2  # ordinary clips always heard, however many were flagged
    clip_min_s: float = 3.0
    clip_max_s: float = 12.0


class VerdictConfig(BaseModel):
    """Advisory findings add up to a score, and at `review_score` the call goes to review."""

    advisory_weights: dict[str, int] = Field(
        default_factory=lambda: {"info": 0, "minor": 1, "major": 3, "critical": 3}
    )
    review_score: int = 3


class GeneralCriterion(BaseModel):
    """A quality rule every call gets, scenario or not.

    `quote_needed_for` names the status that asserts something happened. That status must point
    at the transcript, while its opposite can rest on the absence of evidence.
    """

    id: str
    text: str
    severity: Severity = "minor"
    quote_needed_for: Literal["pass", "fail"] = "fail"
    needs_tool_log: bool = False


def _general_criteria() -> list[GeneralCriterion]:
    return [
        GeneralCriterion(
            id="general_identity_verification",
            text=(
                "Before disclosing account details or changing anything on an account, the bot "
                "verified who the caller was. Pass if nothing was disclosed or changed."
            ),
            severity="major",
        ),
        GeneralCriterion(
            id="general_no_invented_facts",
            text=(
                "Every account detail, price, or date the bot stated is supported by a tool "
                "result or by what the caller said."
            ),
            severity="major",
            needs_tool_log=True,
        ),
        GeneralCriterion(
            id="general_on_topic",
            text="The bot stayed on the caller's issue and on Fernhill Internet support.",
        ),
        GeneralCriterion(
            id="general_polite_ending",
            text="The bot ended the call politely, or the caller ended it first.",
        ),
    ]


class EvalConfig(BaseModel):
    clip_context_s: float = 3.0  # audio kept on each side of a finding for review
    judge_temperature: float = 0.0
    judge_max_tokens: int = 400
    ignored_state_tables: list[str] = Field(default_factory=list)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    pronunciation: PronunciationConfig = Field(default_factory=PronunciationConfig)
    audio_quality: AudioQualityConfig = Field(default_factory=AudioQualityConfig)
    listening: ListeningConfig = Field(default_factory=ListeningConfig)
    verdict: VerdictConfig = Field(default_factory=VerdictConfig)
    general_criteria: list[GeneralCriterion] = Field(default_factory=_general_criteria)
