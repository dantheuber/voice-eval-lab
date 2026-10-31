"""Eval thresholds. Plain models so a profile YAML can override any of them."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "minor", "major", "critical"]


class TranscribeConfig(BaseModel):
    """How a silence-sensitive ASR is fed. See `eval/transcript.py:speech_chunks`."""

    chunk_gap_s: float = 0.8  # speech closer than this goes to the ASR as one clip
    chunk_pad_s: float = 0.2  # audio kept around speech, so first and last words aren't clipped


class TimingConfig(BaseModel):
    min_speech_s: float = 0.1
    min_gap_s: float = 0.2  # shorter pauses stay inside one speech region
    turn_gap_s: float = 0.7  # a speaker's regions this close form one turn
    monologue_gap_s: float = 1.5
    min_overlap_s: float = 0.1  # shorter overlaps are VAD edges, not people talking over each other
    dead_air_s: float = 6.0
    dead_air_major_s: float = 20.0
    # On 18 real calls plain turns land at 1.5 to 2.5 s and tool turns at 4.2 to 4.8 s, so this
    # sits just past a normal tool turn and points at outliers, not at every tool call.
    slow_response_s: float = 5.0
    slow_response_major_s: float = 8.0
    talk_over_s: float = 1.0
    talk_over_major_s: float = 3.0
    barge_in_reaction_s: float = 1.0  # bot still talking this long after the caller cut in
    # Past this share of channel speech outside the timeline's utterances, line noise is reading
    # as speech. On 18 real calls the share was 0.46 with TV noise and at most 0.01 otherwise.
    max_speech_off_timeline: float = 0.25
    min_speech_off_timeline_s: float = 5.0  # a second or two says nothing about the line


class PronunciationConfig(BaseModel):
    low_confidence: float = 0.6  # ASR confidence under this marks a word as doubtful
    minor_mismatch_rate: float = 0.15  # ordinary word errors per utterance, above which it's minor
    max_low_confidence_findings: int = 10
    utterance_slack_s: float = 2.0  # how far around an utterance its words can land


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


class DeliveryConfig(BaseModel):
    """Limits for how speech was delivered, checked as fixed ranges and against the call's median.

    Ranges come from 21 real clips, 12 TTS and 9 human, wideband and through the phone line, with
    the benchmark's planted defects, then 18 real calls of Phonon speech. They are still only that:
    nobody has rated them by ear.

    Rate leaves pauses out, so it reads high: Phonon's calls have medians of 174 to 330 wpm, while
    inside one call 95% of utterances stay under 1.24 times its median. So with enough utterances
    to know the voice, an utterance is judged against the call and the fixed range judges the
    call's median, once. `min_wpm` and `max_wpm` are for an utterance with no call to compare to.
    """

    min_wpm: float = 125.0
    max_wpm: float = 250.0
    min_median_wpm: float = 125.0
    max_median_wpm: float = 340.0
    rate_median_ratio: float = 1.35  # rushed or dragging against the call's own pace
    min_words: int = 4  # shorter utterances can't give a meaningful rate
    min_rate_speech_s: float = 1.0
    min_compare_words: int = 8  # shorter ones still give a rate, too rough to set against the call
    min_compare_speech_s: float = 2.5
    rate_min_snr_db: float = 17.0  # energy VAD loses the pauses past this: real lines read 14 to 16
    pause_s: float = 0.3  # shorter gaps count as speaking
    max_pause_ratio: float = 0.45  # share of an utterance spent in pauses
    min_pauses: int = 3
    chop_min_s: float = 0.02
    chop_max_s: float = 0.3
    chop_cliff_db: float = 35.0
    min_chops: int = 2
    robotic_iqr_st: float = 0.5
    monotone_iqr_st: float = 1.5
    pitch_median_ratio: float = 0.5  # flatter than this share of the call's own movement
    min_voiced_s: float = 0.6
    min_snr_db: float = 15.0
    snr_median_drop_db: float = 8.0
    noise_relative_ceiling_db: float = 30.0  # above this it's clean, whatever the call's median
    abrupt_edge_ms: float = 9.0
    # Real Phonon starts read as abrupt one time in five, so only a pattern is worth a reviewer.
    abrupt_share_to_review: float = 0.5
    abrupt_count_to_review: int = 3
    min_utterances_for_median: int = 3


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
    passes_if_caller_ended: bool = False  # the record already answers this one, no judge needed
    fails_if_changed_unverified: bool = False  # as does a change before verifying, for this one


def _general_criteria() -> list[GeneralCriterion]:
    return [
        GeneralCriterion(
            id="general_identity_verification",
            text=(
                "Before disclosing account details or changing anything on an account, the bot "
                "verified who the caller was. Pass if nothing was disclosed or changed."
            ),
            severity="major",
            fails_if_changed_unverified=True,
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
            passes_if_caller_ended=True,
        ),
    ]


class EvalConfig(BaseModel):
    clip_context_s: float = 3.0  # audio kept on each side of a finding for review
    judge_temperature: float = 0.0
    judge_max_tokens: int = 200  # per item in a request, enough for its quotes and reason
    judge_batch_size: int = 0  # items per request, and 0 asks about them all at once
    ignored_state_tables: list[str] = Field(default_factory=list)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    pronunciation: PronunciationConfig = Field(default_factory=PronunciationConfig)
    audio_quality: AudioQualityConfig = Field(default_factory=AudioQualityConfig)
    delivery: DeliveryConfig = Field(default_factory=DeliveryConfig)
    listening: ListeningConfig = Field(default_factory=ListeningConfig)
    verdict: VerdictConfig = Field(default_factory=VerdictConfig)
    general_criteria: list[GeneralCriterion] = Field(default_factory=_general_criteria)
