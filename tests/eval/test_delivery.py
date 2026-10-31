"""Delivery measurements on audio that really has each defect, wideband and through the phone line.

The limits in `DeliveryConfig` were set on real speech. These tests check that the machinery finds
known defects in audio built with the same transforms, and stays quiet on a clean control.
"""

import json

import numpy as np
import pytest
from evalfakes import (
    RATE,
    FakeAnalyst,
    FakeLLM,
    FakeSTT,
    gated,
    loud_moment,
    robotized,
    synth_voice,
    timed_words,
    with_noise,
    with_pauses,
    write_call,
)

from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    Evidence,
    Finding,
    LineConditions,
    Persona,
    Speaker,
    Utterance,
    Verdict,
    VoiceRef,
    Word,
)
from voicelab.engine.recorder import CallRecorder
from voicelab.eval.config import DeliveryConfig, EvalConfig
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput
from voicelab.eval.evaluators.audio_listening import AudioListeningEvaluator
from voicelab.eval.evaluators.audio_quality import AudioQualityEvaluator
from voicelab.eval.evaluators.delivery import (
    DeliveryEvaluator,
    UtteranceDelivery,
    _measure,
    _pitch_flag,
    _Span,
)
from voicelab.eval.pipeline import default_evaluators, run_evaluation
from voicelab.eval.pitch import pitch_stats, track_pitch
from voicelab.eval.reconcile import reconcile
from voicelab.eval.speech_measures import edge_ms, estimate_snr_db, find_chops, speech_time
from voicelab.eval.vad import Region
from voicelab.eval.verdict import decide_verdict
from voicelab.providers.base import Transcript
from voicelab.telephony import PhoneLine

WORDS = 16
PHONE = LineConditions(codec="mulaw_8k", noise="none")
CHOPS = {"min_s": 0.02, "max_s": 0.3, "cliff_db": 35.0}
EDGE_LIMIT = EvalConfig().delivery.abrupt_edge_ms


def hard_cut(samples: np.ndarray) -> np.ndarray:
    """Trim both ends at loud moments, so speech starts and stops mid-sound."""
    level = np.abs(samples)
    n = len(samples)
    start = n // 8 + int(np.argmax(level[n // 8 : n // 4]))
    end = n - n // 4 + int(np.argmax(level[n - n // 4 : n - n // 8]))
    return samples[start:end]


DEFECTS = {
    "rushed": (lambda seed: synth_voice(WORDS, speed=1.7, seed=seed), ["rushed"]),
    "dragging": (lambda seed: synth_voice(WORDS, speed=0.65, seed=seed), ["dragging"]),
    "chopped": (lambda seed: gated(synth_voice(WORDS, seed=seed)), ["chopped"]),
    "slow_gate": (
        lambda seed: gated(synth_voice(WORDS, seed=seed), every_s=0.4, gap_s=0.2),
        ["chopped"],
    ),
    "noisy": (lambda seed: with_noise(synth_voice(WORDS, seed=seed), 8.0), ["noisy"]),
    "flattened": (lambda seed: synth_voice(WORDS, flat=True, seed=seed), ["robotic"]),
    "robotized": (lambda seed: robotized(synth_voice(WORDS, seed=seed)), ["robotic"]),
    "halting": (lambda seed: with_pauses(synth_voice(WORDS, seed=seed), pause_s=0.9), ["halting"]),
}


def clean(seed: int) -> np.ndarray:
    return synth_voice(WORDS, seed=seed)


def words_over(start_s: float, seconds: float, count: int = WORDS) -> list[Word]:
    each = seconds / count
    return [
        Word(text="word", start_s=start_s + i * each, end_s=start_s + (i + 1) * each)
        for i in range(count)
    ]


def build_context(store, clips, *, line=None, artifact=None, heard=True, config=None, **deps):
    """A call with these bot clips one second apart, recorded through `line` if there is one."""
    recorder = CallRecorder(artifact or CallArtifact(mode=CallMode.LIVE_BOTS), store)
    moment = 1.0
    words: list[Word] = []
    phone = PhoneLine(line, seed=3) if line else None
    for samples in clips:
        clip = AudioClip.from_samples(samples, RATE)
        sent = phone.process(clip) if phone else clip
        recorder.place(Speaker.BOT, sent, moment)
        words += words_over(moment, clip.duration_s)
        recorder.add_event(
            Utterance(speaker=Speaker.BOT, start_s=moment, end_s=moment + clip.duration_s)
        )
        moment += clip.duration_s + 1.0
    artifact = recorder.finish()
    ctx = EvalContext(
        artifact, store.paths(artifact.id), EvalDeps(config=config or EvalConfig(), **deps)
    )
    if heard:
        ctx.transcripts[Speaker.BOT] = Transcript(text="", words=words)
    return ctx


async def measure(ctx: EvalContext) -> EvaluatorOutput:
    out = EvaluatorOutput()
    await DeliveryEvaluator().run(ctx, out)
    return out


def flags(out: EvaluatorOutput) -> list[list[str]]:
    return [item["flags"] for item in out.metrics["bot"]]


LINES = [pytest.param(None, id="wideband"), pytest.param(PHONE, id="phone")]


@pytest.mark.parametrize("line", LINES)
async def test_clean_control_produces_no_findings(store, line):
    out = await measure(build_context(store, [clean(s) for s in range(4)], line=line))
    assert out.findings == [], [f.title for f in out.findings]
    assert flags(out) == [[], [], [], []]


@pytest.mark.parametrize("line", LINES)
@pytest.mark.parametrize("defect", DEFECTS)
async def test_a_defect_among_clean_speech_is_flagged_on_its_utterance_alone(store, defect, line):
    build, expected = DEFECTS[defect]
    clips = [clean(0), clean(1), build(2), clean(3)]
    out = await measure(build_context(store, clips, line=line))
    assert flags(out) == [[], [], expected, []], out.metrics["bot"][2]


@pytest.mark.parametrize("line", LINES)
@pytest.mark.parametrize("defect", DEFECTS)
async def test_a_defect_alone_is_caught_by_the_fixed_limits(store, defect, line):
    build, expected = DEFECTS[defect]
    out = await measure(build_context(store, [build(2)], line=line))
    assert flags(out) == [expected], out.metrics["bot"][0]


@pytest.mark.parametrize("line", LINES)
async def test_hard_cut_edges_are_found(store, line):
    clips = [clean(0), clean(1), hard_cut(clean(2)), clean(3)]
    ctx = build_context(store, clips, line=line)
    out = await measure(ctx)
    found = flags(out)[2]
    assert "abrupt_onset" in found or "abrupt_offset" in found, out.metrics["bot"][2]
    assert flags(out)[0] == flags(out)[1] == flags(out)[3] == []


@pytest.mark.parametrize("noise", ["office", "street", "cafe", "car", "tv"])
async def test_line_noise_is_flagged_as_noise(store, noise):
    line = LineConditions(noise=noise, snr_db=6.0)
    out = await measure(build_context(store, [clean(1), clean(2)], line=line))
    assert all("noisy" in f for f in flags(out)), out.metrics["bot"]


@pytest.mark.parametrize("noise", ["office", "street", "cafe", "car", "tv", "hum"])
async def test_a_quiet_noise_bed_is_not_flagged(store, noise):
    line = LineConditions(noise=noise, snr_db=30.0)
    out = await measure(build_context(store, [clean(1), clean(2)], line=line))
    assert not any("noisy" in f for f in flags(out)), out.metrics["bot"]


async def test_packet_loss_on_the_line_reads_as_dropouts_not_a_clean_voice(store):
    line = LineConditions(packet_loss_pct=15.0)
    out = await measure(build_context(store, [clean(1)], line=line))
    # Concealment fades rather than cutting, so loss may not read as chops, but it mustn't read as
    # a flat voice: repeating a packet doesn't take the pitch movement out of the rest.
    assert "robotic" not in flags(out)[0]


def test_pitch_tracks_a_steady_voice():
    samples = synth_voice(WORDS, flat=True, f0=120.0)
    stats = pitch_stats(track_pitch(samples, RATE))
    assert stats.median_hz == pytest.approx(120.0, rel=0.01)
    assert stats.iqr_st < 0.1


@pytest.mark.parametrize("f0", [100.0, 120.0, 180.0, 220.0])
def test_pitch_survives_a_phone_band_with_no_fundamental(f0):
    clip = AudioClip.from_samples(synth_voice(WORDS, flat=True, f0=f0), RATE)
    phoned = PhoneLine(PHONE).process(clip).resampled(RATE).samples()
    stats = pitch_stats(track_pitch(phoned, RATE))
    assert stats.median_hz == pytest.approx(f0, rel=0.02)
    assert stats.iqr_st < 0.25


def test_pitch_movement_reads_the_same_through_the_line():
    samples = synth_voice(WORDS, seed=2)
    wide = pitch_stats(track_pitch(samples, RATE))
    phoned = PhoneLine(PHONE).process(AudioClip.from_samples(samples, RATE)).resampled(RATE)
    narrow = pitch_stats(track_pitch(phoned.samples(), RATE))
    assert narrow.iqr_st == pytest.approx(wide.iqr_st, abs=1.0)


def test_noise_and_silence_have_no_pitch():
    noise = np.random.default_rng(0).standard_normal(RATE).astype(np.float32) * 0.1
    assert np.isnan(track_pitch(noise, RATE)).mean() > 0.9
    assert np.isnan(track_pitch(np.zeros(RATE, np.float32), RATE)).all()


def test_too_little_audio_has_no_pitch():
    assert len(track_pitch(np.zeros(100, np.float32), RATE)) == 0
    assert pitch_stats(np.full(10, np.nan)) is None


def test_an_octave_error_does_not_read_as_a_lively_voice():
    pitch = np.full(100, 120.0)
    pitch[::4] = 240.0  # a quarter of frames locked onto half the period
    pitch[1::4] = 60.0
    assert pitch_stats(pitch).iqr_st == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("target", [5.0, 10.0, 20.0])
def test_snr_estimate_is_close_to_the_truth(target):
    estimate = estimate_snr_db(with_noise(synth_voice(WORDS), target), RATE)
    assert estimate == pytest.approx(target, abs=4.0)


@pytest.mark.parametrize("target", [8.0, 20.0])
def test_snr_estimate_holds_through_the_phone_band(target):
    clip = AudioClip.from_samples(with_noise(synth_voice(WORDS), target), RATE)
    phoned = PhoneLine(PHONE).process(clip).resampled(RATE).samples()
    assert estimate_snr_db(phoned, RATE) == pytest.approx(target, abs=4.0)


def test_mulaw_alone_does_not_read_as_noise():
    clip = AudioClip.from_samples(synth_voice(WORDS), RATE)
    phoned = PhoneLine(PHONE).process(clip).resampled(RATE).samples()
    assert estimate_snr_db(phoned, RATE) > 20


def test_digital_silence_is_not_the_noise_floor():
    voice = with_noise(synth_voice(WORDS), 10.0)
    padded = np.concatenate([np.zeros(RATE * 3, np.float32), voice, np.zeros(RATE * 3, np.float32)])
    assert estimate_snr_db(padded, RATE) == pytest.approx(estimate_snr_db(voice, RATE), abs=1.0)


def test_too_little_audio_has_no_snr():
    assert estimate_snr_db(np.zeros(50, np.float32), RATE) is None
    assert estimate_snr_db(np.zeros(RATE, np.float32), RATE) is None


def test_chops_are_found_at_their_positions():
    samples = synth_voice(WORDS)
    first, second = loud_moment(samples, 1.0), loud_moment(samples, 2.0)
    samples[first : first + 800] = 0
    samples[second : second + 800] = 0
    chops = find_chops(samples, RATE, **CHOPS)
    assert [c.start_s for c in chops] == pytest.approx([first / RATE, second / RATE], abs=0.011)
    assert chops[0].end_s - chops[0].start_s == pytest.approx(0.05, abs=0.011)


def test_a_smooth_dip_is_not_a_chop():
    samples = synth_voice(WORDS).copy()
    middle = RATE
    n = round(0.08 * RATE)
    shape = np.abs(np.linspace(-1, 1, n)) ** 2
    samples[middle : middle + n] *= shape.astype(np.float32)
    assert find_chops(samples, RATE, **CHOPS) == []


def test_a_gap_longer_than_a_chop_is_a_pause():
    samples = synth_voice(WORDS)
    samples[RATE : 2 * RATE] = 0
    assert find_chops(samples, RATE, **CHOPS) == []


def lead_in(samples: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(round(0.05 * RATE), np.float32), samples])


def test_edge_timing_tells_a_cut_from_a_fade():
    voice = synth_voice(WORDS)
    assert edge_ms(lead_in(hard_cut(voice))[: RATE // 3], RATE) < EDGE_LIMIT
    assert edge_ms(lead_in(voice)[: RATE // 3], RATE) > EDGE_LIMIT


def test_edge_timing_at_the_end():
    voice = hard_cut(synth_voice(WORDS))
    tail = np.concatenate([voice, np.zeros(round(0.05 * RATE), np.float32)])[-RATE // 3 :]
    assert edge_ms(tail, RATE, at_end=True) < EDGE_LIMIT


def test_speech_already_running_has_no_edge_to_time():
    voice = synth_voice(WORDS)
    start = loud_moment(voice, 1.0)
    assert edge_ms(voice[start : start + RATE // 3], RATE) is None


def test_a_plosive_burst_is_not_a_cut():
    # A brief burst, then a gap, then a vowel: fast to arrive, but nothing is held there.
    level = 0.1
    burst = np.full(round(0.004 * RATE), level * 4, np.float32)
    gap = np.zeros(round(0.05 * RATE), np.float32)
    vowel = level * np.sin(2 * np.pi * 200 * np.arange(RATE // 4) / RATE).astype(np.float32)
    assert edge_ms(lead_in(np.concatenate([burst, gap, vowel])), RATE) is None


def test_speech_time_leaves_out_real_pauses_only():
    regions = [Region(0, 1), Region(1.2, 2), Region(3, 4)]
    total, pauses = speech_time(regions, 0.3)
    assert total == pytest.approx(3.0)
    assert pauses == [Region(2, 3)]


async def test_rate_uses_stt_words_over_speech_time(store):
    out = await measure(build_context(store, [clean(1)]))
    item = out.metrics["bot"][0]
    assert item["words"] == WORDS
    assert item["wpm"] == pytest.approx(165, rel=0.1)


async def test_rate_counts_numbers_as_they_are_said(store):
    ctx = build_context(store, [clean(1)], heard=False)
    span = ctx.artifact.timeline[0]
    text = "your total is $49.99 for invoice FH-4419 and pin 4821"
    each = (span.end_s - span.start_s) / len(text.split())
    words = timed_words(text, span.start_s, per_word_s=each)
    ctx.transcripts[Speaker.BOT] = Transcript(text=text, words=words)
    out = await measure(ctx)
    assert out.metrics["bot"][0]["words"] > len(text.split())


async def test_no_transcript_still_measures_the_rest(store):
    out = await measure(build_context(store, [gated(clean(1))], heard=False))
    item = out.metrics["bot"][0]
    assert item["wpm"] is None
    assert item["flags"] == ["chopped"]
    assert "no speaking rate" in out.metrics["wpm_note"]


async def test_too_few_words_gives_no_rate(store):
    ctx = build_context(store, [clean(1)], heard=False)
    ctx.transcripts[Speaker.BOT] = Transcript(text="", words=words_over(1.0, 1.0, count=3))
    out = await measure(ctx)
    assert out.metrics["bot"][0]["wpm"] is None


async def test_noise_stops_rate_from_being_judged(store):
    ctx = build_context(store, [with_noise(synth_voice(WORDS, speed=1.7), 5.0)])
    out = await measure(ctx)
    assert flags(out) == [["noisy"]]


async def test_chops_stop_rate_and_edges_from_being_judged(store):
    out = await measure(build_context(store, [gated(hard_cut(synth_voice(WORDS, speed=1.7)))]))
    assert flags(out) == [["chopped"]]


async def test_limits_are_configurable(store):
    config = EvalConfig()
    config.delivery.max_wpm = 120.0
    config.delivery.rate_median_ratio = 100.0
    out = await measure(build_context(store, [clean(1)], config=config))
    assert flags(out) == [["rushed"]]


async def test_the_call_median_needs_enough_utterances(store):
    config = EvalConfig()
    config.delivery.max_wpm = 1000.0
    two = await measure(
        build_context(store, [clean(0), synth_voice(WORDS, speed=1.5)], config=config)
    )
    assert two.metrics["medians"]["wpm"] is None
    assert flags(two) == [[], []]
    four = await measure(
        build_context(
            store,
            [clean(0), clean(1), clean(2), synth_voice(WORDS, speed=1.5, seed=3)],
            config=config,
        )
    )
    assert flags(four) == [[], [], [], ["rushed"]]
    assert "median" in four.findings[0].detail


async def test_noise_relative_to_the_call_is_flagged_above_the_fixed_limit(store):
    config = EvalConfig()
    config.delivery.min_snr_db = 5.0  # out of the way, so only the median can flag this
    clips = [clean(0), clean(1), clean(2), with_noise(clean(3), 10.0)]
    out = await measure(build_context(store, clips, config=config))
    assert flags(out)[3] == ["noisy"]
    assert "median" in out.findings[0].detail


async def test_every_finding_is_advisory_with_a_bot_clip(store):
    clips = [build(i) for i, (build, _) in enumerate(DEFECTS.values())]
    out = await measure(build_context(store, clips))
    assert out.findings
    for finding in out.findings:
        assert finding.advisory is True
        assert finding.evaluator == "delivery"
        assert finding.evidence[0].speaker == Speaker.BOT
        assert finding.evidence[0].end_s > finding.evidence[0].start_s


async def test_a_flat_voice_all_call_is_one_finding_listing_each_utterance(store):
    clips = [synth_voice(8, flat=True, seed=i) for i in range(7)]
    out = await measure(build_context(store, clips))
    [finding] = [f for f in out.findings if f.category == "robotic"]
    assert len(finding.evidence) == 7
    assert "7 utterances" in finding.title
    assert out.metrics["flag_counts"]["robotic"] == 7
    assert sum("robotic" in f for f in flags(out)) == 7


# Calibrated on 18 real calls ---------------------------------------------------------------------


async def test_a_voice_that_is_fast_all_call_is_not_rushed_on_every_utterance(store):
    """Real Phonon speech runs near 270 wpm with pauses left out. That's the voice, not a defect."""
    clips = [synth_voice(WORDS, speed=1.7, seed=i) for i in range(5)]
    out = await measure(build_context(store, clips))
    assert out.metrics["medians"]["wpm"] > EvalConfig().delivery.max_wpm
    assert flags(out) == [[], [], [], [], []]
    assert out.findings == []


async def test_a_call_median_past_its_limit_is_one_finding(store):
    config = EvalConfig()
    config.delivery.max_median_wpm = 100.0
    clips = [synth_voice(WORDS, speed=1.3, seed=i) for i in range(5)]
    out = await measure(build_context(store, clips, config=config))
    [finding] = out.findings
    assert finding.category == "rushed"
    assert "the whole call" in finding.title
    assert flags(out) == [[], [], [], [], []]


async def test_a_short_utterance_is_not_compared_to_the_call(store):
    """A few words give a wild rate, which is how one sentence read as "dragging"."""
    clips = [clean(0), clean(1), clean(2), synth_voice(6, speed=0.5, seed=3)]
    ctx = build_context(store, clips)
    out = await measure(ctx)
    assert flags(out)[3] == []


async def test_an_interrupted_utterance_has_no_rate_to_judge(store):
    """The span runs to the cut, so words said over it reads as dragging."""
    ctx = build_context(store, [clean(0), clean(1), clean(2), synth_voice(WORDS, speed=0.65)])
    utterance = ctx.artifact.timeline[-1]
    utterance.interrupted = True
    out = await measure(ctx)
    assert flags(out)[3] == []


async def test_rate_needs_a_clean_signal(store):
    """Energy VAD can't find pauses near this noise, so speech time and rate are both off."""
    config = EvalConfig()
    config.delivery.rate_min_snr_db = 200.0
    clips = [clean(0), clean(1), clean(2), synth_voice(WORDS, speed=1.7, seed=3)]
    out = await measure(build_context(store, clips, config=config))
    assert "rushed" not in flags(out)[3]


async def test_pauses_at_sentence_ends_are_not_halting(store):
    """Phonon leaves about 0.9 s between sentences, and that's how an utterance should sound."""
    clip = with_pauses(synth_voice(WORDS), pause_s=0.9)
    ctx = build_context(store, [clip])
    words = ctx.transcripts[Speaker.BOT].words
    plain = await measure(ctx)
    assert flags(plain) == [["halting"]]
    regions = ctx.regions(Speaker.BOT)
    for region in regions[:-1]:
        last = max((w for w in words if w.start_s < region.end_s + 0.1), key=lambda w: w.start_s)
        last.text = "word."
    out = await measure(ctx)
    assert flags(out) == [[]]
    assert out.metrics["bot"][0]["pause_count"] >= 3  # the raw count stays for the charts


async def test_noise_from_the_personas_own_line_is_a_detail(store):
    persona = Persona(
        name="Rushed caller",
        voice=VoiceRef(voice_id="v"),
        line=LineConditions(noise="car", snr_db=12.0),
    )
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS, persona_snapshot=persona)
    clips = [with_noise(synth_voice(WORDS, seed=i), 8.0) for i in range(4)]
    out = await measure(build_context(store, clips, artifact=artifact))
    [finding] = [f for f in out.findings if f.category == "noisy"]
    assert finding.severity == "info"
    assert len(finding.evidence) == 4


async def test_a_rare_abrupt_start_is_a_detail_and_a_common_one_is_not(store):
    """One start in five reads as abrupt on clean Phonon, so one alone says nothing."""
    cut = hard_cut(clean(4))
    rare = await measure(
        build_context(store, [clean(0), clean(1), clean(2), clean(3), clean(5), cut])
    )
    edges = [f for f in rare.findings if f.category.startswith("abrupt")]
    assert edges and all(f.severity == "info" for f in edges)
    common = await measure(build_context(store, [hard_cut(clean(i)) for i in range(4)]))
    onset = next(f for f in common.findings if f.category == "abrupt_onset")
    assert onset.severity == "minor"


async def test_clean_speech_is_not_noisy_however_far_it_sits_under_the_median(store):
    """A real utterance read 38 dB in a call with a 48 dB median, and that isn't noise."""
    config = EvalConfig()
    config.delivery.min_snr_db = 2.0
    config.delivery.noise_relative_ceiling_db = 5.0
    clips = [clean(0), clean(1), clean(2), with_noise(clean(3), 12.0)]
    out = await measure(build_context(store, clips, config=config))
    assert "noisy" not in flags(out)[3]


async def test_two_abrupt_starts_are_not_a_pattern(store):
    out = await measure(
        build_context(store, [hard_cut(clean(0)), hard_cut(clean(1)), clean(2), clean(3)])
    )
    edges = [f for f in out.findings if f.category == "abrupt_onset"]
    assert edges and all(f.severity == "info" for f in edges)


async def test_a_terrible_voice_reaches_review_and_never_fails(store):
    clips = [DEFECTS[d][0](i) for i, d in enumerate(DEFECTS)]
    ctx = build_context(store, clips)
    deps = EvalDeps(stt=FakeSTT(), judge=FakeLLM())
    result = await run_evaluation(ctx.artifact, ctx.paths, deps)
    assert result.verdict != Verdict.FAIL
    delivered = [f for f in result.findings if f.evaluator == "delivery"]
    assert delivered and all(f.clip_start_s is not None for f in delivered)


def test_delivery_cannot_fail_a_call_even_at_critical_severity():
    finding = Finding(evaluator="delivery", category="rushed", severity="critical", title="x")
    finding.advisory = True
    decision = decide_verdict(CallArtifact(mode=CallMode.HUMAN), [], [finding], [], None)
    assert decision.verdict == Verdict.NEEDS_REVIEW


async def test_metrics_are_per_utterance_and_serialize(store):
    ctx = build_context(store, [clean(0), clean(1)])
    out = await measure(ctx)
    json.dumps(out.metrics)
    first = out.metrics["bot"][0]
    assert first["utterance_id"] == ctx.artifact.timeline[0].id
    expected = {
        "speaker", "utterance_id", "start_s", "end_s", "words", "speech_s", "wpm", "pause_count",
        "longest_pause_s", "pause_ratio", "halting_pauses", "halting_ratio", "interrupted",
        "chops", "mid_word_chops", "voiced_s",
        "pitch_median_hz", "pitch_range_st", "pitch_std_st", "pitch_iqr_st", "snr_db",
        "onset_ms", "offset_ms", "flags",
    }  # fmt: skip
    assert set(first) == expected
    assert "300 to 3400 Hz" in out.metrics["limitation"]


async def test_mid_word_chops_use_word_timing(store):
    out = await measure(build_context(store, [gated(clean(1))]))
    item = out.metrics["bot"][0]
    assert 0 < item["mid_word_chops"] <= item["chops"]


async def test_caller_speech_is_measured_but_never_flagged(store):
    recorder = CallRecorder(CallArtifact(mode=CallMode.LIVE_BOTS), store)
    noisy = AudioClip.from_samples(with_noise(synth_voice(WORDS, speed=1.7), 5.0), RATE)
    recorder.place(Speaker.CALLER, noisy, 0.5)
    recorder.add_event(Utterance(speaker=Speaker.CALLER, start_s=0.5, end_s=0.5 + noisy.duration_s))
    artifact = recorder.finish()
    out = await measure(EvalContext(artifact, store.paths(artifact.id), EvalDeps()))
    assert out.metrics["caller"][0]["snr_db"] < 15
    assert out.metrics["caller"][0]["flags"] == []
    assert out.findings == []


async def test_interrupted_utterance_is_not_abrupt(store):
    recorder = CallRecorder(CallArtifact(mode=CallMode.HUMAN), store)
    clip = AudioClip.from_samples(hard_cut(clean(1)), RATE)
    fade_in = lead_in(clean(2))
    recorder.place(Speaker.BOT, AudioClip.from_samples(fade_in[: len(clip.pcm) // 2], RATE), 1.0)
    end = 1.0 + len(clip.pcm) / 2 / RATE
    recorder.add_event(Utterance(speaker=Speaker.BOT, start_s=1.0, end_s=end, interrupted=True))
    recorder.place(Speaker.CALLER, AudioClip.silence(0.1, RATE), 9.0)
    artifact = recorder.finish()
    out = await measure(EvalContext(artifact, store.paths(artifact.id), EvalDeps()))
    assert out.metrics["bot"][0]["offset_ms"] is None
    assert "abrupt_offset" not in out.metrics["bot"][0]["flags"]


async def test_a_hard_stop_audio_quality_reported_is_not_counted_twice(store):
    voice = clean(1)
    tail_cut = voice[: loud_moment(voice, 2.5)]
    ctx = build_context(store, [clean(0), tail_cut, clean(2)])
    quality = EvaluatorOutput()
    await AudioQualityEvaluator().run(ctx, quality)
    assert "abrupt_cut" in [f.category for f in quality.findings]
    ctx.findings.extend(quality.findings)
    out = await measure(ctx)
    assert "abrupt_offset" not in [f.category for f in out.findings]
    assert out.metrics["bot"][1]["offset_ms"] is not None


async def test_upload_without_a_timeline_measures_speech_regions(store):
    artifact = CallArtifact(mode=CallMode.UPLOAD, has_intended_text=False)
    artifact, paths = write_call(store, artifact, bot=[(1.0, 4.0)], duration_s=5.0)
    out = await measure(EvalContext(artifact, paths, EvalDeps()))
    assert len(out.metrics["bot"]) == 1


async def test_nothing_to_measure_is_a_skip(store):
    artifact, paths = write_call(store, duration_s=2.0)
    result = await run_evaluation(artifact, paths, EvalDeps(), evaluators=[DeliveryEvaluator()])
    assert result.evaluators[0].status == "skipped"


def test_delivery_runs_before_the_listener_it_can_overrule():
    names = [e.name for e in default_evaluators()]
    assert names.index("audio_quality") < names.index("delivery") < names.index("audio_listening")
    assert names.index("transcribe") < names.index("delivery")


# When the listener and the measurements disagree, the measurements win ----------------------------


def said(*issues, tone="neutral", delivery="Even pace.", odd=()):
    def reply(audio, prompt):
        return {
            "tone": tone,
            "delivery": delivery,
            "odd_pronunciations": list(odd),
            "issues": list(issues),
        }

    return reply


def issue(title: str, severity: str = "major", detail: str = "") -> dict:
    return {"title": title, "severity": severity, "detail": detail}


async def listen(ctx: EvalContext) -> EvaluatorOutput:
    measured = await measure(ctx)
    ctx.findings.extend(measured.findings)
    heard = EvaluatorOutput()
    await AudioListeningEvaluator().run(ctx, heard)
    return heard


async def test_a_claim_the_measurements_contradict_is_overruled(store):
    ctx = build_context(store, [clean(1)], analyst=FakeAnalyst(said(issue("Speech is rushed"))))
    out = await listen(ctx)
    finding = next(f for f in out.findings if "rushed" in f.title)
    assert finding.severity == "info"
    assert finding.title.startswith("Overruled:")
    assert "wpm" in finding.detail
    assert [d["kind"] for d in out.metrics["disagreements"]] == ["overruled"]
    assert "measured" in out.metrics["disagreements"][0]["note"]


@pytest.mark.parametrize(
    ("claim", "unit"),
    [
        ("Sounds robotic", "st of pitch"),
        ("Background noise is loud", "dB SNR"),
        ("The audio keeps cutting out", "dropouts"),
        ("Speech is too slow", "wpm"),
    ],
)
async def test_each_measurable_claim_is_checked(store, claim, unit):
    out = await listen(build_context(store, [clean(1)], analyst=FakeAnalyst(said(issue(claim)))))
    finding = next(f for f in out.findings if claim in f.title)
    assert finding.severity == "info"
    assert unit in finding.detail


async def test_a_claim_the_measurements_confirm_stands(store):
    ctx = build_context(
        store,
        [synth_voice(WORDS, speed=1.7)],
        analyst=FakeAnalyst(said(issue("Speech is rushed", "minor"))),
    )
    out = await listen(ctx)
    finding = next(f for f in out.findings if "rushed" in f.title)
    assert finding.severity == "minor"
    assert not finding.title.startswith("Overruled")
    assert out.metrics["disagreements"] == []


async def test_a_claim_nothing_measured_is_left_alone(store):
    ctx = build_context(
        store, [clean(1)], analyst=FakeAnalyst(said(issue("Odd stress on one word", "minor")))
    )
    out = await listen(ctx)
    assert out.findings[0].severity == "minor"
    assert out.metrics["disagreements"] == []


async def test_a_claim_with_no_measurement_of_its_own_stands(store):
    # No transcript means no rate, so a rushed claim can't be contradicted.
    ctx = build_context(
        store, [clean(1)], heard=False, analyst=FakeAnalyst(said(issue("Speech is rushed")))
    )
    out = await listen(ctx)
    assert out.findings[0].severity == "major"
    assert out.metrics["disagreements"] == []


async def test_a_defect_the_model_missed_is_a_disagreement(store):
    ctx = build_context(store, [gated(clean(1))], analyst=FakeAnalyst())
    out = await listen(ctx)
    kinds = [d["kind"] for d in out.metrics["disagreements"]]
    assert kinds[0] == "missed"
    assert "chopped" in out.metrics["disagreements"][0]["note"]


async def test_hearing_emotion_in_a_flat_voice_is_a_disagreement(store):
    # The benchmark's failure: a flat clip called "frustrated" because of its words.
    flat = synth_voice(WORDS, flat=True)
    ctx = build_context(store, [flat], analyst=FakeAnalyst(said(tone="frustrated")))
    out = await listen(ctx)
    note = next(d for d in out.metrics["disagreements"] if d["kind"] == "emotion")
    assert "frustrated" in note["note"]
    assert "pitch" in note["note"]


async def test_emotion_in_a_moving_voice_is_no_disagreement(store):
    ctx = build_context(store, [clean(1)], analyst=FakeAnalyst(said(tone="frustrated")))
    out = await listen(ctx)
    assert out.metrics["disagreements"] == []


async def test_a_claim_in_the_free_text_is_checked_too(store):
    ctx = build_context(
        store, [clean(1)], analyst=FakeAnalyst(said(delivery="Rushed and hurried throughout."))
    )
    out = await listen(ctx)
    assert [d["kind"] for d in out.metrics["disagreements"]] == ["overruled"]


async def test_the_summary_says_measurements_won(store):
    analyst = FakeAnalyst(said(issue("Speech is rushed")))
    artifact = CallArtifact(mode=CallMode.HUMAN)
    ctx = build_context(store, [clean(0), clean(1)], artifact=artifact, analyst=analyst)
    words = timed_words(" ".join(["word"] * WORDS), 1.0, per_word_s=0.36)
    stt = FakeSTT(Transcript(text="x", words=words))  # the caller is silent, so it never hears
    deps = EvalDeps(stt=stt, analyst=analyst, judge=FakeLLM())
    result = await run_evaluation(ctx.artifact, ctx.paths, deps)
    assert "Measurements overruled the audio model" in result.summary
    assert "Speech is rushed" in result.summary


async def test_an_overruled_claim_no_longer_counts_toward_review(store):
    analyst = FakeAnalyst(said(issue("Speech is rushed", "major")))
    ctx = build_context(store, [clean(0)], artifact=CallArtifact(mode=CallMode.HUMAN))
    words = timed_words(" ".join(["word"] * WORDS), 1.0, per_word_s=0.36)
    stt = FakeSTT(Transcript(text="x", words=words))
    result = await run_evaluation(
        ctx.artifact, ctx.paths, EvalDeps(stt=stt, analyst=analyst, judge=FakeLLM())
    )
    assert result.verdict == Verdict.PASS, result.summary


def test_reconcile_ignores_a_clip_nothing_measured_overlaps():
    assert reconcile(50.0, 60.0, {"tone": "angry"}, [], []) == []


def test_reconcile_edits_only_the_contradicted_finding():
    item = UtteranceDelivery("bot", None, 0.0, 5.0, wpm=170.0, snr_db=30.0)
    rushed = Finding(evaluator="audio_listening", category="d", severity="major", title="Rushed")
    other = Finding(evaluator="audio_listening", category="d", severity="major", title="Odd stress")
    evidence = Evidence(speaker=Speaker.BOT, start_s=0.0, end_s=5.0)
    rushed.evidence = other.evidence = [evidence]
    notes = reconcile(0.0, 5.0, {}, [rushed, other], [item])
    assert (rushed.severity, other.severity) == ("info", "major")
    assert len(notes) == 1


# The phone-line failures the corpus showed, which the synthetic voice doesn't produce by itself ---


def third_harmonic_voice(f0: float = 125.0, seconds: float = 2.0) -> np.ndarray:
    """What a low voice can look like past the band's 300 Hz edge: mostly its third harmonic."""
    t = np.arange(round(seconds * RATE)) / RATE
    strengths = {3: 1.0, 4: 0.3, 5: 0.25, 6: 0.2, 7: 0.15}
    voice = sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in strengths.items())
    return (0.1 * voice).astype(np.float32)


def test_a_strong_third_harmonic_does_not_triple_the_pitch():
    # The first dip sits at a third of the period, which reads as 375 Hz. The period is deeper.
    stats = pitch_stats(track_pitch(third_harmonic_voice(), RATE))
    assert stats.median_hz == pytest.approx(125.0, rel=0.02)


def test_stray_lag_frames_move_std_but_not_iqr():
    pitch = np.full(100, 125.0)
    pitch[::8] = 125.0 * 1.5  # about what the phone line does to a flat voice
    stats = pitch_stats(pitch)
    assert stats.std_st > 2.0
    assert stats.iqr_st < 0.01


def test_flatness_is_judged_on_iqr_because_std_reads_a_flat_voice_as_lively():
    # Values from real robotized speech through PhoneLine.
    item = UtteranceDelivery("bot", None, 0.0, 5.0, pitch_iqr_st=0.2, pitch_std_st=2.5)
    item.pitch_median_hz = 125.0
    assert _pitch_flag(item, None, DeliveryConfig()).kind == "robotic"


def test_rate_falls_back_to_the_utterance_when_noise_breaks_speech_detection():
    # At 5 dB on real speech VAD finds only fragments, which read as over 1000 wpm.
    noisy = with_noise(clean(1), 3.0)
    clip = AudioClip.from_samples(noisy, RATE)
    span = _Span(0.0, clip.duration_s)
    fragments = [Region(1.0, 1.2), Region(2.0, 2.15)]
    words = words_over(0.0, clip.duration_s)
    item = _measure(span, clip, fragments, words, DeliveryConfig(), Speaker.BOT)
    assert item.speech_s == pytest.approx(clip.duration_s)
    assert item.wpm == pytest.approx(165, rel=0.05)
    assert item.pause_count == 0


async def test_speech_running_into_the_end_of_the_call_has_no_offset_to_judge(store):
    out = await measure(build_context(store, [hard_cut(clean(1))]))
    assert out.metrics["bot"][0]["offset_ms"] is None
    assert "abrupt_offset" not in flags(out)[0]
