import numpy as np
import pytest
from evalfakes import RATE, FakeAnalyst, speech

from voicelab.core.audio import AudioClip
from voicelab.core.models import CallArtifact, CallMode, Evidence, Finding, Speaker, Utterance
from voicelab.eval.config import EvalConfig
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput, Skip
from voicelab.eval.evaluators.audio_listening import AudioListeningEvaluator
from voicelab.eval.evaluators.audio_quality import (
    AudioQualityEvaluator,
    clipped_fraction,
    integrated_lufs,
    spectral_centroid,
)
from voicelab.store.artifacts import ArtifactStore


def sine(hz: float, seconds: float, amp: float) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def faded(seconds: float, amp: float = 0.3, hz: float = 300) -> np.ndarray:
    """A speech-like burst that decays at the end, so it isn't an abrupt cut."""
    tone = sine(hz, seconds, amp)
    n = round(0.08 * RATE)
    tone[-n:] *= np.linspace(1, 0, n)
    return tone


def bot_channel(duration_s: float, *parts: tuple[float, np.ndarray]) -> AudioClip:
    samples = np.zeros(round(duration_s * RATE), dtype=np.float32)
    for start, part in parts:
        first = round(start * RATE)
        samples[first : first + len(part)] += part
    return AudioClip.from_samples(samples, RATE)


def context(store: ArtifactStore, bot: AudioClip, artifact=None, **deps) -> EvalContext:
    artifact = artifact or CallArtifact(mode=CallMode.LIVE_BOTS)
    artifact.duration_s = bot.duration_s
    caller = AudioClip.silence(bot.duration_s, RATE)
    paths = store.write_call(artifact, caller, bot)
    return EvalContext(artifact, paths, EvalDeps(**deps))


async def quality(ctx: EvalContext) -> EvaluatorOutput:
    out = EvaluatorOutput()
    await AudioQualityEvaluator().run(ctx, out)
    return out


def categories(out: EvaluatorOutput) -> list[str]:
    return [f.category for f in out.findings]


@pytest.mark.parametrize("amp", [0.1, 0.316, 0.0316])
def test_lufs_of_a_sine_matches_the_bs1770_reference(amp):
    # BS.1770 reads a 997 Hz sine of peak amplitude A as 20*log10(A) - 3.01 LKFS on one channel.
    expected = 20 * np.log10(amp) - 3.01
    assert integrated_lufs(sine(997, 3.0, amp), RATE) == pytest.approx(expected, abs=0.1)


def test_lufs_ignores_silence_through_gating():
    tone = sine(997, 2.0, 0.1)
    with_gaps = np.concatenate([tone, np.zeros(RATE * 2, dtype=np.float32), tone])
    # Blocks straddling the edges of the gap pull the mean down a little, as in the standard.
    assert integrated_lufs(with_gaps, RATE) == pytest.approx(integrated_lufs(tone, RATE), abs=0.5)


def test_lufs_of_silence_is_none():
    assert integrated_lufs(np.zeros(RATE), RATE) is None


def test_clipped_fraction_counts_samples_at_the_rail():
    samples = np.array([0.1, 0.5, 1.0, -1.0], dtype=np.float32)
    assert clipped_fraction(samples, 0.99) == 0.5


def test_spectral_centroid_of_a_tone_is_its_frequency():
    assert spectral_centroid(sine(1000, 1.0, 0.3), RATE) == pytest.approx(1000, abs=1)


async def test_clean_audio_has_no_findings(store):
    bot = bot_channel(10.0, (1.0, faded(2.0)), (4.0, faded(2.0)), (7.0, faded(2.0)))
    out = await quality(context(store, bot))
    assert out.findings == []
    assert out.metrics["bot"]["clipped_fraction"] == 0
    assert out.metrics["bot"]["lufs"] == pytest.approx(-13.5, abs=2)
    assert out.metrics["mos"] is None
    assert "no MOS predictor" in out.metrics["mos_note"]


async def test_clipping_is_found_and_stays_advisory(store):
    hot = np.clip(sine(300, 2.0, 3.0), -1, 1)
    hot[-1600:] *= np.linspace(1, 0, 1600)
    out = await quality(context(store, bot_channel(5.0, (1.0, hot))))
    finding = next(f for f in out.findings if f.category == "clipping")
    assert finding.severity == "major"
    assert finding.advisory is True
    assert finding.evidence[0].start_s == pytest.approx(1.0, abs=0.03)


async def test_loudness_out_of_range_is_found(store):
    bot = bot_channel(5.0, (1.0, faded(3.0, amp=0.01)))
    out = await quality(context(store, bot))
    assert "loudness" in categories(out)


async def test_abrupt_cut_is_found(store):
    abrupt = sine(300, 2.0, 0.3)
    out = await quality(context(store, bot_channel(5.0, (1.0, abrupt))))
    finding = next(f for f in out.findings if f.category == "abrupt_cut")
    assert finding.evidence[0].end_s == pytest.approx(3.0, abs=0.03)


async def test_abrupt_cut_explained_by_an_interruption_is_not_flagged(store):
    artifact = CallArtifact(
        mode=CallMode.HUMAN,
        timeline=[Utterance(speaker=Speaker.BOT, start_s=1.0, end_s=3.0, interrupted=True)],
    )
    out = await quality(context(store, bot_channel(5.0, (1.0, sine(300, 2.0, 0.3))), artifact))
    assert "abrupt_cut" not in categories(out)


async def test_speech_running_into_the_end_of_the_call_is_not_abrupt(store):
    out = await quality(context(store, bot_channel(3.0, (1.0, sine(300, 2.0, 0.3)))))
    assert "abrupt_cut" not in categories(out)


async def test_long_silence_inside_an_utterance_is_found(store):
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS,
        timeline=[Utterance(speaker=Speaker.BOT, start_s=1.0, end_s=6.0)],
    )
    bot = bot_channel(7.0, (1.0, faded(1.5)), (4.5, faded(1.5)))
    out = await quality(context(store, bot, artifact))
    finding = next(f for f in out.findings if f.category == "internal_silence")
    assert finding.title == "2.0 s of silence inside an utterance"


async def test_a_pause_between_utterances_is_not_internal_silence(store):
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS,
        timeline=[
            Utterance(speaker=Speaker.BOT, start_s=1.0, end_s=2.5),
            Utterance(speaker=Speaker.BOT, start_s=4.5, end_s=6.0),
        ],
    )
    bot = bot_channel(7.0, (1.0, faded(1.5)), (4.5, faded(1.5)))
    out = await quality(context(store, bot, artifact))
    assert "internal_silence" not in categories(out)


async def test_outlier_utterances_are_judged_against_the_call_median(store):
    bot = bot_channel(
        13.0,
        (1.0, faded(2.0, amp=0.1)),
        (4.0, faded(2.0, amp=0.1)),
        (7.0, faded(2.0, amp=0.1)),
        (10.0, faded(2.0, amp=0.5)),
    )
    out = await quality(context(store, bot))
    outliers = [f for f in out.findings if f.category == "level_outlier"]
    assert len(outliers) == 1
    assert "louder" in outliers[0].title
    assert outliers[0].evidence[0].start_s == pytest.approx(10.0, abs=0.03)


async def test_timbre_outlier_is_found(store):
    bot = bot_channel(
        13.0,
        (1.0, faded(2.0)),
        (4.0, faded(2.0)),
        (7.0, faded(2.0)),
        (10.0, faded(2.0, hz=1200)),
    )
    out = await quality(context(store, bot))
    assert categories(out).count("timbre_outlier") == 1


async def test_two_utterances_are_too_few_for_outliers(store):
    bot = bot_channel(7.0, (1.0, faded(2.0, amp=0.03)), (4.0, faded(2.0, amp=0.3)))
    out = await quality(context(store, bot))
    assert "level_outlier" not in categories(out)


async def test_silent_bot_channel_reports_metrics_without_findings(store):
    out = await quality(context(store, AudioClip.silence(2.0, RATE)))
    assert out.findings == []
    assert out.metrics["note"] == "no speech on the bot channel"


async def test_mos_predictor_plugs_in(store):
    class FakeMos:
        name = "fake-mos"

        def predict(self, audio: AudioClip) -> float:
            return 2.4

    bot = bot_channel(4.0, (1.0, faded(2.0)))
    out = await quality(context(store, bot, mos=FakeMos()))
    assert out.metrics["mos"] == 2.4
    assert out.model == "fake-mos"
    assert "low_mos" in categories(out)


async def test_every_quality_finding_is_advisory(store):
    hot = np.clip(sine(300, 2.0, 3.0), -1, 1)
    out = await quality(context(store, bot_channel(5.0, (1.0, hot))))
    assert out.findings
    assert all(f.advisory for f in out.findings)


def flagged(start: float, end: float, severity: str = "minor", title: str = "odd") -> Finding:
    return Finding(
        evaluator="pronunciation",
        category="x",
        severity=severity,
        title=title,
        evidence=[Evidence(speaker=Speaker.BOT, start_s=start, end_s=end)],
    )


async def listen(ctx: EvalContext) -> EvaluatorOutput:
    out = EvaluatorOutput()
    await AudioListeningEvaluator().run(ctx, out)
    return out


def many_turns(count: int = 10) -> AudioClip:
    return speech([(i * 5.0, i * 5.0 + 3.0) for i in range(count)], count * 5.0)


async def test_skips_without_an_analyst(store):
    with pytest.raises(Skip, match="no audio analyst"):
        await listen(context(store, many_turns()))


async def test_skips_with_no_bot_speech(store):
    with pytest.raises(Skip, match="no bot speech"):
        await listen(context(store, AudioClip.silence(5.0, RATE), analyst=FakeAnalyst()))


async def test_flagged_clips_come_first_then_samples(store):
    analyst = FakeAnalyst()
    ctx = context(store, many_turns(), analyst=analyst)
    ctx.findings = [flagged(20.0, 21.0, "minor", "low one"), flagged(40.0, 41.0, "major", "high")]
    out = await listen(ctx)
    clips = out.metrics["clips"]
    assert "high" in clips[0]["why"]
    assert "low one" in clips[1]["why"]
    assert all("sample" in c["why"] for c in clips[2:])
    assert len(clips) == 6
    assert len(analyst.clips) == 6


async def test_samples_are_kept_even_with_many_flagged_clips(store):
    ctx = context(store, many_turns(), analyst=FakeAnalyst())
    ctx.findings = [flagged(i * 5.0, i * 5.0 + 1) for i in range(10)]
    out = await listen(ctx)
    clips = out.metrics["clips"]
    assert sum("flagged" in c["why"] for c in clips) == 4
    assert sum("sample" in c["why"] for c in clips) == 2


async def test_clip_lengths_are_widened_and_capped(store):
    config = EvalConfig()
    analyst = FakeAnalyst()
    ctx = context(store, many_turns(), analyst=analyst, config=config)
    ctx.findings = [flagged(20.0, 20.2), flagged(30.0, 45.0)]
    await listen(ctx)
    assert analyst.clips[0].duration_s == pytest.approx(config.listening.clip_min_s, abs=0.01)
    assert analyst.clips[1].duration_s == pytest.approx(config.listening.clip_max_s, abs=0.01)


async def test_caller_side_findings_are_not_listened_to(store):
    ctx = context(store, many_turns(), analyst=FakeAnalyst())
    ctx.findings = [
        Finding(
            evaluator="timing", category="x", title="caller thing",
            evidence=[Evidence(speaker=Speaker.CALLER, start_s=1.0, end_s=2.0)],
        )
    ]  # fmt: skip
    out = await listen(ctx)
    assert all("sample" in c["why"] for c in out.metrics["clips"])


async def test_findings_are_all_advisory_and_structured(store):
    analyst = FakeAnalyst(
        lambda audio, prompt: {
            "tone": "flat",
            "delivery": "Rushed.",
            "odd_pronunciations": ["Fernhill"],
            "issues": [
                {"title": "Robotic stretch", "severity": "major", "detail": "at the end"},
                {"title": "Odd pause", "severity": "critical"},
            ],
        }
    )
    ctx = context(store, many_turns(2), analyst=analyst)
    out = await listen(ctx)
    assert out.findings
    assert all(f.advisory for f in out.findings)
    assert {f.category for f in out.findings} == {"odd_pronunciation", "delivery"}
    assert "critical" not in {f.severity for f in out.findings}
    assert out.metrics["clips"][0]["tone"] == "flat"


async def test_prompt_says_why_and_gives_the_intended_text(store):
    analyst = FakeAnalyst()
    artifact = CallArtifact(
        mode=CallMode.LIVE_BOTS,
        timeline=[
            Utterance(speaker=Speaker.BOT, start_s=0.0, end_s=3.0, intended_text="Hello there")
        ],
    )
    await listen(context(store, many_turns(1), artifact, analyst=analyst))
    assert "Hello there" in analyst.prompts[0]


async def test_one_bad_clip_does_not_stop_the_rest(store):
    seen = []

    def reply(audio, prompt):
        seen.append(audio)
        if len(seen) == 1:
            raise ConnectionError("hiccup")
        return "not json" if len(seen) == 2 else {"tone": "ok", "delivery": "ok"}

    out = await listen(context(store, many_turns(), analyst=FakeAnalyst(reply)))
    assert len(out.metrics["failed_clips"]) == 2
    assert len(out.metrics["clips"]) == 4


async def test_failing_on_every_clip_is_an_error(store):
    def reply(audio, prompt):
        raise ConnectionError("down")

    ctx = context(store, many_turns(2), analyst=FakeAnalyst(reply))
    with pytest.raises(RuntimeError, match="every clip"):
        await listen(ctx)
