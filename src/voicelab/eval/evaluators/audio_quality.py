"""Signal-level checks on the recording. Pure numpy, judged against the call's own median.

These thresholds are heuristics that nobody has calibrated against listeners yet, so every finding
here is advisory: it can send a call to review but can't fail it.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np

from voicelab.core.audio import AudioClip
from voicelab.core.models import Evidence, Finding, Speaker, Utterance
from voicelab.eval.config import AudioQualityConfig
from voicelab.eval.context import EvalContext, EvaluatorOutput
from voicelab.eval.vad import Region

_ABRUPT_WINDOW_S = 0.02
_END_OF_CALL_MARGIN_S = 0.05
_INTERRUPTED_MARGIN_S = 0.5
_MIN_UTTERANCE_S = 0.3
_MIN_UTTERANCES_FOR_OUTLIERS = 3

# ITU-R BS.1770 K-weighting filters, specified as biquads at 48 kHz.
_SHELF = ((1.53512485958697, -2.69169618940638, 1.19839281085285),
          (1.0, -1.69065929318241, 0.73248077421585))  # fmt: skip
_HIGH_PASS = ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))


def _k_weight_gain(freqs: np.ndarray) -> np.ndarray:
    z = np.exp(-2j * np.pi * freqs / 48000)

    def response(section: tuple[tuple[float, ...], tuple[float, ...]]) -> np.ndarray:
        b, a = section
        return (b[0] + b[1] * z + b[2] * z**2) / (a[0] + a[1] * z + a[2] * z**2)

    return np.abs(response(_SHELF) * response(_HIGH_PASS))


def integrated_lufs(samples: np.ndarray, rate: int) -> float | None:
    """Integrated loudness in the style of BS.1770, or None for silence.

    Filters in the frequency domain, because numpy has no IIR filter and a sample loop over a long
    call is too slow. Evaluating the 48 kHz filter response at each bin keeps the curve exact.
    """
    block = round(0.4 * rate)
    if len(samples) < block:
        return None
    freqs = np.fft.rfftfreq(len(samples), 1 / rate)
    weighted = np.fft.irfft(np.fft.rfft(samples) * _k_weight_gain(freqs), len(samples))
    energy = np.concatenate([[0.0], np.cumsum(np.square(weighted))])
    starts = np.arange(0, len(samples) - block + 1, block // 4)
    power = (energy[starts + block] - energy[starts]) / block

    loud = power[power > 10 ** ((-70 + 0.691) / 10)]
    if len(loud) == 0:
        return None
    gate = np.mean(loud) * 0.1  # the relative gate sits 10 LU under the ungated mean
    kept = loud[loud > gate]
    return float(-0.691 + 10 * np.log10(np.mean(kept)))


def clipped_fraction(samples: np.ndarray, level: float) -> float:
    return float(np.mean(np.abs(samples) >= level)) if len(samples) else 0.0


def spectral_centroid(samples: np.ndarray, rate: int) -> float:
    power = np.abs(np.fft.rfft(samples)) ** 2
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    return float(np.sum(np.fft.rfftfreq(len(samples), 1 / rate) * power) / total)


def _level_db(samples: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(samples)) + 1e-12))


def _speech_samples(clip: AudioClip, regions: list[Region]) -> np.ndarray:
    parts = [clip.slice(r.start_s, r.end_s).samples() for r in regions]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _ends_abruptly(clip: AudioClip, region: Region, ratio: float) -> bool:
    """Speech that stops at full level with silence right after, instead of decaying."""
    tail = clip.slice(region.end_s - _ABRUPT_WINDOW_S, region.end_s).samples()
    after = clip.slice(region.end_s, region.end_s + _ABRUPT_WINDOW_S).samples()
    body = clip.slice(region.start_s, region.end_s).samples()
    if len(tail) == 0 or len(after) == 0:
        return False
    tail_rms = float(np.sqrt(np.mean(np.square(tail))))
    after_rms = float(np.sqrt(np.mean(np.square(after))))
    frame = round(_ABRUPT_WINDOW_S * clip.sample_rate)
    count = len(body) // frame
    if count == 0:
        return False
    frames = body[: count * frame].reshape(count, frame)
    typical = float(np.median(np.sqrt(np.mean(np.square(frames), axis=1))))
    return typical > 0 and tail_rms >= ratio * typical and after_rms < 0.1 * tail_rms


def _finding(
    category: str, title: str, detail: str, severity: str, start_s: float, end_s: float
) -> Finding:
    return Finding(
        evaluator="audio_quality",
        category=category,
        severity=severity,
        advisory=True,
        title=title,
        detail=detail,
        evidence=[Evidence(speaker=Speaker.BOT, start_s=start_s, end_s=end_s)],
    )


def _worst_clipping(clip: AudioClip, regions: list[Region], level: float) -> Region:
    return max(
        regions, key=lambda r: np.sum(np.abs(clip.slice(r.start_s, r.end_s).samples()) >= level)
    )


def _bot_utterances(ctx: EvalContext) -> list[Utterance]:
    utterances = [
        e
        for e in ctx.artifact.timeline
        if isinstance(e, Utterance) and e.speaker == Speaker.BOT and e.end_s > e.start_s
    ]
    if utterances:
        return utterances
    return [
        Utterance(speaker=Speaker.BOT, start_s=r.start_s, end_s=r.end_s)
        for r in ctx.regions(Speaker.BOT)
    ]


@dataclass(frozen=True)
class _UtteranceStats:
    utterance: Utterance
    level_db: float
    centroid_hz: float


def _utterance_stats(clip: AudioClip, utterances: list[Utterance]) -> list[_UtteranceStats]:
    stats = []
    for utterance in utterances:
        samples = clip.slice(utterance.start_s, utterance.end_s).samples()
        if len(samples) / clip.sample_rate >= _MIN_UTTERANCE_S:
            centroid = spectral_centroid(samples, clip.sample_rate)
            stats.append(_UtteranceStats(utterance, _level_db(samples), centroid))
    return stats


def _outlier_findings(stats: list[_UtteranceStats], cfg: AudioQualityConfig) -> list[Finding]:
    if len(stats) < _MIN_UTTERANCES_FOR_OUTLIERS:
        return []
    typical_level = median(s.level_db for s in stats)
    typical_centroid = median(s.centroid_hz for s in stats)
    findings = []
    for s in stats:
        u = s.utterance
        difference = s.level_db - typical_level
        if abs(difference) >= cfg.outlier_level_db:
            direction = "louder" if difference > 0 else "quieter"
            findings.append(
                _finding(
                    "level_outlier",
                    f"Utterance is {abs(difference):.1f} dB {direction} than the call's median",
                    f"Utterance at {u.start_s:.1f} s measured {s.level_db:.1f} dBFS.",
                    "minor",
                    u.start_s,
                    u.end_s,
                )
            )
        if typical_centroid > 0 and s.centroid_hz > 0:
            ratio = max(s.centroid_hz, typical_centroid) / min(s.centroid_hz, typical_centroid)
            if ratio >= cfg.outlier_centroid_ratio:
                findings.append(
                    _finding(
                        "timbre_outlier",
                        f"Utterance timbre is {ratio:.2f}x off the call's median",
                        f"Spectral centroid {s.centroid_hz:.0f} Hz against "
                        f"{typical_centroid:.0f} Hz.",
                        "minor",
                        u.start_s,
                        u.end_s,
                    )
                )
    return findings


def _internal_silence_findings(
    regions: list[Region], utterances: list[Utterance], cfg: AudioQualityConfig
) -> list[Finding]:
    findings = []
    for utterance in utterances:
        inside = [r for r in regions if r.start_s < utterance.end_s and r.end_s > utterance.start_s]
        for previous, region in zip(inside, inside[1:], strict=False):
            gap = region.start_s - previous.end_s
            if gap >= cfg.max_internal_silence_s:
                findings.append(
                    _finding(
                        "internal_silence",
                        f"{gap:.1f} s of silence inside an utterance",
                        f"Quiet from {previous.end_s:.1f} s to {region.start_s:.1f} s.",
                        "minor",
                        previous.end_s,
                        region.start_s,
                    )
                )
    return findings


def _abrupt_findings(
    clip: AudioClip,
    regions: list[Region],
    interrupted_ends: list[float],
    cfg: AudioQualityConfig,
) -> list[Finding]:
    findings = []
    for region in regions:
        expected_cut = any(
            abs(region.end_s - end) <= _INTERRUPTED_MARGIN_S for end in interrupted_ends
        )
        at_call_end = region.end_s >= clip.duration_s - _END_OF_CALL_MARGIN_S
        if expected_cut or at_call_end:
            continue
        if _ends_abruptly(clip, region, cfg.abrupt_end_ratio):
            findings.append(
                _finding(
                    "abrupt_cut",
                    f"Speech stops abruptly at {region.end_s:.1f} s",
                    "The audio ends at full level with no decay, and no interruption explains it.",
                    "minor",
                    max(0.0, region.end_s - 1.0),
                    region.end_s,
                )
            )
    return findings


class AudioQualityEvaluator:
    name = "audio_quality"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        cfg = ctx.config.audio_quality
        out.metrics["caller"] = _channel_metrics(
            ctx.audio(Speaker.CALLER), ctx.regions(Speaker.CALLER), cfg
        )
        clip = ctx.audio(Speaker.BOT)
        regions = ctx.regions(Speaker.BOT)
        out.metrics["bot"] = _channel_metrics(clip, regions, cfg)
        self._mos(ctx, out, clip)
        if not regions:
            out.metrics["note"] = "no speech on the bot channel"
            return

        utterances = _bot_utterances(ctx)
        interrupted_ends = [u.end_s for u in utterances if u.interrupted]
        fraction = out.metrics["bot"]["clipped_fraction"]
        if fraction >= cfg.clip_fraction:
            worst = _worst_clipping(clip, regions, cfg.clip_level)
            severity = "major" if fraction >= cfg.clip_major_fraction else "minor"
            out.findings.append(
                _finding(
                    "clipping",
                    f"{fraction:.2%} of bot speech samples are clipped",
                    f"Worst stretch starts at {worst.start_s:.1f} s.",
                    severity,
                    worst.start_s,
                    worst.end_s,
                )
            )
        loudness = out.metrics["bot"]["lufs"]
        if loudness is not None and not cfg.loudness_low_lufs <= loudness <= cfg.loudness_high_lufs:
            out.findings.append(
                _finding(
                    "loudness",
                    f"Bot loudness {loudness:.1f} LUFS is out of range",
                    f"Expected {cfg.loudness_low_lufs} to {cfg.loudness_high_lufs} LUFS.",
                    "minor",
                    regions[0].start_s,
                    min(regions[0].end_s, regions[0].start_s + 5.0),
                )
            )
        out.findings.extend(_abrupt_findings(clip, regions, interrupted_ends, cfg))
        out.findings.extend(_internal_silence_findings(regions, utterances, cfg))
        out.findings.extend(_outlier_findings(_utterance_stats(clip, utterances), cfg))

    def _mos(self, ctx: EvalContext, out: EvaluatorOutput, clip: AudioClip) -> None:
        predictor = ctx.deps.mos
        if predictor is None:
            out.metrics["mos"] = None
            out.metrics["mos_note"] = "no MOS predictor configured"
            return
        out.model = predictor.name
        score = float(predictor.predict(clip))
        out.metrics["mos"] = round(score, 2)
        if score < ctx.config.audio_quality.low_mos:
            regions = ctx.regions(Speaker.BOT)
            start, end = (regions[0].start_s, regions[-1].end_s) if regions else (0.0, 0.0)
            out.findings.append(
                _finding(
                    "low_mos", f"Predicted MOS {score:.2f}", "Across the whole bot channel.",
                    "minor", start, end,
                )
            )  # fmt: skip


def _channel_metrics(
    clip: AudioClip, regions: list[Region], cfg: AudioQualityConfig
) -> dict[str, Any]:
    speech = _speech_samples(clip, regions)
    loudness = integrated_lufs(speech, clip.sample_rate)
    return {
        "lufs": round(loudness, 2) if loudness is not None else None,
        "clipped_fraction": round(clipped_fraction(speech, cfg.clip_level), 6),
        "peak": round(float(np.max(np.abs(speech))), 4) if len(speech) else 0.0,
    }
