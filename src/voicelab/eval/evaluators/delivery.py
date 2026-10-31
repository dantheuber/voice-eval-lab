"""How the speech was delivered, measured from the signal with no model.

Audio models read the words rather than the voice: in the benchmark LFM3-Audio called a flat clip
"frustrated" and missed all five planted defects. So rate, pauses, chops, pitch movement, noise and
edges are measured here, per utterance. Findings are advisory, because the limits come from a small
corpus and not from listeners, but they beat the audio model when the two disagree.

Only the bot gets findings. Caller speech is measured for the charts, since a noisy or fast caller
is usually the persona working as designed.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

from voicelab.core.audio import AudioClip
from voicelab.core.models import Evidence, Finding, Speaker, Utterance, Word
from voicelab.eval.config import DeliveryConfig
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.normalize import tokenize
from voicelab.eval.pitch import HOP_S, pitch_stats, track_pitch
from voicelab.eval.rollup import roll_up
from voicelab.eval.speech_measures import (
    edge_ms,
    estimate_snr_db,
    find_chops,
    speech_time,
)
from voicelab.eval.vad import Region

LIMITATION = (
    "Delivery limits come from 21 clips, 18 real calls, and no listener ratings. On a phone line, "
    "pitch is read from harmonics because the band removes the fundamental, and SNR covers 300 to "
    "3400 Hz."
)
_WORD_SLACK_S = 0.1
_WORD_EDGE_S = 0.02
_EDGE_LEAD_S = 0.05
_EDGE_BODY_S = 0.3
_SAME_EDGE_S = 0.1
_SENTENCE_MARKS = (".", ",", "?", "!", ";", ":")
_EDGE_KINDS = ("abrupt_onset", "abrupt_offset")
_KIND_NAMES = {
    "rushed": "Speech rushed",
    "dragging": "Speech dragging",
    "robotic": "Pitch is robotic",
    "monotone": "Pitch is monotone",
    "noisy": "Noisy speech",
    "halting": "Halting speech",
    "chopped": "Audio chopped",
    "abrupt_onset": "Speech starts abruptly",
    "abrupt_offset": "Speech stops abruptly",
}


@dataclass
class UtteranceDelivery:
    """One utterance's measurements, in the shape the dashboard charts."""

    speaker: str
    utterance_id: str | None
    start_s: float
    end_s: float
    words: int = 0
    speech_s: float = 0.0
    wpm: float | None = None
    pause_count: int = 0
    longest_pause_s: float = 0.0
    pause_ratio: float = 0.0
    halting_pauses: int = 0  # the pauses that came mid-sentence
    halting_ratio: float = 0.0
    interrupted: bool = False
    chops: int = 0
    mid_word_chops: int = 0
    voiced_s: float = 0.0
    pitch_median_hz: float | None = None
    pitch_range_st: float | None = None
    pitch_std_st: float | None = None
    pitch_iqr_st: float | None = None
    snr_db: float | None = None
    onset_ms: float | None = None
    offset_ms: float | None = None
    flags: list[str] = field(default_factory=list)

    def rounded(self) -> dict[str, Any]:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class _Span:
    start_s: float
    end_s: float
    utterance_id: str | None = None
    interrupted: bool = False


@dataclass(frozen=True)
class _Flag:
    item: UtteranceDelivery
    kind: str
    title: str
    detail: str


def _spans(ctx: EvalContext, speaker: Speaker) -> list[_Span]:
    utterances = [
        e
        for e in ctx.artifact.timeline
        if isinstance(e, Utterance) and e.speaker == speaker and e.end_s > e.start_s
    ]
    if utterances:
        return [_Span(u.start_s, u.end_s, u.id, u.interrupted) for u in utterances]
    return [_Span(r.start_s, r.end_s) for r in ctx.regions(speaker)]


def _inside(regions: list[Region], span: _Span) -> list[Region]:
    clipped = [
        Region(max(r.start_s, span.start_s), min(r.end_s, span.end_s))
        for r in regions
        if r.start_s < span.end_s and r.end_s > span.start_s
    ]
    return clipped


def _spoken_words(words: list[Word]) -> int:
    """Words as said. "$49.99" is one ASR word and several spoken ones, and would read as slow."""
    return sum(t.norm != "." for t in tokenize([w.text for w in words]))


def _measure(
    span: _Span,
    clip: AudioClip,
    regions: list[Region],
    words: list[Word],
    cfg: DeliveryConfig,
    speaker: Speaker,
) -> UtteranceDelivery:
    item = UtteranceDelivery(speaker.value, span.utterance_id, span.start_s, span.end_s)
    item.interrupted = span.interrupted
    rate = clip.sample_rate
    item.snr_db = estimate_snr_db(clip.slice(span.start_s, span.end_s).samples(), rate)
    inside = _inside(regions, span)
    first = inside[0].start_s if inside else span.start_s
    last = inside[-1].end_s if inside else span.end_s
    item.speech_s, pauses = speech_time(inside, cfg.pause_s)
    if _is_noisy(item, cfg) or not inside:
        # Noise near speech level breaks energy VAD, so the utterance's own length is safer.
        item.speech_s, pauses = span.end_s - span.start_s, []
    item.pause_count = len(pauses)
    item.longest_pause_s = max((p.duration_s for p in pauses), default=0.0)
    if last > first:
        item.pause_ratio = sum(p.duration_s for p in pauses) / (last - first)

    said = [
        w
        for w in words
        if span.start_s - _WORD_SLACK_S <= (w.start_s + w.end_s) / 2 <= span.end_s + _WORD_SLACK_S
    ]
    if last > first:
        halting = [p for p in pauses if not _after_a_sentence(p, said)]
        item.halting_pauses = len(halting)
        item.halting_ratio = sum(p.duration_s for p in halting) / (last - first)
    item.words = _spoken_words(said)
    if item.words >= cfg.min_words and item.speech_s >= cfg.min_rate_speech_s:
        item.wpm = item.words / item.speech_s * 60

    speech = clip.slice(first, last).samples()
    chops = find_chops(
        speech, rate, min_s=cfg.chop_min_s, max_s=cfg.chop_max_s, cliff_db=cfg.chop_cliff_db
    )
    item.chops = len(chops)
    item.mid_word_chops = sum(_in_a_word(first + (c.start_s + c.end_s) / 2, said) for c in chops)

    pitch = track_pitch(speech, rate)
    item.voiced_s = float((pitch == pitch).sum() * HOP_S)
    stats = pitch_stats(pitch) if item.voiced_s >= cfg.min_voiced_s else None
    if stats is not None:
        item.pitch_median_hz = stats.median_hz
        item.pitch_range_st = stats.range_st
        item.pitch_std_st = stats.std_st
        item.pitch_iqr_st = stats.iqr_st

    item.onset_ms = edge_ms(clip.slice(first - _EDGE_LEAD_S, first + _EDGE_BODY_S).samples(), rate)
    if not span.interrupted:  # being cut off is the caller's doing, not a fault in the speech
        tail = clip.slice(last - _EDGE_BODY_S, last + _EDGE_LEAD_S).samples()
        item.offset_ms = edge_ms(tail, rate, at_end=True)
    return item


def _after_a_sentence(pause: Region, words: list[Word]) -> bool:
    """Whether the word before this pause ended a sentence or a clause, so it's punctuation."""
    before = [w for w in words if w.start_s < pause.start_s + _WORD_SLACK_S]
    return bool(before) and before[-1].text.endswith(_SENTENCE_MARKS)


def _in_a_word(moment: float, words: list[Word]) -> bool:
    return any(w.start_s + _WORD_EDGE_S < moment < w.end_s - _WORD_EDGE_S for w in words)


def _measure_channel(
    ctx: EvalContext, speaker: Speaker, spans: list[_Span], cfg: DeliveryConfig
) -> list[UtteranceDelivery]:
    clip = ctx.audio(speaker)
    regions = ctx.regions(speaker)
    transcript = ctx.transcripts.get(speaker)
    words = transcript.words if transcript else []
    return [_measure(span, clip, regions, words, cfg, speaker) for span in spans]


def _median(values: list[float | None], cfg: DeliveryConfig) -> float | None:
    known = [v for v in values if v is not None]
    return median(known) if len(known) >= cfg.min_utterances_for_median else None


def _is_noisy(item: UtteranceDelivery, cfg: DeliveryConfig) -> bool:
    return item.snr_db is not None and item.snr_db < cfg.min_snr_db


def _is_chopped(item: UtteranceDelivery, cfg: DeliveryConfig) -> bool:
    return item.chops >= cfg.min_chops


def _rate_is_measurable(item: UtteranceDelivery, cfg: DeliveryConfig) -> bool:
    """Noise and chops break the speech timing rate rests on, and get reported themselves. A cut
    off utterance's span runs on past the words said in it."""
    noisy = item.snr_db is not None and item.snr_db < cfg.rate_min_snr_db
    return not (item.wpm is None or noisy or _is_chopped(item, cfg) or item.interrupted)


def _rate_flag(item: UtteranceDelivery, typical: float | None, cfg: DeliveryConfig) -> _Flag | None:
    if not _rate_is_measurable(item, cfg):
        return None
    if typical is None:
        return _fixed_rate_flag(item, cfg)
    if item.words < cfg.min_compare_words or item.speech_s < cfg.min_compare_speech_s:
        return None
    ratio = item.wpm / typical
    if ratio >= cfg.rate_median_ratio:
        kind = "rushed"
    elif ratio <= 1 / cfg.rate_median_ratio:
        kind = "dragging"
    else:
        return None
    why = f"{ratio:.2f}x the call's median of {typical:.0f} wpm"
    return _Flag(item, kind, f"Speech {kind}: {item.wpm:.0f} wpm", f"Rate is {why}.")


def _fixed_rate_flag(item: UtteranceDelivery, cfg: DeliveryConfig) -> _Flag | None:
    if item.wpm > cfg.max_wpm:
        kind, why = "rushed", f"over the {cfg.max_wpm:.0f} wpm limit"
    elif item.wpm < cfg.min_wpm:
        kind, why = "dragging", f"under the {cfg.min_wpm:.0f} wpm limit"
    else:
        return None
    return _Flag(item, kind, f"Speech {kind}: {item.wpm:.0f} wpm", f"Rate is {why}.")


def _call_rate_flag(
    items: list[UtteranceDelivery], typical: float | None, cfg: DeliveryConfig
) -> _Flag | None:
    """A voice that's fast all call is one thing to say, about the call."""
    if typical is None:
        return None
    if typical > cfg.max_median_wpm:
        kind, why = "rushed", f"over the {cfg.max_median_wpm:.0f} wpm limit for a call"
    elif typical < cfg.min_median_wpm:
        kind, why = "dragging", f"under the {cfg.min_median_wpm:.0f} wpm limit for a call"
    else:
        return None
    title = f"Speech {kind} the whole call: {typical:.0f} wpm"
    return _Flag(max(items, key=lambda i: i.words), kind, title, f"The median is {why}.")


def _pitch_flag(
    item: UtteranceDelivery, typical: float | None, cfg: DeliveryConfig
) -> _Flag | None:
    iqr = item.pitch_iqr_st
    if iqr is None:
        return None
    measured = f"Pitch moved {iqr:.2f} semitones (IQR) around {item.pitch_median_hz:.0f} Hz"
    if iqr < cfg.robotic_iqr_st:
        kind, why = "robotic", f"under the {cfg.robotic_iqr_st} st limit"
    elif iqr < cfg.monotone_iqr_st:
        kind, why = "monotone", f"under the {cfg.monotone_iqr_st} st limit"
    elif typical and iqr <= typical * cfg.pitch_median_ratio:
        kind, why = "monotone", f"against the call's median of {typical:.2f} st"
    else:
        return None
    return _Flag(item, kind, f"Pitch is {kind}: {iqr:.2f} st of movement", f"{measured}, {why}.")


def _noise_flag(
    item: UtteranceDelivery, typical: float | None, cfg: DeliveryConfig
) -> _Flag | None:
    snr = item.snr_db
    if snr is None:
        return None
    if snr < cfg.min_snr_db:
        why = f"under the {cfg.min_snr_db:.0f} dB limit"
    elif (
        typical is not None
        and snr <= typical - cfg.snr_median_drop_db
        and snr < cfg.noise_relative_ceiling_db
    ):
        why = f"{typical - snr:.0f} dB under the call's median"
    else:
        return None
    return _Flag(item, "noisy", f"Noisy speech: {snr:.0f} dB SNR", f"Estimated SNR is {why}.")


def _pause_flag(item: UtteranceDelivery, cfg: DeliveryConfig) -> _Flag | None:
    if item.halting_pauses < cfg.min_pauses or item.halting_ratio < cfg.max_pause_ratio:
        return None
    share = f"{item.halting_ratio:.0%}"
    title = f"Halting speech: {item.halting_pauses} pauses mid-sentence"
    return _Flag(item, "halting", title, f"{share} of the utterance was silence.")


def _chop_flag(item: UtteranceDelivery, cfg: DeliveryConfig) -> _Flag | None:
    if not _is_chopped(item, cfg):
        return None
    detail = f"{item.chops} cliff-edged dropouts, {item.mid_word_chops} of them inside a word."
    return _Flag(item, "chopped", f"Audio chopped {item.chops} times", detail)


def _edge_flags(item: UtteranceDelivery, cuts: list[float], cfg: DeliveryConfig) -> list[_Flag]:
    flags: list[_Flag] = []
    # A gate's first edge is the chop again, and noise that starts with the clip isn't the voice.
    if _is_chopped(item, cfg) or _is_noisy(item, cfg):
        return flags
    if item.onset_ms is not None and item.onset_ms < cfg.abrupt_edge_ms:
        detail = f"Full level arrived in {item.onset_ms:.0f} ms."
        flags.append(_Flag(item, "abrupt_onset", "Speech starts abruptly", detail))
    # Audio quality already reports a hard stop, and one defect shouldn't count twice.
    reported = any(abs(cut - item.end_s) <= _SAME_EDGE_S for cut in cuts)
    if item.offset_ms is not None and item.offset_ms < cfg.abrupt_edge_ms and not reported:
        detail = f"Level fell away in {item.offset_ms:.0f} ms."
        flags.append(_Flag(item, "abrupt_offset", "Speech stops abruptly", detail))
    return flags


def _flags(
    items: list[UtteranceDelivery], ctx: EvalContext, cfg: DeliveryConfig
) -> tuple[list[_Flag], dict[str, float | None]]:
    medians = {
        "wpm": _median([i.wpm for i in items], cfg),
        "pitch_iqr_st": _median([i.pitch_iqr_st for i in items], cfg),
        "snr_db": _median([i.snr_db for i in items], cfg),
    }
    cuts = [
        e.end_s
        for f in ctx.findings
        if f.evaluator == "audio_quality" and f.category == "abrupt_cut"
        for e in f.evidence
        if e.end_s is not None
    ]
    flags: list[_Flag] = []
    for item in items:
        checks = [
            _rate_flag(item, medians["wpm"], cfg),
            _pause_flag(item, cfg),
            _chop_flag(item, cfg),
            _pitch_flag(item, medians["pitch_iqr_st"], cfg),
            _noise_flag(item, medians["snr_db"], cfg),
            *_edge_flags(item, cuts, cfg),
        ]
        found = [f for f in checks if f is not None]
        item.flags = [f.kind for f in found]
        flags.extend(found)
    whole_call = _call_rate_flag(items, medians["wpm"], cfg) if items else None
    if whole_call is not None:
        flags.append(whole_call)
    return flags, medians


def _worth_a_reviewer(
    flag: _Flag, counts: dict[str, int], utterances: int, ctx: EvalContext, cfg: DeliveryConfig
) -> bool:
    """Whether this says something about the bot. Details stay in the call, out of the way."""
    if flag.kind == "noisy":
        # A persona's line is noisy on purpose, and the bot didn't make that noise.
        persona = ctx.artifact.persona_snapshot
        return persona is None or persona.line.noise == "none"
    if flag.kind in _EDGE_KINDS:
        seen = counts[flag.kind]
        return seen >= cfg.abrupt_count_to_review and (
            seen / max(1, utterances) >= cfg.abrupt_share_to_review
        )
    return True


def _rollup_title(group: list[Finding]) -> str:
    return f"{_KIND_NAMES[group[0].category]} on {len(group)} utterances"


def _finding(flag: _Flag, *, reviewed: bool = True) -> Finding:
    item = flag.item
    return Finding(
        evaluator="delivery",
        category=flag.kind,
        severity="minor" if reviewed else "info",
        advisory=True,
        title=flag.title,
        detail=flag.detail,
        evidence=[Evidence(speaker=Speaker.BOT, start_s=item.start_s, end_s=item.end_s)],
    )


class DeliveryEvaluator:
    name = "delivery"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        cfg = ctx.config.delivery
        bot_spans = _spans(ctx, Speaker.BOT)
        caller_spans = _spans(ctx, Speaker.CALLER)
        if not bot_spans and not caller_spans:
            raise Skip("no speech to measure")
        # A long call is seconds of numpy work, and a live call may share this event loop.
        bot = await asyncio.to_thread(_measure_channel, ctx, Speaker.BOT, bot_spans, cfg)
        caller = await asyncio.to_thread(_measure_channel, ctx, Speaker.CALLER, caller_spans, cfg)

        flags, medians = _flags(bot, ctx, cfg)
        ctx.delivery = bot
        counts: dict[str, int] = {}
        for flag in flags:
            counts[flag.kind] = counts.get(flag.kind, 0) + 1
        # A voice that's flat throughout is one problem, not one finding per sentence.
        findings = [
            _finding(flag, reviewed=_worth_a_reviewer(flag, counts, len(bot), ctx, cfg))
            for flag in flags
        ]
        out.findings.extend(roll_up(findings, key=lambda f: f.category, title=_rollup_title))
        out.metrics.update(
            {
                "bot": [item.rounded() for item in bot],
                "caller": [item.rounded() for item in caller],
                "medians": {k: round(v, 3) if v else v for k, v in medians.items()},
                "flag_counts": counts,
                "limitation": LIMITATION,
            }
        )
        if Speaker.BOT not in ctx.transcripts:
            out.metrics["wpm_note"] = "no transcript of the bot channel, so no speaking rate"
