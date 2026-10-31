"""Conversation timing, measured from the audio itself and cross-checked against the timeline."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from voicelab.core.models import BargeInEvent, Evidence, Finding, Speaker, Utterance
from voicelab.eval.config import TimingConfig
from voicelab.eval.context import EvalContext, EvaluatorOutput
from voicelab.eval.vad import Region

_SYNTHETIC_REASON = (
    "timing_is_real is False: this was a turn-based call, so its pauses come from the scheduler"
)
_INFERRED_REASON = "channels were inferred from a mono recording, so they can't overlap"


@dataclass(frozen=True)
class Overlap:
    start_s: float
    end_s: float
    started_by: Speaker

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Latency:
    caller_end_s: float
    bot_start_s: float

    @property
    def latency_s(self) -> float:
        return self.bot_start_s - self.caller_end_s


@dataclass(frozen=True)
class BargeIn:
    caller_start_s: float
    bot_end_s: float

    @property
    def reaction_s(self) -> float:
        return self.bot_end_s - self.caller_start_s


def merge_turns(regions: list[Region], gap_s: float) -> list[Region]:
    """Join regions separated by less than `gap_s` into one turn."""
    turns: list[Region] = []
    for region in regions:
        if turns and region.start_s - turns[-1].end_s < gap_s:
            turns[-1] = Region(turns[-1].start_s, region.end_s)
        else:
            turns.append(region)
    return turns


def total_speech(regions: list[Region]) -> float:
    return sum(r.duration_s for r in regions)


def find_overlaps(caller: list[Region], bot: list[Region], *, min_s: float = 0.0) -> list[Overlap]:
    overlaps = []
    for c in caller:
        for b in bot:
            start, end = max(c.start_s, b.start_s), min(c.end_s, b.end_s)
            if end - start > 0 and end - start >= min_s:
                started_by = Speaker.BOT if c.start_s < b.start_s else Speaker.CALLER
                overlaps.append(Overlap(start, end, started_by))
    return sorted(overlaps, key=lambda o: o.start_s)


def find_silences(caller: list[Region], bot: list[Region]) -> list[Region]:
    """Stretches where nobody speaks, between the first and last speech of the call."""
    speech = sorted([*caller, *bot], key=lambda r: r.start_s)
    silences = []
    reached = speech[0].end_s if speech else 0.0
    for region in speech[1:]:
        if region.start_s > reached:
            silences.append(Region(reached, region.start_s))
        reached = max(reached, region.end_s)
    return silences


def longest_monologue(regions: list[Region], other: list[Region], gap_s: float) -> Region | None:
    """Longest run of one speaker's turn that the other side didn't speak during."""
    best: Region | None = None
    for turn in merge_turns(regions, gap_s):
        start = turn.start_s
        for interjection in other:
            if turn.start_s < interjection.start_s < turn.end_s:
                best = _longer(best, Region(start, interjection.start_s))
                start = interjection.end_s
        if start < turn.end_s:
            best = _longer(best, Region(start, turn.end_s))
    return best


def _longer(a: Region | None, b: Region) -> Region:
    return b if a is None or b.duration_s > a.duration_s else a


def response_latencies(caller_turns: list[Region], bot_turns: list[Region]) -> list[Latency]:
    """For each caller turn the bot answered, the gap from caller stopping to bot starting."""
    latencies = []
    for i, turn in enumerate(caller_turns):
        next_caller = caller_turns[i + 1].start_s if i + 1 < len(caller_turns) else float("inf")
        reply = next((b for b in bot_turns if b.start_s >= turn.end_s), None)
        if reply is not None and reply.start_s < next_caller:
            latencies.append(Latency(turn.end_s, reply.start_s))
    return latencies


def find_barge_ins(
    caller_turns: list[Region], bot_turns: list[Region], *, min_overlap_s: float
) -> list[BargeIn]:
    """Moments the caller started over the bot, with when the bot's audio actually stopped."""
    barge_ins = []
    for turn in caller_turns:
        bot = next((b for b in bot_turns if b.start_s < turn.start_s < b.end_s), None)
        if bot is not None and min(bot.end_s, turn.end_s) - turn.start_s >= min_overlap_s:
            barge_ins.append(BargeIn(turn.start_s, bot.end_s))
    return barge_ins


def _rounded(value: float) -> float:
    return round(value, 3)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


class TimingEvaluator:
    name = "timing"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        cfg = ctx.config.timing
        caller = ctx.regions(Speaker.CALLER)
        bot = ctx.regions(Speaker.BOT)
        duration = ctx.artifact.duration_s or ctx.audio(Speaker.BOT).duration_s
        caller_turns = merge_turns(caller, cfg.turn_gap_s)
        bot_turns = merge_turns(bot, cfg.turn_gap_s)
        real = ctx.artifact.timing_is_real
        inferred = ctx.channels_inferred

        silences = find_silences(caller, bot)
        out.metrics.update(_talk_metrics(caller, bot, duration))
        out.metrics["dead_air"] = {
            "total_s": _rounded(total_speech(silences)),
            "longest_s": _rounded(max((s.duration_s for s in silences), default=0.0)),
            "count_over_threshold": sum(s.duration_s >= cfg.dead_air_s for s in silences),
            "synthetic": not real,
        }
        monologue = _monologue_metrics(caller, bot, cfg.monologue_gap_s)
        out.metrics["longest_monologue"] = monologue

        skipped: dict[str, str] = {}
        if inferred:
            skipped["overlap"] = _INFERRED_REASON
            skipped["barge_in"] = _INFERRED_REASON
            out.metrics["overlap"] = None
        else:
            overlaps = find_overlaps(caller, bot, min_s=cfg.min_overlap_s)
            out.metrics["overlap"] = {
                "count": len(overlaps),
                "total_s": _rounded(sum(o.duration_s for o in overlaps)),
                "bot_started": sum(o.started_by == Speaker.BOT for o in overlaps),
                "caller_started": sum(o.started_by == Speaker.CALLER for o in overlaps),
            }

        if real:
            latencies = response_latencies(caller_turns, bot_turns)
            out.metrics["response_latency"] = _latency_metrics(ctx, latencies)
            out.findings.extend(_dead_air_findings(silences, cfg))
            out.findings.extend(_slow_response_findings(latencies, cfg))
        else:
            skipped["latency"] = _SYNTHETIC_REASON
            skipped["barge_in"] = _SYNTHETIC_REASON
            out.metrics["response_latency"] = None

        if real and not inferred:
            barge_ins = find_barge_ins(caller_turns, bot_turns, min_overlap_s=cfg.min_overlap_s)
            out.metrics["barge_in"] = _barge_in_metrics(ctx, barge_ins)
            out.findings.extend(_slow_barge_in_findings(barge_ins, cfg))
            out.findings.extend(_talk_over_findings(caller_turns, bot_turns, cfg))
        else:
            out.metrics["barge_in"] = None
        out.metrics["skipped"] = skipped


def _talk_metrics(caller: list[Region], bot: list[Region], duration: float) -> dict[str, Any]:
    caller_s, bot_s = total_speech(caller), total_speech(bot)
    talking = caller_s + bot_s
    return {
        "duration_s": _rounded(duration),
        "caller_talk_s": _rounded(caller_s),
        "bot_talk_s": _rounded(bot_s),
        "talk_ratio_bot": _rounded(bot_s / talking) if talking else None,
    }


def _monologue_metrics(caller: list[Region], bot: list[Region], gap_s: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for speaker, mine, theirs in (("caller", caller, bot), ("bot", bot, caller)):
        longest = longest_monologue(mine, theirs, gap_s)
        result[speaker] = (
            {"start_s": _rounded(longest.start_s), "duration_s": _rounded(longest.duration_s)}
            if longest
            else None
        )
    return result


def _latency_metrics(ctx: EvalContext, latencies: list[Latency]) -> dict[str, Any]:
    values = [item.latency_s for item in latencies]
    reported = [
        e.ttfb_ms
        for e in ctx.artifact.timeline
        if isinstance(e, Utterance) and e.ttfb_ms is not None
    ]
    return {
        "turns": [
            {
                "caller_end_s": _rounded(item.caller_end_s),
                "bot_start_s": _rounded(item.bot_start_s),
                "latency_s": _rounded(item.latency_s),
            }
            for item in latencies
        ],
        "median_s": _rounded(median(values)) if values else None,
        "p95_s": _rounded(_p95(values)) if values else None,
        "max_s": _rounded(max(values)) if values else None,
        "reported_ttfb_median_ms": _rounded(median(reported)) if reported else None,
    }


def _barge_in_metrics(ctx: EvalContext, barge_ins: list[BargeIn]) -> dict[str, Any]:
    reported = [e for e in ctx.artifact.timeline if isinstance(e, BargeInEvent)]
    reactions = [b.reaction_s for b in barge_ins]
    return {
        "events": [
            {
                "caller_start_s": _rounded(b.caller_start_s),
                "bot_end_s": _rounded(b.bot_end_s),
                "reaction_s": _rounded(b.reaction_s),
            }
            for b in barge_ins
        ],
        "median_reaction_s": _rounded(median(reactions)) if reactions else None,
        "max_reaction_s": _rounded(max(reactions)) if reactions else None,
        # The audio is the measure, and the engine's own count shows where the two disagree.
        "reported_by_engine": len(reported),
    }


def _severity(value: float, major_at: float) -> str:
    return "major" if value >= major_at else "minor"


def _dead_air_findings(silences: list[Region], cfg: TimingConfig) -> list[Finding]:
    return [
        Finding(
            evaluator="timing",
            category="dead_air",
            severity=_severity(s.duration_s, cfg.dead_air_major_s),
            title=f"{s.duration_s:.1f} s of dead air",
            detail=f"Nobody spoke from {s.start_s:.1f} s to {s.end_s:.1f} s.",
            evidence=[Evidence(start_s=s.start_s, end_s=s.end_s)],
        )
        for s in silences
        if s.duration_s >= cfg.dead_air_s
    ]


def _slow_response_findings(latencies: list[Latency], cfg: TimingConfig) -> list[Finding]:
    return [
        Finding(
            evaluator="timing",
            category="slow_response",
            severity=_severity(item.latency_s, cfg.slow_response_major_s),
            title=f"Bot took {item.latency_s:.1f} s to respond",
            detail=(
                f"The caller stopped at {item.caller_end_s:.1f} s and the bot's audio started "
                f"at {item.bot_start_s:.1f} s."
            ),
            evidence=[
                Evidence(speaker=Speaker.BOT, start_s=item.caller_end_s, end_s=item.bot_start_s)
            ],
        )
        for item in latencies
        if item.latency_s >= cfg.slow_response_s
    ]


def _slow_barge_in_findings(barge_ins: list[BargeIn], cfg: TimingConfig) -> list[Finding]:
    return [
        Finding(
            evaluator="timing",
            category="slow_barge_in",
            severity=_severity(b.reaction_s, cfg.talk_over_major_s),
            title=f"Bot kept talking for {b.reaction_s:.1f} s after the caller cut in",
            detail=(
                f"The caller started at {b.caller_start_s:.1f} s and the bot's audio ran until "
                f"{b.bot_end_s:.1f} s."
            ),
            evidence=[Evidence(speaker=Speaker.BOT, start_s=b.caller_start_s, end_s=b.bot_end_s)],
        )
        for b in barge_ins
        if b.reaction_s >= cfg.barge_in_reaction_s
    ]


def _talk_over_findings(
    caller_turns: list[Region], bot_turns: list[Region], cfg: TimingConfig
) -> list[Finding]:
    findings = []
    for bot in bot_turns:
        caller = next((c for c in caller_turns if c.start_s < bot.start_s < c.end_s), None)
        if caller is None:
            continue
        overlap = min(bot.end_s, caller.end_s) - bot.start_s
        if overlap >= cfg.talk_over_s:
            findings.append(
                Finding(
                    evaluator="timing",
                    category="talk_over",
                    severity=_severity(overlap, cfg.talk_over_major_s),
                    title=f"Bot talked over the caller for {overlap:.1f} s",
                    detail=(
                        f"The bot started at {bot.start_s:.1f} s while the caller was speaking."
                    ),
                    evidence=[
                        Evidence(
                            speaker=Speaker.BOT, start_s=bot.start_s, end_s=bot.start_s + overlap
                        )
                    ],
                )
            )
    return findings
