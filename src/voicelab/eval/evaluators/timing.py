"""Conversation timing, measured from the audio itself and cross-checked against the timeline.

Merging a speaker's regions into turns smooths over pauses inside a sentence, but a pause someone
else spoke in is where a turn ended, so turns never merge across the other side. Interruptions are
measured against the raw speech regions: a merged turn would invent speech during the pause.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from voicelab.core.models import (
    BargeInEvent,
    CallArtifact,
    Evidence,
    Finding,
    Speaker,
    Utterance,
)
from voicelab.eval.config import TimingConfig
from voicelab.eval.context import EvalContext, EvaluatorOutput
from voicelab.eval.rollup import roll_up
from voicelab.eval.vad import Region

_SYNTHETIC_REASON = (
    "timing_is_real is False: this was a turn-based call, so its pauses come from the scheduler"
)
_INFERRED_REASON = "channels were guessed from a mono recording, so who spoke when isn't known"
_WAITS = ("slow_response", "unanswered_turn")
_NOISE_REASON = "noise on this line reads as speech on the bot channel, so its turns can't be timed"


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


@dataclass(frozen=True)
class Unanswered:
    caller_end_s: float
    waited_s: float  # silence from the bot until the caller spoke again or the call ended


def merge_turns(regions: list[Region], gap_s: float, other: Sequence[Region] = ()) -> list[Region]:
    """Join regions less than `gap_s` apart into one turn, unless `other` spoke in between."""
    turns: list[Region] = []
    for region in regions:
        if turns and _same_turn(turns[-1], region, gap_s, other):
            turns[-1] = Region(turns[-1].start_s, region.end_s)
        else:
            turns.append(region)
    return turns


def _same_turn(first: Region, second: Region, gap_s: float, other: Sequence[Region]) -> bool:
    if second.start_s - first.end_s >= gap_s:
        return False
    return not any(o.start_s < second.start_s and o.end_s > first.end_s for o in other)


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


def find_silences(caller: list[Region], bot: list[Region], *, end_s: float) -> list[Region]:
    """Stretches where nobody speaks, from the first speech through the end of the call.

    The tail counts, because a bot that never answers leaves nothing but tail.
    """
    speech = sorted([*caller, *bot], key=lambda r: r.start_s)
    if not speech:
        return [Region(0.0, end_s)] if end_s > 0 else []
    silences = []
    reached = speech[0].end_s
    for region in speech[1:]:
        if region.start_s > reached:
            silences.append(Region(reached, region.start_s))
        reached = max(reached, region.end_s)
    if end_s > reached:
        silences.append(Region(reached, end_s))
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


def unanswered_turns(
    caller_turns: list[Region], bot_regions: list[Region], *, end_s: float
) -> list[Unanswered]:
    """Caller turns the bot said nothing after, before the caller spoke again or the call ended."""
    items = []
    for i, turn in enumerate(caller_turns):
        until = caller_turns[i + 1].start_s if i + 1 < len(caller_turns) else end_s
        if not any(b.start_s < until and b.end_s > turn.end_s for b in bot_regions):
            items.append(Unanswered(turn.end_s, max(0.0, until - turn.end_s)))
    return items


def find_barge_ins(
    caller_regions: list[Region], bot_regions: list[Region], *, min_overlap_s: float
) -> list[BargeIn]:
    """Each stretch of bot speech the caller cut into, and when that speech actually stopped."""
    barge_ins = []
    for bot in bot_regions:
        for caller in caller_regions:
            cuts_in = bot.start_s < caller.start_s < bot.end_s
            if cuts_in and min(bot.end_s, caller.end_s) - caller.start_s >= min_overlap_s:
                barge_ins.append(BargeIn(caller.start_s, bot.end_s))
                break
    return barge_ins


def noise_reads_as_speech(bot: list[Region], artifact: CallArtifact, cfg: TimingConfig) -> bool:
    """Whether the bot channel shows speech the bot's own timeline can't account for.

    A speech-like noise bed such as a TV passes the energy VAD. Everything measured from those
    regions would be wrong, and the turn timing is what can fail a call.
    """
    said = [
        Region(e.start_s, e.end_s)
        for e in artifact.timeline
        if isinstance(e, Utterance) and e.speaker == Speaker.BOT
    ]
    total = total_speech(bot)
    if not said or total <= 0:
        return False
    covered = sum(
        max(0.0, min(region.end_s, span.end_s) - max(region.start_s, span.start_s))
        for region in bot
        for span in said
    )
    outside = total - covered
    return outside / total > cfg.max_speech_off_timeline and (
        outside >= cfg.min_speech_off_timeline_s
    )


def _rounded(value: float) -> float:
    return round(value, 3)


def _rollup_title(group: list[Finding]) -> str:
    durations = [(e.end_s or 0.0) - (e.start_s or 0.0) for f in group for e in f.evidence]
    longest = max(durations, default=0.0)
    names = {
        "dead_air": f"{len(group)} stretches of dead air, the longest {longest:.1f} s",
        "slow_response": f"{len(group)} slow responses, the slowest {longest:.1f} s",
        "unanswered_turn": f"{len(group)} turns left unanswered, the longest {longest:.1f} s",
        "slow_barge_in": f"Bot slow to stop on {len(group)} barge-ins, up to {longest:.1f} s",
        "talk_over": f"Bot talked over the caller {len(group)} times, up to {longest:.1f} s",
    }
    return names[group[0].category]


def _rolled_up(findings: list[Finding]) -> list[Finding]:
    """One finding per kind of problem, with the longest occurrence first."""
    ordered = sorted(_without_repeats(findings), key=_duration, reverse=True)
    return roll_up(ordered, key=lambda f: f.category, title=_rollup_title)


def _without_repeats(findings: list[Finding]) -> list[Finding]:
    """A wait the bot owed an answer in is already reported, so it isn't dead air as well."""
    waits = [f.evidence[0] for f in findings if f.category in _WAITS]

    def explained(silence: Evidence) -> bool:
        for wait in waits:
            shared = min(silence.end_s, wait.end_s) - max(silence.start_s, wait.start_s)
            if shared >= (silence.end_s - silence.start_s) / 2:
                return True
        return False

    return [f for f in findings if f.category != "dead_air" or not explained(f.evidence[0])]


def _duration(finding: Finding) -> float:
    evidence = finding.evidence[0]
    return (evidence.end_s or 0.0) - (evidence.start_s or 0.0)


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
        caller_turns = merge_turns(caller, cfg.turn_gap_s, bot)
        bot_turns = merge_turns(bot, cfg.turn_gap_s, caller)
        noisy = noise_reads_as_speech(bot, ctx.artifact, cfg)
        real = ctx.artifact.timing_is_real
        inferred = ctx.channels_inferred

        silences = find_silences(caller, bot, end_s=duration)
        out.metrics.update(_talk_metrics(caller, bot, duration))
        out.metrics["dead_air"] = {
            "total_s": _rounded(total_speech(silences)),
            "longest_s": _rounded(max((s.duration_s for s in silences), default=0.0)),
            "count_over_threshold": sum(s.duration_s >= cfg.dead_air_s for s in silences),
            "synthetic": not real,
        }
        monologue = _monologue_metrics(caller, bot, cfg.monologue_gap_s)
        out.metrics["longest_monologue"] = monologue
        unanswered = unanswered_turns(caller_turns, bot, end_s=duration)
        out.metrics["unanswered_turns"] = [
            {"caller_end_s": _rounded(u.caller_end_s), "waited_s": _rounded(u.waited_s)}
            for u in unanswered
        ]

        skipped: dict[str, str] = {}
        if inferred or noisy:
            reason = _INFERRED_REASON if inferred else _NOISE_REASON
            skipped["overlap"] = reason
            skipped["barge_in"] = reason
            out.metrics["overlap"] = None
        else:
            overlaps = find_overlaps(caller, bot, min_s=cfg.min_overlap_s)
            out.metrics["overlap"] = {
                "count": len(overlaps),
                "total_s": _rounded(sum(o.duration_s for o in overlaps)),
                "bot_started": sum(o.started_by == Speaker.BOT for o in overlaps),
                "caller_started": sum(o.started_by == Speaker.CALLER for o in overlaps),
            }

        if real and not noisy:
            out.findings.extend(_dead_air_findings(silences, cfg))
        if real and not noisy and not inferred:
            latencies = response_latencies(caller_turns, bot_turns)
            out.metrics["response_latency"] = _latency_metrics(ctx, latencies)
            out.findings.extend(_slow_response_findings(latencies, cfg))
            out.findings.extend(_unanswered_findings(unanswered, cfg))
        else:
            reason = _SYNTHETIC_REASON if not real else _NOISE_REASON if noisy else _INFERRED_REASON
            skipped["latency"] = reason
            skipped["barge_in"] = reason
            out.metrics["response_latency"] = None

        if real and not inferred and not noisy:
            barge_ins = find_barge_ins(caller, bot, min_overlap_s=cfg.min_overlap_s)
            out.metrics["barge_in"] = _barge_in_metrics(ctx, barge_ins)
            out.findings.extend(_slow_barge_in_findings(barge_ins, cfg))
            out.findings.extend(_talk_over_findings(caller_turns, bot_turns, cfg))
        else:
            out.metrics["barge_in"] = None
        out.metrics["skipped"] = skipped
        out.findings[:] = _rolled_up(out.findings)


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


def _unanswered_findings(items: list[Unanswered], cfg: TimingConfig) -> list[Finding]:
    return [
        Finding(
            evaluator="timing",
            category="unanswered_turn",
            severity=_severity(item.waited_s, cfg.slow_response_major_s),
            title=f"Bot left the caller unanswered for {item.waited_s:.1f} s",
            detail=(
                f"The caller stopped at {item.caller_end_s:.1f} s and the bot said nothing "
                "before the caller spoke again or the call ended."
            ),
            evidence=[
                Evidence(
                    speaker=Speaker.BOT,
                    start_s=item.caller_end_s,
                    end_s=item.caller_end_s + item.waited_s,
                )
            ],
        )
        for item in items
        if item.waited_s >= cfg.slow_response_s
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
