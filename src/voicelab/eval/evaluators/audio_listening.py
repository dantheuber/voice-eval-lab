"""An audio-capable model listens to a few bot clips.

These models agree with human listeners less than 80% of the time, so everything here is advisory:
it points a person at a clip and never decides a call. Where it contradicts the `delivery`
measurements of the same clip, the measurements win, as `eval.reconcile` works out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voicelab.core.models import Evidence, Finding, Speaker, Utterance
from voicelab.eval.config import ListeningConfig
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.judging import parse_json_reply, render_prompt
from voicelab.eval.reconcile import reconcile

LISTENING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tone": {"type": "string"},
        "delivery": {"type": "string"},
        "odd_pronunciations": {"type": "array", "items": {"type": "string"}},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "severity": {"type": "string", "enum": ["info", "minor", "major"]},
                },
                "required": ["title", "severity"],
            },
        },
    },
    "required": ["tone", "delivery", "odd_pronunciations", "issues"],
}

_SEVERITY_RANK = {"critical": 0, "major": 1, "minor": 2, "info": 3}
_ALLOWED_SEVERITIES = ("info", "minor", "major")


@dataclass(frozen=True)
class _Clip:
    start_s: float
    end_s: float
    why: str

    def overlaps(self, other: _Clip) -> bool:
        return self.start_s < other.end_s and other.start_s < self.end_s


def _fit(
    start_s: float, end_s: float, duration_s: float, cfg: ListeningConfig
) -> tuple[float, float]:
    """Widen a short moment to a hearable length and cap a long one, staying inside the call."""
    middle = (start_s + end_s) / 2
    length = min(max(end_s - start_s, cfg.clip_min_s), cfg.clip_max_s)
    start = max(0.0, middle - length / 2)
    return start, min(duration_s, start + length)


def _flagged_clips(ctx: EvalContext, duration_s: float, cfg: ListeningConfig) -> list[_Clip]:
    clips: list[_Clip] = []
    ordered = sorted(ctx.findings, key=lambda f: _SEVERITY_RANK[f.severity])
    for finding in ordered:
        for evidence in finding.evidence:
            if evidence.speaker != Speaker.BOT or evidence.start_s is None:
                continue
            end_s = evidence.end_s if evidence.end_s is not None else evidence.start_s
            clip = _Clip(
                *_fit(evidence.start_s, end_s, duration_s, cfg),
                f"flagged by {finding.evaluator}: {finding.title}",
            )
            if not any(clip.overlaps(kept) for kept in clips):
                clips.append(clip)
    return clips


def _sample_clips(
    ctx: EvalContext, taken: list[_Clip], count: int, cfg: ListeningConfig, duration_s: float
) -> list[_Clip]:
    """Evenly spaced stretches of ordinary bot speech, for a view of the rest of the call."""
    candidates = []
    for region in ctx.regions(Speaker.BOT):
        clip = _Clip(*_fit(region.start_s, region.end_s, duration_s, cfg), "sample of bot speech")
        if not any(clip.overlaps(kept) for kept in taken):
            candidates.append(clip)
    if len(candidates) <= count:
        return candidates
    step = len(candidates) / count
    return [candidates[int(i * step)] for i in range(count)]


def pick_clips(ctx: EvalContext, duration_s: float, cfg: ListeningConfig) -> list[_Clip]:
    flagged = _flagged_clips(ctx, duration_s, cfg)[: max(0, cfg.max_clips - cfg.min_samples)]
    room = cfg.max_clips - len(flagged)
    return flagged + _sample_clips(ctx, flagged, room, cfg, duration_s)


def _expected_text(ctx: EvalContext, clip: _Clip) -> str:
    texts = [
        e.spoken_text or e.intended_text
        for e in ctx.artifact.timeline
        if isinstance(e, Utterance)
        and e.speaker == Speaker.BOT
        and e.start_s < clip.end_s
        and clip.start_s < e.end_s
    ]
    return " ".join(t for t in texts if t) or "not known"


def _clip_findings(clip: _Clip, payload: dict[str, Any]) -> list[Finding]:
    evidence = [Evidence(speaker=Speaker.BOT, start_s=clip.start_s, end_s=clip.end_s)]
    findings = []
    odd = [str(word) for word in payload.get("odd_pronunciations") or []]
    if odd:
        findings.append(
            Finding(
                evaluator="audio_listening",
                category="odd_pronunciation",
                severity="minor",
                advisory=True,
                title=f"Sounded wrong to the listening model: {', '.join(odd)}",
                evidence=evidence,
            )
        )
    for issue in payload.get("issues") or []:
        if not isinstance(issue, dict) or not issue.get("title"):
            continue
        severity = issue.get("severity")
        findings.append(
            Finding(
                evaluator="audio_listening",
                category="delivery",
                severity=severity if severity in _ALLOWED_SEVERITIES else "minor",
                advisory=True,
                title=str(issue["title"]),
                detail=str(issue.get("detail") or ""),
                evidence=evidence,
            )
        )
    return findings


class AudioListeningEvaluator:
    name = "audio_listening"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        analyst = ctx.deps.analyst
        if analyst is None:
            raise Skip(
                "no audio listener configured: none beat chance in benchmarks, and delivery "
                "measurements cover this"
            )
        out.model = analyst.name
        audio = ctx.audio(Speaker.BOT)
        clips = pick_clips(ctx, audio.duration_s, ctx.config.listening)
        if not clips:
            raise Skip("no bot speech to listen to")

        heard: list[dict[str, Any]] = []
        failures: list[str] = []
        disagreements: list[dict[str, Any]] = []
        for position, clip in enumerate(clips, start=1):
            await ctx.note_progress(f"clip {position} of {len(clips)}")
            prompt = render_prompt(
                "audio_listening", why=clip.why, expected=_expected_text(ctx, clip)
            )
            try:
                reply = await analyst.analyze(
                    audio.slice(clip.start_s, clip.end_s), prompt, json_schema=LISTENING_SCHEMA
                )
                payload = parse_json_reply(reply)
            except Exception as exc:  # one unusable clip leaves the others worth keeping
                failures.append(f"{clip.start_s:.1f} s: {exc}")
                continue
            findings = _clip_findings(clip, payload)
            disagreements += reconcile(clip.start_s, clip.end_s, payload, findings, ctx.delivery)
            out.findings.extend(findings)
            heard.append(
                {
                    "start_s": round(clip.start_s, 3),
                    "end_s": round(clip.end_s, 3),
                    "why": clip.why,
                    "tone": str(payload.get("tone") or ""),
                    "delivery": str(payload.get("delivery") or ""),
                }
            )
        out.metrics["clips"] = heard
        out.metrics["disagreements"] = disagreements
        if failures:
            out.metrics["failed_clips"] = failures
        if not heard:
            raise RuntimeError(f"analyst failed on every clip: {failures[0]}")
