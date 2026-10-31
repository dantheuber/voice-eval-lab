"""What the bot was meant to say against what an independent ASR heard.

The comparison runs over the whole call at once, not utterance by utterance. ASR timestamps drift,
and cutting the words into per-utterance windows would turn that drift into false mismatches on
the very tokens that fail a call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voicelab.core.models import CallMode, Evidence, Finding, Speaker, Utterance, Word
from voicelab.eval.config import PronunciationConfig
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.normalize import Aligned, Lexicon, Token, align, tokenize

LIMITATION = (
    "This compares against an ASR, which can silently correct a mispronounced word to the one it "
    "expects. A clean result doesn't prove a word sounded right."
)
_CONFIDENCE_ONLY_MODES = (CallMode.HUMAN, CallMode.UPLOAD)
_MAX_LISTED_WORDS = 6


@dataclass
class _Spoken:
    """One bot utterance with a text to compare against."""

    utterance: Utterance
    words: list[str]


@dataclass
class _Group:
    """The aligned tokens of one critical source word, such as all four digits of a price."""

    spoken: _Spoken
    kind: str
    token: Token
    items: list[Aligned]
    position: int  # index of the first item in the aligned list

    @property
    def expected(self) -> str:
        return " ".join(self.spoken.words[self.token.start : self.token.end])

    @property
    def mismatched(self) -> bool:
        return any(not item.matched for item in self.items)


def _reference_text(utterance: Utterance) -> str | None:
    if utterance.interrupted and utterance.spoken_text:
        return utterance.spoken_text
    return utterance.intended_text or utterance.spoken_text


def _spoken_utterances(ctx: EvalContext) -> list[_Spoken]:
    spoken = []
    for event in ctx.artifact.timeline:
        if isinstance(event, Utterance) and event.speaker == Speaker.BOT:
            text = _reference_text(event)
            if text and text.strip():
                spoken.append(_Spoken(event, text.split()))
    return spoken


def _known_names(ctx: EvalContext) -> set[str]:
    names: set[str] = set()
    persona = ctx.artifact.persona_snapshot
    if persona is not None:
        names.update(persona.name.split())
    if ctx.scenario is not None:
        for value in ctx.scenario.facts.values():
            names.update(w for w in value.split() if w.istitle() and w.isalpha())
    return names


def _trim_cut_off(tokens: list[Aligned], cut_off: bool) -> list[Aligned]:
    """Drop what an interrupted bot never got to say, since nobody can hear unspoken words."""
    if not cut_off:
        return tokens
    last = max((i for i, item in enumerate(tokens) if item.matched), default=-1)
    return tokens[: last + 1]


def _heard_words(items: list[Aligned], words: list[Word]) -> list[Word]:
    indices = sorted({i for item in items for h in item.hyp for i in range(h.start, h.end)})
    return [words[i] for i in indices]


def _time_span(
    group_items: list[Aligned], aligned: list[Aligned], position: int, words: list[Word],
    utterance: Utterance,
) -> tuple[float, float]:  # fmt: skip
    """Where a token was, or for a word nobody heard, the gap between its matched neighbors."""
    heard = _heard_words(group_items, words)
    if heard:
        return heard[0].start_s, heard[-1].end_s
    before = next(
        (words[a.hyp[-1].end - 1].end_s for a in reversed(aligned[:position]) if a.matched), None
    )
    after = next(
        (
            words[a.hyp[0].start].start_s
            for a in aligned[position + len(group_items) :]
            if a.matched
        ),
        None,
    )
    start = max(before, utterance.start_s) if before is not None else utterance.start_s
    end = min(after, utterance.end_s) if after is not None else utterance.end_s
    return start, max(start, end)


def _groups(aligned: list[Aligned], owners: list[_Spoken], kept: list[bool]) -> list[_Group]:
    groups: list[_Group] = []
    current: _Group | None = None
    for position, item in enumerate(aligned):
        kind = item.ref.critical
        if kind is None or not kept[position]:
            current = None
            continue
        owner = owners[position]
        if (
            current is not None
            and current.spoken is owner
            and current.token.start == item.ref.start
            and current.kind == kind
        ):
            current.items.append(item)
        else:
            current = _Group(owner, kind, item.ref, [item], position)
            groups.append(current)
    return groups


def _mismatch_finding(group: _Group, heard: str, span: tuple[float, float]) -> Finding:
    return Finding(
        evaluator="pronunciation",
        category="critical_token_mismatch",
        severity="major",
        title=f"Critical {group.kind} differs from the script: {group.expected}",
        detail=(
            f'The bot was meant to say "{group.expected}". The independent ASR heard '
            f'"{heard or "nothing"}".'
        ),
        evidence=[Evidence(quote=heard, speaker=Speaker.BOT, start_s=span[0], end_s=span[1])],
    )


def _low_confidence_critical_finding(
    group: _Group, heard: list[Word], span: tuple[float, float]
) -> Finding:
    lowest = min(w.confidence for w in heard if w.confidence is not None)
    quote = " ".join(w.text for w in heard)
    return Finding(
        evaluator="pronunciation",
        category="low_confidence_critical",
        severity="minor",
        advisory=True,
        title=f"Critical {group.kind} heard with low confidence: {group.expected}",
        detail=f'The ASR matched "{quote}", but its confidence was only {lowest:.2f}.',
        evidence=[Evidence(quote=quote, speaker=Speaker.BOT, start_s=span[0], end_s=span[1])],
    )


def _ordinary_finding(
    spoken: _Spoken, items: list[Aligned], words: list[Word], cfg: PronunciationConfig
) -> Finding | None:
    missed = [item for item in items if not item.matched and item.ref.critical is None]
    if not items or not missed:
        return None
    rate = len(missed) / len(items)
    pairs = [
        f'"{spoken.words[a.ref.start]}" heard as "{_heard_text(a, words)}"'
        for a in missed[:_MAX_LISTED_WORDS]
    ]
    utterance = spoken.utterance
    return Finding(
        evaluator="pronunciation",
        category="word_mismatch",
        severity="minor" if rate >= cfg.minor_mismatch_rate else "info",
        advisory=True,
        title=f"{len(missed)} of {len(items)} words differ from the script",
        detail="; ".join(pairs),
        evidence=[Evidence(speaker=Speaker.BOT, start_s=utterance.start_s, end_s=utterance.end_s)],
    )


def _heard_text(item: Aligned, words: list[Word]) -> str:
    return " ".join(w.text for w in _heard_words([item], words)) or "nothing"


def _confidence_findings(
    words: list[Word], tokens: list[Token], cfg: PronunciationConfig
) -> list[Finding]:
    """Flag doubtful stretches. This needs no script, so it works on any recording."""
    critical = {i for t in tokens if t.critical for i in range(t.start, t.end)}
    spans: list[list[int]] = []
    for i, word in enumerate(words):
        if word.confidence is None or word.confidence >= cfg.low_confidence:
            continue
        if spans and spans[-1][-1] == i - 1:
            spans[-1].append(i)
        else:
            spans.append([i])
    spans.sort(
        key=lambda span: (
            not any(i in critical for i in span),
            min(words[i].confidence for i in span),
        )
    )
    findings = []
    for span in spans[: cfg.max_low_confidence_findings]:
        is_critical = any(i in critical for i in span)
        quote = " ".join(words[i].text for i in span)
        lowest = min(words[i].confidence for i in span)
        findings.append(
            Finding(
                evaluator="pronunciation",
                category="low_confidence",
                severity="minor" if is_critical else "info",
                advisory=True,
                title=f"Low ASR confidence on: {quote}",
                detail=f"Lowest word confidence {lowest:.2f}.",
                evidence=[
                    Evidence(
                        quote=quote,
                        speaker=Speaker.BOT,
                        start_s=words[span[0]].start_s,
                        end_s=words[span[-1]].end_s,
                    )
                ],
            )
        )
    return findings


class PronunciationEvaluator:
    name = "pronunciation"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        transcript = ctx.transcripts.get(Speaker.BOT)
        if transcript is None:
            raise Skip("no independent transcript of the bot channel")
        cfg = ctx.config.pronunciation
        out.metrics["limitation"] = LIMITATION
        lexicon = Lexicon.load(ctx.deps.lexicon_path)
        words = transcript.words
        hyp = tokenize([w.text for w in words], lexicon=lexicon)
        spoken = _spoken_utterances(ctx)

        if spoken:
            self._compare(ctx, out, spoken, words, hyp, lexicon, cfg)
        else:
            out.metrics["compared"] = False
        if not spoken or ctx.artifact.mode in _CONFIDENCE_ONLY_MODES:
            self._confidence_pass(out, words, hyp, cfg)

    def _compare(
        self,
        ctx: EvalContext,
        out: EvaluatorOutput,
        spoken: list[_Spoken],
        words: list[Word],
        hyp: list[Token],
        lexicon: Lexicon,
        cfg: PronunciationConfig,
    ) -> None:
        names = _known_names(ctx)
        watch = ctx.scenario.pronunciation_watch if ctx.scenario else []
        ref: list[Token] = []
        owners: list[_Spoken] = []
        for item in spoken:
            tokens = tokenize(item.words, lexicon=lexicon, known_names=names, watch=watch)
            ref.extend(tokens)
            owners.extend([item] * len(tokens))
        aligned = align(ref, hyp)

        kept = [True] * len(aligned)
        by_utterance: dict[int, list[int]] = {}
        for position, owner in enumerate(owners):
            by_utterance.setdefault(id(owner), []).append(position)
        ordinary_total = 0
        for item in spoken:
            positions = by_utterance.get(id(item), [])
            items = [aligned[p] for p in positions]
            remaining = len(_trim_cut_off(items, item.utterance.interrupted))
            for p in positions[remaining:]:
                kept[p] = False
            items = items[:remaining]
            ordinary_total += sum(a.ref.critical is None for a in items)
            finding = _ordinary_finding(item, items, words, cfg)
            if finding is not None:
                out.findings.append(finding)

        groups = _groups(aligned, owners, kept)
        by_kind: dict[str, dict[str, int]] = {}
        for group in groups:
            tally = by_kind.setdefault(group.kind, {"total": 0, "mismatched": 0})
            tally["total"] += 1
            heard = _heard_words(group.items, words)
            span = _time_span(group.items, aligned, group.position, words, group.spoken.utterance)
            if group.mismatched:
                tally["mismatched"] += 1
                text = " ".join(w.text for w in heard)
                out.findings.append(_mismatch_finding(group, text, span))
            elif any(w.confidence is not None and w.confidence < cfg.low_confidence for w in heard):
                out.findings.append(_low_confidence_critical_finding(group, heard, span))

        counted = [a for a, keep in zip(aligned, kept, strict=True) if keep]
        out.metrics.update(_comparison_metrics(counted, groups, by_kind))

    def _confidence_pass(
        self, out: EvaluatorOutput, words: list[Word], hyp: list[Token], cfg: PronunciationConfig
    ) -> None:
        if words and all(w.confidence is None for w in words):
            out.metrics["low_confidence_pass"] = "skipped: the ASR gave no word confidence"
            return
        findings = _confidence_findings(words, hyp, cfg)
        out.metrics["low_confidence_flags"] = len(findings)
        out.findings.extend(findings)


def _comparison_metrics(
    counted: list[Aligned], groups: list[_Group], by_kind: dict[str, dict[str, int]]
) -> dict[str, Any]:
    matched = sum(a.matched for a in counted)
    return {
        "compared": True,
        "tokens_compared": len(counted),
        "tokens_matched": matched,
        "token_accuracy": round(matched / len(counted), 4) if counted else None,
        "critical_tokens": len(groups),
        "critical_mismatches": sum(g.mismatched for g in groups),
        "critical_by_kind": by_kind,
    }
