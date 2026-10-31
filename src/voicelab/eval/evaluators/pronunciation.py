"""What the bot was meant to say against what an independent ASR heard.

One ASR disagreeing with the script is a suspicion, because ASRs make mistakes on their own. So a
critical mismatch is advisory unless a second engine confirms it: the live STT, recorded as
`heard_text`, has to have heard the same wrong token. Then it's corroborated and can fail a call.

The comparison runs over the whole call at once, not utterance by utterance. ASR timestamps drift,
and cutting the words into per-utterance windows would turn that drift into false mismatches on
the very tokens that fail a call.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from voicelab.core.models import (
    CallArtifact,
    CallMode,
    Evidence,
    Finding,
    Speaker,
    ToolCallEvent,
    Utterance,
    Word,
)
from voicelab.eval.config import PronunciationConfig
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.normalize import Aligned, Lexicon, Token, align, sounds_alike, tokenize
from voicelab.eval.rollup import roll_up
from voicelab.providers.base import Transcript

LIMITATION = (
    "This compares against an ASR, which can silently correct a mispronounced word to the one it "
    "expects. A clean result doesn't prove a word sounded right."
)
_CONFIDENCE_ONLY_MODES = (CallMode.HUMAN, CallMode.UPLOAD)
_MAX_LISTED_WORDS = 6
_SPELLING_KINDS = ("name", "lexicon", "url", "watch")  # numbers and ids can't sound alike
_FOLLOWING_WORDS = 2  # a term the ASR split can have the rest of it just after the aligned word


@dataclass
class _Spoken:
    """One bot utterance with a text to compare against."""

    utterance: Utterance
    words: list[str]
    tokens: list[Token] = field(default_factory=list)
    offset: int = 0  # where these tokens start in the whole call's reference


@dataclass(frozen=True)
class _Confirmation:
    """Whether a second engine backs up a mismatch, and what to tell the reviewer about it."""

    corroborated: bool
    note: str


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


def _customer_names(artifact: CallArtifact) -> set[str]:
    """Names from customer rows the tools returned. Plans have a `name` too, but no customer."""
    names: set[str] = set()
    for event in artifact.timeline:
        if isinstance(event, ToolCallEvent) and isinstance(event.result, dict):
            if "customer_id" in event.result and isinstance(event.result.get("name"), str):
                names.update(event.result["name"].split())
    return names


def _known_names(ctx: EvalContext) -> set[str]:
    """Words that are people in this call. Watch terms and the lexicon are critical separately."""
    names = _customer_names(ctx.artifact)
    persona = ctx.artifact.persona_snapshot
    if persona is not None:  # a persona's name can be a label: "Polite and overly chatty caller"
        names.update(w for w in persona.name.split() if w.istitle())
    if ctx.scenario is not None:
        for key, value in ctx.scenario.facts.items():
            if "name" in key.lower():
                names.update(w for w in value.split() if w.isalpha())
    return names


def _aligned_alone(
    spoken: _Spoken, hyp: list[Token], words: list[Word], slack_s: float
) -> list[Aligned]:
    """This utterance against only the words heard around it.

    Aligning the whole call copes with drifting timestamps, but phrases the bot repeats ("one
    moment, let me check that") can line up with the wrong utterance and leave this one with
    nothing. When that happens, its own stretch of the recording is the better comparison.
    """
    start, end = spoken.utterance.start_s - slack_s, spoken.utterance.end_s + slack_s
    nearby = [t for t in hyp if t.start < len(words) and start <= words[t.start].start_s <= end]
    return align(spoken.tokens, nearby)


def _better_of(whole: list[Aligned], alone: list[Aligned]) -> list[Aligned]:
    """Prefer the whole-call result unless the utterance's own stretch clearly matched more."""
    return alone if sum(a.matched for a in alone) > sum(a.matched for a in whole) else whole


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


def _canonical(words: list[str], lexicon: Lexicon) -> str:
    """One string for a token however it was written, so "$49.98" equals its spoken form."""
    return "".join(t.norm for t in tokenize(words, lexicon=lexicon))


def _live_words(group: _Group, lexicon: Lexicon) -> list[str] | None:
    """What the live STT heard where this token belongs. None if it left no transcript."""
    heard_text = group.spoken.utterance.heard_text
    if not heard_text or not heard_text.strip():
        return None
    words = heard_text.split()
    aligned = align(group.spoken.tokens, tokenize(words, lexicon=lexicon))
    start = group.position - group.spoken.offset
    items = aligned[start : start + len(group.items)]
    indices = sorted({i for item in items for h in item.hyp for i in range(h.start, h.end)})
    return [words[i] for i in indices]


def _same_sound(said: list[str], heard: list[str], lexicon: Lexicon) -> bool:
    return sounds_alike(_canonical(said, lexicon), _canonical(heard, lexicon))


def _spelled_differently(
    group: _Group, heard: list[Word], words: list[Word], lexicon: Lexicon
) -> bool:
    """Whether what looks like a mismatch is the same sound written another way."""
    if group.kind not in _SPELLING_KINDS or not heard:
        return False
    said = group.expected.split()
    after = words.index(heard[-1]) + 1
    longer = [[*heard, *words[after : after + extra]] for extra in range(_FOLLOWING_WORDS + 1)]
    # A name inside an address aligns with the whole address, so try its words one at a time.
    candidates = [*longer, *([word] for word in heard)]
    return any(_same_sound(said, [w.text for w in c], lexicon) for c in candidates)


def _same_engine(ctx: EvalContext, transcript: Transcript) -> bool:
    live = ctx.artifact.providers.get("stt_live")
    evaluated = transcript.model or (ctx.deps.stt.name if ctx.deps.stt else "")
    return bool(live and evaluated) and live.rsplit("/", 1)[-1] == evaluated.rsplit("/", 1)[-1]


def _confirm(
    group: _Group, heard: list[Word], ctx: EvalContext, transcript: Transcript, lexicon: Lexicon
) -> _Confirmation:
    live = _live_words(group, lexicon)
    if live is None:
        return _Confirmation(False, "ASR only: no second engine heard this utterance.")
    if _same_engine(ctx, transcript):
        note = "ASR only: the live and eval transcripts came from the same engine, so their "
        return _Confirmation(False, note + "agreement isn't independent.")
    live_text = _canonical(live, lexicon)
    expected_text = _canonical(group.expected.split(), lexicon)
    if live_text == expected_text or sounds_alike(expected_text, live_text):
        return _Confirmation(False, "ASR only: the live engine heard it as scripted.")
    eval_text = _canonical([w.text for w in heard], lexicon)
    if eval_text and (live_text == eval_text or sounds_alike(eval_text, live_text)):
        note = f'Corroborated: the live and eval engines both heard "{" ".join(live)}".'
        return _Confirmation(True, note)
    return _Confirmation(
        False, f'ASR only: the live engine heard "{" ".join(live) or "nothing"}" instead.'
    )


def _mismatch_finding(
    group: _Group, heard: str, span: tuple[float, float], confirmation: _Confirmation
) -> Finding:
    corroborated = confirmation.corroborated
    label = "corroborated" if corroborated else "asr_only"
    return Finding(
        evaluator="pronunciation",
        category=f"critical_token_{label}",
        severity="major",
        advisory=not corroborated,
        title=f"Critical {group.kind} differs from the script: {group.expected}",
        detail=(
            f'The bot was meant to say "{group.expected}". The independent ASR heard '
            f'"{heard or "nothing"}". {confirmation.note}'
        ),
        evidence=[Evidence(quote=heard, speaker=Speaker.BOT, start_s=span[0], end_s=span[1])],
    )


def _low_confidence_critical_finding(
    group: _Group, heard: list[Word], span: tuple[float, float]
) -> Finding:
    """The word matched, so nothing is wrong. It's a detail for anyone checking this clip."""
    lowest = min(w.confidence for w in heard if w.confidence is not None)
    quote = " ".join(w.text for w in heard)
    return Finding(
        evaluator="pronunciation",
        category="low_confidence_critical",
        severity="info",
        advisory=True,
        title=f"Critical {group.kind} heard with low confidence: {group.expected}",
        detail=f'"{quote}" matched at confidence {lowest:.2f}',
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


def _ordinary_title(group: list[Finding]) -> str:
    return f"Ordinary words differ from the script in {len(group)} utterances"


def _confidence_title(group: list[Finding]) -> str:
    return f"{len(group)} critical words matched at low confidence"


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
            self._compare(ctx, out, spoken, transcript, hyp, lexicon, cfg)
        else:
            out.metrics["compared"] = False
        if not spoken or ctx.artifact.mode in _CONFIDENCE_ONLY_MODES:
            self._confidence_pass(out, words, hyp, cfg)

    def _compare(
        self,
        ctx: EvalContext,
        out: EvaluatorOutput,
        spoken: list[_Spoken],
        transcript: Transcript,
        hyp: list[Token],
        lexicon: Lexicon,
        cfg: PronunciationConfig,
    ) -> None:
        words = transcript.words
        names = _known_names(ctx)
        watch = ctx.scenario.pronunciation_watch if ctx.scenario else []
        ref: list[Token] = []
        owners: list[_Spoken] = []
        for item in spoken:
            item.tokens = tokenize(item.words, lexicon=lexicon, known_names=names, watch=watch)
            item.offset = len(ref)
            ref.extend(item.tokens)
            owners.extend([item] * len(item.tokens))
        aligned = align(ref, hyp)

        kept = [True] * len(aligned)
        by_utterance: dict[int, list[int]] = {}
        for position, owner in enumerate(owners):
            by_utterance.setdefault(id(owner), []).append(position)
        ordinary: list[Finding] = []
        for item in spoken:
            positions = by_utterance.get(id(item), [])
            items = _better_of(
                [aligned[p] for p in positions],
                _aligned_alone(item, hyp, words, cfg.utterance_slack_s),
            )
            for position, better in zip(positions, items, strict=True):
                aligned[position] = better
            remaining = len(_trim_cut_off(items, item.utterance.interrupted))
            for p in positions[remaining:]:
                kept[p] = False
            finding = _ordinary_finding(item, items[:remaining], words, cfg)
            if finding is not None:
                ordinary.append(finding)

        groups = _groups(aligned, owners, kept)
        by_kind: dict[str, dict[str, int]] = {}
        corroborated = 0
        spelled = 0
        doubtful: list[Finding] = []
        for group in groups:
            tally = by_kind.setdefault(group.kind, {"total": 0, "mismatched": 0})
            tally["total"] += 1
            heard = _heard_words(group.items, words)
            span = _time_span(group.items, aligned, group.position, words, group.spoken.utterance)
            if group.mismatched and _spelled_differently(group, heard, words, lexicon):
                spelled += 1
                for item in group.items:
                    aligned[group.position + group.items.index(item)] = replace(item, matched=True)
            elif group.mismatched:
                tally["mismatched"] += 1
                text = " ".join(w.text for w in heard)
                confirmation = _confirm(group, heard, ctx, transcript, lexicon)
                corroborated += confirmation.corroborated
                out.findings.append(_mismatch_finding(group, text, span, confirmation))
            elif any(w.confidence is not None and w.confidence < cfg.low_confidence for w in heard):
                doubtful.append(_low_confidence_critical_finding(group, heard, span))
        out.findings.extend(roll_up(ordinary, key=lambda _: "all", title=_ordinary_title))
        out.findings.extend(roll_up(doubtful, key=lambda _: "all", title=_confidence_title))

        counted = [a for a, keep in zip(aligned, kept, strict=True) if keep]
        out.metrics.update(_comparison_metrics(counted, groups, by_kind, spelled))
        out.metrics["critical_corroborated"] = corroborated

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
    counted: list[Aligned],
    groups: list[_Group],
    by_kind: dict[str, dict[str, int]],
    spelled: int,
) -> dict[str, Any]:
    matched = sum(a.matched for a in counted)
    return {
        "compared": True,
        "tokens_compared": len(counted),
        "tokens_matched": matched,
        "token_accuracy": round(matched / len(counted), 4) if counted else None,
        "critical_tokens": len(groups),
        "critical_mismatches": sum(kind["mismatched"] for kind in by_kind.values()),
        "sound_alike": spelled,  # critical tokens the ASR spelled differently, same sound
        "critical_by_kind": by_kind,
    }
