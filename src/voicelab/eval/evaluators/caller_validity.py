"""Whether the caller bot did its job. Failure here is a broken test, not a bot failure."""

from __future__ import annotations

import re
from typing import Any

from voicelab.agents.speech import normalize
from voicelab.core.models import (
    Beat,
    CallMode,
    CriterionResult,
    Evidence,
    Finding,
    Speaker,
    Utterance,
)
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.judging import (
    JudgeItem,
    JudgeStatus,
    Judgment,
    ask_items,
    ask_json,
    call_record,
    item_lines,
    render_prompt,
    verify_judgment,
)
from voicelab.eval.normalize import Lexicon, tokenize
from voicelab.eval.rollup import roll_up
from voicelab.eval.transcript import TranscriptIndex, estimate_words, render_transcript
from voicelab.eval.vad import Region
from voicelab.eval.verdict import FACTS_CRITERION_ID, MATERIAL_FACT_BREACH
from voicelab.providers.base import Transcript

FACTS_TEXT = "Caller stayed within its fact sheet"
_BOT_TO_BOT = (CallMode.LIVE_BOTS, CallMode.TURN_BASED)


def facts_schema(facts: dict[str, str]) -> dict[str, Any]:
    """A reply about this sheet. A breach names the line it contradicts, if it contradicts one."""
    breach = {
        "type": "object",
        "properties": {
            "quote": {"type": "string", "maxLength": 200},
            "contradicts": {"type": "string", "enum": [*facts, _NO_SHEET_LINE]},
            "why": {"type": "string", "maxLength": 200},
            "material": {"type": "boolean"},
        },
        "required": ["quote", "contradicts", "why", "material"],
    }
    return {
        "type": "object",
        "properties": {
            "breaches": {"type": "array", "items": breach, "maxItems": 5},
            "reason": {"type": "string", "maxLength": 240},
            "status": {"type": "string", "enum": ["pass", "fail", "unknown"]},
        },
        "required": ["breaches", "reason", "status"],
    }


_NO_SHEET_LINE = "none"
_FACTS_REPLY_ITEMS = 3  # the fact check's reply can run to a few breaches
_ONE_LINE_S = 3600.0  # a script line stays on one line, however long the speaker paused
_DETAIL_KINDS = ("number", "money", "id", "url")
_PUNCTUATION = "\"'`.,!?;:()[]{}<>"
_NAME_INTRODUCTIONS = ("is", "am", "i'm", "it's", "called", "named")

# A checkpoint is what the caller did, so the caller's words prove it. These say what happened to
# the caller instead, and there the bot's own words are the proof.
_RECEIVED_CHECKPOINT = re.compile(r"\b(?:received|was told|were told|was given)\b", re.IGNORECASE)


def checkpoint_actor(beat: Beat) -> Speaker | None:
    """Whose words can prove this beat. None means either side's."""
    text = beat.checkpoint or beat.instruction
    return None if _RECEIVED_CHECKPOINT.search(text) else Speaker.CALLER


class CallerValidityEvaluator:
    name = "caller_validity"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        if ctx.artifact.mode not in _BOT_TO_BOT:
            raise Skip(f"only bot-to-bot calls have a scripted caller, this is {ctx.artifact.mode}")
        scenario = ctx.scenario
        if scenario is None:
            raise Skip("no scenario, so the caller had no script to follow")
        judge = ctx.deps.judge
        missing = None if judge else "no judge LLM configured"
        if not any(t.words for t in ctx.transcripts.values()):
            missing = missing or "no independent transcript"
        if missing:
            out.criteria.extend(
                CriterionResult(
                    criterion_id=b.id, kind="beat", text=b.instruction, status="unknown",
                    reason=missing,
                )
                for b in scenario.beats
            )  # fmt: skip
            raise Skip(missing)
        out.model = judge.name

        context = f"{scenario.title}. The caller wants: {scenario.goal}"
        record = call_record(context, ctx.artifact, ctx.transcripts)

        def prompt_for(batch: list[JudgeItem]) -> str:
            return render_prompt(
                "caller_beats", record=record, goal=scenario.goal, steps=item_lines(batch)
            )

        asked = [JudgeItem(b.id, _beat_line(b)) for b in scenario.beats]
        replies = await ask_items(judge, asked, prompt_for, ctx.config, note=ctx.note_progress)
        out.model = replies.model or out.model
        reached: list[str] = []
        missed: list[str] = []
        undecided: list[str] = []
        missed_findings: list[Finding] = []
        for beat in scenario.beats:
            payload = replies.payloads.get(beat.id)
            if payload is None:
                undecided.append(beat.id)
                out.criteria.append(
                    CriterionResult(
                        criterion_id=beat.id, kind="beat", text=beat.instruction, status="unknown",
                        reason=f"judge failed: {replies.errors.get(beat.id, 'no answer')}",
                    )
                )  # fmt: skip
                continue
            judgment = verify_judgment(
                payload, ctx.index, quote_needed_for="pass", actor=checkpoint_actor(beat)
            )
            out.criteria.append(
                CriterionResult(
                    criterion_id=beat.id,
                    kind="beat",
                    text=beat.instruction,
                    status=judgment.status,
                    reason=judgment.reason,
                    evidence=judgment.evidence,
                )
            )
            {"pass": reached, "fail": missed, "unknown": undecided}[judgment.status].append(beat.id)
            if judgment.status == "fail":
                checkpoint = beat.checkpoint or beat.instruction
                missed_findings.append(
                    Finding(
                        evaluator=self.name,
                        category="beat_missed",
                        severity="major" if beat.required else "minor",
                        # Verdict logic reads this as invalid. It says nothing about the bot.
                        advisory=True,
                        title=f"Caller missed checkpoint: {checkpoint}",
                        detail=f"{checkpoint} {judgment.reason}",
                        evidence=judgment.evidence,
                        criterion_id=beat.id,
                    )
                )
        out.findings.extend(
            roll_up(
                missed_findings,
                key=lambda f: f.category,
                title=lambda group: f"Caller missed {len(group)} checkpoints",
            )
        )
        if scenario.facts:
            await self._check_facts(ctx, out)
        out.metrics.update({"reached": reached, "missed": missed, "unknown": undecided})
        if scenario.beats and not replies.payloads:
            raise RuntimeError(f"judge failed on all {len(scenario.beats)} beats")

    async def _check_facts(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        """Whether the caller kept to its sheet, read from the words it was given to say.

        This call came from a script, so the transcript's only contribution here would be its
        mistakes: a mishearing of "Nguyen" reads as a made-up name, and a made-up name makes the
        run invalid. With no script, as on an older recording, the transcript has to do.
        """
        lines = _script_lines(ctx.artifact.timeline)
        said = {Speaker.CALLER: lines} if lines else ctx.transcripts
        index = TranscriptIndex(said) if lines else ctx.index
        prompt = render_prompt(
            "caller_facts",
            facts=_sheet(ctx.scenario.facts),
            transcript=render_transcript(said, gap_s=_ONE_LINE_S),
        )
        try:
            payload, out.model = await ask_json(
                ctx.deps.judge,
                prompt,
                facts_schema(ctx.scenario.facts),
                ctx.config,
                items=_FACTS_REPLY_ITEMS,
            )
        except Exception as exc:
            out.criteria.append(_facts_criterion("unknown", f"judge failed: {exc}", []))
            return
        listed = [b for b in payload.get("breaches") or [] if isinstance(b, dict)]
        bot = ctx.transcripts.get(Speaker.BOT)

        def worth_keeping(breach: dict[str, Any]) -> bool:
            # A quote that isn't the caller's stays, so verification can throw the reply out.
            quote = breach.get("quote")
            found = index.locate(quote, Speaker.CALLER) if isinstance(quote, str) else None
            if found is None or found.speaker != Speaker.CALLER:
                return True
            return _is_a_breach(breach, quote, found, ctx.scenario.facts, bot)

        breaches = [b for b in listed if worth_keeping(b)]
        cited = {**payload, "quotes": [b.get("quote") for b in breaches]}
        if listed and not breaches:
            cited = {
                **cited,
                "status": "pass",
                "reason": "Nothing the judge listed is a new detail.",
            }
        judgment = verify_judgment(cited, index, quote_needed_for="fail", actor=Speaker.CALLER)
        status, reason = judgment.status, judgment.reason
        problem = _breach_problem(judgment, breaches)
        if problem:
            status, reason = "unknown", f"{problem} {reason}"
        out.criteria.append(_facts_criterion(status, reason, judgment.evidence))
        if status == "fail":
            pairs = zip(breaches, judgment.evidence, strict=True)
            out.findings.extend(_breach_finding(b, e, reason) for b, e in pairs)


def _script_lines(timeline: list[Any]) -> Transcript | None:
    """What the caller bot said, as a transcript. Words it never got to play don't count.

    A cut-off line keeps only the part that played, and with no record of that part it's left out.
    """
    spoken = [
        event
        for event in timeline
        if isinstance(event, Utterance) and event.speaker == Speaker.CALLER and _said(event)
    ]
    words = [
        word
        for utterance in spoken
        for word in estimate_words(
            _said(utterance), [Region(utterance.start_s, utterance.end_s)], utterance.end_s
        )
    ]
    return Transcript(text=" ".join(w.text for w in words), words=words) if words else None


def _said(utterance: Utterance) -> str | None:
    return utterance.spoken_text if utterance.interrupted else utterance.intended_text


def _sheet(facts: dict[str, str]) -> str:
    """The sheet with each value's spoken form, because that's what a script line will contain."""
    lines = []
    for key, value in facts.items():
        spoken = normalize(value).spoken
        lines.append(f"- {key}: {value}" + (f" (said aloud: {spoken})" if spoken != value else ""))
    return "\n".join(lines)


def _is_a_breach(
    breach: dict[str, Any],
    quote: str,
    found: Evidence,
    facts: dict[str, str],
    bot: Transcript | None,
) -> bool:
    """Whether a claimed breach is a statement the caller had no right to make.

    A question states nothing. A statement that contradicts a line of the sheet is a breach whether
    or not it names a number, as "my card is valid" against an expired card is, unless it holds
    that line's own value. Otherwise it has to name a concrete detail the caller couldn't have,
    which means off the sheet and not something the bot had already told it.
    """
    if quote.rstrip().endswith("?"):
        return False
    line = breach.get("contradicts")
    if line in facts:
        return _spelled(facts[line]) not in _spelled(quote)
    return _names_something_new(quote, _known_details(facts, bot, found.start_s))


def _spelled(text: str) -> str:
    return "".join(t.norm for t in tokenize(text.split(), lexicon=Lexicon.load()))


def _known_details(facts: dict[str, str], bot: Transcript | None, before_s: float | None) -> str:
    """What the caller may say at this moment: its sheet, and what the bot had told it by then."""
    sheet = [*facts.values(), *(normalize(value).spoken for value in facts.values())]
    told = [
        word.text
        for word in (bot.words if bot else [])
        if before_s is not None and word.start_s < before_s
    ]
    return _spelled(" ".join([*sheet, *told]))


def _names_something_new(quote: Any, known: str) -> bool:
    """Whether a claimed breach states a concrete detail the caller had no way to have.

    A breach has to be a number, an id, an address, or a name. A judge left to itself calls a
    question about the price or a dog a breach, and a breach voids the test.
    """
    if not isinstance(quote, str):
        return False
    lexicon = Lexicon.load()
    words = quote.split()
    details: dict[tuple[int, int], list[str]] = {}
    for token in tokenize(words, lexicon=lexicon):
        if token.critical in _DETAIL_KINDS:
            details.setdefault((token.start, token.end), []).append(token.norm)
    if any("".join(detail) not in known for detail in details.values()):
        return True
    return any(
        _is_new_name(word, words[position - 1] if position else "", known, lexicon)
        for position, word in enumerate(words)
    )


def _is_new_name(word: str, before: str, known: str, lexicon: Lexicon) -> bool:
    """A name the sheet doesn't have, given as who someone is. A chatty caller's dog has one too."""
    bare = word.strip(_PUNCTUATION)
    capitalized = len(bare) > 1 and bare[0].isupper() and bare[1:].islower() and bare.isalpha()
    if not capitalized or before.lower().strip(_PUNCTUATION) not in _NAME_INTRODUCTIONS:
        return False
    return "".join(t.norm for t in tokenize([bare], lexicon=lexicon)) not in known


def _beat_line(beat: Beat) -> str:
    return f"{beat.instruction} Proof it was reached: {beat.checkpoint or beat.instruction}"


def _breach_problem(judgment: Judgment, breaches: list[dict[str, Any]]) -> str | None:
    """Why a reply can't be taken at face value. Every breach has to stand up on its own."""
    if judgment.status == "pass" and breaches:
        return "The judge passed the caller but listed breaches."
    if judgment.status != "fail":
        return None
    if len(judgment.evidence) != len(breaches):
        return "A breach came without a usable quote."
    if any(e.speaker != Speaker.CALLER for e in judgment.evidence):
        return "The cited words were not the caller's."
    if not all(isinstance(b.get("material"), bool) for b in breaches):
        return "The judge didn't say whether every breach was material."
    return None


def _breach_finding(breach: dict[str, Any], evidence: Evidence, reason: str) -> Finding:
    material = breach["material"]
    label = "material detail" if material else "minor detail"
    return Finding(
        evaluator="caller_validity",
        category=MATERIAL_FACT_BREACH if material else "fact_sheet_slip",
        severity="major" if material else "minor",
        # Verdict logic reads these. They describe the test, so they never count against the bot.
        advisory=True,
        title=f"Caller went outside its fact sheet on a {label}: {evidence.quote}",
        detail=str(breach.get("why") or reason),
        evidence=[evidence],
        criterion_id=FACTS_CRITERION_ID,
    )


def _facts_criterion(status: JudgeStatus, reason: str, evidence: list[Evidence]) -> CriterionResult:
    return CriterionResult(
        criterion_id=FACTS_CRITERION_ID,
        kind="beat",
        text=FACTS_TEXT,
        status=status,
        reason=reason,
        evidence=evidence,
    )
