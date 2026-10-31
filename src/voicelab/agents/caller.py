"""The caller bot: a scripted customer that talks like a person, plus a text-only call runner.

A free-running LLM caller is too cooperative and invents details, so this one is held to the
scenario. It may state only what is on the fact sheet, it works through ordered beats that this
code tracks, and the LLM only chooses the words. Each turn is a structured reply saying what to
say, whether the current beat is done, and whether to hang up.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from voicelab.agents.speech import SpeechText
from voicelab.agents.support import SupportBot
from voicelab.company.db import diff_against_seed
from voicelab.core.models import (
    BargeInEvent,
    Beat,
    DbChange,
    Persona,
    Scenario,
    Speaker,
    SystemEvent,
    TimelineEvent,
    Utterance,
)
from voicelab.providers.base import LLM, ChatMessage

MAX_TURNS_REASON = "max_turns"
MAX_TURNS_LINE = "Look, I have to go. I'll call back later."
REDIRECT_LINE = "Sorry, um, can we get back to why I called?"
OPTIONAL_BEAT_TURNS = 2  # an optional beat that goes nowhere for this long gets dropped
TRANSCRIPT_LINES = 20  # keeps the prompt small enough for a little local model

_PATIENCE = {
    1: "You have no patience and it shows fast.",
    2: "You get annoyed quickly.",
    3: "You are reasonably patient.",
    4: "You are patient.",
    5: "You are very patient and rarely get upset.",
}
_VERBOSITY = {
    1: "You use as few words as you can.",
    2: "You keep it short.",
    3: "You talk normally.",
    4: "You tend to over-explain.",
    5: "You ramble and make small talk, but small talk never adds account facts.",
}
_TECH = {
    1: 'You don\'t know technical words. You say things like "the email thing".',
    2: "You only know basic computer words.",
    3: "You are fine with everyday tech words.",
    4: "You know your way around hosting and email settings.",
    5: "You use exact technical terms like MX record, TTL, and IMAP.",
}

_ASSISTANT_TALK = re.compile(
    r"how (?:can|may) i (?:help|assist)|anything else i can|happy to help|let me know if you"
    r"|i can help you with|as an ai",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_DIGITS = re.compile(r"\d[\d\s().-]*\d")
_SEPARATORS = re.compile(r"[\s().-]+")


class _Reply(BaseModel):
    """The JSON the LLM returns each turn."""

    say: str = ""
    completed_beat: str | None = None
    hang_up: bool = False
    hang_up_reason: str = ""


@dataclass(frozen=True)
class CallerLine:
    """One caller turn. `text` may be empty when the caller just hangs up."""

    text: str
    completed_beat: str | None = None
    hang_up: bool = False
    hang_up_reason: str = ""


@dataclass(frozen=True)
class Interruption:
    """Whether to cut in on a bot turn, and how far through its speech (0 to 1)."""

    interrupt: bool
    after_fraction: float


def _parse_reply(text: str) -> _Reply | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        reply = _Reply.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None
    return reply if reply.say.strip() or reply.hang_up else None


def _squash(text: str) -> str:
    return _SEPARATORS.sub("", text.lower())


class CallerBot:
    """Produces the caller's next line from what it heard. It only knows what it heard."""

    def __init__(
        self,
        llm: LLM,
        persona: Persona,
        scenario: Scenario,
        *,
        seed: int = 0,
        temperature: float = 0.7,
        max_tokens: int = 160,
    ) -> None:
        self.llm = llm
        self.persona = persona
        self.scenario = scenario
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.transcript: list[tuple[str, str]] = []
        self.turns = 0
        self.completed_beats: list[str] = []
        self.hang_up_reason = ""
        self.hung_up = False
        self._beat_index = 0
        self._stalled_turns = 0  # consecutive turns where no beat got done

    @property
    def current_beat(self) -> Beat | None:
        beats = self.scenario.beats
        return beats[self._beat_index] if self._beat_index < len(beats) else None

    @property
    def finished_beats(self) -> bool:
        return bool(self.scenario.beats) and self.current_beat is None

    @property
    def patience_limit(self) -> int:
        """Stalled turns before the caller gives up."""
        return self.persona.patience * 2 + 3

    def decide_interruption(self, bot_turn: int) -> Interruption:
        """Decide whether to cut in on a bot turn. The same seed and turn always agree."""
        rng = random.Random(f"{self.seed}:{self.scenario.id}:{bot_turn}")
        return Interruption(
            interrupt=rng.random() < self.persona.interruption_rate,
            after_fraction=rng.uniform(0.2, 0.7),
        )

    async def next_line(self, heard: str) -> CallerLine:
        """Produce the caller's next line. `max_turns` counts these lines, not both sides."""
        if self.hung_up:
            return CallerLine("", hang_up=True, hang_up_reason=self.hang_up_reason)
        if heard.strip():
            self.transcript.append(("Agent", heard.strip()))
        if self.turns >= self.scenario.max_turns:
            return self._conclude(CallerLine(MAX_TURNS_LINE), MAX_TURNS_REASON)
        self.turns += 1
        was_finished = self.finished_beats
        reply = await self._ask()
        line = CallerLine(reply.say.strip(), completed_beat=self._track_beat(reply))
        if reply.hang_up:
            return self._conclude(line, reply.hang_up_reason or "caller decided to hang up")
        if was_finished:
            return self._conclude(line, "finished")
        if self._stalled_turns >= self.patience_limit:
            return self._conclude(line, "out of patience")
        self.transcript.append(("You", line.text))
        return line

    def _conclude(self, line: CallerLine, reason: str) -> CallerLine:
        self.hung_up = True
        self.hang_up_reason = reason
        if line.text:
            self.transcript.append(("You", line.text))
        return CallerLine(line.text, line.completed_beat, True, reason)

    def _track_beat(self, reply: _Reply) -> str | None:
        """Advance on the current beat only, so a confused model can't skip ahead."""
        beat = self.current_beat
        if beat and reply.completed_beat == beat.id:
            self.completed_beats.append(beat.id)
            self._beat_index += 1
            self._stalled_turns = 0
            return beat.id
        self._stalled_turns += 1
        if beat and not beat.required and self._stalled_turns >= OPTIONAL_BEAT_TURNS:
            self._beat_index += 1
            self._stalled_turns = 0
        return None

    async def _ask(self) -> _Reply:
        correction = ""
        for _ in range(2):
            response = await self.llm.chat(
                self._messages(correction),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                json_schema=self._schema(),
            )
            reply = _parse_reply(response.text)
            if reply is None:
                correction = "Your last reply was not valid JSON with a non-empty say field."
                continue
            problem = self._problem_with(reply.say)
            if problem is None:
                return reply
            correction = problem
        return _Reply(say=REDIRECT_LINE)

    def _problem_with(self, say: str) -> str | None:
        """Catch the two failures that ruin a test run: acting like an agent, inventing facts."""
        if _ASSISTANT_TALK.search(say):
            return "You sounded like a support agent. You are the customer, so don't offer help."
        invented = self._invented_details(say)
        if invented:
            return (
                f"You said {', '.join(invented)}, which is not on your fact sheet. "
                "Say you don't have that instead."
            )
        return None

    def _invented_details(self, say: str) -> list[str]:
        known = _squash(
            " ".join([*self.scenario.facts.values(), *(line for _, line in self.transcript)])
        )
        invented = [email for email in _EMAIL.findall(say) if _squash(email) not in known]
        for token in _DIGITS.findall(_EMAIL.sub(" ", say)):
            if _squash(token) in known:
                continue
            parts = [part for part in _SEPARATORS.split(token) if part]
            # Spaced digits are one number, while "4821 97214" is two numbers side by side.
            numbers = [_squash(token)] if all(len(p) == 1 for p in parts) else parts
            if any(len(number) >= 3 and number not in known for number in numbers):
                invented.append(token.strip())
        return invented

    def _schema(self) -> dict[str, Any]:
        beat = self.current_beat
        return {
            "type": "object",
            "properties": {
                "say": {"type": "string"},
                "completed_beat": {"enum": [beat.id, None] if beat else [None]},
                "hang_up": {"type": "boolean"},
                "hang_up_reason": {"type": "string"},
            },
            "required": ["say", "completed_beat", "hang_up", "hang_up_reason"],
        }

    def _messages(self, correction: str) -> list[ChatMessage]:
        lines = [f"{who}: {text}" for who, text in self.transcript[-TRANSCRIPT_LINES:]]
        transcript = "\n".join(lines) or "(The call just connected.)"
        request = f"The call so far:\n{transcript}\n\nReply with JSON only."
        if correction:
            request += f"\n\nYour last reply was rejected. {correction}"
        return [
            ChatMessage(role="system", content=self.system_prompt()),
            ChatMessage(role="user", content=request),
        ]

    def system_prompt(self) -> str:
        persona = self.persona
        identity = [
            f"You are {persona.age} years old." if persona.age else "",
            f"Your mood: {persona.temperament}.",
            _PATIENCE[persona.patience],
            _VERBOSITY[persona.verbosity],
            _TECH[persona.tech_savviness],
            persona.speaking_style,
        ]
        facts = "\n".join(f"- {key}: {value}" for key, value in self.scenario.facts.items())
        return f"""\
You are a customer on the phone with Fernhill Internet support. You are not an assistant.

Who you are:
{" ".join(part for part in identity if part)}

Why you called: {self.scenario.goal}

Your fact sheet. These are the only facts you know:
{facts or "- (nothing)"}
If you are asked for anything not on this sheet, say you don't know it or don't have it.
Never guess or make up a detail. Give a fact only when asked, or when your step calls for it.

{self._step_text()}

Rules:
- You are the customer. Never help the agent, summarize for it, or offer to do its job.
- If the agent goes off topic, don't follow. Bring it back to your current step.
- Talk like a real person on a phone. Fragments and fillers are fine. No lists.
- Usually one or two sentences.

Reply as JSON with these fields:
- say: your next words
- completed_beat: the current step id if this turn finishes that step, otherwise null
- hang_up: true only if you are done or giving up
- hang_up_reason: why you are hanging up, or an empty string"""

    def _step_text(self) -> str:
        beat = self.current_beat
        if beat is None:
            if self.finished_beats:
                return "You got through everything. Say a short goodbye and hang up."
            return "Your step: explain why you called and get it sorted out."
        step = f"Your current step (id {beat.id}): {beat.instruction}"
        if self._stalled_turns + 1 >= self.patience_limit:
            return f"{step}\nYou have had enough. Say so and hang up."
        if self._stalled_turns > self.persona.patience:
            return f"{step}\nThis is going nowhere and you are getting annoyed. Let it show."
        return step


# Text-only calls -------------------------------------------------------------------------------


@dataclass
class TextCallResult:
    """A finished text call, in the same event shapes a voice call produces."""

    timeline: list[TimelineEvent]
    end_reason: str  # caller_hung_up, bot_ended, or max_turns
    detail: str
    completed_beats: list[str]
    db_changes: list[DbChange]


@dataclass
class _TextLine:
    """Stands in for the phone line. Time is synthetic and comes from word counts."""

    seconds_per_word: float
    now: float = 0.0
    timeline: list[TimelineEvent] = field(default_factory=list)

    def system(self, kind: str, detail: str = "") -> None:
        self.timeline.append(SystemEvent(at_s=self.now, kind=kind, detail=detail))

    def caller_speaks(self, text: str) -> None:
        start = self.now
        self.now += len(text.split()) * self.seconds_per_word
        self.timeline.append(
            Utterance(
                speaker=Speaker.CALLER,
                start_s=start,
                end_s=self.now,
                intended_text=text,
                spoken_text=text,
                heard_text=text,
            )
        )

    def bot_speaks(self, segments: list[SpeechText], cut_at: float | None) -> str:
        """Play the bot's turn and return what reached the caller."""
        intended = " ".join(segment.spoken for segment in segments)
        words = intended.split()
        spoken = intended
        if cut_at is not None:
            spoken = " ".join(words[: max(1, int(len(words) * cut_at))])
        start = self.now
        self.now += len(spoken.split()) * self.seconds_per_word
        utterance = Utterance(
            speaker=Speaker.BOT,
            start_s=start,
            end_s=self.now,
            intended_text=intended,
            spoken_text=spoken,
            heard_text=spoken,
            interrupted=cut_at is not None,
        )
        self.timeline.append(utterance)
        if cut_at is not None:
            self.timeline.append(
                BargeInEvent(
                    at_s=self.now,
                    bot_stopped_at_s=self.now,
                    interrupted_utterance_id=utterance.id,
                )
            )
        return spoken


async def run_text_call(
    support: SupportBot, caller: CallerBot, *, seconds_per_word: float = 0.4
) -> TextCallResult:
    """Run both bots with text in place of audio until someone ends the call.

    The caller hears the normalized spoken form, the closest text gets to what crosses a real line.
    Barge-ins follow the caller's interruption decisions, and timestamps are synthetic.
    """
    line = _TextLine(seconds_per_word)
    support.clock = lambda: line.now
    support.on_event = line.timeline.append
    line.system("call_started")

    def bot_turn(segments: list[SpeechText], number: int) -> str:
        decision = caller.decide_interruption(number)
        cut_at = decision.after_fraction if decision.interrupt and not support.ended else None
        heard = line.bot_speaks(segments, cut_at)
        if cut_at is not None:
            support.on_interrupted(heard)
        return heard

    turn = 0
    heard = bot_turn([support.greet()], turn)
    while True:
        said = await caller.next_line(heard)
        if said.text:
            line.caller_speaks(said.text)
        if said.completed_beat:
            line.system("beat_completed", said.completed_beat)
        if said.hang_up:
            at_limit = said.hang_up_reason == MAX_TURNS_REASON
            end_reason = MAX_TURNS_REASON if at_limit else "caller_hung_up"
            detail = said.hang_up_reason
            break
        turn += 1
        segments = [segment async for segment in support.respond(said.text)]
        heard = bot_turn(segments, turn)
        if support.ended:
            end_reason, detail = "bot_ended", support.end_reason
            break
    line.system(end_reason, detail)
    line.system("call_ended")
    return TextCallResult(
        timeline=line.timeline,
        end_reason=end_reason,
        detail=detail,
        completed_beats=list(caller.completed_beats),
        db_changes=diff_against_seed(support.dispatcher.db),
    )
