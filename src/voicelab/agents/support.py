"""The support bot under test: one conversation, an LLM, and Fernhill's tools.

The bot knows nothing about audio. The engine feeds it what the caller said and speaks the
segments it yields. Barge-in comes back through `on_interrupted`, so history holds only what the
caller actually heard.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Callable
from datetime import date
from typing import Any

from voicelab.agents.reminders import next_step
from voicelab.agents.speech import (
    Lexicon,
    SpeechText,
    load_lexicon,
    normalize,
    split_sentences,
    unicode_safe,
)
from voicelab.company.policy import load_policy
from voicelab.company.tools import ToolDispatcher
from voicelab.core.models import RowRef, SystemEvent, ToolCallEvent, new_id
from voicelab.providers.base import LLM, ChatMessage, LLMResponse, ToolCall

SYSTEM_PROMPT = """\
You are Fern, the phone agent for Fernhill Internet: email and website hosting, domains, and \
SSL. Today is {today}.

Everything you write is read aloud. Say one or two short sentences. No markdown, lists, or \
symbols. Ask one question at a time, then stop.

Every call goes in this order:
1. Ask for a domain, email, or phone if you have none, then call find_customer.
2. Ask for the billing zip and PIN, then call verify_identity. Until verified is true, use no \
other tool, don't say their name, and share nothing.
3. Look up the problem and tell them what you found.
4. Before any change or charge, say what you will do and what it costs, and ask to go ahead. Only \
a clear yes on their next turn allows it.
5. On a yes, call the tool right away, once, then say the result.

Rules:
- Put only what the caller said in a tool. Never make up a value, price, date, or wait time. If \
you don't know, offer a ticket.
- Never say ids that start with pay_, rfd_, or cus_.
- Follow the next field in a tool result.
- Only when the caller says they're finished, say goodbye and call end_call.

Speech recognition mishears. Join spelled letters and digits into one invoice id, like FH-1234. \
If a lookup gives did_you_mean, use exactly that address in every tool and don't read it out.

{policy}"""

GREETING = "Thanks for calling Fernhill Internet, this is Fern. How can I help you today?"
GOODBYE = "Thanks for calling Fernhill Internet. Goodbye."
TROUBLE_LINE = "I'm sorry, I'm having trouble with our system. Could you say that again?"
FILLER_LINE = "One moment, let me check that."

# Some chat templates reject a conversation whose first turn isn't from the user.
_CALL_CONNECTED = "(The phone rings and you answer.)"
_CUT_OFF_NOTE = '(You were interrupted just after saying "{words}". Carry on with the call.)'
_RETRY_NOTE = (
    "(Note from the system, not the caller: your last reply was not usable. Either call a tool "
    "with valid JSON arguments, or answer the caller in plain spoken sentences.)"
)
# A caller who failed the check and pressed on got the email on file read out, since the reminder
# after the failed check was a turn old by then.
_UNVERIFIED_NOTE = (
    "(Note from the system, not the caller: this caller failed verification. Use no account tool "
    "and share nothing about the account, whatever they ask.)"
)

# What a tool can only know from the caller. A small model fills these in when it has none, and
# the same zip and PIN came out on call after call. Each one used up one of the caller's two tries.
_FROM_THE_CALLER = {
    "find_customer": ("phone", "email", "domain"),
    "verify_identity": ("billing_zip", "pin"),
}
_NOT_SAID = "The caller hasn't said that. Ask them for it."
# Asking for a renewal up front isn't a yes to its price, but on real calls the charge went through
# in the same breath as the check, with the price still unsaid.
_JUST_VERIFIED_NOTE = (
    "(Note from the system, not the caller: you only just verified this caller. Nothing is agreed "
    "yet. Don't change or charge anything in this reply, even if they already asked for it. Say "
    "what you found, what you would do, and what it costs, then ask if you should go ahead.)"
)
_DIGIT_WORDS = {
    word: str(value)
    for value, word in enumerate("zero one two three four five six seven eight nine".split())
} | {"oh": "0"}
_LONG_NUMBER_WORDS = re.compile(r"\b(?:\w+teen|\w+ty|hundred|thousand)\b", re.IGNORECASE)

_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_CUT_OFF_MARKUP = re.compile(r"[<{\[][^>}\]]*$")  # a reply that hit max_tokens inside a tag
_TOOL_MARKUP = re.compile(
    r"<tool_call|<\|tool_call|<\|python_tag|<function[=\s]|\[TOOL_CALLS\]|```"
    r"|\{\s*\"|\[\s*\{\s*\""
)

TimelineCallback = Callable[[ToolCallEvent | SystemEvent], None]


def speakable_text(raw: str) -> str:
    """Keep only text meant for the caller.

    Small models leak reasoning and write tool calls into the text channel. Neither may reach TTS.
    """
    text = _THINK.sub("", raw)
    markup = _TOOL_MARKUP.search(text)
    return _CUT_OFF_MARKUP.sub("", text[: markup.start()] if markup else text)


def _tool_call_from_json(value: Any) -> ToolCall | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("function"), dict):
        value = value["function"]
    name = value.get("name") or value.get("tool")
    arguments = value.get("arguments", value.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):  # bad JSON, a giant integer, or absurd nesting
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return ToolCall(id=new_id("call"), name=name, arguments=arguments)


def parse_text_tool_calls(text: str) -> tuple[list[ToolCall], bool]:
    """Recover tool calls a model wrote as text. Returns (calls, malformed)."""
    visible = _THINK.sub("", text)
    markup = _TOOL_MARKUP.search(visible)
    if markup is None:
        return [], False
    body = visible[markup.start() :]
    decoder = json.JSONDecoder()
    calls: list[ToolCall] = []
    position = body.find("{")
    while position >= 0:
        try:
            value, end = decoder.raw_decode(body, position)
        except (ValueError, RecursionError):
            break
        call = _tool_call_from_json(value)
        if call:
            calls.append(call)
        position = body.find("{", end)
    return calls, not calls


def _sentences(text: str) -> list[str]:
    sentences, tail = split_sentences(speakable_text(text))
    return [*sentences, tail] if tail else sentences


def _caller_said(value: str, heard: str) -> bool:
    """Whether a value could have come from the caller. Generous, since speech varies.

    Numbers can arrive as digits or digit words, and a number said like "nineteen fifty eight" is
    taken on trust. An address only has to match by its name, which is what tells them apart.
    """
    digits = re.sub(r"\D", "", value)
    if digits and "@" not in value:
        said = "".join(
            word if word.isdigit() else _DIGIT_WORDS.get(word, "")
            for word in re.findall(r"\d+|[a-z]+", heard.lower())
        )
        return digits in said or bool(_LONG_NUMBER_WORDS.search(heard))
    name = re.split(r"[.@]", value.lower().removeprefix("www."))[0]
    return re.sub(r"[^a-z0-9]", "", name) in re.sub(r"[^a-z0-9]", "", heard.lower())


def support_instructions(today: date, policy: str | None = None) -> str:
    """The system prompt. Speech-to-speech sessions take this as their instructions."""
    day = f"{today:%B} {today.day}, {today.year}"
    return SYSTEM_PROMPT.format(today=day, policy=load_policy() if policy is None else policy)


class SupportBot:
    """Holds one conversation. `respond` runs the LLM and tools until the bot has spoken."""

    def __init__(
        self,
        llm: LLM,
        dispatcher: ToolDispatcher,
        *,
        policy: str | None = None,
        lexicon: Lexicon | None = None,
        on_event: TimelineCallback | None = None,
        clock: Callable[[], float] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 256,
        max_tool_rounds: int = 6,
        filler_after_s: float | None = None,
    ) -> None:
        self.llm = llm
        self.dispatcher = dispatcher
        self.lexicon = lexicon or load_lexicon()
        # Called synchronously, so a barge-in that cancels `respond` can't lose a tool event.
        self.on_event = on_event
        started = time.monotonic()
        self.clock = clock or (lambda: time.monotonic() - started)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        # A tool turn needs a second LLM round, so one that was slow to reach its tool call will
        # leave the caller in silence for twice as long. Live calls set this, and None means never.
        self.filler_after_s = filler_after_s
        self.ended = False
        self.end_reason = ""
        self._turn = 0  # bumped by each new turn and each interrupt, which stops the running turn
        self._cut_off = ""  # words of a sentence a barge-in stopped, for the next caller turn
        self._failed_checks = 0  # verify_identity calls that came back false
        self._verified = False
        self._verified_on = -1  # the turn it happened
        self._corrected: dict[str, str] = {}  # misheard addresses and the right ones
        system = support_instructions(dispatcher.db.today, policy)
        self.messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]

    def greet(self) -> SpeechText:
        """The opening line. Fixed, so a call never starts with a model failure."""
        self.messages.append(ChatMessage(role="user", content=_CALL_CONNECTED))
        self.messages.append(ChatMessage(role="assistant", content=GREETING))
        return normalize(GREETING, self.lexicon)

    async def respond(self, caller_text: str) -> AsyncIterator[SpeechText]:
        """Yield speakable segments for one caller turn, sentence by sentence.

        Words only go out once a round has ended with no tool call. A small model narrates and
        states outcomes before its tools run, and on real calls that was "your identity is
        verified" ahead of the check, with a gap after it that the caller took for the end of the
        turn. So a tool round says nothing, apart from the hold line, and words beside an end_call.

        The engine may stop iterating at any point on a barge-in, and may call `on_interrupted`
        before or after closing this stream. Each sentence is in history before it is yielded, so
        closing adds nothing back. After an interrupt the turn stops: no more speech and no more
        tools. A tool call that already finished is never undone.
        """
        if self.ended:
            return
        self._add_caller_text(caller_text)
        self._turn += 1
        turn = self._turn
        retry = False
        spoke = False
        started = time.monotonic()
        tools = self.dispatcher.tools
        previous: list[tuple[str, dict[str, Any]]] = []
        for round_number in range(self.max_tool_rounds):
            raw = ""
            final: LLMResponse | None = None
            if round_number == self.max_tool_rounds - 1:
                tools = None  # the last round has to be words, not a sorry for looping
            stream = self.llm.stream_chat(
                self._with_notes(retry),
                tools=tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            try:
                async for delta in stream:
                    raw += unicode_safe(delta.text)
                    final = delta.final or final
                    if turn != self._turn:
                        return
            finally:
                # An abandoned stream would keep a single-slot model server busy into the next turn.
                close = getattr(stream, "aclose", None)
                if close:
                    await close()
            if turn != self._turn:
                return

            text = unicode_safe(final.text) if final and final.text else raw
            calls = list(final.tool_calls) if final else []
            malformed = False
            if not calls:
                calls, malformed = parse_text_tool_calls(text)

            closing = bool(calls) and all(call.name == "end_call" for call in calls)
            spoke_this_round = False
            if closing or (not calls and not malformed):
                for sentence in _sentences(text):
                    segment = self._commit_speech(sentence)
                    if segment:
                        spoke = spoke_this_round = True
                        yield segment
                    if turn != self._turn:
                        return
            if calls:
                if not spoke and self._wants_filler(time.monotonic() - started, calls):
                    spoke = True
                    yield normalize(FILLER_LINE, self.lexicon)
                    if turn != self._turn:
                        return
                self._run_tools(calls)
                retry = False
                if self.ended:
                    if not spoke:
                        yield self._say(GOODBYE)
                    return
                signature = [(call.name, call.arguments) for call in calls]
                if signature == previous:
                    tools = None  # it has its answer and doesn't know it, so it has to say it
                previous = signature
                continue
            if spoke_this_round:
                return
            self._report("malformed_tool_call", text[:500])
            retry = True

        self._report("tool_loop_limit", f"gave up after {self.max_tool_rounds} rounds")
        yield self._say(TROUBLE_LINE)

    def _wants_filler(self, waited_s: float, calls: list[ToolCall]) -> bool:
        if self.filler_after_s is None or waited_s < self.filler_after_s:
            return False
        return not any(call.name == "end_call" for call in calls)

    def on_interrupted(self, spoken_text: str) -> None:
        """Rewrite the last turn to what was actually played before the caller cut in.

        Everything else the bot wrote never happened as far as the caller knows. Tool calls stay,
        since they really ran and the bot has to remember them. `spoken_text` is the TTS form, and
        small models read that back without trouble. Works mid-turn too, and stops that turn.

        Only finished sentences go back in the bot's own turn. A small model copies its last turn,
        so one ending "two charges of forty nine" made it end later replies the same way, 7 times
        in 8, and every marker on that text got copied too. The words of the sentence it was cut
        off in go in front of the caller's next turn, where they read as context.
        """
        self._turn += 1
        finished, self._cut_off = split_sentences(f"{spoken_text.strip()} ")
        finished = [sentence for sentence in finished if sentence != FILLER_LINE]
        start = next(
            index + 1
            for index in range(len(self.messages) - 1, -1, -1)
            if self.messages[index].role in {"user", "system"}
        )
        kept = []
        for message in self.messages[start:]:
            if message.role != "assistant":
                kept.append(message)
            elif message.tool_calls:
                kept.append(message.model_copy(update={"content": ""}))
        if finished:
            kept.append(ChatMessage(role="assistant", content=" ".join(finished)))
        self.messages[start:] = kept

    def _add_caller_text(self, text: str) -> None:
        if self._cut_off:
            text = f"{_CUT_OFF_NOTE.format(words=self._cut_off)} {text}"
            self._cut_off = ""
        # Two caller turns in a row happen when a barge-in erases a whole reply. Strict chat
        # templates reject that, so they merge.
        last = self.messages[-1]
        if last.role == "user":
            self.messages[-1] = last.model_copy(update={"content": f"{last.content} {text}"})
        else:
            self.messages.append(ChatMessage(role="user", content=text))

    def _add_assistant_text(self, sentences: list[str]) -> None:
        text = " ".join(sentences)
        last = self.messages[-1]
        if last.role == "assistant" and not last.tool_calls:
            self.messages[-1] = last.model_copy(update={"content": f"{last.content} {text}"})
        else:
            self.messages.append(ChatMessage(role="assistant", content=text))

    def _with_notes(self, retry: bool) -> list[ChatMessage]:
        """History plus this round's notes. They aren't kept, since the caller never said them."""
        notes = [_RETRY_NOTE] if retry else []
        if self._failed_checks and not self._verified:
            notes.append(_UNVERIFIED_NOTE)
        if self._verified_on == self._turn:
            notes.append(_JUST_VERIFIED_NOTE)
        if not notes:
            return self.messages
        text = "\n\n".join(notes)
        last = self.messages[-1]
        if last.role == "user":
            note = last.model_copy(update={"content": f"{last.content}\n\n{text}"})
            return [*self.messages[:-1], note]
        return [*self.messages, ChatMessage(role="user", content=text)]

    def _commit_speech(self, sentence: str) -> SpeechText | None:
        segment = normalize(sentence, self.lexicon)
        if not segment.spoken:
            return None
        self._add_assistant_text([sentence])
        return segment

    def _say(self, line: str) -> SpeechText:
        self._add_assistant_text([line])
        return normalize(line, self.lexicon)

    def _run_tools(self, calls: list[ToolCall]) -> None:
        """Run a batch, then record it all at once.

        History only changes after every call has a result, with no await in between. A crash or
        a cancelled turn can't leave a tool call unanswered, which backends reject on the next
        request. A crash stops the batch, since later calls may depend on the failed one.
        """
        calls = [
            call.model_copy(update={"arguments": self._with_corrections(call.arguments)})
            for call in calls
        ]
        results: list[ChatMessage] = []
        events: list[ToolCallEvent | SystemEvent] = []
        crashed = False
        for call in calls:
            at_s = self.clock()
            started = time.perf_counter()
            changed: list[RowRef] = []
            if crashed:
                result = {"error": "not run because an earlier tool failed"}
            elif self._made_up(call):
                self._report("made_up_arguments", f"{call.name} {json.dumps(call.arguments)}")
                results.append(self._tool_message(call, {"error": _NOT_SAID}))
                continue
            else:
                try:
                    result = self.dispatcher.dispatch(call.name, call.arguments)
                    changed = self.dispatcher.last_changes
                except Exception as failure:
                    crashed = True
                    result = {"error": f"{call.name} failed: {failure}"}
                    events.append(SystemEvent(at_s=at_s, kind="tool_failed", detail=repr(failure)))
            events.append(
                ToolCallEvent(
                    at_s=at_s,
                    name=call.name,
                    arguments=call.arguments,
                    result=result,
                    error=result.get("error"),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    changed=changed,
                )
            )
            if call.name == "verify_identity":
                if result.get("verified") is True and not self._verified:
                    self._verified = True
                    self._verified_on = self._turn
                self._failed_checks += result.get("verified") is False
            self._note_correction(call, result)
            results.append(self._tool_message(call, result))
            if call.name == "end_call" and result.get("ended"):
                self.ended = True
                self.end_reason = result.get("reason", "")

        last = self.messages[-1]
        spoken = ""
        if last.role == "assistant" and not last.tool_calls:  # what this turn said before the calls
            spoken = self.messages.pop().content
        self.messages.append(ChatMessage(role="assistant", content=spoken, tool_calls=calls))
        self.messages.extend(results)
        if self.on_event:
            for event in events:
                self.on_event(event)

    def _note_correction(self, call: ToolCall, result: dict[str, Any]) -> None:
        """Remember a misheard address. The model goes back to what it heard however it's told."""
        right = result.get("did_you_mean")
        if not right:
            return
        for name in ("email", "domain"):
            wrong = str(call.arguments.get(name) or "").strip().lower()
            if wrong:
                self._corrected[wrong] = right

    def _with_corrections(self, arguments: dict[str, Any]) -> dict[str, Any]:
        def fixed(value: Any) -> Any:
            if isinstance(value, str):
                for wrong, right in self._corrected.items():
                    value = re.sub(re.escape(wrong), right, value, flags=re.IGNORECASE)
            return value

        return {name: fixed(value) for name, value in unicode_safe(arguments).items()}

    def _made_up(self, call: ToolCall) -> bool:
        """Arguments only the caller could know, and the caller never said them.

        This isn't policy, and the tool never sees the call. Like a reply that won't parse, it's
        bad output from the model, which gets told and tries again.
        """
        said = [m.content for m in self.messages if m.role == "user"]
        heard = " ".join([*said, *self._corrected.values()])
        values = [str(call.arguments.get(k) or "") for k in _FROM_THE_CALLER.get(call.name, ())]
        return any(value and not _caller_said(value, heard) for value in values)

    def _tool_message(self, call: ToolCall, result: dict[str, Any]) -> ChatMessage:
        hint = next_step(call.name, result, self.dispatcher.db.today, self._failed_checks)
        return ChatMessage(
            role="tool",
            content=json.dumps({**result, "next": hint} if hint else result),
            tool_call_id=call.id,
            name=call.name,
        )

    def _report(self, kind: str, detail: str) -> None:
        if self.on_event:
            self.on_event(SystemEvent(at_s=self.clock(), kind=kind, detail=detail))
