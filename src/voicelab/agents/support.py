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
from voicelab.core.models import SystemEvent, ToolCallEvent, new_id
from voicelab.providers.base import LLM, ChatMessage, LLMResponse, ToolCall

SYSTEM_PROMPT = """\
You are Fern, the phone support agent for Fernhill Internet. Fernhill sells email hosting, \
website hosting, domains, and SSL certificates. Today is {today}.

You are on a live phone call, and everything you write is read aloud.
- Use plain spoken sentences. No markdown, lists, or symbols.
- Say one or two short sentences, then stop.
- Ask one question at a time.
- Look facts up with tools. Never guess account details.
- Before changing anything, say what you will do and wait for a clear yes.
- If you can't do something, say so and open a ticket.
- When the caller is finished, say goodbye and call end_call.

{policy}"""

GREETING = "Thanks for calling Fernhill Internet, this is Fern. How can I help you today?"
GOODBYE = "Thanks for calling Fernhill Internet. Goodbye."
TROUBLE_LINE = "I'm sorry, I'm having trouble with our system. Could you say that again?"

# Some chat templates reject a conversation whose first turn isn't from the user.
_CALL_CONNECTED = "(The phone rings and you answer.)"
_RETRY_NOTE = (
    "(Note from the system, not the caller: your last reply was not usable. Either call a tool "
    "with valid JSON arguments, or answer the caller in plain spoken sentences.)"
)

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


def _unsaid(text: str, said: list[str], *, include_tail: bool = False) -> list[str]:
    sentences, tail = split_sentences(speakable_text(text))
    if include_tail and tail:
        sentences.append(tail)
    return sentences[len(said) :]


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
        self.ended = False
        self.end_reason = ""
        self._turn = 0  # bumped by each new turn and each interrupt, which stops the running turn
        system = support_instructions(dispatcher.db.today, policy)
        self.messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]

    def greet(self) -> SpeechText:
        """The opening line. Fixed, so a call never starts with a model failure."""
        self.messages.append(ChatMessage(role="user", content=_CALL_CONNECTED))
        self.messages.append(ChatMessage(role="assistant", content=GREETING))
        return normalize(GREETING, self.lexicon)

    async def respond(self, caller_text: str) -> AsyncIterator[SpeechText]:
        """Yield speakable segments for one caller turn, sentence by sentence.

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
        segments_spoken = 0
        for _ in range(self.max_tool_rounds):
            said: list[str] = []
            raw = ""
            final: LLMResponse | None = None
            spoken_before_round = segments_spoken
            stream = self.llm.stream_chat(
                self._retry_messages() if retry else self.messages,
                tools=self.dispatcher.tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            try:
                async for delta in stream:
                    raw += unicode_safe(delta.text)
                    final = delta.final or final
                    for sentence in _unsaid(raw, said):
                        if turn != self._turn:
                            return
                        said.append(sentence)
                        segment = self._commit_speech(sentence)
                        if segment:
                            segments_spoken += 1
                            yield segment
                text = unicode_safe(final.text) if final and final.text else raw
                for sentence in _unsaid(text, said, include_tail=True):
                    if turn != self._turn:
                        return
                    said.append(sentence)
                    segment = self._commit_speech(sentence)
                    if segment:
                        segments_spoken += 1
                        yield segment
            finally:
                # An abandoned stream would keep a single-slot model server busy into the next turn.
                close = getattr(stream, "aclose", None)
                if close:
                    await close()
            if turn != self._turn:
                return

            calls = list(final.tool_calls) if final else []
            malformed = False
            if not calls:
                calls, malformed = parse_text_tool_calls(text)

            if calls:
                self._run_tools(calls)
                retry = False
                if self.ended:
                    if not segments_spoken:
                        yield self._say(GOODBYE)
                    return
                continue
            if segments_spoken > spoken_before_round and not malformed:
                return
            self._report("malformed_tool_call", text[:500])
            retry = True

        self._report("tool_loop_limit", f"gave up after {self.max_tool_rounds} rounds")
        yield self._say(TROUBLE_LINE)

    def on_interrupted(self, spoken_text: str) -> None:
        """Rewrite the last turn to what was actually played before the caller cut in.

        Everything else the bot wrote never happened as far as the caller knows. Tool calls stay,
        since they really ran and the bot has to remember them. `spoken_text` is the TTS form, and
        small models read that back without trouble. Works mid-turn too, and stops that turn.
        """
        self._turn += 1
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
        if spoken_text.strip():
            kept.append(ChatMessage(role="assistant", content=spoken_text.strip()))
        self.messages[start:] = kept

    def _add_caller_text(self, text: str) -> None:
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

    def _retry_messages(self) -> list[ChatMessage]:
        """History plus a one-time note. The note isn't kept, since the caller never said it."""
        last = self.messages[-1]
        if last.role == "user":
            note = last.model_copy(update={"content": f"{last.content}\n\n{_RETRY_NOTE}"})
            return [*self.messages[:-1], note]
        return [*self.messages, ChatMessage(role="user", content=_RETRY_NOTE)]

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
            call.model_copy(update={"arguments": unicode_safe(call.arguments)}) for call in calls
        ]
        results: list[ChatMessage] = []
        events: list[ToolCallEvent | SystemEvent] = []
        crashed = False
        for call in calls:
            at_s = self.clock()
            started = time.perf_counter()
            if crashed:
                result = {"error": "not run because an earlier tool failed"}
            else:
                try:
                    result = self.dispatcher.dispatch(call.name, call.arguments)
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
                )
            )
            results.append(
                ChatMessage(
                    role="tool",
                    content=json.dumps(result),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
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

    def _report(self, kind: str, detail: str) -> None:
        if self.on_event:
            self.on_event(SystemEvent(at_s=self.clock(), kind=kind, detail=detail))
