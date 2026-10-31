"""The support bot as one speech-to-speech model, for profiles with `support_engine: realtime`.

The model hears audio directly, so STT runs beside it only to give the call a transcript and to
rebuild context after a cut-in. Pipecat still does VAD and turn detection, since the server has
neither. This path has no `SupportBot`, so tool calls are validated here and run one at a time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    InputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from voicelab.agents.support import GREETING
from voicelab.company.tools import ToolDispatcher
from voicelab.core.models import RowRef, ToolCallEvent
from voicelab.engine.live.brains import BotBrain
from voicelab.engine.speech_ledger import Segment
from voicelab.engine.tool_checks import tool_call_problem
from voicelab.providers.base import TTS, RealtimeEvent, RealtimeVoice, ToolCall

logger = logging.getLogger(__name__)

_PRE_ROLL_S = 0.4  # VAD fires after speech starts, and the words before that still count
_END_SLACK_S = 0.1  # the output drops a partial last chunk


class RealtimeBrain(BotBrain):
    role = "realtime"

    def __init__(
        self,
        voice: RealtimeVoice,
        *,
        dispatcher: ToolDispatcher,
        instructions: str,
        greeter: TTS,
        greeter_voice: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.voice = voice
        self.dispatcher = dispatcher
        self.instructions = (
            f"{instructions}\n\nYou have already answered the call by saying: {GREETING}"
        )
        self.greeter = greeter
        self.greeter_voice = greeter_voice
        self.session: Any = None
        self._reader: asyncio.Task | None = None
        self._sender: asyncio.Task | None = None
        self._audio_out: asyncio.Queue[bytes] = asyncio.Queue()
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_s = 0.0
        self._user_talking = False
        self._held_for_cut = False
        self._resampler = create_stream_resampler()
        self._in_rate = 16000
        self._reply = Segment("")  # every response in a turn runs together on the ledger
        self._calls: list[RealtimeEvent] = []
        self._reply_audio_started = False
        self._response_id: str | None = None
        self._ending: str | None = None

    async def on_start(self) -> None:
        try:
            self.session = await self.voice.connect(
                instructions=self.instructions, tools=self.dispatcher.tools
            )
        except Exception as failure:  # a busy server is the usual cause: one session per process
            await self.fail(f"realtime failed to connect: {failure!r}")
            return
        self._reader = self.create_task(self._read_events())
        self._sender = self.create_task(self._send_audio())

    async def greet(self) -> None:
        """The opening line is fixed, so a call never starts with a model failure."""
        if self.session is None:
            return
        self.begin_turn()
        self.turn_open = True
        self._reply = self.ledger.start_segment(GREETING)
        self.run(self._play_greeting())

    async def _play_greeting(self) -> None:
        await self.push_frame(TTSStartedFrame())
        async for clip in self.greeter.stream(GREETING, voice_id=self.greeter_voice):
            self._reply.produced_s += clip.duration_s
            await self._push_audio(clip.pcm, clip.sample_rate)
        await self.push_frame(TTSStoppedFrame())
        self.task = None
        self._reply.finished = True

    async def _push_audio(self, pcm: bytes, rate: int) -> None:
        await self.push_frame(TTSAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=1))

    def on_audio_in(self, frame: InputAudioRawFrame) -> None:
        self._in_rate = frame.sample_rate
        if self._user_talking and not self._held_for_cut:
            self._audio_out.put_nowait(frame.audio)
            return
        self._pre_roll.append(frame.audio)
        self._pre_roll_s += len(frame.audio) / 2 / frame.sample_rate
        while self._pre_roll_s > _PRE_ROLL_S and len(self._pre_roll) > 1:
            dropped = self._pre_roll.popleft()
            self._pre_roll_s -= len(dropped) / 2 / frame.sample_rate

    def _release_pre_roll(self) -> None:
        while self._pre_roll:
            self._audio_out.put_nowait(self._pre_roll.popleft())
        self._pre_roll_s = 0.0

    async def _send_audio(self) -> None:
        while True:
            pcm = await self._audio_out.get()
            audio = await self._resampler.resample(pcm, self._in_rate, self.voice.input_sample_rate)
            if audio:
                await self.session.send_audio(audio)

    async def on_far_side_started(self) -> None:
        await super().on_far_side_started()
        self._user_talking = True
        if self.turn_open:
            self._held_for_cut = True  # audio sent now would die with the socket being replaced
        else:
            self._release_pre_roll()

    async def on_far_side_finished(self) -> None:
        """Local turn detection decided the caller is done, so ask for the reply now.

        This doesn't wait for the transcript. STT feeds the record and any later session rebuild,
        and this is the latency that matters.
        """
        self._user_talking = False
        self.barged_at_s = None
        if self.call_over:
            return
        self.begin_turn()
        self._reply = self.ledger.start_segment("")
        self._reply_audio_started = False
        self._response_id = None
        self.turn_open = True
        await self.session.commit_turn()

    def on_transcript(self, text: str) -> None:
        self.session.record_user_text(text)
        self.note_heard(text)
        self._log_caller_turn(text)

    async def stop_model(self, played_ms: int) -> None:
        await self.stop_work()
        self._calls = []
        while not self._audio_out.empty():  # the sender is waiting on this queue, so keep it
            self._audio_out.get_nowait()
        await self.session.interrupt(response_id=self._response_id, played_ms=played_ms)

    async def forget_unheard(self, played_text: str) -> None:
        """`interrupt` already rebuilt the session, so just send what the caller said since."""
        self._held_for_cut = False
        self._release_pre_roll()

    async def _read_events(self) -> None:
        async for event in self.session.events():
            await self._handle(event)

    async def _handle(self, event: RealtimeEvent) -> None:
        if self.cut_pending:
            return  # still in the pipe from the response being cut. Its tools must not run
        if event.response_id:
            self._response_id = event.response_id
        if event.type == "audio":
            if not event.audio_pcm16:
                return
            if not self._reply_audio_started:
                self._reply_audio_started = True
                await self.push_frame(TTSStartedFrame())
            rate = self.voice.output_sample_rate
            self._reply.produced_s += len(event.audio_pcm16) / 2 / rate
            await self._push_audio(event.audio_pcm16, rate)
        elif event.type == "bot_transcript":
            self._reply.text += event.text
        elif event.type == "tool_call":
            self._calls.append(event)
        elif event.type == "response_done":
            await self._finish_response()
        elif event.type == "error":
            await self.fail(f"realtime failed: {event.text}")

    async def _finish_response(self) -> None:
        calls, self._calls = self._calls, []
        if calls:
            self.run(self._run_tools(calls))
            return
        if self._reply_audio_started:
            await self.push_frame(TTSStoppedFrame())
        self._reply.finished = True
        await self.on_speech_paused()

    async def _run_tools(self, calls: list[RealtimeEvent]) -> None:
        """In order, one at a time. A later call may depend on an earlier result."""
        for event in calls:
            await self._run_tool(event)
        await self.session.continue_after_tools()

    async def _run_tool(self, event: RealtimeEvent) -> None:
        call: ToolCall = event.tool_call
        problem = event.text or tool_call_problem(call, self.dispatcher.tools)
        at_s = self.log.now()
        changed: list[RowRef] = []
        if problem:
            result: dict[str, Any] = {"error": problem}
        else:
            try:
                result = self.dispatcher.dispatch(call.name, call.arguments)
                changed = self.dispatcher.last_changes
            except Exception as failure:
                self.log.system("tool_failed", repr(failure))
                result = {"error": f"{call.name} failed: {failure}"}
        self.log.event(
            ToolCallEvent(
                at_s=at_s,
                name=call.name,
                arguments=call.arguments,
                result=result,
                error=result.get("error"),
                changed=changed,
            )
        )
        if call.name == "end_call" and result.get("ended"):
            self._ending = result.get("reason", "")
        await self.session.send_tool_result(call.id, json.dumps(result))

    async def on_speech_paused(self) -> None:
        # The model finishes long before its audio does, so `response_done` proves little.
        audio_out = self.ledger.played_s >= self._reply.produced_s - _END_SLACK_S
        if not self.turn_open or self.cut_pending or not self._reply.finished:
            return
        if self.bot_speaking or not audio_out:
            return
        self.close_turn(interrupted=False)
        if self._ending is not None:
            await self.on_call_ended(self._ending)

    async def close(self) -> None:
        for task in (self._sender, self._reader):
            if task is not None and not task.done():
                await self.cancel_task(task)
        if self.session is not None:
            await self.session.close()
