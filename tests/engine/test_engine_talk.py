"""Mic calls over real WebRTC: an aiortc client in this process plays the browser."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame
from enginefakes import Messages, Scripted, stack

from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    BargeInEvent,
    CallArtifact,
    CallMode,
    CallStatus,
    Speaker,
    Utterance,
)
from voicelab.engine.contract import TalkRequest
from voicelab.engine.live.parts import simple_settings
from voicelab.engine.live.talk import create_live_talk_session
from voicelab.engine.service import create_talk_session
from voicelab.providers.base import Transcript
from voicelab.store.artifacts import ArtifactStore

RATE = 48000
FRAME = 960


class Mic(AudioStreamTrack):
    """Silence until told to talk, then a loud tone, as a microphone would send."""

    def __init__(self) -> None:
        super().__init__()
        self.talking = False
        self._pts = 0

    async def recv(self) -> AudioFrame:
        await asyncio.sleep(FRAME / RATE)
        if self.talking:
            t = np.arange(self._pts, self._pts + FRAME) / RATE
            samples = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        else:
            samples = np.zeros(FRAME, dtype=np.int16)
        frame = AudioFrame.from_ndarray(samples[None, :], format="s16", layout="mono")
        frame.sample_rate = RATE
        frame.pts = self._pts
        self._pts += FRAME
        return frame


class HeardWhatever:
    """Reports a fixed transcript for any speech, since the mic here sends a tone."""

    name = "fixed-stt"

    def __init__(self, text: str = "Hold on please.") -> None:
        self.text = text
        self.clips: list[AudioClip] = []

    async def transcribe(self, audio, *, language="en", hints=None):
        self.clips.append(audio)
        return Transcript(text=self.text)


class Browser:
    def __init__(self) -> None:
        self.pc = RTCPeerConnection()
        self.mic = Mic()
        self.pc.addTrack(self.mic)
        self.messages: list[dict[str, Any]] = []
        self.loudest = 0.0
        self.channel = self.pc.createDataChannel("chat")
        self.channel.on("message", self._on_message)
        self.channel.on("open", self._say_ready)
        self._drain: asyncio.Task | None = None

        @self.pc.on("track")
        def on_track(track):
            self._drain = asyncio.ensure_future(self._listen(track))

    def _say_ready(self) -> None:
        ready = {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": str(uuid.uuid4()),
            "data": {"version": "1.0.0", "about": {"library": "test"}},
        }
        self.channel.send(json.dumps(ready))

    def _on_message(self, raw: str) -> None:
        try:
            self.messages.append(json.loads(raw))
        except ValueError:
            pass

    async def _listen(self, track) -> None:
        while True:
            frame = await track.recv()
            level = np.abs(frame.to_ndarray()).max() / 32768
            self.loudest = max(self.loudest, float(level))

    async def offer(self, session) -> None:
        await self.pc.setLocalDescription(await self.pc.createOffer())
        local = self.pc.localDescription
        answer = await session.handle_offer(local.sdp, local.type)
        await self.pc.setRemoteDescription(RTCSessionDescription(answer["sdp"], answer["type"]))

    async def close(self) -> None:
        if self._drain is not None:
            self._drain.cancel()
        await self.pc.close()


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path)


async def wait_until(condition, timeout: float = 15.0) -> None:
    async def poll() -> None:
        while not condition():  # noqa: ASYNC110  (conditions read other tasks' plain state)
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


def utterances(artifact: CallArtifact, speaker: Speaker | None = None) -> list[Utterance]:
    found = [e for e in artifact.timeline if isinstance(e, Utterance)]
    return [u for u in found if speaker is None or u.speaker == speaker]


async def open_call(store: ArtifactStore, *, emit=None, **adapters):
    built = stack(stt_live=adapters.pop("stt_live", HeardWhatever()), **adapters)
    session = await create_live_talk_session(
        TalkRequest("call_talk"),
        stack=built,
        store=store,
        emit=emit or Messages(),
        settings=simple_settings(),
    )
    browser = Browser()
    await browser.offer(session)
    return session, browser


async def test_the_bot_greets_a_caller_who_connects_and_the_call_is_recorded(store):
    session, browser = await open_call(store, llm_support=Scripted(default="Okay."))

    await wait_until(lambda: browser.loudest > 0.02)
    artifact = await session.end()
    await browser.close()

    assert artifact.mode == CallMode.HUMAN and artifact.timing_is_real is True
    assert artifact.status == CallStatus.EVALUATING
    assert store.paths("call_talk").recording.exists()
    bot = AudioClip.from_file(store.paths("call_talk").bot)
    assert np.abs(bot.samples()).max() > 0.02


async def test_talking_over_the_bot_stops_it_and_leaves_only_what_was_played(store):
    support = Scripted(default="All right.")
    stt = HeardWhatever()
    session, browser = await open_call(store, llm_support=support, stt_live=stt)

    await wait_until(lambda: browser.loudest > 0.02)
    await asyncio.sleep(1.5)
    browser.mic.talking = True
    await asyncio.sleep(0.8)
    browser.mic.talking = False
    await wait_until(lambda: bool(support.requests))
    await asyncio.sleep(1.5)
    artifact = await session.end()
    await browser.close()

    greeting = utterances(artifact, Speaker.BOT)[0]
    assert greeting.interrupted and greeting.spoken_text != greeting.intended_text
    assert greeting.intended_text.startswith(greeting.spoken_text)
    barge = next(e for e in artifact.timeline if isinstance(e, BargeInEvent))
    assert barge.interrupted_utterance_id == greeting.id
    assert 0 <= barge.bot_stopped_at_s - barge.at_s < 0.6

    caller = utterances(artifact, Speaker.CALLER)[0]
    assert caller.heard_text == "Hold on please." and caller.intended_text is None
    history = [m for m in support.requests[0] if m.role == "assistant"]
    assert history[-1].content == greeting.spoken_text
    assert utterances(artifact, Speaker.BOT)[1].intended_text == "All right."


async def test_noise_that_turns_out_to_be_nothing_is_counted_as_a_false_barge_in(store):
    support = Scripted(default="All right.")
    session, browser = await open_call(store, llm_support=support, stt_live=HeardWhatever(text=""))

    await wait_until(lambda: browser.loudest > 0.02)
    await asyncio.sleep(1.0)
    browser.mic.talking = True
    await asyncio.sleep(0.5)
    browser.mic.talking = False
    await wait_until(lambda: session.artifact.metadata.get("false_barge_ins", 0) > 0, 10)
    artifact = await session.end()
    await browser.close()

    assert artifact.metadata["false_barge_ins"] >= 1
    assert support.requests == []


async def test_events_reach_the_browser_as_voicelab_server_messages(store):
    from enginefakes import uses

    plan = uses("list_plans", text="Let me check.")
    session, browser = await open_call(store, llm_support=Scripted(plan, default="Several."))

    await wait_until(lambda: browser.loudest > 0.02)
    await asyncio.sleep(1.0)
    browser.mic.talking = True
    await asyncio.sleep(0.5)
    browser.mic.talking = False

    def saw_tool() -> bool:
        payloads = [
            m.get("data", {}).get("voicelab", {})
            for m in browser.messages
            if m.get("type") == "server-message"
        ]
        return any(p.get("event", {}).get("name") == "list_plans" for p in payloads)

    await wait_until(saw_tool)
    await session.end()
    await browser.close()

    kinds = {m.get("type") for m in browser.messages}
    assert {"bot-ready", "user-started-speaking", "user-transcription"} <= kinds


async def test_the_api_hears_when_the_browser_goes_away(store):
    messages = Messages()
    session, browser = await open_call(store, emit=messages, llm_support=Scripted(default="Okay."))

    await wait_until(lambda: browser.loudest > 0.02)
    await browser.close()
    await wait_until(lambda: any(m["type"] == "disconnected" for m in messages.items), 40)
    artifact = await session.end()

    assert artifact.status == CallStatus.EVALUATING
    assert store.read_artifact("call_talk").status == CallStatus.EVALUATING


async def test_ending_twice_returns_the_same_artifact(store):
    session, browser = await open_call(store, llm_support=Scripted(default="Okay."))

    first = await session.end()
    second = await session.end()
    await browser.close()

    assert first is second


async def test_a_session_that_never_connected_still_ends(store):
    session = await create_talk_session(
        TalkRequest("call_talk"), stack=stack(), store=store, emit=Messages()
    )

    artifact = await session.end()

    assert artifact.status == CallStatus.EVALUATING
    assert store.paths("call_talk").call_json.exists()


async def test_trickled_ice_candidates_are_accepted(store):
    session, browser = await open_call(store, llm_support=Scripted(default="Okay."))

    await session.add_ice_candidates(
        [
            {
                "candidate": "candidate:1 1 udp 2122260223 127.0.0.1 54321 typ host",
                "sdp_mid": "0",
                "sdp_mline_index": 0,
            },
            {"candidate": "", "sdp_mid": "0", "sdp_mline_index": 0},
        ]
    )
    await session.end()
    await browser.close()


async def test_candidates_before_any_offer_are_ignored(store):
    session = await create_talk_session(
        TalkRequest("call_talk"), stack=stack(), store=store, emit=Messages()
    )

    await session.add_ice_candidates([{"candidate": "x"}])

    await session.end()
