"""Live calls on real models, end to end through real audio.

Run with the `local-default` services up: `uv run pytest -m models tests/engine`. Whisper is the
independent ear, so these check what is really in the recording, not what the engine believes.
LLMs are stochastic, so the conversation tests assert outcomes a working stack should reach, and
one failure means look at the transcript, not necessarily that the engine broke.
"""

import asyncio
import re
from collections import deque
from pathlib import Path

import httpx
import numpy as np
import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame

from voicelab.agents import load_personas, load_scenarios
from voicelab.core.audio import AudioClip
from voicelab.core.models import BargeInEvent, CallMode, CallStatus, Speaker, Utterance
from voicelab.core.profiles import build_stack, load_profiles
from voicelab.engine.contract import BotCallRequest, TalkRequest
from voicelab.engine.service import create_talk_session, run_bot_call
from voicelab.eval.evaluators.state import satisfying_changes
from voicelab.store.artifacts import ArtifactStore

pytestmark = pytest.mark.models

PROFILE = "local-default"
EVAL_ONLY = {"judge"}
_MIC_RATE = 48000
_MIC_FRAME = 960


def require_servers() -> None:
    for service in load_profiles()[PROFILE].services:
        if service.name in EVAL_ONLY:
            continue
        try:
            httpx.get(f"http://127.0.0.1:{service.port}{service.health_path}", timeout=2)
        except httpx.HTTPError:
            pytest.skip(f"{service.name} isn't running on port {service.port}")


async def ignore(_message) -> None:
    return None


async def live_call(store: ArtifactStore, persona_id: str, scenario_id: str, *, seed: int = 1):
    require_servers()
    stack = build_stack(load_profiles()[PROFILE])
    persona = load_personas()[persona_id]
    request = BotCallRequest(
        call_id=f"call_{scenario_id}_{seed}",
        mode=CallMode.LIVE_BOTS,
        persona=persona,
        scenario=load_scenarios()[scenario_id],
        seed=seed,
    )
    artifact = await asyncio.wait_for(
        run_bot_call(request, stack=stack, store=store, emit=ignore), 480
    )
    return artifact, stack


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def overlap(heard: str, said: str) -> float:
    heard_words = set(words(heard))
    wanted = words(said)
    return sum(word in heard_words for word in wanted) / max(1, len(wanted))


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path)


async def test_a_spoken_refund_reaches_the_database(store):
    artifact, _ = await live_call(store, "per_skeptical_bookkeeper", "scn_double_charge_refund")

    assert artifact.status == CallStatus.EVALUATING, artifact.error
    scenario = artifact.scenario_snapshot
    added = next(e for e in scenario.expected_state if e.op == "row_added")
    assert satisfying_changes(added, artifact.db_changes), artifact.db_changes


async def test_what_is_on_disk_is_what_the_timeline_says(store):
    artifact, stack = await live_call(store, "per_chatty_polite", "scn_locked_mailbox", seed=2)

    bot = AudioClip.from_file(store.paths(artifact.id).bot)
    turns = [u for u in artifact.timeline if isinstance(u, Utterance) and u.speaker == Speaker.BOT]
    assert len(turns) >= 2
    for turn in turns[:3]:
        heard = await stack.stt_eval.transcribe(bot.slice(turn.start_s, turn.end_s))
        assert overlap(heard.text, turn.spoken_text) > 0.6, (heard.text, turn.spoken_text)


async def test_a_real_barge_in_stops_the_bot_where_the_artifact_says(store):
    artifact, stack = await live_call(store, "per_rushed_interrupter", "scn_locked_mailbox", seed=3)

    barges = [e for e in artifact.timeline if isinstance(e, BargeInEvent)]
    assert barges, "the caller never cut in"
    barge = barges[0]
    assert barge.bot_stopped_at_s - barge.at_s < 1.5
    cut = next(event for event in artifact.timeline if event.id == barge.interrupted_utterance_id)
    assert cut.interrupted and cut.spoken_text != cut.intended_text

    bot = AudioClip.from_file(store.paths(artifact.id).bot)
    played = await stack.stt_eval.transcribe(bot.slice(cut.start_s, cut.end_s))
    unplayed = cut.intended_text[len(cut.spoken_text) :]
    assert overlap(played.text, cut.spoken_text) > 0.6, (played.text, cut.spoken_text)
    if len(words(unplayed)) >= 4:
        assert overlap(played.text, unplayed) < 0.7, (played.text, unplayed)


# Mic calls, with an aiortc client playing the browser ---------------------------------------------


class SpeechMic(AudioStreamTrack):
    """Plays queued audio in real time, and silence otherwise, as a microphone would send it."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: deque[np.ndarray] = deque()
        self.gain = 1.0
        self._pts = 0

    def say(self, clip: AudioClip) -> None:
        samples = clip.resampled(_MIC_RATE).samples()
        for start in range(0, len(samples), _MIC_FRAME):
            self.queue.append(samples[start : start + _MIC_FRAME])

    async def recv(self) -> AudioFrame:
        await asyncio.sleep(_MIC_FRAME / _MIC_RATE)
        block = np.zeros(_MIC_FRAME, dtype=np.float32)
        if self.queue:
            chunk = self.queue.popleft()
            block[: len(chunk)] = chunk * self.gain
        frame = AudioFrame.from_ndarray(
            (np.clip(block, -1, 1) * 32767).astype(np.int16)[None, :], format="s16", layout="mono"
        )
        frame.sample_rate = _MIC_RATE
        frame.pts = self._pts
        self._pts += _MIC_FRAME
        return frame


class Phone:
    """A browser. With `echo_gain`, what it plays comes back into its mic, as speakers would do."""

    def __init__(self, *, echo_gain: float = 0.0) -> None:
        self.pc = RTCPeerConnection()
        self.mic = SpeechMic()
        self.pc.addTrack(self.mic)
        self.echo_gain = echo_gain
        self.loudest = 0.0
        self._task: asyncio.Task | None = None
        self.pc.on("track", self._on_track)

    def _on_track(self, track) -> None:
        self._task = asyncio.ensure_future(self._listen(track))

    async def _listen(self, track) -> None:
        while True:
            frame = await track.recv()
            samples = frame.to_ndarray().astype(np.float32).reshape(-1) / 32768
            self.loudest = max(self.loudest, float(np.abs(samples).max()))
            if self.echo_gain and frame.sample_rate:
                mono = (
                    samples[:: frame.layout.channels] if len(frame.layout.channels) > 1 else samples
                )
                clip = AudioClip.from_samples(mono * self.echo_gain, frame.sample_rate)
                self.mic.say(clip)

    async def dial(self, session) -> None:
        await self.pc.setLocalDescription(await self.pc.createOffer())
        offer = self.pc.localDescription
        answer = await session.handle_offer(offer.sdp, offer.type)
        await self.pc.setRemoteDescription(RTCSessionDescription(answer["sdp"], answer["type"]))

    async def hang_up(self) -> None:
        if self._task is not None:
            self._task.cancel()
        await self.pc.close()


async def _wait_for(condition, limit_s: float = 30.0) -> None:
    async with asyncio.timeout(limit_s):
        while not condition():  # noqa: ASYNC110  (conditions read plain state from other tasks)
            await asyncio.sleep(0.02)


async def _mic_session(store: ArtifactStore, phone: Phone):
    require_servers()
    stack = build_stack(load_profiles()[PROFILE])
    session = await create_talk_session(
        TalkRequest("call_mic"), stack=stack, store=store, emit=ignore
    )
    await phone.dial(session)
    await _wait_for(lambda: phone.loudest > 0.02)
    return session, stack


async def test_a_real_voice_on_the_mic_cuts_the_bot_off(store):
    phone = Phone()
    session, stack = await _mic_session(store, phone)
    words_said = "Sorry to cut in. I forgot the password for my work email."
    speech = await stack.tts_caller.synthesize(words_said, voice_id="default_female_1")

    await asyncio.sleep(1.5)
    phone.mic.say(speech)
    await _wait_for(
        lambda: any(
            isinstance(e, Utterance) and e.speaker == Speaker.CALLER
            for e in session.artifact.timeline
        )  # fmt: skip
    )
    await asyncio.sleep(1.0)
    artifact = await session.end()
    await phone.hang_up()

    greeting = next(u for u in artifact.timeline if isinstance(u, Utterance))
    assert greeting.interrupted and greeting.spoken_text != greeting.intended_text
    barge = next(e for e in artifact.timeline if isinstance(e, BargeInEvent))
    assert barge.bot_stopped_at_s - barge.at_s < 1.0
    caller = next(
        u for u in artifact.timeline if isinstance(u, Utterance) and u.speaker == Speaker.CALLER
    )
    assert overlap(caller.heard_text, words_said) > 0.6, caller.heard_text
    bot = AudioClip.from_file(store.paths("call_mic").bot)
    played = await stack.stt_eval.transcribe(bot.slice(greeting.start_s, greeting.end_s + 0.2))
    assert overlap(played.text, greeting.spoken_text) > 0.5, played.text


@pytest.mark.parametrize("gain", [0.05, 0.2])
async def test_the_bot_does_not_interrupt_itself_through_the_speakers(store, gain):
    phone = Phone(echo_gain=gain)
    session, _ = await _mic_session(store, phone)

    await asyncio.sleep(7.0)  # the whole greeting plays, and comes back into the mic
    artifact = await session.end()
    await phone.hang_up()

    assert not any(isinstance(e, BargeInEvent) for e in artifact.timeline)
    assert artifact.metadata.get("false_barge_ins", 0) == 0
    greeting = next(u for u in artifact.timeline if isinstance(u, Utterance))
    assert not greeting.interrupted
