"""Live calls on real models, end to end through real audio.

Run with the `local-default` services up: `uv run pytest -m models tests/engine`. Whisper is the
independent ear, so these check what is really in the recording, not what the engine believes.
LLMs are stochastic, so the conversation tests assert outcomes a working stack should reach, and
one failure means look at the transcript, not necessarily that the engine broke.
"""

import asyncio
import re
from pathlib import Path

import httpx
import pytest

from voicelab.agents import load_personas, load_scenarios
from voicelab.core.audio import AudioClip
from voicelab.core.models import BargeInEvent, CallMode, CallStatus, Speaker, Utterance
from voicelab.core.profiles import build_stack, load_profiles
from voicelab.engine.contract import BotCallRequest
from voicelab.engine.service import run_bot_call
from voicelab.eval.evaluators.state import satisfying_changes
from voicelab.store.artifacts import ArtifactStore

pytestmark = pytest.mark.models

PROFILE = "local-default"
EVAL_ONLY = {"judge"}


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
    cut = next(
        event for event in artifact.timeline if event.id == barge.interrupted_utterance_id
    )
    assert cut.interrupted and cut.spoken_text != cut.intended_text

    bot = AudioClip.from_file(store.paths(artifact.id).bot)
    played = await stack.stt_eval.transcribe(bot.slice(cut.start_s, cut.end_s))
    unplayed = cut.intended_text[len(cut.spoken_text) :]
    assert overlap(played.text, cut.spoken_text) > 0.6, (played.text, cut.spoken_text)
    if len(words(unplayed)) >= 4:
        assert overlap(played.text, unplayed) < 0.7, (played.text, unplayed)
