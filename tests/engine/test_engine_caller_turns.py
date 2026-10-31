"""When the caller bot holds its tongue: the bot's turn can have gaps in it."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from voicelab.agents.caller import CallerLine, Interruption
from voicelab.engine.live.brains import _HOLD_ON, CallerBrain
from voicelab.engine.live.parts import LiveSettings
from voicelab.engine.speech_ledger import PlayoutLedger


class _Caller(CallerBrain):
    def create_task(self, coroutine, name=None):
        return asyncio.ensure_future(coroutine)

    async def cancel_task(self, task, *_):
        task.cancel()


def _brain(next_line) -> _Caller:
    return _Caller(
        SimpleNamespace(
            next_line=next_line,
            decide_interruption=lambda _: Interruption(False, 0.5),
        ),
        stt=None,
        on_hang_up=None,
        log=SimpleNamespace(now=lambda: 0.0, hear=lambda *_: None),
        ledger=PlayoutLedger(),
        settings=LiveSettings(caller_patience_s=30.0),
        fail=None,
    )


async def _replying_brain() -> _Caller:
    """A caller partway through working out a reply."""
    thinking = asyncio.Event()

    async def next_line(heard: str) -> CallerLine:
        thinking.set()
        await asyncio.sleep(30)
        return CallerLine("Okay.")

    brain = _brain(next_line)
    brain.on_transcript("I found your account.")
    await brain.on_far_side_finished()
    await thinking.wait()
    return brain


async def test_a_reply_being_worked_out_stops_when_the_bot_carries_on():
    brain = await _replying_brain()
    await brain.on_far_side_started()
    assert not brain.working
    await brain._stop_patience()


async def test_a_reply_already_being_spoken_is_left_to_finish():
    brain = await _replying_brain()
    brain.turn_open = True
    await brain.on_far_side_started()
    assert brain.working
    await brain.stop_work()


@pytest.mark.parametrize(
    "heard",
    [
        "One moment, let me check that.",
        "One moment.",
        "Let me check that",
        "Let me look that up for you.",
        "Hold on.",
    ],
)
def test_asking_to_wait_is_heard_as_a_hold(heard):
    assert _HOLD_ON.search(heard)


@pytest.mark.parametrize(
    "heard",
    ["What's your billing zip?", "The site is back online.", "Let me know if that works for you."],
)
def test_a_finished_reply_is_not_a_hold(heard):
    assert not _HOLD_ON.search(heard)
