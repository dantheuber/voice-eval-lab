import numpy as np
import pytest

from voicelab.core.models import CallArtifact, CallMode, Speaker, Utterance
from voicelab.engine.echo import STEP_S, EchoDetector, LevelContour
from voicelab.engine.live.log import LiveLog
from voicelab.engine.speech_ledger import PlayoutLedger
from voicelab.engine.tool_checks import tool_call_problem
from voicelab.providers.base import ToolCall, ToolDef
from voicelab.store.artifacts import ArtifactStore

RATE = 16000


def babble(seconds: float, seed: int, *, gain: float = 0.2) -> np.ndarray:
    """Noise under a slow random level contour, which is all the detector looks at."""
    rng = np.random.default_rng(seed)
    steps = round(seconds / 0.05)
    contour = np.repeat(rng.uniform(0, 1, steps) ** 2, round(0.05 * RATE))
    return (gain * contour * rng.standard_normal(len(contour))).astype(np.float32)


def frames(samples: np.ndarray, size: int = 320):
    for start in range(0, len(samples) - size + 1, size):
        yield samples[start : start + size]


# Playout ledger ----------------------------------------------------------------------------------


def turn() -> PlayoutLedger:
    ledger = PlayoutLedger()
    for text in ("one two", "three four"):
        segment = ledger.start_segment(text)
        segment.produced_s = 1.0
        segment.finished = True
    return ledger


def test_nothing_is_spoken_until_audio_goes_out():
    assert turn().spoken_text == ""


def test_spoken_text_follows_audio_sent_not_audio_made():
    ledger = turn()

    ledger.add_played(1.1)

    assert ledger.spoken_text == "one two"
    assert ledger.intended_text == "one two three four"


def test_a_sentence_still_in_tts_is_measured_by_its_length():
    ledger = PlayoutLedger()
    ledger.start_segment("one two three four five six seven eight nine ten").produced_s = 0.2

    ledger.add_played(0.6)

    assert ledger.spoken_text == "one two"


def test_tts_finishing_a_sentence_after_a_new_turn_began_is_harmless():
    ledger = PlayoutLedger()
    in_flight = ledger.start_segment("from the old turn")

    ledger.reset()
    in_flight.produced_s += 0.5
    in_flight.finished = True

    assert ledger.intended_text == ""


def test_reset_starts_a_new_turn():
    ledger = turn()
    ledger.add_played(2.0)

    ledger.reset()

    assert ledger.spoken_text == "" and ledger.intended_text == ""


# Echo --------------------------------------------------------------------------------------------


def run_echo_check(detector: EchoDetector, far: np.ndarray, mic: np.ndarray) -> float:
    """Play both directions in step and return how much of the mic counted as echo."""
    flags = []
    for out, heard in zip(frames(far), frames(mic), strict=True):
        detector.add_outgoing(out, RATE)
        flags.append(detector.is_echo(heard, RATE))
    return float(np.mean(flags[50:]))


@pytest.mark.parametrize("delay_s", [0.0, 0.12, 0.3])
def test_the_bots_own_voice_coming_back_quieter_and_late_is_echo(delay_s):
    far = babble(3.0, seed=1)
    delay = round(delay_s * RATE)
    mic = np.concatenate([np.zeros(delay, dtype=np.float32), far[: len(far) - delay] * 0.15])

    assert run_echo_check(EchoDetector(), far, mic) > 0.9


def test_a_person_talking_over_the_bot_is_not_echo():
    far = babble(3.0, seed=1)
    person = babble(3.0, seed=2)

    assert run_echo_check(EchoDetector(), far, person) < 0.1


def test_a_person_loud_over_the_echo_is_not_echo():
    far = babble(3.0, seed=1)
    person = babble(3.0, seed=2) + far * 0.05

    assert run_echo_check(EchoDetector(), far, person) < 0.15


def test_nothing_is_echo_while_the_bot_is_silent():
    detector = EchoDetector()

    flags = [detector.is_echo(frame, RATE) for frame in frames(babble(2.0, seed=3))]

    assert not any(flags)


def test_echo_is_caught_before_vad_could_fire_on_it():
    far = babble(1.0, seed=1)
    detector = EchoDetector()

    flags = []
    for out, heard in zip(frames(far), frames(far * 0.2), strict=True):
        detector.add_outgoing(out, RATE)
        flags.append(detector.is_echo(heard, RATE))

    assert all(flags[3:10])  # VAD needs 200 ms, which is ten of these frames


def test_a_person_cutting_in_as_the_bot_starts_is_not_blanked():
    far = babble(1.0, seed=1, gain=0.1)
    person = babble(1.0, seed=2, gain=0.4)
    detector = EchoDetector()

    flags = []
    for out, heard in zip(frames(far), frames(person), strict=True):
        detector.add_outgoing(out, RATE)
        flags.append(detector.is_echo(heard, RATE))

    assert np.mean(flags[3:15]) < 0.3


def test_echo_is_watched_for_just_after_the_bot_stops():
    far = np.concatenate([babble(2.0, seed=1), np.zeros(RATE, dtype=np.float32)])
    mic = far * 0.15
    detector = EchoDetector()

    flags = []
    for out, heard in zip(frames(far), frames(mic), strict=True):
        detector.add_outgoing(out, RATE)
        flags.append(detector.is_echo(heard, RATE))

    assert not any(flags[-20:])


def test_contours_work_for_any_frame_size():
    contour = LevelContour(100)
    steps = sum(contour.add(np.ones(77, dtype=np.float32), RATE) for _ in range(100))

    assert steps == (77 * 100) // round(STEP_S * RATE)
    assert contour.levels[-1] == pytest.approx(1.0)


# Tool checks -------------------------------------------------------------------------------------

TOOLS = [
    ToolDef(
        name="get_invoice",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "years": {"type": "integer"},
            },
            "required": ["invoice_id"],
        },
    )
]


def problem(name: str = "get_invoice", **arguments) -> str | None:
    return tool_call_problem(ToolCall(id="c", name=name, arguments=arguments), TOOLS)


def test_a_good_call_has_no_problem():
    assert problem(invoice_id="FH-4419") is None


def test_an_unknown_tool_is_a_problem():
    assert "no tool" in problem("get_invoices", invoice_id="x")


def test_a_missing_required_argument_is_a_problem():
    assert "missing invoice_id" in problem(years=1)
    assert "missing invoice_id" in problem(invoice_id="")


def test_template_syntax_in_a_value_is_a_problem():
    assert "template" in problem(invoice_id="[customer_id]\n</function>")


def test_a_near_miss_on_type_is_left_to_the_tool_to_coerce():
    assert problem(invoice_id=4419) is None
    assert problem(invoice_id="FH-4419", years="1") is None


def test_a_real_type_mismatch_is_a_problem():
    assert "wrong type" in problem(invoice_id=["FH-4419"])


def test_stray_arguments_are_tolerated():
    assert problem(invoice_id="FH-4419", extra="field") is None


# What the far side heard -------------------------------------------------------------------------


async def _ignore(_message) -> None:
    return None


def _log(tmp_path) -> LiveLog:
    artifact = CallArtifact(mode=CallMode.LIVE_BOTS)
    return LiveLog(artifact, ArtifactStore(tmp_path), _ignore)


def _turn(start_s: float, end_s: float) -> Utterance:
    return Utterance(speaker=Speaker.CALLER, start_s=start_s, end_s=end_s, intended_text="x")


async def test_several_stt_segments_join_the_one_turn_they_came_from(tmp_path):
    log = _log(tmp_path)

    turn = log.utterance(_turn(1.0, 5.0), far_side_listens=True)
    log.hear(Speaker.CALLER, "Hello there.", 2.0)
    log.hear(Speaker.CALLER, "My email is locked.", 5.1)

    assert turn.heard_text == "Hello there. My email is locked."


async def test_segments_that_arrive_before_the_turn_closes_are_kept_for_it(tmp_path):
    log = _log(tmp_path)

    log.hear(Speaker.CALLER, "Early words.", 2.0)
    turn = log.utterance(_turn(1.0, 5.0), far_side_listens=True)

    assert turn.heard_text == "Early words."


async def test_a_later_turns_words_are_not_credited_to_an_earlier_turn(tmp_path):
    log = _log(tmp_path)

    first = log.utterance(_turn(1.0, 3.0), far_side_listens=True)
    log.hear(Speaker.CALLER, "First.", 3.1)
    log.hear(Speaker.CALLER, "Second.", 9.0)
    second = log.utterance(_turn(8.0, 9.0), far_side_listens=True)

    assert first.heard_text == "First." and second.heard_text == "Second."


async def test_a_turn_nobody_heard_is_recorded_as_heard_empty(tmp_path):
    log = _log(tmp_path)
    log.start()

    turn = log.utterance(_turn(1.0, 3.0), far_side_listens=True)
    await log.close()

    assert turn.heard_text == ""


async def test_a_mic_callers_turn_goes_out_without_waiting_for_anyone(tmp_path):
    sent = []

    async def keep(message) -> None:
        sent.append(message)

    log = LiveLog(CallArtifact(mode=CallMode.HUMAN), ArtifactStore(tmp_path), keep)
    log.start()

    log.utterance(_turn(1.0, 3.0), far_side_listens=False)
    await log.close()

    assert [message["type"] for message in sent] == ["turn"]
