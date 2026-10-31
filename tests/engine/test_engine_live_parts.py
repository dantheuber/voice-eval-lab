import numpy as np
import pytest

from voicelab.engine.echo import STEP_S, EchoDetector, LevelContour
from voicelab.engine.speech_ledger import PlayoutLedger
from voicelab.engine.tool_checks import tool_call_problem
from voicelab.providers.base import ToolCall, ToolDef

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
    ledger.start_segment("one two")
    ledger.add_audio(1.0)
    ledger.finish_segment()
    ledger.start_segment("three four")
    ledger.add_audio(1.0)
    ledger.finish_segment()
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
    ledger.start_segment("one two three four five six seven eight nine ten")
    ledger.add_audio(0.2)

    ledger.add_played(0.6)

    assert ledger.spoken_text == "one two"


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
