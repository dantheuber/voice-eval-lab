import numpy as np

from voicelab.core.audio import AudioClip


def clip(seconds: float = 2.0, rate: int = 16000) -> AudioClip:
    return AudioClip.from_samples(np.full(round(seconds * rate), 0.25, dtype=np.float32), rate)


def test_slice_returns_requested_range():
    assert clip().slice(0.5, 1.5).duration_s == 1.0


def test_negative_end_is_empty_not_wrapped():
    assert clip().slice(0, -1).n_samples == 0


def test_end_before_start_is_empty():
    assert clip().slice(1.5, 0.5).n_samples == 0


def test_bounds_clamp_to_clip():
    assert clip().slice(-1, 5).duration_s == 2.0
    assert clip().slice(3, 4).n_samples == 0
