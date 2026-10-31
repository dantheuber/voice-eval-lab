import json

import numpy as np
import pytest
import soundfile as sf

from voicelab.core.audio import AudioClip, read_stereo_wav
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    Speaker,
    SystemEvent,
    ToolCallEvent,
    Utterance,
)
from voicelab.engine.recorder import CallRecorder
from voicelab.store.artifacts import ArtifactStore

RATE = 16000


def constant(level: float, seconds: float, rate: int = RATE) -> AudioClip:
    return AudioClip.from_samples(np.full(round(seconds * rate), level, dtype=np.float32), rate)


def tone(hz: float, seconds: float, rate: int = RATE, amp: float = 0.4) -> AudioClip:
    t = np.arange(round(seconds * rate)) / rate
    return AudioClip.from_samples((amp * np.sin(2 * np.pi * hz * t)).astype(np.float32), rate)


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path)


@pytest.fixture
def recorder(store) -> CallRecorder:
    return CallRecorder(CallArtifact(mode=CallMode.TURN_BASED, timing_is_real=False), store)


def channels(store: ArtifactStore, artifact: CallArtifact) -> tuple[np.ndarray, np.ndarray]:
    left, right = read_stereo_wav(store.paths(artifact.id).recording)
    return left.samples(), right.samples()


def test_finish_writes_every_artifact_file(recorder, store):
    recorder.place(Speaker.CALLER, tone(440, 0.5), 0.0)
    artifact = recorder.finish()
    paths = store.paths(artifact.id)
    assert paths.dir == store.data_dir / "calls" / artifact.id
    for path in (paths.recording, paths.caller, paths.bot, paths.call_json, paths.peaks):
        assert path.exists()
    assert store.read_artifact(artifact.id).id == artifact.id


def test_channels_land_left_and_right_at_their_positions(recorder, store):
    recorder.place(Speaker.CALLER, constant(0.5, 0.5), 1.0)
    recorder.place(Speaker.BOT, constant(-0.25, 0.5), 2.0)
    artifact = recorder.finish()
    assert artifact.duration_s == pytest.approx(2.5)
    left, right = channels(store, artifact)
    assert left.shape == right.shape
    assert np.count_nonzero(left[:RATE]) == 0
    assert left[RATE] == pytest.approx(0.5, abs=0.001)
    assert np.count_nonzero(left[int(1.5 * RATE) :]) == 0
    assert np.count_nonzero(right[: 2 * RATE]) == 0
    assert right[2 * RATE] == pytest.approx(-0.25, abs=0.001)


def test_per_channel_files_match_the_stereo_recording(recorder, store):
    recorder.place(Speaker.CALLER, tone(300, 1.0), 0.0)
    recorder.place(Speaker.BOT, tone(500, 0.5), 0.5)
    artifact = recorder.finish()
    paths = store.paths(artifact.id)
    left, right = channels(store, artifact)
    assert sf.info(str(paths.recording)).channels == 2
    assert sf.info(str(paths.recording)).samplerate == RATE
    assert np.allclose(AudioClip.from_file(paths.caller).samples(), left, atol=1e-3)
    assert np.allclose(AudioClip.from_file(paths.bot).samples(), right, atol=1e-3)


def test_gaps_are_silence(recorder, store):
    recorder.place(Speaker.BOT, constant(0.3, 0.2), 0.0)
    recorder.place(Speaker.BOT, constant(0.3, 0.2), 1.0)
    artifact = recorder.finish()
    _, bot = channels(store, artifact)
    assert np.count_nonzero(bot[int(0.2 * RATE) + 1 : RATE - 1]) == 0
    assert bot[RATE + 10] > 0.29


def test_overlap_on_one_channel_is_mixed_and_clipped(recorder, store):
    recorder.place(Speaker.BOT, constant(0.3, 1.0), 0.0)
    recorder.place(Speaker.BOT, constant(0.2, 1.0), 0.5)
    recorder.place(Speaker.BOT, constant(0.9, 0.2), 0.7)
    artifact = recorder.finish()
    _, bot = channels(store, artifact)
    assert bot[int(0.2 * RATE)] == pytest.approx(0.3, abs=0.001)
    assert bot[int(0.6 * RATE)] == pytest.approx(0.5, abs=0.001)
    assert bot[int(0.8 * RATE)] == pytest.approx(1.0, abs=0.001)


def test_truncate_cuts_a_clip_at_the_interruption(recorder, store):
    recorder.place(Speaker.BOT, constant(0.3, 3.0), 1.0)
    recorder.truncate(Speaker.BOT, 2.0)
    assert recorder.channel_end_s(Speaker.BOT) == pytest.approx(2.0)
    artifact = recorder.finish()
    assert artifact.duration_s == pytest.approx(2.0)
    _, bot = channels(store, artifact)
    assert bot[int(1.9 * RATE)] == pytest.approx(0.3, abs=0.001)
    assert len(bot) == 2 * RATE


def test_truncate_one_placement_leaves_later_audio_alone(recorder, store):
    first = recorder.place(Speaker.BOT, constant(0.3, 3.0), 0.0)
    recorder.place(Speaker.BOT, constant(0.3, 1.0), 2.5)
    recorder.truncate(Speaker.BOT, 1.0, placement=first)
    artifact = recorder.finish()
    _, bot = channels(store, artifact)
    assert np.count_nonzero(bot[RATE + 1 : int(2.5 * RATE) - 1]) == 0
    assert bot[int(3.0 * RATE)] == pytest.approx(0.3, abs=0.001)


def test_truncate_before_a_clip_starts_removes_all_of_it(recorder, store):
    recorder.place(Speaker.CALLER, constant(0.2, 1.0), 0.0)
    placement = recorder.place(Speaker.BOT, constant(0.3, 1.0), 5.0)
    recorder.truncate(Speaker.BOT, 4.0, placement=placement)
    artifact = recorder.finish()
    _, bot = channels(store, artifact)
    assert np.count_nonzero(bot) == 0


def test_truncate_keeps_only_the_shortest_cut(recorder):
    placement = recorder.place(Speaker.BOT, constant(0.3, 3.0), 0.0)
    recorder.truncate(Speaker.BOT, 1.0)
    recorder.truncate(Speaker.BOT, 2.0)
    assert placement.end_s == pytest.approx(1.0)


def test_inputs_are_resampled_to_the_common_rate(recorder, store):
    recorder.place(Speaker.CALLER, tone(440, 1.0, rate=8000), 0.0)
    recorder.place(Speaker.BOT, tone(440, 1.0, rate=48000), 0.0)
    artifact = recorder.finish()
    left, right = channels(store, artifact)
    assert sf.info(str(store.paths(artifact.id).recording)).samplerate == RATE
    assert len(left) == len(right) == RATE
    for channel in (left, right):
        assert np.argmax(np.abs(np.fft.rfft(channel))) == pytest.approx(440, abs=1)


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def live(store, clock) -> CallRecorder:
    return CallRecorder(CallArtifact(mode=CallMode.HUMAN), store, clock=clock)


def test_live_frames_with_jittery_timestamps_join_without_gaps(live, store):
    frame = constant(0.3, 0.02)
    jitter = [0.0, 0.003, -0.004, 0.002, -0.001] * 10
    for i, offset in enumerate(jitter):
        live.write_frame(Speaker.CALLER, frame, at_s=i * 0.02 + offset)
    artifact = live.finish()
    caller, _ = channels(store, artifact)
    assert artifact.duration_s == pytest.approx(1.0)
    assert np.all(caller[10:-10] > 0.29)


def test_live_gap_longer_than_the_snap_stays_silent(live, store):
    live.write_frame(Speaker.BOT, constant(0.3, 0.02), at_s=0.0)
    position = live.write_frame(Speaker.BOT, constant(0.3, 0.02), at_s=0.5)
    assert position == 0.5
    _, bot = channels(store, live.finish())
    assert np.count_nonzero(bot[int(0.03 * RATE) : int(0.49 * RATE)]) == 0
    assert bot[int(0.51 * RATE)] > 0.29


def test_live_frames_default_to_the_recorder_clock(live, clock):
    live.start()
    clock.t += 2.0
    assert live.now() == pytest.approx(2.0)
    assert live.write_frame(Speaker.CALLER, constant(0.3, 0.02)) == pytest.approx(2.0)


def test_live_frames_sent_ahead_of_the_clock_queue_up(live):
    live.start()
    positions = [live.write_frame(Speaker.BOT, constant(0.3, 0.02)) for _ in range(5)]
    assert positions == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.08])


def test_live_frames_resample_as_one_run_without_boundary_artifacts(live, store):
    clip = tone(440, 1.0, rate=8000)
    for i in range(0, len(clip.pcm), 320):
        live.write_frame(Speaker.BOT, AudioClip(clip.pcm[i : i + 320], 8000), at_s=i / 16000)
    _, bot = channels(store, live.finish())
    expected = clip.resampled(RATE).samples()
    n = min(len(bot), len(expected))
    assert np.max(np.abs(bot[:n] - expected[:n])) < 0.01


def test_barge_in_on_a_live_channel_removes_queued_audio(live, store):
    live.start()
    for _ in range(100):  # two seconds queued ahead of the clock
        live.write_frame(Speaker.BOT, constant(0.3, 0.02))
    live.truncate(Speaker.BOT, 0.5)
    position = live.write_frame(Speaker.BOT, constant(0.1, 0.02), at_s=1.0)
    assert position == 1.0
    _, bot = channels(store, live.finish())
    assert bot[int(0.4 * RATE)] > 0.29
    assert np.count_nonzero(bot[int(0.51 * RATE) : int(0.99 * RATE)]) == 0
    assert bot[int(1.01 * RATE)] == pytest.approx(0.1, abs=0.001)


def test_timeline_events_are_collected_in_time_order(recorder):
    recorder.add_event(ToolCallEvent(at_s=3.0, name="lookup"))
    recorder.add_event(Utterance(speaker=Speaker.BOT, start_s=1.0, end_s=2.0))
    recorder.add_event(SystemEvent(at_s=0.0, kind="call_started"))
    artifact = recorder.finish()
    assert [e.type for e in artifact.timeline] == ["system", "utterance", "tool_call"]


def test_duration_covers_events_past_the_last_audio(recorder, store):
    recorder.place(Speaker.CALLER, constant(0.3, 1.0), 0.0)
    recorder.add_event(SystemEvent(at_s=2.5, kind="call_ended"))
    artifact = recorder.finish()
    assert artifact.duration_s == pytest.approx(2.5)
    left, right = channels(store, artifact)
    assert len(left) == len(right) == round(2.5 * RATE)


def test_finish_keeps_call_metadata_and_sets_times(store, clock):
    artifact = CallArtifact(mode=CallMode.HUMAN, profile="local-qwen")
    recorder = CallRecorder(artifact, store, clock=clock)
    recorder.start()
    recorder.write_frame(Speaker.CALLER, constant(0.3, 0.02))
    finished = recorder.finish()
    stored = store.read_artifact(finished.id)
    assert stored.profile == "local-qwen"
    assert stored.started_at is not None
    assert stored.ended_at is not None


def test_empty_call_still_writes_a_valid_artifact(recorder, store):
    artifact = recorder.finish()
    assert artifact.duration_s == 0.0
    assert store.read_artifact(artifact.id).duration_s == 0.0


def test_peaks_track_each_channel(recorder, store):
    recorder.place(Speaker.CALLER, constant(0.5, 1.0), 0.0)
    recorder.place(Speaker.BOT, constant(0.25, 1.0), 1.0)
    artifact = recorder.finish()
    peaks = json.loads(store.paths(artifact.id).peaks.read_text())
    assert peaks["peaks_per_second"] == 50
    assert peaks["duration_s"] == pytest.approx(2.0)
    caller, bot = peaks["channels"]["caller"], peaks["channels"]["bot"]
    assert len(caller) == len(bot) == 100
    assert caller[10] == pytest.approx(0.5, abs=0.002)
    assert caller[60] == 0
    assert bot[10] == 0
    assert bot[60] == pytest.approx(0.25, abs=0.002)
