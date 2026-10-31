import io

import numpy as np
import pytest
import soundfile as sf

from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    EvalResult,
    Speaker,
    SystemEvent,
    Utterance,
    Verdict,
)
from voicelab.eval.vad import speech_regions
from voicelab.store.artifacts import (
    SEPARATION_EVENT,
    ArtifactStore,
    EnergySeparator,
    Separation,
    SpeakerSegment,
)

RATE = 16000


def tone(seconds: float, hz: float = 300.0, amp: float = 0.3, rate: int = RATE) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(seconds: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(round(seconds * rate), dtype=np.float32)


def wav_bytes(samples: np.ndarray, rate: int = RATE, fmt: str = "WAV") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format=fmt)
    return buf.getvalue()


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path)


@pytest.fixture
def stored_call(store) -> CallArtifact:
    artifact = CallArtifact(mode=CallMode.TURN_BASED, duration_s=2.0)
    caller = AudioClip.from_samples(np.full(2 * RATE, 0.5, dtype=np.float32), RATE)
    bot = AudioClip.from_samples(np.full(2 * RATE, -0.25, dtype=np.float32), RATE)
    store.write_call(artifact, caller, bot)
    return artifact


def test_artifact_roundtrip_keeps_typed_timeline(store):
    artifact = CallArtifact(
        mode=CallMode.HUMAN,
        timeline=[
            Utterance(speaker=Speaker.BOT, start_s=0.0, end_s=1.0, intended_text="Hello"),
            SystemEvent(at_s=1.0, kind="call_ended"),
        ],
    )
    store.write_artifact(artifact)
    loaded = store.read_artifact(artifact.id)
    assert isinstance(loaded.timeline[0], Utterance)
    assert isinstance(loaded.timeline[1], SystemEvent)
    assert loaded.model_dump() == artifact.model_dump()


def test_eval_roundtrip(store, stored_call):
    assert store.read_eval(stored_call.id) is None
    result = EvalResult(call_id=stored_call.id, verdict=Verdict.PASS, summary="ok")
    store.write_eval(result)
    assert store.read_eval(stored_call.id).verdict == Verdict.PASS


def test_writes_leave_no_temporary_files(store, stored_call):
    names = {p.name for p in store.paths(stored_call.id).dir.iterdir()}
    assert names == {"recording.wav", "caller.wav", "bot.wav", "call.json", "peaks.json"}


@pytest.mark.parametrize("bad_id", ["../outside", "a/b", "", "..", "call.json"])
def test_call_ids_cannot_escape_the_data_directory(store, bad_id):
    with pytest.raises(ValueError, match="invalid call id"):
        store.paths(bad_id)


def test_channels_of_different_lengths_and_rates_are_matched(store):
    artifact = CallArtifact(mode=CallMode.TURN_BASED)
    caller = AudioClip.from_samples(tone(1.0, rate=8000), 8000)
    bot = AudioClip.from_samples(tone(2.0), RATE)
    paths = store.write_call(artifact, caller, bot)
    assert sf.info(str(paths.caller)).frames == sf.info(str(paths.bot)).frames == 2 * RATE
    assert sf.info(str(paths.caller)).samplerate == RATE


def test_clip_of_one_channel(store, stored_call):
    clip = AudioClip.from_wav_bytes(store.clip_wav(stored_call.id, 0.5, 1.5, "caller"))
    assert clip.sample_rate == RATE
    assert clip.duration_s == pytest.approx(1.0)
    assert clip.samples()[100] == pytest.approx(0.5, abs=0.001)


def test_clip_of_the_stereo_mix_keeps_both_channels(store, stored_call):
    data, rate = sf.read(io.BytesIO(store.clip_wav(stored_call.id, 0.0, 1.0)))
    assert rate == RATE
    assert data.shape == (RATE, 2)
    assert data[100, 0] == pytest.approx(0.5, abs=0.001)
    assert data[100, 1] == pytest.approx(-0.25, abs=0.001)


def test_clip_range_is_clamped_to_the_call(store, stored_call):
    clip = AudioClip.from_wav_bytes(store.clip_wav(stored_call.id, -1.0, 99.0, "bot"))
    assert clip.duration_s == pytest.approx(2.0)


def test_empty_clip_range_is_an_error(store, stored_call):
    with pytest.raises(ValueError, match="empty clip range"):
        store.clip_wav(stored_call.id, 5.0, 6.0, "bot")


def test_stereo_import_is_caller_left_bot_right(store):
    data = np.stack([tone(1.0, 300), tone(1.0, 700)], axis=1)
    artifact = store.import_recording(wav_bytes(data), filename="call.wav")
    assert artifact.mode == CallMode.UPLOAD
    assert artifact.has_tool_log is False
    assert artifact.has_intended_text is False
    assert artifact.duration_s == pytest.approx(1.0)
    paths = store.paths(artifact.id)
    caller = AudioClip.from_file(paths.caller).samples()
    assert np.argmax(np.abs(np.fft.rfft(caller))) == pytest.approx(300, abs=1)
    assert not any(
        isinstance(e, SystemEvent) and e.kind == SEPARATION_EVENT for e in artifact.timeline
    )
    assert store.read_artifact(artifact.id).id == artifact.id


def test_stereo_import_can_swap_channels(store):
    data = np.stack([tone(1.0, 300), tone(1.0, 700)], axis=1)
    artifact = store.import_recording(wav_bytes(data), swap_channels=True)
    caller = AudioClip.from_file(store.paths(artifact.id).caller).samples()
    assert np.argmax(np.abs(np.fft.rfft(caller))) == pytest.approx(700, abs=1)


def test_import_resamples_and_reads_other_formats(store):
    data = tone(1.0, 300, rate=44100)
    artifact = store.import_recording(wav_bytes(data, 44100, "FLAC"))
    assert sf.info(str(store.paths(artifact.id).bot)).samplerate == RATE


def test_stereo_import_records_speech_as_utterances(store):
    left = np.concatenate([tone(1.0), silence(2.0)])
    right = np.concatenate([silence(2.0), tone(1.0)])
    artifact = store.import_recording(wav_bytes(np.stack([left, right], axis=1)))
    utterances = [e for e in artifact.timeline if isinstance(e, Utterance)]
    assert [u.speaker for u in utterances] == [Speaker.CALLER, Speaker.BOT]
    assert utterances[1].start_s == pytest.approx(2.0, abs=0.03)


def test_import_rejects_files_that_are_not_audio(store):
    with pytest.raises(ValueError, match="can't read audio"):
        store.import_recording(b"this is not audio")


def test_import_rejects_empty_audio(store):
    with pytest.raises(ValueError, match="no audio"):
        store.import_recording(wav_bytes(np.zeros(0, dtype=np.float32)))


def two_level_call() -> np.ndarray:
    """Loud first speaker and a quieter second, taking two turns each."""
    gap = silence(0.6)
    parts = [tone(1.0, amp=0.4), gap, tone(1.0, amp=0.1), gap, tone(1.0, amp=0.4), gap]
    return np.concatenate([*parts, tone(1.0, amp=0.1)])


def test_mono_import_splits_turns_by_level_with_first_speaker_as_bot(store):
    artifact = store.import_recording(wav_bytes(two_level_call()))
    event = next(e for e in artifact.timeline if getattr(e, "kind", "") == SEPARATION_EVENT)
    assert "not diarization" in event.detail
    assert artifact.providers["speaker_separation"] == "energy_level"
    speakers = [e.speaker for e in artifact.timeline if isinstance(e, Utterance)]
    assert speakers == [Speaker.BOT, Speaker.CALLER, Speaker.BOT, Speaker.CALLER]
    paths = store.paths(artifact.id)
    bot_regions = speech_regions(AudioClip.from_file(paths.bot))
    caller_regions = speech_regions(AudioClip.from_file(paths.caller))
    assert [round(r.start_s, 1) for r in bot_regions] == [0.0, 3.2]
    assert [round(r.start_s, 1) for r in caller_regions] == [1.6, 4.8]


def test_mono_at_one_level_falls_back_to_alternating_at_pauses():
    samples = np.concatenate([tone(1.0), silence(0.8), tone(1.0), silence(0.8), tone(1.0)])
    separation = EnergySeparator().separate(AudioClip.from_samples(samples, RATE))
    assert separation.method == "energy_alternation"
    speakers = [s.speaker for s in separation.segments]
    assert speakers == [Speaker.BOT, Speaker.CALLER, Speaker.BOT]


def test_short_pause_keeps_the_same_speaker_when_alternating():
    samples = np.concatenate([tone(1.0), silence(0.35), tone(1.0), silence(0.8), tone(1.0)])
    separation = EnergySeparator().separate(AudioClip.from_samples(samples, RATE))
    assert [s.speaker for s in separation.segments] == [Speaker.BOT, Speaker.BOT, Speaker.CALLER]


def test_mono_with_no_speech_produces_no_segments():
    separation = EnergySeparator().separate(AudioClip.silence(1.0, RATE))
    assert separation.segments == []


def test_stereo_with_identical_channels_is_treated_as_mono(store):
    mono = two_level_call()
    artifact = store.import_recording(wav_bytes(np.stack([mono, mono], axis=1)))
    assert "speaker_separation" in artifact.providers


def test_a_diarizer_can_replace_the_energy_separator(store):
    class Diarizer:
        def separate(self, audio: AudioClip) -> Separation:
            segments = [SpeakerSegment(Speaker.CALLER, 0.0, audio.duration_s)]
            return Separation(segments, "fake_diarizer", "a real model did this")

    artifact = store.import_recording(wav_bytes(tone(1.0)), separator=Diarizer())
    assert artifact.providers["speaker_separation"] == "fake_diarizer"
    paths = store.paths(artifact.id)
    assert np.abs(AudioClip.from_file(paths.caller).samples()).max() > 0.2
    assert np.abs(AudioClip.from_file(paths.bot).samples()).max() == 0


def test_speech_regions_on_noise_only_channel_finds_nothing():
    rng = np.random.default_rng(0)
    noise = (0.01 * rng.standard_normal(RATE * 3)).astype(np.float32)
    assert speech_regions(AudioClip.from_samples(noise, RATE)) == []


def test_speech_regions_track_speech_over_a_noise_floor():
    rng = np.random.default_rng(0)
    speech = np.concatenate([silence(1.0), tone(1.0, amp=0.3), silence(1.0)])
    mixed = speech + (0.005 * rng.standard_normal(len(speech))).astype(np.float32)
    regions = speech_regions(AudioClip.from_samples(mixed, RATE))
    assert len(regions) == 1
    assert regions[0].start_s == pytest.approx(1.0, abs=0.03)
    assert regions[0].end_s == pytest.approx(2.0, abs=0.03)


def test_speech_regions_merge_short_gaps_and_drop_blips():
    samples = np.concatenate(
        [tone(0.5), silence(0.1), tone(0.5), silence(0.5), tone(0.05), silence(0.5)]
    )
    regions = speech_regions(AudioClip.from_samples(samples, RATE))
    assert len(regions) == 1
    assert regions[0].end_s == pytest.approx(1.1, abs=0.03)
