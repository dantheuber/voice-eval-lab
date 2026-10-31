import io
import tracemalloc

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
from voicelab.store import artifacts
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


# Uploads are checked before decoding and read in bounded blocks ----------------------------------


def flac_file(tmp_path, seconds: float, *, rate: int = RATE, channels: int = 1) -> str:
    """Compressed silence: tiny on disk, large once decoded."""
    path = tmp_path / "upload.flac"
    data = np.zeros((round(seconds * rate), channels), dtype=np.float32)
    sf.write(str(path), data, rate, format="FLAC")
    return str(path)


def test_a_recording_over_the_length_limit_is_refused_before_decoding(tmp_path, monkeypatch):
    path = flac_file(tmp_path, 3.0)
    store = ArtifactStore(tmp_path, max_upload_s=2.0)

    def must_not_decode(*args, **kwargs):
        raise AssertionError("decoded audio before checking its size")

    monkeypatch.setattr(sf, "read", must_not_decode)
    monkeypatch.setattr(sf.SoundFile, "read", must_not_decode)
    with pytest.raises(ValueError, match="longer than the 2 s limit"):
        store.import_recording(path)
    assert not (tmp_path / "calls").exists()


def test_a_recording_with_too_many_decoded_samples_is_refused(tmp_path, monkeypatch):
    # Within the length limit, but eight channels at 96 kHz decode to far more than a phone call.
    path = flac_file(tmp_path, 1.0, rate=96000, channels=8)
    monkeypatch.setattr(artifacts, "MAX_DECODED_SAMPLES", 100_000)
    with pytest.raises(ValueError, match="decodes to too many samples"):
        ArtifactStore(tmp_path).import_recording(path)


def test_decoding_stops_if_a_file_runs_past_its_header(tmp_path, monkeypatch):
    path = flac_file(tmp_path, 1.0)

    class Lying:
        def __init__(self, inner):
            self._inner = inner
            self.samplerate, self.channels = inner.samplerate, inner.channels
            self.frames = 10  # claims to be tiny, and then keeps producing audio

        def read(self, *args, **kwargs):
            return self._inner.read(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._inner.close()

    real = sf.SoundFile
    monkeypatch.setattr(artifacts.sf, "SoundFile", lambda target: Lying(real(target)))
    with pytest.raises(ValueError, match="longer than the"):
        ArtifactStore(tmp_path, max_upload_s=0.5).import_recording(path)


def test_accepted_recordings_are_read_in_bounded_blocks(tmp_path, monkeypatch):
    path = flac_file(tmp_path, 30.0, rate=44100, channels=2)
    largest: list[int] = []
    real_read = sf.SoundFile.read

    def spying_read(self, frames=-1, *args, **kwargs):
        assert frames > 0, "unbounded read"
        largest.append(frames)
        return real_read(self, frames, *args, **kwargs)

    def must_not_decode(*args, **kwargs):
        raise AssertionError("whole-file decode")

    monkeypatch.setattr(sf, "read", must_not_decode)
    monkeypatch.setattr(sf.SoundFile, "read", spying_read)
    artifact = ArtifactStore(tmp_path).import_recording(path)
    assert artifact.duration_s == pytest.approx(30.0, abs=0.01)
    assert len(largest) > 1
    assert max(largest) * 2 <= artifacts.BLOCK_SAMPLES


def test_block_decoding_keeps_the_audio_intact(tmp_path):
    rate = 44100
    samples = np.concatenate([tone(9.0, 300, rate=rate), tone(9.0, 700, rate=rate)])
    stereo = np.stack([samples, samples[::-1]], axis=1)
    path = tmp_path / "long.flac"
    sf.write(str(path), stereo, rate, format="FLAC")
    store = ArtifactStore(tmp_path)
    artifact = store.import_recording(str(path))
    caller = AudioClip.from_file(store.paths(artifact.id).caller)
    expected = AudioClip.from_samples(samples, rate).resampled(RATE).samples()
    assert caller.sample_rate == RATE
    assert caller.n_samples == pytest.approx(len(expected), abs=2)
    n = min(caller.n_samples, len(expected))
    assert np.max(np.abs(caller.samples()[:n] - expected[:n])) < 0.02


def test_dual_mono_is_noticed_across_blocks(tmp_path):
    # Same audio on both sides until a difference far past the first block.
    rate = 16000
    base = tone(20.0, 300, rate=rate)
    other = base.copy()
    other[rate * 18 :] = 0
    path = tmp_path / "almost.flac"
    sf.write(str(path), np.stack([base, other], axis=1), rate, format="FLAC")
    artifact = ArtifactStore(tmp_path).import_recording(str(path))
    assert "speaker_separation" not in artifact.providers


@pytest.mark.parametrize("channels", [1, 2])
def test_import_memory_stays_near_the_size_of_the_output(tmp_path, channels):
    # Five minutes at 48 kHz decodes to 115 MB per channel as floats. The result is 9.6 MB of PCM
    # per channel at 16 kHz, so the peak has to reflect that, not the decoded size.
    seconds, rate = 300, 48000
    path = tmp_path / "long.flac"
    with sf.SoundFile(str(path), "w", rate, channels, format="FLAC") as flac:
        block = np.zeros((rate, channels), dtype=np.float32)
        block[::7] = 0.001
        for _ in range(seconds):
            flac.write(block)
    store = ArtifactStore(tmp_path)
    tracemalloc.start()
    artifact = store.import_recording(str(path))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert artifact.duration_s == pytest.approx(seconds, abs=0.01)
    assert peak < 60_000_000


def test_a_long_call_writes_without_whole_file_float_copies(tmp_path):
    seconds = 300
    clip = AudioClip.from_samples(np.full(RATE * seconds, 0.1, dtype=np.float32), RATE)
    store = ArtifactStore(tmp_path)
    tracemalloc.start()
    store.write_call(CallArtifact(mode=CallMode.HUMAN), clip, clip)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < len(clip.pcm)


def test_block_writing_keeps_stereo_and_peaks_exact(tmp_path):
    rate_samples = artifacts.BLOCK_SAMPLES + 12345
    left = np.linspace(-0.5, 0.5, rate_samples, dtype=np.float32)
    caller = AudioClip.from_samples(left, RATE)
    bot = AudioClip.from_samples(-left, RATE)
    store = ArtifactStore(tmp_path)
    paths = store.write_call(CallArtifact(mode=CallMode.HUMAN), caller, bot)
    stereo, rate = sf.read(str(paths.recording), dtype="int16")
    assert rate == RATE
    assert stereo[:, 0].tobytes() == caller.pcm
    assert stereo[:, 1].tobytes() == bot.pcm
    assert sf.read(str(paths.caller), dtype="int16")[0].tobytes() == caller.pcm
    peaks = store.read_peaks(paths.dir.name)["channels"]["caller"]
    size = RATE // artifacts.PEAKS_PER_SECOND
    samples = np.abs(caller.samples())
    expected = [round(float(samples[i : i + size].max()), 3) for i in range(0, len(samples), size)]
    assert peaks == pytest.approx(expected, abs=0.001)


def test_speech_regions_are_unchanged_across_a_block_boundary():
    # Speech straddling the internal block size has to read as one region.
    boundary_s = (1 << 18) / RATE
    samples = np.concatenate([silence(boundary_s - 1.0), tone(2.0), silence(1.0)])
    regions = speech_regions(AudioClip.from_samples(samples, RATE))
    assert len(regions) == 1
    assert regions[0].start_s == pytest.approx(boundary_s - 1.0, abs=0.03)
    assert regions[0].end_s == pytest.approx(boundary_s + 1.0, abs=0.03)
