from collections.abc import AsyncIterator

import numpy as np
import pytest

from voicelab.core.audio import AudioClip
from voicelab.core.models import LineConditions
from voicelab.telephony import LINE_RATE, PhoneLine, Transport, mulaw_decode, mulaw_encode
from voicelab.telephony.noise import make_noise

CLEAN_LINE = LineConditions(codec="mulaw_8k", noise="none", packet_loss_pct=0)


def tone(hz: float, seconds: float = 1.0, rate: int = 16000, amp: float = 0.3) -> AudioClip:
    t = np.arange(round(seconds * rate)) / rate
    return AudioClip.from_samples((amp * np.sin(2 * np.pi * hz * t)).astype(np.float32), rate)


def speech_like(seconds: float = 4.0, rate: int = 16000, amp: float = 0.2) -> AudioClip:
    """A 1 kHz tone gated on and off, so the line's level tracker has pauses to ignore."""
    t = np.arange(round(seconds * rate)) / rate
    gate = (np.sin(2 * np.pi * 2 * t) > -0.5).astype(np.float32)
    return AudioClip.from_samples(amp * gate * np.sin(2 * np.pi * 1000 * t), rate)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples))))


def stream(line: PhoneLine, clip: AudioClip, frame_s: float = 0.02) -> AudioClip:
    size = round(frame_s * clip.sample_rate) * 2
    out = AudioClip(b"", LINE_RATE)
    for i in range(0, len(clip.pcm), size):
        out = out + line.process_frame(AudioClip(clip.pcm[i : i + size], clip.sample_rate))
    return out + line.flush()


def test_mulaw_matches_g711_reference_points():
    assert mulaw_encode(np.array([0]))[0] == 0xFF
    assert mulaw_encode(np.array([32767]))[0] == 0x80
    assert mulaw_encode(np.array([-32768]))[0] == 0x00
    assert mulaw_decode(np.array([0xFF]))[0] == 0
    assert mulaw_decode(np.array([0x80]))[0] == 32124
    assert mulaw_decode(np.array([0x00]))[0] == -32124


def test_mulaw_codes_decode_to_themselves():
    codes = np.arange(256, dtype=np.uint8)
    roundtrip = mulaw_encode(mulaw_decode(codes))
    # 0x7F is negative zero, which decodes to 0 and re-encodes as positive zero (0xFF).
    assert np.array_equal(roundtrip[codes != 0x7F], codes[codes != 0x7F])


def test_mulaw_roundtrip_error_grows_with_level_and_stays_small():
    pcm = (np.sin(np.linspace(0, 200, 8000)) * 8000).astype(np.int16)
    decoded = mulaw_decode(mulaw_encode(pcm))
    error = pcm.astype(np.float64) - decoded
    snr_db = 20 * np.log10(rms(pcm.astype(np.float64)) / rms(error))
    assert snr_db > 30
    assert np.abs(decoded[np.abs(pcm) < 100] - pcm[np.abs(pcm) < 100]).max() <= 8


def test_output_is_8khz_with_the_same_duration():
    out = PhoneLine(CLEAN_LINE).process(tone(1000))
    assert out.sample_rate == LINE_RATE
    assert out.duration_s == pytest.approx(1.0, abs=0.001)


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 44100, 48000])
def test_tone_survives_resampling_at_common_rates(rate):
    out = PhoneLine(CLEAN_LINE).process(tone(1000, rate=rate)).samples()[800:]
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak_hz = np.argmax(spectrum) * LINE_RATE / len(out)
    assert peak_hz == pytest.approx(1000, abs=5)
    assert rms(out) == pytest.approx(0.3 / np.sqrt(2), rel=0.1)


@pytest.mark.parametrize("hz", [100, 3900])
def test_band_limit_removes_out_of_band_tones(hz):
    out = PhoneLine(CLEAN_LINE).process(tone(hz)).samples()[800:]
    assert rms(out) < (0.3 / np.sqrt(2)) * 10 ** (-15 / 20)


def test_band_limit_passes_mid_band():
    out = PhoneLine(CLEAN_LINE).process(tone(1500)).samples()[800:]
    assert 20 * np.log10(rms(out) / (0.3 / np.sqrt(2))) > -1


def test_codec_none_passes_through_unchanged():
    clip = tone(1000)
    line = PhoneLine(LineConditions(codec="none"))
    assert line.process(clip).pcm == clip.pcm
    assert line.process_frame(clip) is clip


def test_codec_none_still_applies_configured_noise_at_input_rate():
    clip = tone(1000)
    out = PhoneLine(LineConditions(codec="none", noise="office", snr_db=10)).process(clip)
    assert out.sample_rate == clip.sample_rate
    assert out.n_samples == clip.n_samples
    assert out.pcm != clip.pcm


def test_same_seed_gives_identical_output_and_different_seed_does_not():
    conditions = LineConditions(noise="cafe", snr_db=15, packet_loss_pct=5)
    clip = speech_like()
    first = PhoneLine(conditions, seed=7).process(clip)
    again = PhoneLine(conditions, seed=7).process(clip)
    other = PhoneLine(conditions, seed=8).process(clip)
    assert first.pcm == again.pcm
    assert first.pcm != other.pcm


@pytest.mark.parametrize("rate", [16000, 24000, 44100])
@pytest.mark.parametrize("frame_s", [0.01, 0.02, 0.03])
def test_streaming_equals_batch(rate, frame_s):
    conditions = LineConditions(noise="street", snr_db=20, packet_loss_pct=3)
    clip = speech_like(seconds=1.5, rate=rate)
    batch = PhoneLine(conditions, seed=3).process(clip)
    streamed = stream(PhoneLine(conditions, seed=3), clip, frame_s)
    assert streamed.n_samples == batch.n_samples
    assert streamed.pcm == batch.pcm


def test_each_20ms_frame_returns_one_20ms_packet():
    clip = tone(1000, seconds=0.2)
    line = PhoneLine(CLEAN_LINE)
    sizes = [
        line.process_frame(AudioClip(clip.pcm[i : i + 640], 16000)).n_samples
        for i in range(0, len(clip.pcm), 640)
    ]
    assert sizes == [160] * 10


def test_short_frames_wait_for_a_whole_packet():
    line = PhoneLine(CLEAN_LINE)
    first = line.process_frame(AudioClip(b"\x00\x00" * 160, 16000))
    second = line.process_frame(AudioClip(b"\x00\x00" * 160, 16000))
    assert (first.n_samples, second.n_samples) == (0, 160)


def test_successive_clips_get_fresh_noise():
    line = PhoneLine(LineConditions(noise="street", snr_db=10), seed=1)
    silence = AudioClip.silence(0.5, 16000)
    assert line.process(silence).pcm != line.process(silence).pcm


@pytest.mark.parametrize("kind", ["office", "street", "cafe", "car", "tv", "hum"])
@pytest.mark.parametrize("target_db", [10.0, 20.0])
def test_noise_snr_is_close_to_target(kind, target_db):
    conditions = LineConditions(noise=kind, snr_db=target_db)
    clip = speech_like(seconds=6.0)
    clean = PhoneLine(CLEAN_LINE, seed=2).process(clip).samples()
    noisy = PhoneLine(conditions, seed=2).process(clip).samples()
    # The target is relative to active speech, and the gate holds the tone on 2/3 of the time.
    speech_power = np.mean(np.square(clean)) / (2 / 3)
    noise_power = np.mean(np.square(noisy - clean))
    measured = 10 * np.log10(speech_power / noise_power)
    assert measured == pytest.approx(target_db, abs=1.5)


def test_noise_is_present_while_the_speaker_is_silent():
    conditions = LineConditions(noise="office", snr_db=15)
    out = PhoneLine(conditions, seed=1).process(AudioClip.silence(1.0, 16000)).samples()
    assert rms(out) > 0.001


def test_noise_level_follows_the_speech_level():
    conditions = LineConditions(noise="street", snr_db=15)
    quiet = speech_like(amp=0.05)
    loud = speech_like(amp=0.4)

    def noise_rms(clip: AudioClip) -> float:
        clean = PhoneLine(CLEAN_LINE, seed=4).process(clip).samples()
        return rms(PhoneLine(conditions, seed=4).process(clip).samples() - clean)

    assert noise_rms(loud) > 3 * noise_rms(quiet)


@pytest.mark.parametrize("kind", ["office", "street", "cafe", "car", "tv", "hum"])
def test_noise_loops_have_unit_rms_after_band_limiting(kind):
    rng = np.random.default_rng(0)
    loop = make_noise(kind, 8000, rng, band=(300, 3400), seconds=5.0)
    assert rms(loop) == pytest.approx(1.0, rel=1e-3)
    assert np.isfinite(loop).all()
    spectrum = np.abs(np.fft.rfft(loop))
    freqs = np.fft.rfftfreq(len(loop), 1 / 8000)
    assert spectrum[freqs < 290].max() < 1e-3 * spectrum.max()


def test_noise_kinds_sound_different():
    rng = np.random.default_rng(0)
    car = make_noise("car", 8000, rng, seconds=5.0)
    hum = make_noise("hum", 8000, rng, seconds=5.0)
    freqs = np.fft.rfftfreq(len(car), 1 / 8000)
    spectrum = np.abs(np.fft.rfft(hum))
    assert freqs[np.argmax(spectrum)] == pytest.approx(60, abs=1)
    car_spectrum = np.abs(np.fft.rfft(car)) ** 2
    assert car_spectrum[freqs < 500].sum() > 10 * car_spectrum[freqs > 2000].sum()


def test_unknown_noise_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown noise kind"):
        make_noise("rain", 8000, np.random.default_rng(0))


def test_packet_loss_rate_matches_configuration():
    line = PhoneLine(LineConditions(packet_loss_pct=10), seed=5)
    line.process(tone(1000, seconds=30))
    assert line.packets_sent == 1500
    assert line.packets_lost / line.packets_sent == pytest.approx(0.10, abs=0.02)


def test_no_packet_loss_means_no_lost_packets():
    line = PhoneLine(CLEAN_LINE)
    assert line.process(tone(1000)).n_samples == 8000
    assert line.packets_lost == 0


def test_concealment_fades_a_long_gap_to_silence():
    # At 30% loss, runs of 4 or more lost packets happen often enough over a minute of audio.
    conditions = LineConditions(packet_loss_pct=30)
    clip = tone(1000, seconds=60)
    clean = PhoneLine(CLEAN_LINE).process(clip).samples()
    lossy = PhoneLine(conditions, seed=1).process(clip).samples()
    silent_packets = [
        i for i in range(len(lossy) // 160) if rms(lossy[i * 160 : (i + 1) * 160]) < 1e-4
    ]
    assert silent_packets, "expected at least one packet past the fade"
    assert rms(clean[400:]) > 0.1


def test_single_lost_packet_is_concealed_rather_than_silent():
    conditions = LineConditions(packet_loss_pct=5)
    line = PhoneLine(conditions, seed=2)
    out = line.process(tone(1000, seconds=10)).samples()
    assert line.packets_lost > 0
    packet_levels = [rms(out[i * 160 : (i + 1) * 160]) for i in range(5, len(out) // 160)]
    assert sum(level < 1e-4 for level in packet_levels) < line.packets_lost / 2


def test_latency_is_reported_from_conditions():
    assert PhoneLine(LineConditions(one_way_latency_ms=150)).latency_s == pytest.approx(0.15)


def test_flush_on_an_unused_line_returns_empty():
    assert PhoneLine(CLEAN_LINE).flush().n_samples == 0


def test_a_transport_can_be_implemented_from_the_protocol():
    class Loopback:
        sample_rate = 8000

        async def send(self, frame: AudioClip) -> None: ...

        def receive(self) -> AsyncIterator[AudioClip]: ...

        async def hang_up(self) -> None: ...

    assert isinstance(Loopback(), Transport)
