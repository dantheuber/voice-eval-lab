"""Simulated phone line: what audio goes through between the two ends of a call.

The stages follow a real call. Room noise joins the speech before the codec, so quantization and
packet loss hit both. Audio moves through the line in 20 ms packets, as RTP would carry it, and
every stage is stateful, so a whole clip and the same clip fed in frames give identical output.
"""

from __future__ import annotations

import math

import numpy as np

from voicelab.core.audio import AudioClip
from voicelab.core.models import LineConditions
from voicelab.telephony.noise import make_noise

LINE_RATE = 8000
PACKET_S = 0.02
BAND_HZ = (300.0, 3400.0)

# Lost packets fade to silence over three packets, the 60 ms G.711 Appendix I allows for.
_FADE_PER_PACKET = 1 / 3

# Before any speech is heard, noise is scaled against a typical active speech level. The weight
# is how many packets that assumption counts for against the real level measured afterward.
_NOMINAL_SPEECH_RMS = 10 ** (-26 / 20)
_NOMINAL_WEIGHT = 5
_LEVEL_GATE_DB = 16.0  # packets this far under the running level are pauses, as in ITU P.56
_LEVEL_FLOOR_RMS = 10 ** (-50 / 20)


# G.711 mu-law ------------------------------------------------------------------------------------

_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635


def mulaw_encode(pcm: np.ndarray) -> np.ndarray:
    """G.711 mu-law: 16-bit linear samples to 8-bit codes."""
    linear = np.asarray(pcm, dtype=np.int32)
    sign = (linear < 0).astype(np.int32) << 7
    magnitude = np.minimum(np.abs(linear), _MULAW_CLIP) + _MULAW_BIAS
    exponent = np.floor(np.log2(magnitude)).astype(np.int32) - 7
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """G.711 mu-law: 8-bit codes to 16-bit linear samples."""
    inverted = ~np.asarray(codes, dtype=np.int32) & 0xFF
    exponent = (inverted >> 4) & 0x07
    mantissa = inverted & 0x0F
    magnitude = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    return np.where(inverted & 0x80, -magnitude, magnitude).astype(np.int16)


# Line stages -------------------------------------------------------------------------------------


class _Resampler:
    """Streaming low-pass and interpolate.

    soxr's stream holds back 50 to 100 ms and releases audio in bursts of about 500 samples, which
    a 20 ms live line can't absorb. This one has a fixed 4 ms delay and emits on every call.
    """

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        half = round(in_rate * 0.004)
        t = np.arange(-half, half + 1)
        cutoff = 0.475 * min(in_rate, out_rate) / in_rate  # cycles per sample, just under Nyquist
        taps = 2 * cutoff * np.sinc(2 * cutoff * t) * np.hanning(len(t))
        self._taps = taps / taps.sum()
        self._history = np.zeros(len(t) - 1)
        self._last = 0.0
        self._consumed = 0
        self._produced = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self.in_rate == self.out_rate:
            return samples
        if len(samples) == 0:
            return np.zeros(0)
        joined = np.concatenate([self._history, samples])
        filtered = np.concatenate([[self._last], np.convolve(joined, self._taps, "valid")])
        self._history = joined[-len(self._history) :]
        self._last = float(filtered[-1])

        total = self._consumed + len(samples)
        wanted = (total - 1) * self.out_rate // self.in_rate + 1
        # Integer math, because float positions drift over a long call.
        outputs = np.arange(self._produced, wanted, dtype=np.int64)
        whole = outputs * self.in_rate // self.out_rate
        fraction = (outputs * self.in_rate % self.out_rate) / self.out_rate
        index = whole - self._consumed + 1  # +1 for the sample carried over in `filtered`
        following = np.minimum(index + 1, len(filtered) - 1)
        self._consumed = total
        self._produced = wanted
        return filtered[index] * (1 - fraction) + filtered[following] * fraction


class _BandLimit:
    """Fourth-order Butterworth high-pass and low-pass, as cascaded biquads.

    A sample loop, since numpy has no IIR filter and scipy isn't a dependency. Measured on this
    machine, the whole line costs about 0.1 ms per 20 ms frame.
    """

    def __init__(self, rate: int, low_hz: float, high_hz: float) -> None:
        self._sections: list[tuple[float, float, float, float, float]] = []
        for q in (0.5412, 1.3066):
            self._sections.append(_biquad(rate, low_hz, q, highpass=True))
            self._sections.append(_biquad(rate, high_hz, q, highpass=False))
        self._state = [[0.0, 0.0] for _ in self._sections]

    def process(self, samples: np.ndarray) -> np.ndarray:
        values = samples.tolist()
        for (b0, b1, b2, a1, a2), state in zip(self._sections, self._state, strict=True):
            z1, z2 = state
            for i, x in enumerate(values):
                y = b0 * x + z1
                z1 = b1 * x - a1 * y + z2
                z2 = b2 * x - a2 * y
                values[i] = y
            state[0], state[1] = z1, z2
        return np.array(values)


def _biquad(
    rate: int, hz: float, q: float, *, highpass: bool
) -> tuple[float, float, float, float, float]:
    """Normalized (b0, b1, b2, a1, a2) from the RBJ audio EQ cookbook."""
    w0 = 2 * math.pi * hz / rate
    cos_w0, alpha = math.cos(w0), math.sin(w0) / (2 * q)
    edge = (1 + cos_w0) / 2 if highpass else (1 - cos_w0) / 2
    b1 = -2 * edge if highpass else 2 * edge
    a0 = 1 + alpha
    return edge / a0, b1 / a0, edge / a0, -2 * cos_w0 / a0, (1 - alpha) / a0


class _SpeechLevel:
    """Running active-speech RMS. Noise is set relative to it, and live calls can't look ahead."""

    def __init__(self) -> None:
        self._energy = _NOMINAL_SPEECH_RMS**2 * _NOMINAL_WEIGHT
        self._count = _NOMINAL_WEIGHT

    @property
    def rms(self) -> float:
        return math.sqrt(self._energy / self._count)

    def update(self, packet: np.ndarray) -> None:
        power = float(np.mean(np.square(packet)))
        gate = max(_LEVEL_FLOOR_RMS, self.rms * 10 ** (-_LEVEL_GATE_DB / 20))
        if power > gate**2:
            self._energy += power
            self._count += 1


class _Concealer:
    """Drops packets and hides the gap by repeating the last good packet while fading out."""

    def __init__(self, loss_pct: float, rng: np.random.Generator) -> None:
        self._probability = loss_pct / 100
        self._rng = rng
        self._last = np.zeros(1)
        self._run = 0
        self.sent = 0
        self.lost = 0

    def process(self, packet: np.ndarray) -> np.ndarray:
        self.sent += 1
        if self._rng.random() >= self._probability:
            self._last = packet
            self._run = 0
            return packet
        self.lost += 1
        start = max(0.0, 1 - self._run * _FADE_PER_PACKET)
        end = max(0.0, 1 - (self._run + 1) * _FADE_PER_PACKET)
        self._run += 1
        return np.resize(self._last, len(packet)) * np.linspace(start, end, len(packet))


# The line ----------------------------------------------------------------------------------------


class PhoneLine:
    """One direction of a call. Use a separate instance for each direction.

    `process_frame` is for live audio and returns whole 20 ms packets as they complete, so 20 ms in
    gives 20 ms out, and a frame that doesn't complete a packet returns an empty clip. `process`
    is for finished clips. Both share state, so a sequence of clips hears continuous noise rather
    than the same noise restarted each time, and two lines with one seed produce the same output
    for the same calls.

    Output is 8 kHz under `mulaw_8k`. `codec="none"` skips resampling, band-limiting and companding
    and keeps the input rate. Noise and packet loss are separate settings and still apply, so with
    no noise and no loss, `codec="none"` returns its input unchanged.

    The line doesn't delay audio itself: `latency_s` reports what the caller should schedule.
    """

    def __init__(self, conditions: LineConditions | None = None, seed: int = 0) -> None:
        self.conditions = conditions or LineConditions()
        self.seed = seed
        noise_seed, loss_seed = np.random.SeedSequence(seed).spawn(2)
        self._noise_rng = np.random.default_rng(noise_seed)
        self._concealer = _Concealer(
            self.conditions.packet_loss_pct, np.random.default_rng(loss_seed)
        )
        self._level = _SpeechLevel()
        self._band_limit = _BandLimit(LINE_RATE, *BAND_HZ) if self._uses_codec else None
        self._resampler: _Resampler | None = None
        self._rate: int | None = None
        self._noise: np.ndarray | None = None
        self._noise_position = 0
        self._pending = np.zeros(0)

    @property
    def _uses_codec(self) -> bool:
        return self.conditions.codec == "mulaw_8k"

    @property
    def _passthrough(self) -> bool:
        c = self.conditions
        return not self._uses_codec and c.noise == "none" and c.packet_loss_pct == 0

    @property
    def latency_s(self) -> float:
        """One-way delay for this line. The transport or call engine applies it."""
        return self.conditions.one_way_latency_ms / 1000

    @property
    def packets_sent(self) -> int:
        return self._concealer.sent

    @property
    def packets_lost(self) -> int:
        return self._concealer.lost

    def process(self, clip: AudioClip) -> AudioClip:
        """Degrade a finished clip."""
        return self.process_frame(clip) + self.flush()

    def process_frame(self, frame: AudioClip) -> AudioClip:
        """Degrade one streaming frame, returning the packets it completed."""
        if self._passthrough:
            return frame
        self._prepare(frame.sample_rate)
        resampled = self._resampler.process(frame.samples().astype(np.float64))
        self._pending = np.concatenate([self._pending, resampled])
        size = round(self._rate * PACKET_S)
        count = len(self._pending) // size
        chunks = [self._pending[i * size : (i + 1) * size] for i in range(count)]
        packets = [self._process_packet(chunk) for chunk in chunks]
        self._pending = self._pending[count * size :]
        return self._to_clip(packets)

    def flush(self) -> AudioClip:
        """Emit the partial packet still held. Call at the end of a clip or stream."""
        if self._passthrough or self._rate is None:
            return AudioClip(b"", self._rate or LINE_RATE)
        packets = [self._process_packet(self._pending)] if len(self._pending) else []
        self._pending = np.zeros(0)
        return self._to_clip(packets)

    def _prepare(self, input_rate: int) -> None:
        output_rate = LINE_RATE if self._uses_codec else input_rate
        if self._resampler is None or self._resampler.in_rate != input_rate:
            self._resampler = _Resampler(input_rate, output_rate)
        if output_rate != self._rate:
            self._rate = output_rate
            self._noise = self._render_noise(output_rate)

    def _render_noise(self, rate: int) -> np.ndarray | None:
        if self.conditions.noise == "none":
            return None
        band = BAND_HZ if self._uses_codec else None
        noise = make_noise(self.conditions.noise, rate, self._noise_rng, band=band)
        self._noise_position = int(self._noise_rng.integers(0, len(noise)))
        return noise

    def _process_packet(self, packet: np.ndarray) -> np.ndarray:
        if self._band_limit is not None:
            packet = self._band_limit.process(packet)
        if self._noise is not None:
            packet = packet + self._next_noise(packet)
        if self._uses_codec:
            pcm = np.round(np.clip(packet, -1.0, 1.0) * 32767).astype(np.int16)
            packet = mulaw_decode(mulaw_encode(pcm)) / 32768.0
        if self.conditions.packet_loss_pct > 0:
            packet = self._concealer.process(packet)
        return packet

    def _next_noise(self, packet: np.ndarray) -> np.ndarray:
        self._level.update(packet)
        indices = np.arange(self._noise_position, self._noise_position + len(packet))
        self._noise_position = int(indices[-1] + 1) % len(self._noise)
        gain = self._level.rms / 10 ** (self.conditions.snr_db / 20)
        return self._noise.take(indices, mode="wrap") * gain

    def _to_clip(self, packets: list[np.ndarray]) -> AudioClip:
        joined = np.concatenate(packets) if packets else np.zeros(0)
        return AudioClip.from_samples(joined, self._rate)
