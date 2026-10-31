"""Call artifacts on disk, one directory per call.

    data/calls/<call_id>/
        recording.wav   stereo, caller left, bot right
        caller.wav      caller channel
        bot.wav         bot channel
        call.json       the CallArtifact
        peaks.json      downsampled amplitude per channel, for waveform rendering
        eval.json       the EvalResult, once evaluated

All audio here is 16 kHz, and every WAV is the same length so timestamps line up across files.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol

import numpy as np
import soundfile as sf
import soxr

from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    EvalResult,
    Speaker,
    SystemEvent,
    TimelineEvent,
    Utterance,
    utcnow,
)
from voicelab.eval.vad import Region, speech_regions

CALL_SAMPLE_RATE = 16000
PEAKS_PER_SECOND = 50

# Compressed audio can decode to thousands of times its size, so an upload is judged by what it
# decodes to, from its header, before any of it is decoded.
MAX_UPLOAD_S = 2 * 3600.0
MAX_DECODED_SAMPLES = round(MAX_UPLOAD_S) * 48000 * 2
BLOCK_SAMPLES = 1 << 18

# A SystemEvent of this kind marks channels that were inferred from a mono mix, not recorded apart.
SEPARATION_EVENT = "speaker_separation"

Channel = Literal["caller", "bot", "stereo"]

_SAFE_CALL_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class CallPaths:
    dir: Path

    @property
    def recording(self) -> Path:
        return self.dir / "recording.wav"

    @property
    def caller(self) -> Path:
        return self.dir / "caller.wav"

    @property
    def bot(self) -> Path:
        return self.dir / "bot.wav"

    @property
    def call_json(self) -> Path:
        return self.dir / "call.json"

    @property
    def peaks(self) -> Path:
        return self.dir / "peaks.json"

    @property
    def eval_json(self) -> Path:
        return self.dir / "eval.json"


# Speaker separation ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeakerSegment:
    speaker: Speaker
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Separation:
    segments: list[SpeakerSegment]
    method: str  # recorded in the artifact as the "speaker_separation" provider
    note: str  # what was done and how far to trust it, shown with the call


class SpeakerSeparator(Protocol):
    """Decides who spoke when in a mono mix. A diarizer plugs in here, returning segments."""

    def separate(self, audio: AudioClip) -> Separation: ...


class EnergySeparator:
    """Best-effort split of a mono call into turns. This is not diarization.

    Finds speech regions, then assigns each whole region to one speaker, so overlapping talk lands
    on a single channel. If regions fall into two loudness groups at least `min_level_gap_db`
    apart, which is common when one party comes through a phone line, the groups are the speakers.
    Otherwise it assumes people take turns and swaps speaker at each pause of `turn_pause_s` or
    more. The first speaker is labeled bot, because support agents answer with a greeting.

    Labels can be wrong or swapped, and overlap can't be measured from the result.
    """

    def __init__(self, *, min_level_gap_db: float = 4.0, turn_pause_s: float = 0.5) -> None:
        self.min_level_gap_db = min_level_gap_db
        self.turn_pause_s = turn_pause_s

    def separate(self, audio: AudioClip) -> Separation:
        regions = speech_regions(audio, min_gap_s=0.3)
        if not regions:
            return Separation([], "energy_none", "No speech found in the mono recording.")
        levels = [_level_db(audio.slice(r.start_s, r.end_s)) for r in regions]
        groups = self._level_groups(levels)
        if groups is not None:
            first_group = groups[0]
            speakers = [Speaker.BOT if g == first_group else Speaker.CALLER for g in groups]
            method, how = "energy_level", "two loudness groups"
        else:
            speakers = self._alternate(regions)
            method, how = "energy_alternation", "assumed turn-taking at pauses"
        segments = [
            SpeakerSegment(speaker, r.start_s, r.end_s)
            for speaker, r in zip(speakers, regions, strict=True)
        ]
        note = (
            f"Mono recording split into turns by {how}, with the first speaker labeled bot. "
            "This is not diarization: labels may be wrong or swapped, and overlap can't be "
            "measured."
        )
        return Separation(segments, method, note)

    def _level_groups(self, levels: list[float]) -> list[int] | None:
        values = np.array(levels)
        low, high = float(values.min()), float(values.max())
        for _ in range(20):  # two-means settles in a few rounds on one dimension
            is_high = np.abs(values - high) < np.abs(values - low)
            if not is_high.any() or is_high.all():
                return None
            low, high = float(values[~is_high].mean()), float(values[is_high].mean())
        if high - low < self.min_level_gap_db:
            return None
        return is_high.astype(int).tolist()

    def _alternate(self, regions: list[Region]) -> list[Speaker]:
        speaker = Speaker.BOT
        speakers = [speaker]
        for previous, region in zip(regions, regions[1:], strict=False):
            if region.start_s - previous.end_s >= self.turn_pause_s:
                speaker = Speaker.CALLER if speaker == Speaker.BOT else Speaker.BOT
            speakers.append(speaker)
        return speakers


def _level_db(clip: AudioClip) -> float:
    return float(10 * np.log10(np.mean(np.square(clip.samples())) + 1e-12))


# Store -------------------------------------------------------------------------------------------


class ArtifactStore:
    def __init__(
        self, data_dir: str | Path = "data", *, max_upload_s: float = MAX_UPLOAD_S
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_upload_s = max_upload_s

    def paths(self, call_id: str) -> CallPaths:
        # Call ids arrive from URLs, and this is what keeps them inside the data directory.
        if not _SAFE_CALL_ID.match(call_id):
            raise ValueError(f"invalid call id {call_id!r}")
        return CallPaths(self.data_dir / "calls" / call_id)

    def write_call(self, artifact: CallArtifact, caller: AudioClip, bot: AudioClip) -> CallPaths:
        """Write the five files of a finished call. Channels are resampled and padded to match."""
        paths = self.paths(artifact.id)
        paths.dir.mkdir(parents=True, exist_ok=True)
        caller, bot = _matched_channels(caller, bot)
        _write_stereo(paths.recording, caller, bot)
        _write_mono(paths.caller, caller)
        _write_mono(paths.bot, bot)
        _write_json(paths.peaks, _peaks(caller, bot))
        self.write_artifact(artifact)
        return paths

    def write_artifact(self, artifact: CallArtifact) -> None:
        paths = self.paths(artifact.id)
        paths.dir.mkdir(parents=True, exist_ok=True)
        _write_text(paths.call_json, artifact.model_dump_json(indent=2))

    def read_artifact(self, call_id: str) -> CallArtifact:
        return CallArtifact.model_validate_json(self.paths(call_id).call_json.read_text())

    def write_eval(self, result: EvalResult) -> None:
        _write_text(self.paths(result.call_id).eval_json, result.model_dump_json(indent=2))

    def read_eval(self, call_id: str) -> EvalResult | None:
        path = self.paths(call_id).eval_json
        return EvalResult.model_validate_json(path.read_text()) if path.exists() else None

    def read_peaks(self, call_id: str) -> dict[str, Any]:
        return json.loads(self.paths(call_id).peaks.read_text())

    def clip_wav(
        self, call_id: str, start_s: float, end_s: float, channel: Channel = "stereo"
    ) -> bytes:
        """Cut a time range as WAV bytes. The range is clamped to the call."""
        paths = self.paths(call_id)
        source = {"stereo": paths.recording, "caller": paths.caller, "bot": paths.bot}[channel]
        with sf.SoundFile(str(source)) as wav:
            first = min(max(0, round(start_s * wav.samplerate)), wav.frames)
            last = min(max(first, round(end_s * wav.samplerate)), wav.frames)
            if last <= first:
                raise ValueError(f"empty clip range {start_s} to {end_s}")
            wav.seek(first)
            samples = wav.read(last - first, dtype="float32", always_2d=True)
            rate = wav.samplerate
        buf = io.BytesIO()
        sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def import_recording(
        self,
        source: str | Path | bytes | BinaryIO,
        *,
        filename: str = "",
        separator: SpeakerSeparator | None = None,
        swap_channels: bool = False,
    ) -> CallArtifact:
        """Turn an uploaded recording into an artifact.

        Stereo files are taken as caller left, bot right, unless `swap_channels` is set. A mono
        file, or a stereo file with the same audio on both sides, goes through `separator`. The
        result is recorded as a "speaker_separation" SystemEvent so evaluators can tell inferred
        channels from recorded ones.

        Raises ValueError for anything that isn't audio, or that decodes past the upload limits.
        """
        left, right = _decode_upload(source, self.max_upload_s)
        artifact = CallArtifact(
            mode=CallMode.UPLOAD,
            has_tool_log=False,
            has_intended_text=False,
            started_at=None,
            ended_at=utcnow(),
            timeline=[SystemEvent(at_s=0.0, kind="upload_imported", detail=filename)],
        )
        if right is not None:
            caller, bot = (right, left) if swap_channels else (left, right)
            events: list[TimelineEvent] = [
                *_utterances(Speaker.CALLER, speech_regions(caller)),
                *_utterances(Speaker.BOT, speech_regions(bot)),
            ]
        else:
            separation = (separator or EnergySeparator()).separate(left)
            caller, bot = _split_channels(left, separation)
            artifact.providers["speaker_separation"] = separation.method
            events = [SystemEvent(at_s=0.0, kind=SEPARATION_EVENT, detail=separation.note)]
            for segment in separation.segments:
                events.extend(
                    _utterances(segment.speaker, [Region(segment.start_s, segment.end_s)])
                )
        artifact.timeline.extend(sorted(events, key=_event_time))
        artifact.duration_s = max(caller.duration_s, bot.duration_s)
        self.write_call(artifact, caller, bot)
        return artifact


def _event_time(event: TimelineEvent) -> float:
    return event.start_s if isinstance(event, Utterance) else event.at_s


def _utterances(speaker: Speaker, regions: list[Region]) -> list[Utterance]:
    return [Utterance(speaker=speaker, start_s=r.start_s, end_s=r.end_s) for r in regions]


class _CallChannel:
    """One channel, resampled to the call rate as blocks arrive."""

    def __init__(self, rate: int) -> None:
        self._resampler = (
            None
            if rate == CALL_SAMPLE_RATE
            else soxr.ResampleStream(rate, CALL_SAMPLE_RATE, 1, dtype="float32", quality="HQ")
        )
        self._pcm = bytearray()

    def add(self, block: np.ndarray, *, last: bool = False) -> None:
        if self._resampler is not None:
            block = self._resampler.resample_chunk(block, last=last)
        self._pcm += AudioClip.from_samples(block, CALL_SAMPLE_RATE).pcm

    def finish(self) -> AudioClip:
        self.add(np.zeros(0, dtype=np.float32), last=True)
        return AudioClip(bytes(self._pcm), CALL_SAMPLE_RATE)


def _check_upload_size(frames: int, rate: int, channels: int, max_seconds: float) -> None:
    if frames > max_seconds * rate:
        raise ValueError(f"recording is longer than the {max_seconds:g} s limit")
    if frames * channels > MAX_DECODED_SAMPLES:
        raise ValueError(f"recording decodes to too many samples, over {MAX_DECODED_SAMPLES}")


def _decode_upload(
    source: str | Path | bytes | BinaryIO, max_seconds: float
) -> tuple[AudioClip, AudioClip | None]:
    """Decode to call-rate channels in fixed blocks. The second is None for one-channel audio."""
    target = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        with sf.SoundFile(target) as audio:
            return _decode_blocks(audio, max_seconds)
    except sf.LibsndfileError as exc:
        raise ValueError(f"can't read audio: {exc}") from exc


def _decode_blocks(audio: sf.SoundFile, max_seconds: float) -> tuple[AudioClip, AudioClip | None]:
    rate, channels = audio.samplerate, audio.channels
    _check_upload_size(audio.frames, rate, channels, max_seconds)
    left = _CallChannel(rate)
    right = _CallChannel(rate) if channels >= 2 else None
    block_frames = max(1, BLOCK_SAMPLES // channels)
    decoded = 0
    same_on_both_sides = True
    while True:
        block = audio.read(block_frames, dtype="float32", always_2d=True)
        if len(block) == 0:
            break
        decoded += len(block)
        # Headers can understate the audio, so the limits hold while decoding as well.
        _check_upload_size(decoded, rate, channels, max_seconds)
        left.add(block[:, 0])
        if right is not None:
            right.add(block[:, 1])
            same_on_both_sides = same_on_both_sides and _is_dual_mono(block)
    if decoded == 0:
        raise ValueError("recording has no audio")
    return left.finish(), None if right is None or same_on_both_sides else right.finish()


def _is_dual_mono(block: np.ndarray) -> bool:
    return bool(np.max(np.abs(block[:, 0] - block[:, 1])) < 1e-3)


def _pcm(clip: AudioClip) -> np.ndarray:
    """The clip's samples without copying them. Long calls can't afford a float copy per step."""
    return np.frombuffer(clip.pcm, dtype=np.int16)


def _split_channels(mono: AudioClip, separation: Separation) -> tuple[AudioClip, AudioClip]:
    samples = _pcm(mono)
    channels = {Speaker.CALLER: np.zeros_like(samples), Speaker.BOT: np.zeros_like(samples)}
    for segment in separation.segments:
        start = round(segment.start_s * mono.sample_rate)
        end = round(segment.end_s * mono.sample_rate)
        channels[segment.speaker][start:end] = samples[start:end]
    return (
        AudioClip(channels[Speaker.CALLER].tobytes(), mono.sample_rate),
        AudioClip(channels[Speaker.BOT].tobytes(), mono.sample_rate),
    )


def _write_mono(path: Path, clip: AudioClip) -> None:
    sf.write(str(path), _pcm(clip), clip.sample_rate, format="WAV", subtype="PCM_16")


def _write_stereo(path: Path, left: AudioClip, right: AudioClip) -> None:
    a, b = _pcm(left), _pcm(right)
    with sf.SoundFile(str(path), "w", left.sample_rate, 2, format="WAV", subtype="PCM_16") as wav:
        for start in range(0, len(a), BLOCK_SAMPLES):
            end = start + BLOCK_SAMPLES
            wav.write(np.stack([a[start:end], b[start:end]], axis=1))


def _matched_channels(caller: AudioClip, bot: AudioClip) -> tuple[AudioClip, AudioClip]:
    caller = caller.resampled(CALL_SAMPLE_RATE)
    bot = bot.resampled(CALL_SAMPLE_RATE)
    length = max(len(caller.pcm), len(bot.pcm))
    return (
        AudioClip(caller.pcm.ljust(length, b"\x00"), CALL_SAMPLE_RATE),
        AudioClip(bot.pcm.ljust(length, b"\x00"), CALL_SAMPLE_RATE),
    )


def _peaks(caller: AudioClip, bot: AudioClip) -> dict[str, Any]:
    return {
        "peaks_per_second": PEAKS_PER_SECOND,
        "duration_s": caller.duration_s,
        "channels": {"caller": _channel_peaks(caller), "bot": _channel_peaks(bot)},
    }


def _channel_peaks(clip: AudioClip) -> list[float]:
    """Peak absolute amplitude per bin, rounded because JSON size drives dashboard load time."""
    size = clip.sample_rate // PEAKS_PER_SECOND
    samples = _pcm(clip)
    step = size * max(1, BLOCK_SAMPLES // size)  # whole bins, so none straddles two blocks
    peaks: list[float] = []
    for start in range(0, len(samples), step):
        block = np.abs(samples[start : start + step].astype(np.int32))
        count = int(np.ceil(len(block) / size))
        bins = np.zeros(count * size, dtype=np.int32)
        bins[: len(block)] = block
        peaks.extend(round(float(p) / 32768.0, 3) for p in bins.reshape(count, size).max(axis=1))
    return peaks


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, separators=(",", ":")))


def _write_text(path: Path, text: str) -> None:
    # The dashboard polls these files while eval runs, so readers must never see half a file.
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)
