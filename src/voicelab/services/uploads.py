"""Outside recordings: import, optional context, and the mono channel-map flow.

`ArtifactStore.import_recording` makes the artifact. This adds what a person supplied alongside the
file, and keeps the original so that a different channel map can re-import it without an upload.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Literal

import soundfile as sf

from voicelab.core.audio import AudioClip
from voicelab.core.models import CallArtifact, CallMode, Speaker, ToolCallEvent, Utterance
from voicelab.services.errors import Conflict, Invalid, NotFound
from voicelab.store.artifacts import (
    SEPARATION_EVENT,
    ArtifactStore,
    EnergySeparator,
    Separation,
    SpeakerSegment,
)
from voicelab.store.calls import CallIndex
from voicelab.store.catalog import Catalog

ALLOWED_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
ChannelMap = Literal["caller_left", "caller_right", "mono"]
_MOVED_FILES = ("recording.wav", "caller.wav", "bot.wav", "peaks.json")
_DERIVED_FILES = ("eval.json", "speech.json")
_ROUGH_EVAL_RATE = 0.5  # eval seconds per second of audio, until uploads have a measured figure
_ROUGH_EVAL_BASE_S = 15.0


@dataclass
class Probe:
    duration_s: float
    channels: int
    sample_rate: int
    channel_map: str
    size_bytes: int


def safe_filename(filename: str) -> str:
    """Browsers send a bare name. A path here is an attack or a bug, not a file to tidy up."""
    if not filename or any(mark in filename for mark in ("/", "\\", "\x00", "..")):
        raise Invalid(f"bad file name {filename!r}")
    if not filename.lower().endswith(ALLOWED_SUFFIXES):
        raise Invalid(f"unsupported file type, use one of {', '.join(ALLOWED_SUFFIXES)}")
    return filename


def parse_intended(text: str) -> list[tuple[str, float | None]]:
    """One bot turn per line, or JSON `[{ "text", "startS"? }]`."""
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            items = json.loads(text)
            return [(str(item["text"]), item.get("startS")) for item in items]
        except (ValueError, KeyError, TypeError) as exc:
            raise Invalid(f"intendedTranscript isn't valid: {exc}") from exc
    return [(line.strip(), None) for line in text.splitlines() if line.strip()]


def parse_tool_log(text: str) -> list[ToolCallEvent]:
    """One JSON tool call per line."""
    events = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            events.append(
                ToolCallEvent(
                    at_s=float(item.get("atS", item.get("at_s", 0.0))),
                    name=item["name"],
                    arguments=item.get("arguments") or {},
                    result=item.get("result"),
                    error=item.get("error"),
                    duration_ms=item.get("durationMs", item.get("duration_ms")),
                )
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise Invalid(f"toolLog line {number} isn't a tool call: {exc}") from exc
    return events


def apply_intended(artifact: CallArtifact, lines: list[tuple[str, float | None]]) -> None:
    """Attach intended text to the bot's turns: by time when given, otherwise in order. Turns here
    come from VAD, so their count won't match the script, and spare lines join the last turn."""
    turns = [e for e in artifact.timeline if isinstance(e, Utterance) and e.speaker == Speaker.BOT]
    if not turns or not lines:
        return
    next_turn = 0
    for text, start_s in lines:
        if start_s is not None:
            turn = min(turns, key=lambda t: abs(t.start_s - start_s))
        else:
            turn = turns[min(next_turn, len(turns) - 1)]
            next_turn += 1
        turn.intended_text = f"{turn.intended_text} {text}" if turn.intended_text else text
    artifact.has_intended_text = True


def _event_time(event: Any) -> float:
    return event.start_s if isinstance(event, Utterance) else event.at_s


class _SwappedLabels:
    """A mono split has no true left or right, so "caller on the right" swaps who is who."""

    def __init__(self) -> None:
        self.inner = EnergySeparator()

    def separate(self, audio: AudioClip) -> Separation:
        found = self.inner.separate(audio)
        other = {Speaker.CALLER: Speaker.BOT, Speaker.BOT: Speaker.CALLER}
        segments = [SpeakerSegment(other[s.speaker], s.start_s, s.end_s) for s in found.segments]
        return Separation(segments, found.method, f"{found.note} Labels swapped by the user.")


class UploadService:
    def __init__(self, store: ArtifactStore, calls: CallIndex, catalog: Catalog) -> None:
        self.store = store
        self.calls = calls
        self.catalog = catalog

    def create(
        self,
        data: bytes,
        filename: str,
        *,
        profile: str | None,
        scenario_id: str | None = None,
        bot_version: str | None = None,
        intended: str = "",
        tool_log: str = "",
    ) -> tuple[CallArtifact, Probe]:
        filename = safe_filename(filename)
        context = {
            "filename": filename,
            "scenario_id": scenario_id or None,
            "bot_version": bot_version or None,
            "intended": intended,
            "tool_log": tool_log,
        }
        # Checked before anything lands on disk.
        self._apply_context(CallArtifact(mode=CallMode.UPLOAD), context)
        try:
            info = sf.info(io.BytesIO(data))
            artifact = self.store.import_recording(data, filename=filename)
        except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
            raise Invalid(f"can't read this recording: {exc}") from exc
        self._apply_context(artifact, context)
        artifact.profile = profile
        paths = self.store.paths(artifact.id)
        (paths.dir / f"original{_suffix(filename)}").write_bytes(data)
        context["was_mono"] = SEPARATION_EVENT in artifact.providers
        (paths.dir / "upload.json").write_text(json.dumps(context))
        self.store.write_artifact(artifact)
        scenario = artifact.scenario_snapshot
        self.calls.add(
            id=artifact.id,
            scenario_id=artifact.scenario_id,
            scenario_title=scenario.title if scenario else None,
            mode=CallMode.UPLOAD.value,
            status="queued",
            profile=profile,
            timing_is_real=int(artifact.timing_is_real),
            duration_s=artifact.duration_s,
        )
        return artifact, Probe(
            duration_s=round(info.duration, 3),
            channels=info.channels,
            sample_rate=info.samplerate,
            # The store sends mono and dual-mono files through separation and notes that it did.
            channel_map="mono" if SEPARATION_EVENT in artifact.providers else "unknown",
            size_bytes=len(data),
        )

    def set_channel_map(self, call_id: str, channel_map: ChannelMap) -> None:
        """Re-import the original with this map. Earlier results described other channels, so
        they go."""
        row = self._upload(call_id)
        if row["status"] in ("running", "evaluating"):
            raise Conflict("this upload is being evaluated")
        paths = self.store.paths(call_id)
        context = json.loads((paths.dir / "upload.json").read_text())
        original = paths.dir / f"original{_suffix(context['filename'])}"
        previous = self.store.read_artifact(call_id)
        source, swap, separator = _mapped_source(
            original.read_bytes(), channel_map, context["was_mono"]
        )
        fresh = self.store.import_recording(
            source, filename=context["filename"], separator=separator, swap_channels=swap
        )
        fresh_paths = self.store.paths(fresh.id)
        for name in _MOVED_FILES:
            os.replace(fresh_paths.dir / name, paths.dir / name)
        shutil.rmtree(fresh_paths.dir)
        for name in _DERIVED_FILES:
            (paths.dir / name).unlink(missing_ok=True)
        fresh.id = call_id
        fresh.profile = previous.profile
        self._apply_context(fresh, context)
        self.store.write_artifact(fresh)
        self.calls.update(
            call_id,
            status="queued",
            verdict=None,
            invalid_reason=None,
            why="",
            findings=0,
            open_findings=0,
            duration_s=fresh.duration_s,
            reply_p50_s=None,
            overlaps=None,
        )
        self.calls.db.run("DELETE FROM call_findings WHERE call_id = ?", (call_id,))

    def preview(self, call_id: str, start_s: float, duration_s: float) -> bytes:
        self._upload(call_id)
        if start_s < 0 or duration_s <= 0:
            raise Invalid("preview needs startS >= 0 and durationS > 0")
        try:
            return self.store.clip_wav(call_id, start_s, start_s + duration_s)
        except ValueError as exc:
            raise Invalid(str(exc)) from exc

    def estimate_s(self, duration_s: float) -> float:
        return round(_ROUGH_EVAL_BASE_S + duration_s * _ROUGH_EVAL_RATE, 1)

    def require(self, call_id: str) -> dict[str, Any]:
        return self._upload(call_id)

    def _upload(self, call_id: str) -> dict[str, Any]:
        row = self.calls.get(call_id)
        if row is None or row["mode"] != CallMode.UPLOAD.value:
            raise NotFound(f"no upload {call_id}")
        return row

    def _apply_context(self, artifact: CallArtifact, context: dict[str, Any]) -> None:
        if context["scenario_id"]:
            found = self.catalog.scenario(context["scenario_id"])
            if found is None:
                raise Invalid(f"no scenario {context['scenario_id']}")
            artifact.scenario_id = context["scenario_id"]
            artifact.scenario_snapshot = found[0]
        if context["bot_version"]:
            artifact.providers["bot_version"] = context["bot_version"]
        apply_intended(artifact, parse_intended(context["intended"]))
        tools = parse_tool_log(context["tool_log"])
        if tools:
            artifact.timeline = sorted([*artifact.timeline, *tools], key=_event_time)
            artifact.has_tool_log = True


def _suffix(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower()


def _mapped_source(
    data: bytes, channel_map: ChannelMap, was_mono: bool
) -> tuple[bytes, bool, _SwappedLabels | None]:
    """Returns what to import, whether to swap stereo channels, and the separator to use."""
    if channel_map == "mono":
        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return AudioClip.from_samples(samples.mean(axis=1), rate).to_wav_bytes(), False, None
    if was_mono:
        return data, False, _SwappedLabels() if channel_map == "caller_right" else None
    return data, channel_map == "caller_right", None
