"""Speech regions for the review screen. Derived once and cached beside the call's peaks."""

from __future__ import annotations

import json
from typing import Any

from voicelab.core.audio import AudioClip
from voicelab.core.models import CallArtifact, CallStatus
from voicelab.eval.config import TimingConfig
from voicelab.eval.evaluators.timing import find_overlaps
from voicelab.eval.vad import Region, speech_regions
from voicelab.store.artifacts import SEPARATION_EVENT, ArtifactStore

_SETTLED = (CallStatus.DONE, CallStatus.FAILED)


def _pairs(regions: list[Region]) -> list[list[float]]:
    return [[round(r.start_s, 3), round(r.end_s, 3)] for r in regions]


def speech_data(store: ArtifactStore, artifact: CallArtifact) -> dict[str, Any]:
    """Returns `{"speech": {caller, bot}, "overlaps": [...]}`."""
    paths = store.paths(artifact.id)
    cache = paths.dir / "speech.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not paths.caller.exists() or not paths.bot.exists():
        return {"speech": {"caller": [], "bot": []}, "overlaps": []}
    timing = TimingConfig()
    options = {"min_speech_s": timing.min_speech_s, "min_gap_s": timing.min_gap_s}
    caller = speech_regions(AudioClip.from_file(paths.caller), **options)
    bot = speech_regions(AudioClip.from_file(paths.bot), **options)
    # Channels split from a mono mix can't overlap, and showing none would imply it was measured.
    inferred = SEPARATION_EVENT in artifact.providers
    overlaps = [] if inferred else find_overlaps(caller, bot, min_s=timing.min_overlap_s)
    data = {
        "speech": {"caller": _pairs(caller), "bot": _pairs(bot)},
        "overlaps": [[round(o.start_s, 3), round(o.end_s, 3)] for o in overlaps],
    }
    if artifact.status in _SETTLED:
        cache.write_text(json.dumps(data, separators=(",", ":")))
    return data
