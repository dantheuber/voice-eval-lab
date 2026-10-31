#!/usr/bin/env python3
"""Refresh the eval regression set from local calibration calls.

The recordings stay in data/calibration, which is gitignored. What a judge reads is small and
synthetic, so that goes in the repo: each labeled call's artifact and transcripts, and the replies
a real judge gave, which tests/eval/test_judge_calibration.py replays with no model server.

Usage: scripts/export_calibration.py RUN_NAME
RUN_NAME is a directory under data/calibration/runs holding one replies file per call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from voicelab.agents.seeds import load_scenarios
from voicelab.core.models import CallArtifact

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = ROOT / "data" / "calibration"
FIXTURES = ROOT / "tests" / "eval" / "fixtures"


def _compact_words(transcript: dict) -> list[list]:
    return [
        [w["text"], round(w["start_s"], 2), round(w["end_s"], 2), round(w["confidence"] or 0, 2)]
        for w in transcript["words"]
    ]


def main(run: str) -> None:
    labels = yaml.safe_load((FIXTURES / "judge_labels.yaml").read_text())["calls"]
    scenarios = load_scenarios()
    target = FIXTURES / "calibration"
    target.mkdir(exist_ok=True)
    for call_id in labels:
        artifact = CallArtifact.model_validate_json(
            (CALIBRATION / "calls" / call_id / "call.json").read_text()
        )
        # Recorded snapshots predate rule-to-beat links, and the seeds have them.
        artifact.scenario_snapshot = scenarios[artifact.scenario_id]
        transcripts = json.loads((CALIBRATION / "transcripts" / f"{call_id}.json").read_text())
        replies = json.loads((CALIBRATION / "runs" / run / f"{call_id}.json").read_text())
        fixture = {
            "artifact": artifact.model_dump(mode="json"),
            "words": {speaker: _compact_words(t) for speaker, t in transcripts.items()},
            "replies": [entry["text"] for entry in replies["log"]],
        }
        (target / f"{call_id}.json").write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"wrote {len(labels)} calls to {target}")


if __name__ == "__main__":
    main(sys.argv[1])
