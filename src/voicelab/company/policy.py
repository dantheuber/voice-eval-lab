"""The policy and troubleshooting text the support bot is given."""

from __future__ import annotations

from pathlib import Path

from voicelab.company.db import SEEDS_DIR

POLICY_FILE = SEEDS_DIR / "policy.md"


def load_policy(path: Path = POLICY_FILE) -> str:
    return Path(path).read_text().strip()
