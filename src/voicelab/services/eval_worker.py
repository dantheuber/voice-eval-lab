"""Subprocess entry point for one evaluation.

Reads an `EvalJob` as JSON on stdin and writes `eval.json` through the artifact store. Each stdout
line is one JSON progress message. The process is niced, and it builds its own adapters, because
HTTP clients can't be shared across processes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from voicelab.core.profiles import Profile
from voicelab.services.evaluation import deps_from_profile, evaluate_call, job_from_json
from voicelab.store.artifacts import ArtifactStore

_NICENESS = 10


def _emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _load_profile(directory: Path, profile_id: str | None) -> Profile | None:
    if profile_id is None:
        return None
    for path in sorted(directory.glob("*.yaml")):
        profile = Profile.model_validate(yaml.safe_load(path.read_text()))
        if profile.id == profile_id:
            return profile
    raise ValueError(f"no profile {profile_id!r} in {directory}")


async def _main(arguments: argparse.Namespace) -> None:
    job = job_from_json(sys.stdin.read())
    store = ArtifactStore(arguments.data_dir)
    profile = _load_profile(Path(arguments.profiles_dir), job.profile)
    await evaluate_call(store, job, deps_from_profile(profile, job.config), _emit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--profiles-dir", required=True)
    arguments = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    os.nice(_NICENESS)
    asyncio.run(_main(arguments))


if __name__ == "__main__":
    main()
