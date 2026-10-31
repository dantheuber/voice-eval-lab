from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import Rig, build_rig  # noqa: E402


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    built = build_rig(tmp_path)
    with built.client:
        yield built


@pytest.fixture
def client(rig: Rig):
    return rig.client


def wait_for(check: Callable[[], Any], timeout_s: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {check}")
        time.sleep(0.01)


@pytest.fixture
def waiter():
    return wait_for


PAIR = {"scenarioId": "scn_dns_mx_fix", "personaId": "per_sysadmin"}


@pytest.fixture
def start_run(client):
    def start(**overrides: Any) -> dict[str, Any]:
        body = {
            "pairs": [PAIR],
            "reps": 1,
            "profile": "local-default",
            "mode": "auto",
            "line": "persona",
            "concurrency": 1,
            "label": None,
            **overrides,
        }
        response = client.post("/api/runs", json=body)
        assert response.status_code == 201, response.text
        return response.json()

    return start


@pytest.fixture
def finished_call(client, start_run, waiter):
    """Run one call through to `done` and return its summary."""

    def make(**overrides: Any) -> dict[str, Any]:
        run_id = start_run(**overrides)["run"]["id"]
        waiter(lambda: client.get(f"/api/runs/{run_id}").json()["run"]["status"] == "done")
        return client.get(f"/api/runs/{run_id}").json()["calls"][0]

    return make
