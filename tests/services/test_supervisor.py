from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
import time
from pathlib import Path

import pytest

from voicelab.core.profiles import LocalService
from voicelab.services.supervisor import ServiceSupervisor

SERVER = Path(__file__).parent / "fake_server.py"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def service(port: int, *flags: str, name: str = "svc", timeout_s: float = 30.0) -> LocalService:
    return LocalService(
        name=name,
        command=[sys.executable, str(SERVER), str(port), *flags],
        port=port,
        startup_timeout_s=timeout_s,
    )


@pytest.fixture
async def supervisor(tmp_path):
    instance = ServiceSupervisor(
        tmp_path / "logs",
        poll_s=0.05,
        min_backoff_s=0.05,
        max_backoff_s=0.1,
        stop_grace_s=2.0,
        failed_polls=3,
    )
    yield instance
    await instance.stop_all()


async def until(check, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as stat:
        return stat.read().split()[2] != "Z"


async def test_a_listed_service_is_off_until_started(supervisor):
    await supervisor.apply([service(free_port())], start=False)
    status = supervisor.statuses()["svc"]
    assert status.state == "off"
    assert status.pid is None


async def test_loading_reports_elapsed_then_ok_with_usual(supervisor):
    await supervisor.apply([service(free_port(), "--ready-after", "0.4")], start=True)
    await until(lambda: supervisor.statuses()["svc"].state == "loading")
    await until(lambda: (supervisor.statuses()["svc"].elapsed_s or 0) > 0)
    assert await supervisor.wait_ready(["svc"], 10.0)
    status = supervisor.statuses()["svc"]
    assert status.state == "ok"
    assert status.elapsed_s is None
    assert status.usual_s >= 0.4
    assert status.pid is not None
    assert status.managed is True


async def test_load_time_survives_a_restart_of_the_app(supervisor, tmp_path):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(["svc"], 10.0)
    later = ServiceSupervisor(tmp_path / "logs")
    await later.apply([service(free_port())], start=False)
    assert later.statuses()["svc"].usual_s is not None


async def test_crash_restarts_the_service(supervisor):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(["svc"], 10.0)
    first = supervisor.statuses()["svc"].pid
    os.kill(first, signal.SIGKILL)
    await until(lambda: supervisor.statuses()["svc"].restarts == 1)
    assert await supervisor.wait_ready(["svc"], 10.0)
    status = supervisor.statuses()["svc"]
    assert status.pid != first
    assert status.state == "ok"


async def test_crash_at_startup_is_retried_and_reported(supervisor, tmp_path):
    marker = str(tmp_path / "crashed")
    await supervisor.apply([service(free_port(), "--crash-marker", marker)], start=True)
    await until(lambda: "exited with code 3" in supervisor.statuses()["svc"].detail)
    assert supervisor.statuses()["svc"].state == "down"
    assert await supervisor.wait_ready(["svc"], 10.0)


async def test_a_server_that_never_gets_healthy_is_restarted(supervisor):
    target = service(free_port(), "--ready-after", "60", timeout_s=0.2)
    await supervisor.apply([target], start=True)
    await until(lambda: "not healthy after" in supervisor.statuses()["svc"].detail)
    assert supervisor.statuses()["svc"].restarts >= 1


async def test_a_hung_server_is_restarted(supervisor):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(["svc"], 10.0)
    first = supervisor.statuses()["svc"].pid
    os.kill(first, signal.SIGSTOP)
    await until(lambda: supervisor.statuses()["svc"].restarts == 1, timeout_s=20.0)
    assert not alive(first)
    assert await supervisor.wait_ready(["svc"], 10.0)


async def test_missing_command_is_down_not_a_crash_of_the_app(supervisor):
    broken = LocalService(name="svc", command=["/no/such/binary"], port=free_port())
    await supervisor.apply([broken], start=True)
    await until(lambda: "could not start" in supervisor.statuses()["svc"].detail)
    assert supervisor.statuses()["svc"].state == "down"


async def test_stop_all_ends_what_it_started(supervisor):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(["svc"], 10.0)
    pid = supervisor.statuses()["svc"].pid
    await supervisor.stop_all()
    await until(lambda: not alive(pid))
    assert supervisor.statuses()["svc"].state == "off"


async def test_stop_reaches_the_servers_children(supervisor, tmp_path):
    pids = tmp_path / "pids"
    port = free_port()
    script = f"{sys.executable} {SERVER} {port} --pid-file {pids} & echo $! >> {pids}.shell; wait"
    wrapped = LocalService(name="svc", command=["sh", "-c", script], port=port)
    await supervisor.apply([wrapped], start=True)
    assert await supervisor.wait_ready(["svc"], 10.0)
    child = int(pids.read_text().split()[0])
    await supervisor.stop_all()
    await until(lambda: not alive(child))


async def test_never_kills_a_server_it_did_not_start(supervisor):
    port = free_port()
    foreign = await asyncio.create_subprocess_exec(
        sys.executable, str(SERVER), str(port), stdout=asyncio.subprocess.PIPE
    )
    try:
        await foreign.stdout.readline()
        await supervisor.apply([service(port)], start=True)
        await until(lambda: supervisor.statuses()["svc"].state == "ok")
        status = supervisor.statuses()["svc"]
        assert status.managed is False
        assert status.pid is None
        assert "not started by voicelab" in status.detail
        await supervisor.restart("svc")
        await supervisor.stop_all()
        await asyncio.sleep(0.2)
        assert foreign.returncode is None
    finally:
        foreign.kill()
        await foreign.wait()


async def test_a_port_held_by_something_else_is_down_and_left_alone(supervisor):
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        await supervisor.apply([service(port)], start=True)
        await until(lambda: supervisor.statuses()["svc"].state == "down")
        assert f"port {port} is held" in supervisor.statuses()["svc"].detail
        assert supervisor.statuses()["svc"].pid is None


async def test_apply_keeps_unchanged_servers_running(supervisor):
    keep = service(free_port(), name="keep")
    drop = service(free_port(), name="drop")
    await supervisor.apply([keep, drop], start=True)
    assert await supervisor.wait_ready(["keep", "drop"], 10.0)
    kept_pid = supervisor.statuses()["keep"].pid
    dropped_pid = supervisor.statuses()["drop"].pid
    replacement = service(free_port(), name="new")
    await supervisor.apply([keep, replacement], start=True)
    assert await supervisor.wait_ready(["keep", "new"], 10.0)
    assert supervisor.statuses()["keep"].pid == kept_pid
    assert "drop" not in supervisor.statuses()
    await until(lambda: not alive(dropped_pid))


async def test_a_changed_definition_restarts_that_server(supervisor):
    port = free_port()
    await supervisor.apply([service(port)], start=True)
    assert await supervisor.wait_ready(["svc"], 10.0)
    first = supervisor.statuses()["svc"].pid
    await supervisor.apply([service(port, "--ready-after", "0")], start=True)
    assert await supervisor.wait_ready(["svc"], 10.0)
    assert supervisor.statuses()["svc"].pid != first


async def test_restart_replaces_the_process(supervisor):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(["svc"], 10.0)
    first = supervisor.statuses()["svc"].pid
    await supervisor.restart("svc")
    assert await supervisor.wait_ready(["svc"], 10.0)
    assert supervisor.statuses()["svc"].pid != first


async def test_output_goes_to_a_log_file(supervisor, tmp_path):
    await supervisor.apply([service(free_port(), name="odd/name")], start=True)
    assert await supervisor.wait_ready(["odd/name"], 10.0)
    log = tmp_path / "logs" / "odd_name.log"
    await until(lambda: log.exists() and "listening" in log.read_text())


async def test_wait_ready_gives_up(supervisor):
    await supervisor.apply([service(free_port(), "--ready-after", "60")], start=True)
    assert not await supervisor.wait_ready(["svc"], 0.1)


async def test_changes_are_announced(tmp_path):
    changes = []
    instance = ServiceSupervisor(tmp_path, poll_s=0.05, on_change=lambda: changes.append(1))
    try:
        await instance.apply([service(free_port())], start=True)
        assert await instance.wait_ready(["svc"], 10.0)
        assert len(changes) >= 2  # loading, then ok
    finally:
        await instance.stop_all()
