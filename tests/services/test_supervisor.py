from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
import time
from pathlib import Path

import pytest
import yaml

from voicelab.core.profiles import LocalService, Profile
from voicelab.services.errors import ModelNotReady
from voicelab.services.profiles import ProfileManager
from voicelab.services.settings import Settings
from voicelab.services.supervisor import ServiceSupervisor, group_members
from voicelab.store.catalog import Catalog
from voicelab.store.sqlite import Database

SERVER = Path(__file__).parent / "fake_server.py"
CHILD = Path(__file__).parent / "fake_child.py"


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


def running(instance: ServiceSupervisor, *names: str) -> list[LocalService]:
    """The definitions these names run under, for `wait_ready`."""
    return [instance.statuses()[name].service for name in names]


def alive_command(pid: int) -> list[str]:
    return Path(f"/proc/{pid}/cmdline").read_text().split("\0")


def _pid_in(path: Path) -> int | None:
    text = path.read_text().strip() if path.exists() else ""
    return int(text) if text else None


async def test_a_listed_service_is_off_until_started(supervisor):
    await supervisor.apply([service(free_port())], start=False)
    status = supervisor.statuses()["svc"]
    assert status.state == "off"
    assert status.pid is None


async def test_loading_reports_elapsed_then_ok_with_usual(supervisor):
    await supervisor.apply([service(free_port(), "--ready-after", "0.4")], start=True)
    await until(lambda: supervisor.statuses()["svc"].state == "loading")
    await until(lambda: (supervisor.statuses()["svc"].elapsed_s or 0) > 0)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    status = supervisor.statuses()["svc"]
    assert status.state == "ok"
    assert status.elapsed_s is None
    assert status.usual_s >= 0.4
    assert status.pid is not None
    assert status.managed is True


async def test_load_time_survives_a_restart_of_the_app(supervisor, tmp_path):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    later = ServiceSupervisor(tmp_path / "logs")
    await later.apply([service(free_port())], start=False)
    assert later.statuses()["svc"].usual_s is not None


async def test_crash_restarts_the_service(supervisor):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    first = supervisor.statuses()["svc"].pid
    os.kill(first, signal.SIGKILL)
    await until(lambda: supervisor.statuses()["svc"].restarts == 1)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    status = supervisor.statuses()["svc"]
    assert status.pid != first
    assert status.state == "ok"


async def test_crash_at_startup_is_retried_and_reported(supervisor, tmp_path):
    marker = str(tmp_path / "crashed")
    await supervisor.apply([service(free_port(), "--crash-marker", marker)], start=True)
    await until(lambda: "exited with code 3" in supervisor.statuses()["svc"].detail)
    assert supervisor.statuses()["svc"].state == "down"
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)


async def test_a_server_that_never_gets_healthy_is_restarted(supervisor):
    target = service(free_port(), "--ready-after", "60", timeout_s=0.2)
    await supervisor.apply([target], start=True)
    await until(lambda: "not healthy after" in supervisor.statuses()["svc"].detail)
    assert supervisor.statuses()["svc"].restarts >= 1


async def test_a_hung_server_is_restarted(supervisor):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    first = supervisor.statuses()["svc"].pid
    os.kill(first, signal.SIGSTOP)
    await until(lambda: supervisor.statuses()["svc"].restarts == 1, timeout_s=20.0)
    assert not alive(first)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)


async def test_missing_command_is_down_not_a_crash_of_the_app(supervisor):
    broken = LocalService(name="svc", command=["/no/such/binary"], port=free_port())
    await supervisor.apply([broken], start=True)
    await until(lambda: "could not start" in supervisor.statuses()["svc"].detail)
    assert supervisor.statuses()["svc"].state == "down"


async def test_stop_all_ends_what_it_started(supervisor):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
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
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
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
    assert await supervisor.wait_ready(running(supervisor, "keep", "drop"), 10.0)
    kept_pid = supervisor.statuses()["keep"].pid
    dropped_pid = supervisor.statuses()["drop"].pid
    replacement = service(free_port(), name="new")
    await supervisor.apply([keep, replacement], start=True)
    assert await supervisor.wait_ready(running(supervisor, "keep", "new"), 10.0)
    assert supervisor.statuses()["keep"].pid == kept_pid
    assert "drop" not in supervisor.statuses()
    await until(lambda: not alive(dropped_pid))


async def test_a_changed_definition_restarts_that_server(supervisor):
    port = free_port()
    await supervisor.apply([service(port)], start=True)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    first = supervisor.statuses()["svc"].pid
    await supervisor.apply([service(port, "--ready-after", "0")], start=True)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    assert supervisor.statuses()["svc"].pid != first


async def test_restart_replaces_the_process(supervisor):
    await supervisor.apply([service(free_port())], start=True)
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    first = supervisor.statuses()["svc"].pid
    await supervisor.restart("svc")
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    assert supervisor.statuses()["svc"].pid != first


async def test_output_goes_to_a_log_file(supervisor, tmp_path):
    await supervisor.apply([service(free_port(), name="odd/name")], start=True)
    assert await supervisor.wait_ready(running(supervisor, "odd/name"), 10.0)
    log = tmp_path / "logs" / "odd_name.log"
    await until(lambda: log.exists() and "listening" in log.read_text())


# Profiles sharing a slot --------------------------------------------------------------------------


def _manager(tmp_path, supervisor, profiles):
    directory = tmp_path / "profiles"
    directory.mkdir(exist_ok=True)
    for profile in profiles:
        (directory / f"{profile.id}.yaml").write_text(yaml.safe_dump(profile.model_dump()))
    settings = Settings(data_dir=tmp_path, profiles_dir=directory)
    catalog = Catalog(Database(tmp_path / "x.sqlite3"))
    return ProfileManager(settings, catalog, supervisor, stack_factory=lambda profile: profile)


def _profile(profile_id, *services):
    role = {"type": "fake", "base_url": f"http://127.0.0.1:{services[0].port}/v1"}
    return Profile(id=profile_id, label=profile_id, roles={"llm_support": role}, services=services)


async def test_asking_for_another_profile_runs_that_profiles_server(supervisor, tmp_path):
    """Same name, same port, different model: the old server must not answer for the new one."""
    port = free_port()
    first = _profile("default", service(port, "--ready-after", "0", name="llm"))
    second = _profile("fast", service(port, "--ready-after", "0.01", name="llm"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.ready(manager.require("default"), 10.0)
    old = supervisor.statuses()["llm"].pid
    manager.release(manager.require("default"))
    await manager.ready(manager.require("fast"), 10.0)
    status = supervisor.statuses()["llm"]
    assert status.pid != old
    assert status.state == "ok"
    assert "--ready-after 0.01" in " ".join(alive_command(status.pid))


async def test_a_new_name_on_a_taken_port_replaces_the_old_server(supervisor, tmp_path):
    port = free_port()
    first = _profile("default", service(port, name="old-name"))
    second = _profile("fast", service(port, name="new-name"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.ready(manager.require("default"), 10.0)
    manager.release(manager.require("default"))
    await manager.ready(manager.require("fast"), 10.0)
    assert "old-name" not in supervisor.statuses()
    assert supervisor.statuses()["new-name"].managed is True


async def test_readiness_is_checked_against_the_requested_definition(supervisor):
    port = free_port()
    running = service(port)
    await supervisor.apply([running], start=True)
    assert await supervisor.wait_ready([running], 10.0)
    assert not await supervisor.wait_ready([service(port, "--ready-after", "1")], 0.1)


async def test_health_does_not_credit_a_different_server(supervisor, tmp_path):
    port = free_port()
    first = _profile("default", service(port, name="llm"))
    second = _profile("fast", service(port, "--ready-after", "0", name="llm"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.ready(manager.require("default"), 10.0)
    llm = next(h for h in manager.health(manager.require("fast")) if h.role == "llm_support")
    assert llm.state == "off"


async def test_work_in_flight_keeps_its_servers(supervisor, tmp_path):
    """Replacing a server under a running call would be worse than the wrong label."""
    port = free_port()
    first = _profile("default", service(port, name="llm"))
    second = _profile("fast", service(port, "--ready-after", "0", name="llm"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.ready(manager.require("default"), 10.0)
    pid = supervisor.statuses()["llm"].pid
    with pytest.raises(ModelNotReady, match="in use"):
        await manager.ready(manager.require("fast"), 0.05)
    with pytest.raises(ModelNotReady, match="in use"):
        await manager.require_ready(manager.require("fast"), ["llm_support"])
    assert supervisor.statuses()["llm"].pid == pid


async def test_a_waiting_profile_runs_once_the_other_lets_go(supervisor, tmp_path):
    port = free_port()
    first = _profile("default", service(port, name="llm"))
    second = _profile("fast", service(port, "--ready-after", "0", name="llm"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.ready(manager.require("default"), 10.0)
    waiting = asyncio.create_task(manager.ready(manager.require("fast"), 10.0))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    manager.release(manager.require("default"))
    await asyncio.wait_for(waiting, 10.0)
    assert supervisor.statuses()["llm"].state == "ok"


# Process groups ----------------------------------------------------------------------------------


def _orphaning_service(port: int, marker: Path, *child_flags: str) -> LocalService:
    """A wrapper that starts a server and exits, leaving the child in its process group."""
    child = f"{sys.executable} {CHILD} {marker} {' '.join(child_flags)}"
    # It waits for the child to be up, so a signal can't land while Python is still starting.
    script = f"{child} & until [ -s {marker} ]; do sleep 0.01; done; exit 3"
    return LocalService(name="svc", command=["sh", "-c", script], port=port)


async def test_children_of_a_crashed_server_are_cleaned_up(supervisor, tmp_path):
    marker = tmp_path / "child.pid"
    await supervisor.apply([_orphaning_service(free_port(), marker)], start=True)
    pid = await until(lambda: _pid_in(marker))
    await until(lambda: not alive(pid), timeout_s=5.0)


async def test_a_child_that_ignores_sigterm_is_killed_after_the_grace(tmp_path):
    instance = ServiceSupervisor(
        tmp_path / "logs", poll_s=0.05, min_backoff_s=30.0, stop_grace_s=0.3
    )
    marker = tmp_path / "child.pid"
    try:
        await instance.apply([_orphaning_service(free_port(), marker, "--ignore-term")], start=True)
        pid = await until(lambda: _pid_in(marker))
        await until(lambda: not alive(pid), timeout_s=5.0)
    finally:
        await instance.stop_all()


async def test_group_cleanup_leaves_other_processes_alone(supervisor, tmp_path):
    bystander = await asyncio.create_subprocess_exec("sleep", "30")
    marker = tmp_path / "child.pid"
    try:
        await supervisor.apply([_orphaning_service(free_port(), marker)], start=True)
        pid = await until(lambda: _pid_in(marker))
        await until(lambda: not alive(pid), timeout_s=5.0)
        assert bystander.returncode is None
    finally:
        bystander.kill()
        await bystander.wait()


def test_a_group_is_only_ours_if_its_session_is_too():
    """The guard against a recycled id: this test's own group isn't a session we started."""
    assert not group_members(os.getpgrp() + 4_000_000)
    assert os.getpid() not in group_members(1)


# Logs --------------------------------------------------------------------------------------------


async def test_a_running_server_cannot_grow_its_log_without_bound(tmp_path):
    instance = ServiceSupervisor(
        tmp_path / "logs", poll_s=0.05, stop_grace_s=2.0, log_max_bytes=20_000, log_backups=2
    )
    port = free_port()
    chatty = LocalService(
        name="svc",
        command=[sys.executable, str(SERVER), str(port), "--chatter", "2000"],
        port=port,
    )
    try:
        await instance.apply([chatty], start=True)
        assert await instance.wait_ready([chatty], 10.0)
        log = tmp_path / "logs" / "svc.log"
        await until(lambda: Path(f"{log}.2").exists(), timeout_s=15.0)
        files = list((tmp_path / "logs").glob("svc.log*"))
        assert len(files) <= 3
    finally:
        await instance.stop_all()


async def test_wait_ready_gives_up(supervisor):
    await supervisor.apply([service(free_port(), "--ready-after", "60")], start=True)
    assert not await supervisor.wait_ready(running(supervisor, "svc"), 0.1)


async def test_changes_are_announced(tmp_path):
    changes = []
    instance = ServiceSupervisor(tmp_path, poll_s=0.05, on_change=lambda: changes.append(1))
    try:
        await instance.apply([service(free_port())], start=True)
        assert await instance.wait_ready(running(instance, "svc"), 10.0)
        assert len(changes) >= 2  # loading, then ok
    finally:
        await instance.stop_all()
