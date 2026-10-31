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
from voicelab.services import supervisor as supervisor_module
from voicelab.services.errors import ModelNotReady, ServiceError
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
    foreign = await foreign_server(port, "--ready-after", "0")
    try:
        await supervisor.apply([service(port)], start=True)
        await until(lambda: supervisor.statuses()["svc"].state == "down")
        status = supervisor.statuses()["svc"]
        assert status.managed is False
        assert status.pid is None
        await supervisor.restart("svc")
        await supervisor.stop_all()
        await asyncio.sleep(0.2)
        assert foreign.returncode is None
    finally:
        foreign.kill()
        await foreign.wait()


def port_is_open(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


async def foreign_server(port: int, *flags: str, cwd: Path | None = None):
    """A healthy server voicelab didn't start, once it's listening."""
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(SERVER), str(port), *flags, stdout=asyncio.subprocess.PIPE, cwd=cwd
    )
    await process.stdout.readline()
    await until(lambda: port_is_open(port))  # it says so before it binds
    return process


async def test_a_healthy_foreign_server_is_not_taken_for_ours(supervisor):
    """A Qwen server on the LFM port answers health checks, and results got labeled LFM."""
    port = free_port()
    foreign = await foreign_server(port, "--ready-after", "0")
    try:
        await supervisor.apply([service(port)], start=True)
        await until(lambda: supervisor.statuses()["svc"].state == "down")
        status = supervisor.statuses()["svc"]
        assert f"port {port} is held by" in status.detail
        assert f"pid {foreign.pid}" in status.detail
        assert status.managed is False
        assert status.pid is None
        assert not await supervisor.wait_ready(running(supervisor, "svc"), 0.1)
        await supervisor.stop_all()
        assert foreign.returncode is None
    finally:
        foreign.kill()
        await foreign.wait()


async def test_the_same_command_from_another_checkout_is_not_ours(supervisor, tmp_path):
    """Relative model paths mean another directory can be another model."""
    port = free_port()
    foreign = await foreign_server(port, cwd=tmp_path)
    try:
        await supervisor.apply([service(port)], start=True)
        await until(lambda: supervisor.statuses()["svc"].state == "down")
        assert f"pid {foreign.pid}" in supervisor.statuses()["svc"].detail
    finally:
        foreign.kill()
        await foreign.wait()


async def test_a_server_left_from_an_earlier_voicelab_is_used(tmp_path):
    """Same command, same directory. It's what a killed voicelab leaves behind, and it's the right
    model, so it's watched and never signalled."""
    instance = ServiceSupervisor(tmp_path / "logs", poll_s=0.05, cwd=Path.cwd())
    port = free_port()
    left_behind = await foreign_server(port)
    try:
        await instance.apply([service(port)], start=True)
        await until(lambda: instance.statuses()["svc"].state == "ok")
        status = instance.statuses()["svc"]
        assert status.managed is False
        assert status.pid is None
        await instance.stop_all()
        await asyncio.sleep(0.2)
        assert left_behind.returncode is None
    finally:
        await instance.stop_all()
        left_behind.kill()
        await left_behind.wait()


async def test_a_conflict_clears_once_the_port_is_free(supervisor):
    port = free_port()
    foreign = await foreign_server(port, "--ready-after", "0")
    await supervisor.apply([service(port)], start=True)
    await until(lambda: supervisor.statuses()["svc"].state == "down")
    foreign.kill()
    await foreign.wait()
    assert await supervisor.wait_ready(running(supervisor, "svc"), 10.0)
    assert supervisor.statuses()["svc"].managed is True


def test_a_port_holder_is_found_by_its_listening_socket():
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        found = supervisor_module.port_holder(holder.getsockname()[1])
    assert found.pid == os.getpid()
    assert found.name


def test_a_free_port_has_no_holder():
    assert supervisor_module.port_holder(free_port()) is None


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


async def test_a_run_waiting_on_a_held_port_says_who_holds_it(supervisor, tmp_path):
    port = free_port()
    foreign = await foreign_server(port, "--ready-after", "0")
    try:
        manager = _manager(tmp_path, supervisor, [_profile("default", service(port, name="llm"))])
        with pytest.raises(ModelNotReady, match=f"pid {foreign.pid}"):
            await manager.ready(manager.require("default"), 0.3)
        llm = next(h for h in manager.health(manager.require("default")) if h.role == "llm_support")
        assert llm.state == "down"
        assert f"pid {foreign.pid}" in llm.detail
    finally:
        foreign.kill()
        await foreign.wait()


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


# Swapping profiles -------------------------------------------------------------------------------


def _crashes(port: int, name: str = "llm") -> LocalService:
    return LocalService(name=name, command=[sys.executable, "-c", "raise SystemExit(3)"], port=port)


def _checkout(tmp_path, supervisor, profiles, manifest: str = "") -> ProfileManager:
    """A manager whose profiles sit in a checkout with a setup manifest, as the real ones do."""
    root = tmp_path / "checkout"
    directory = root / "config" / "profiles"
    directory.mkdir(parents=True)
    (root / "scripts" / "setup").mkdir(parents=True)
    (root / "scripts" / "setup" / "manifest.tsv").write_text(manifest)
    for profile in profiles:
        (directory / f"{profile.id}.yaml").write_text(yaml.safe_dump(profile.model_dump()))
    supervisor.cwd = root
    settings = Settings(data_dir=tmp_path, profiles_dir=directory)
    catalog = Catalog(Database(tmp_path / "x.sqlite3"))
    return ProfileManager(settings, catalog, supervisor, stack_factory=lambda profile: profile)


async def test_a_swap_waits_for_the_new_servers_and_then_saves(supervisor, tmp_path):
    port = free_port()
    first = _profile("default", service(port, name="llm"))
    second = _profile("fast", service(port, "--ready-after", "0.3", name="llm"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.activate("default")
    await manager.activate("fast")
    assert manager.active_id == "fast"
    assert supervisor.statuses()["llm"].state == "ok"


async def test_a_swap_that_cannot_start_puts_the_old_server_back(supervisor, tmp_path):
    port = free_port()
    first = _profile("default", service(port, name="llm"))
    second = _profile("fast", _crashes(port))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.activate("default")

    with pytest.raises(ServiceError, match="exited with code 3") as refused:
        await manager.activate("fast")

    assert "default" in refused.value.message
    assert manager.active_id == "default"
    assert await supervisor.wait_ready(manager.require("default").services, 10.0)
    assert supervisor.statuses()["llm"].pid is not None


async def test_a_failed_swap_leaves_servers_it_did_not_replace_running(supervisor, tmp_path):
    kept_port, new_port = free_port(), free_port()
    kept = service(kept_port, name="only-in-default")
    first = _profile("default", kept)
    second = _profile("fast", _crashes(new_port, "only-in-fast"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.activate("default")
    pid = supervisor.statuses()["only-in-default"].pid

    with pytest.raises(ServiceError):
        await manager.activate("fast")

    assert supervisor.statuses()["only-in-default"].pid == pid
    assert "only-in-fast" not in supervisor.statuses()


async def test_a_good_swap_drops_servers_the_new_profile_does_not_use(supervisor, tmp_path):
    first = _profile("default", service(free_port(), name="old"))
    second = _profile("fast", service(free_port(), name="new"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.activate("default")
    pid = supervisor.statuses()["old"].pid
    await manager.activate("fast")
    assert set(supervisor.statuses()) == {"new"}
    await until(lambda: not alive(pid))


async def test_a_held_port_fails_a_swap_at_once_and_names_the_holder(supervisor, tmp_path):
    held = free_port()
    foreign = await foreign_server(held, "--ready-after", "0")
    first = _profile("default", service(free_port(), name="llm"))
    second = _profile("fast", service(held, name="asr"))
    manager = _manager(tmp_path, supervisor, [first, second])
    try:
        await manager.activate("default")
        with pytest.raises(ServiceError, match=f"pid {foreign.pid}"):
            await asyncio.wait_for(manager.activate("fast"), 5.0)
        assert manager.active_id == "default"
    finally:
        foreign.kill()
        await foreign.wait()


async def test_a_failed_swap_does_not_start_what_was_never_running(supervisor, tmp_path):
    """With --lazy nothing has started, and a failed swap shouldn't change that."""
    idle = service(free_port(), name="idle")
    first = _profile("default", idle)
    second = _profile("fast", _crashes(free_port(), "broken"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await supervisor.apply([idle], start=False)
    with pytest.raises(ServiceError):
        await manager.activate("fast")
    assert supervisor.statuses()["idle"].state == "off"
    assert "broken" not in supervisor.statuses()


async def test_nothing_else_uses_the_servers_during_a_swap(supervisor, tmp_path):
    port = free_port()
    first = _profile("default", service(port, name="llm"))
    second = _profile("fast", service(port, "--ready-after", "0.4", name="llm"))
    manager = _manager(tmp_path, supervisor, [first, second])
    await manager.activate("default")
    swap = asyncio.create_task(manager.activate("fast"))
    await asyncio.sleep(0.05)
    assert manager.in_use
    with pytest.raises(ModelNotReady, match="in use"):
        await manager.require_ready(manager.require("default"), ["llm_support"])
    waiting = asyncio.create_task(manager.ready(manager.require("default"), 10.0))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    await swap
    await asyncio.wait_for(waiting, 10.0)


async def test_a_missing_program_stops_nothing(supervisor, tmp_path):
    port = free_port()
    first = _profile("default", service(port, name="llm"))
    binary = ".runtime/nemo/bin/nemo-speech"
    broken = LocalService(name="llm", command=[binary, "serve"], port=port)
    second = _profile("local-fast", broken)
    manager = _checkout(tmp_path, supervisor, [first, second])
    await manager.activate("default")
    pid = supervisor.statuses()["llm"].pid

    with pytest.raises(ServiceError, match="nemo-speech") as refused:
        await manager.activate("local-fast")

    assert refused.value.detail["missing"] == [binary]
    assert manager.active_id == "default"
    assert supervisor.statuses()["llm"].pid == pid


async def test_a_missing_model_stops_nothing(supervisor, tmp_path):
    port = free_port()
    first = _profile("default", service(port, name="llm"))
    model = ".runtime/models/LFM3.gguf"
    swapped = service(port, name="llm")
    swapped.command += ["-m", model]
    manager = _checkout(tmp_path, supervisor, [first, _profile("local-fast", swapped)])
    await manager.activate("default")
    pid = supervisor.statuses()["llm"].pid
    with pytest.raises(ServiceError, match="LFM3.gguf"):
        await manager.activate("local-fast")
    assert supervisor.statuses()["llm"].pid == pid


MANIFEST = """\
# need: required files stop run.sh.
#profiles\tneed\tpath\tbytes\trepo\trevision\tfile\thash
local-fast,local-realtime\trequired\t.runtime/hf/voice.safetensors\t4\tr\tv\tf\th
local-fast\trequired\t.runtime/models/parakeet.gguf\t4\tr\tv\tf\th
local-fast\ton-demand\t.runtime/hf/optional.safetensors\t4\tr\tv\tf\th
local-default\trequired\t.runtime/models/qwen.gguf\t4\tr\tv\tf\th
"""


def _with_file(root: Path, path: str, text: str = "data") -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)


async def test_required_files_come_from_the_setup_manifest(supervisor, tmp_path):
    """Phonon's voices are named nowhere on its command line, so the command can't show them."""
    fast = Profile(id="local-fast", label="Fast", roles={})
    manager = _checkout(tmp_path, supervisor, [fast], MANIFEST)
    _with_file(supervisor.cwd, ".runtime/models/parakeet.gguf", "short")

    with pytest.raises(ServiceError) as refused:
        await manager.activate("local-fast")

    assert refused.value.detail == {
        "profile": "local-fast",
        "missing": [".runtime/hf/voice.safetensors", ".runtime/models/parakeet.gguf (incomplete)"],
        "setup": "scripts/setup.sh --profile local-fast",
    }
    assert "scripts/setup.sh --profile local-fast" in refused.value.message
    assert manager.catalog.setting("active_profile") is None


async def test_a_complete_install_passes_and_on_demand_files_may_be_missing(supervisor, tmp_path):
    fast = Profile(id="local-fast", label="Fast", roles={})
    manager = _checkout(tmp_path, supervisor, [fast], MANIFEST)
    _with_file(supervisor.cwd, ".runtime/hf/voice.safetensors")
    _with_file(supervisor.cwd, ".runtime/models/parakeet.gguf")
    await manager.activate("local-fast")
    assert manager.active_id == "local-fast"


async def test_a_profile_setup_does_not_cover_says_to_check_its_paths(supervisor, tmp_path):
    custom = LocalService(name="x", command=["/no/such/server"], port=free_port())
    manager = _checkout(tmp_path, supervisor, [_profile("mine", custom)], MANIFEST)
    with pytest.raises(ServiceError, match="/no/such/server") as refused:
        await manager.activate("mine")
    assert refused.value.detail["setup"] is None
    assert "setup.sh" not in refused.value.message


async def test_a_program_on_the_path_counts_as_installed(supervisor, tmp_path):
    on_path = LocalService(name="x", command=["sleep", "0"], port=free_port())
    manager = _checkout(tmp_path, supervisor, [_profile("mine", on_path)])
    manager.require_usable(manager.require("mine"))


def _with_adapters(profile_id: str, service_: LocalService, model: str) -> Profile:
    role = {"type": "fake", "model": model, "base_url": f"http://127.0.0.1:{service_.port}/v1"}
    return Profile(id=profile_id, label=profile_id, roles={"stt_live": role}, services=[service_])


async def test_a_role_on_a_model_that_isnt_installed_changes_nothing(supervisor, tmp_path):
    port = free_port()
    good = _with_adapters("default", service(port, name="asr"), "qwen")
    absent = LocalService(name="parakeet", command=[".runtime/no-such-server"], port=free_port())
    other = _with_adapters("fast", absent, "parakeet")
    manager = _manager(tmp_path, supervisor, [good, other])
    await manager.activate("default")
    pid = supervisor.statuses()["asr"].pid
    path = manager.paths["default"]
    before = path.read_text()

    with pytest.raises(ServiceError, match="no-such-server"):
        await manager.set_role("stt_live", "fake:parakeet")

    assert path.read_text() == before
    assert supervisor.statuses()["asr"].pid == pid


async def test_a_role_swap_that_cannot_start_keeps_the_yaml_and_the_server(supervisor, tmp_path):
    port = free_port()
    good = _with_adapters("default", service(port, name="asr"), "qwen")
    other = _with_adapters("fast", _crashes(free_port(), "parakeet"), "parakeet")
    manager = _manager(tmp_path, supervisor, [good, other])
    await manager.activate("default")
    pid = supervisor.statuses()["asr"].pid
    path = manager.paths["default"]
    before = path.read_text()

    with pytest.raises(ServiceError, match="exited with code 3"):
        await manager.set_role("stt_live", "fake:parakeet")

    assert path.read_text() == before
    assert supervisor.statuses()["asr"].pid == pid
    assert "parakeet" not in supervisor.statuses()


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
