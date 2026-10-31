"""Starts and watches a profile's model servers.

Each `LocalService` runs in its own process group with its log in `data/logs/`. A crash restarts it
with growing backoff, and a server that stops answering health checks is treated as crashed. The
only processes ever signalled are ones this supervisor spawned: a server already on the port is
watched and reported, never touched, because the machine runs other people's model servers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import socket
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from voicelab.core.profiles import LocalService
from voicelab.services.logs import LOG_BACKUPS, LOG_MAX_BYTES, rotate_log
from voicelab.services.settings import REPO_ROOT

logger = logging.getLogger(__name__)

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_GROUP_POLL_S = 0.05
_DEAD_STATES = ("Z", "X")


@dataclass
class ServiceStatus:
    service: LocalService  # the definition actually running under this name
    name: str
    port: int
    state: str  # ok, loading, down, off
    detail: str
    pid: int | None
    managed: bool  # False when something else started it
    elapsed_s: float | None  # time spent loading so far, while loading
    usual_s: float | None  # the last measured load time
    restarts: int


@dataclass
class _Entry:
    service: LocalService
    state: str = "off"
    detail: str = "not started"
    managed: bool = True
    loading_since: float | None = None
    restarts: int = 0
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class ServiceSupervisor:
    def __init__(
        self,
        log_dir: Path,
        *,
        on_change: Callable[[], None] | None = None,
        poll_s: float = 2.0,
        min_backoff_s: float = 1.0,
        max_backoff_s: float = 60.0,
        stable_s: float = 60.0,
        failed_polls: int = 5,
        stop_grace_s: float = 10.0,
        cwd: Path = REPO_ROOT,
        log_max_bytes: int = LOG_MAX_BYTES,
        log_backups: int = LOG_BACKUPS,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.on_change = on_change
        self.poll_s = poll_s
        self.min_backoff_s = min_backoff_s
        self.max_backoff_s = max_backoff_s
        self.stable_s = stable_s  # healthy this long and a later crash starts backoff over
        self.failed_polls = failed_polls
        self.stop_grace_s = stop_grace_s
        self.cwd = cwd
        self.log_max_bytes = log_max_bytes
        self.log_backups = log_backups
        self._lock = asyncio.Lock()  # one change at a time: a stop awaits, and ports must be free
        self._entries: dict[str, _Entry] = {}
        # Load times survive restarts so "usually 40 s" is there on the first load of a session.
        self._times_path = self.log_dir / "service_times.json"
        self._usual = self._read_times()
        self._http: httpx.AsyncClient | None = None

    # Public ------------------------------------------------------------------------------------

    def statuses(self) -> dict[str, ServiceStatus]:
        return {name: self._status(entry) for name, entry in self._entries.items()}

    async def apply(self, services: Iterable[LocalService], *, start: bool) -> None:
        """Make `services` the supervised set. A server whose definition didn't change keeps
        running, so swapping one role doesn't reload every model."""
        wanted = {service.name: service for service in services}
        async with self._lock:
            for name, entry in list(self._entries.items()):
                if wanted.get(name) != entry.service:
                    await self._stop(entry)
                    del self._entries[name]
            for name, service in wanted.items():
                self._entries.setdefault(name, _Entry(service))
        if start:
            self.ensure_started(wanted)
        self._changed()

    async def reconcile(self, services: Iterable[LocalService]) -> None:
        """Make these definitions the ones running, and start them. Whatever holds one of their
        names or ports under another definition stops first, or it would answer for the wrong
        model. Servers outside that set keep running."""
        wanted = list(services)
        async with self._lock:
            for name, entry in list(self._entries.items()):
                if entry.service in wanted:
                    continue
                if any(s.name == name or s.port == entry.service.port for s in wanted):
                    await self._stop(entry)
                    del self._entries[name]
            for service in wanted:
                self._entries.setdefault(service.name, _Entry(service))
        self.ensure_started(service.name for service in wanted)

    def ensure_started(self, names: Iterable[str] | None = None) -> None:
        for name in self._entries if names is None else names:
            entry = self._entries[name]
            if entry.task is None or entry.task.done():
                entry.task = asyncio.create_task(self._supervise(entry), name=f"service-{name}")

    async def restart(self, name: str) -> None:
        async with self._lock:
            await self._stop(self._entries[name])
        self.ensure_started([name])

    async def stop_all(self) -> None:
        await asyncio.gather(*(self._stop(entry) for entry in self._entries.values()))
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def wait_ready(self, services: Iterable[LocalService], timeout_s: float) -> bool:
        wanted = list(services)
        deadline = time.monotonic() + timeout_s
        while not all(self._is_ready(service) for service in wanted):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(0.25, self.poll_s))
        return True

    def _is_ready(self, service: LocalService) -> bool:
        entry = self._entries.get(service.name)
        return entry is not None and entry.service == service and entry.state == "ok"

    # Supervision -------------------------------------------------------------------------------

    async def _supervise(self, entry: _Entry) -> None:
        backoff = self.min_backoff_s
        service = entry.service
        while True:
            if await self._answers(service):
                entry.managed = False
                self._set(entry, "ok", "running, not started by voicelab")
                await asyncio.sleep(self.poll_s)
                continue
            entry.managed = True
            if _port_taken(service.port):
                self._set(entry, "down", f"port {service.port} is held by another process")
                await asyncio.sleep(self.poll_s)
                continue
            reason, healthy_s = await self._run_once(entry)
            if healthy_s >= self.stable_s:
                backoff = self.min_backoff_s
            entry.restarts += 1
            logger.warning("service %s: %s", service.name, reason)
            self._set(entry, "down", f"{reason}, restarting in {backoff:.0f} s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)

    async def _run_once(self, entry: _Entry) -> tuple[str, float]:
        """Run the server until it fails. Returns why, and how long it was healthy."""
        service = entry.service
        try:
            process = await self._spawn(entry)
        except OSError as exc:
            return f"could not start: {exc}", 0.0
        started = time.monotonic()
        entry.loading_since = started
        self._set(entry, "loading", "loading")
        ready_at: float | None = None
        failures = 0
        try:
            while True:
                try:
                    await asyncio.wait_for(process.wait(), self.poll_s)
                except TimeoutError:
                    pass
                await asyncio.to_thread(self._rotate, entry)
                healthy_s = time.monotonic() - ready_at if ready_at else 0.0
                if process.returncode is not None:
                    return f"exited with code {process.returncode}", healthy_s
                if await self._answers(service):
                    failures = 0
                    if ready_at is None:
                        ready_at = time.monotonic()
                        self._remember_load_time(service.name, ready_at - started)
                        entry.loading_since = None
                        self._set(entry, "ok", "ready")
                elif ready_at is None:
                    if time.monotonic() - started > service.startup_timeout_s:
                        limit = service.startup_timeout_s
                        return f"not healthy after {limit:.0f} s", 0.0
                else:
                    failures += 1
                    if failures >= self.failed_polls:
                        return "stopped answering health checks", healthy_s
        finally:
            entry.loading_since = None
            await self._terminate(entry)

    async def _spawn(self, entry: _Entry) -> asyncio.subprocess.Process:
        service = entry.service
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._rotate(entry)
        with self._log_path(service).open("ab") as log:
            log.write(f"\n=== starting {service.command} at {time.ctime()}\n".encode())
            log.flush()
            # A new session makes the server a group leader, so stopping it reaches its children.
            entry.process = await asyncio.create_subprocess_exec(
                *service.command,
                cwd=self.cwd,
                env={**os.environ, **service.env},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        return entry.process

    def _log_path(self, service: LocalService) -> Path:
        return self.log_dir / f"{_UNSAFE_NAME.sub('_', service.name)}.log"

    def _rotate(self, entry: _Entry) -> None:
        rotate_log(
            self._log_path(entry.service), max_bytes=self.log_max_bytes, backups=self.log_backups
        )

    async def _terminate(self, entry: _Entry) -> None:
        """Stop everything left in the server's process group, including when the server itself
        already died and left workers behind."""
        process, entry.process = entry.process, None
        if process is None:
            return
        group = process.pid
        _signal_group(group, signal.SIGTERM)
        deadline = time.monotonic() + self.stop_grace_s
        while time.monotonic() < deadline:
            if not await asyncio.to_thread(group_members, group):
                break
            await asyncio.sleep(_GROUP_POLL_S)
        _signal_group(group, signal.SIGKILL)
        if process.returncode is None:
            await process.wait()

    async def _stop(self, entry: _Entry) -> None:
        task, entry.task = entry.task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._terminate(entry)
        self._set(entry, "off", "stopped")

    async def _answers(self, service: LocalService) -> bool:
        if self._http is None:
            # trust_env off: a desktop's proxy settings must not catch loopback health checks.
            self._http = httpx.AsyncClient(timeout=2.0, trust_env=False)
        try:
            url = f"http://127.0.0.1:{service.port}{service.health_path}"
            return (await self._http.get(url)).is_success
        except httpx.HTTPError:
            return False

    # State -------------------------------------------------------------------------------------

    def _status(self, entry: _Entry) -> ServiceStatus:
        process = entry.process
        loading = entry.state == "loading" and entry.loading_since is not None
        return ServiceStatus(
            service=entry.service,
            name=entry.service.name,
            port=entry.service.port,
            state=entry.state,
            detail=entry.detail,
            pid=process.pid if process is not None and process.returncode is None else None,
            managed=entry.managed,
            elapsed_s=round(time.monotonic() - entry.loading_since, 1) if loading else None,
            usual_s=self._usual.get(entry.service.name),
            restarts=entry.restarts,
        )

    def _set(self, entry: _Entry, state: str, detail: str) -> None:
        if (entry.state, entry.detail) != (state, detail):
            entry.state, entry.detail = state, detail
            self._changed()

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def _read_times(self) -> dict[str, float]:
        try:
            return json.loads(self._times_path.read_text())
        except (OSError, ValueError):
            return {}

    def _remember_load_time(self, name: str, seconds: float) -> None:
        self._usual[name] = round(seconds, 1)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._times_path.write_text(json.dumps(self._usual))


def _port_taken(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def group_members(group: int) -> set[int]:
    """Live processes in a server's group.

    Servers start as the leader of a new session, so a member's group and session both equal the
    server's pid. Asking for both is what makes signalling a group safe once its leader is gone:
    that number could in principle go to an unrelated process, which wouldn't lead a matching
    session and group. A process that called setsid has left the group, and we can't claim it.
    """
    members = set()
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat.read_text()
        except OSError:  # it exited while we looked
            continue
        # Field 2 is the command name, which may itself hold spaces and parentheses.
        fields = text.rpartition(")")[2].split()
        state, process_group, session = fields[0], int(fields[2]), int(fields[3])
        if process_group == group and session == group and state not in _DEAD_STATES:
            members.add(int(stat.parent.name))
    return members


def _signal_group(group: int, signum: signal.Signals) -> None:
    if group_members(group):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signum)
