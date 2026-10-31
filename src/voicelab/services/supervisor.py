"""Starts and watches a profile's model servers.

Each `LocalService` runs in its own process group with its log in `data/logs/`. A crash restarts it
with growing backoff, and a server that stops answering health checks is treated as crashed. The
only processes ever signalled are ones this supervisor spawned: a server already on the port is
watched and reported, never touched, because the machine runs other people's model servers.
"""

from __future__ import annotations

import asyncio
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
from voicelab.services.settings import REPO_ROOT

logger = logging.getLogger(__name__)

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass
class ServiceStatus:
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
        for name, entry in list(self._entries.items()):
            if wanted.get(name) != entry.service:
                await self._stop(entry)
                del self._entries[name]
        for name, service in wanted.items():
            self._entries.setdefault(name, _Entry(service))
        if start:
            self.ensure_started(wanted)
        self._changed()

    def add(self, services: Iterable[LocalService]) -> None:
        """Supervise these too, and start them."""
        names = []
        for service in services:
            self._entries.setdefault(service.name, _Entry(service))
            names.append(service.name)
        self.ensure_started(names)

    def ensure_started(self, names: Iterable[str] | None = None) -> None:
        for name in self._entries if names is None else names:
            entry = self._entries[name]
            if entry.task is None or entry.task.done():
                entry.task = asyncio.create_task(self._supervise(entry), name=f"service-{name}")

    async def restart(self, name: str) -> None:
        entry = self._entries[name]
        await self._stop(entry)
        self.ensure_started([name])

    async def stop_all(self) -> None:
        await asyncio.gather(*(self._stop(entry) for entry in self._entries.values()))
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def wait_ready(self, names: Iterable[str], timeout_s: float) -> bool:
        names = list(names)
        deadline = time.monotonic() + timeout_s
        while not all(self._entries[name].state == "ok" for name in names):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(0.25, self.poll_s))
        return True

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
        log_path = self.log_dir / f"{_UNSAFE_NAME.sub('_', service.name)}.log"
        with log_path.open("ab") as log:
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

    async def _terminate(self, entry: _Entry) -> None:
        process, entry.process = entry.process, None
        if process is None or process.returncode is not None:
            return
        _signal_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), self.stop_grace_s)
        except TimeoutError:
            _signal_group(process, signal.SIGKILL)
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


def _signal_group(process: asyncio.subprocess.Process, signum: signal.Signals) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass
