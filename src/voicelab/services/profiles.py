"""The active model profile: its adapters, its servers, and role health."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from voicelab.core.profiles import (
    OPTIONAL_ROLES,
    ROLE_INTERFACES,
    LocalService,
    Profile,
    ProviderStack,
    build_stack,
)
from voicelab.services.errors import Conflict, Invalid, ModelNotReady, NotFound
from voicelab.services.settings import Settings
from voicelab.services.supervisor import ServiceStatus, ServiceSupervisor
from voicelab.store.catalog import Catalog
from voicelab.store.sqlite import timestamp

logger = logging.getLogger(__name__)

ROLE_LABELS = {
    "stt_live": "STT, live",
    "stt_eval": "STT, post-call eval",
    "tts_bot": "TTS, bot voice",
    "tts_caller": "TTS, caller voice",
    "llm_support": "support bot LLM",
    "llm_caller": "caller LLM",
    "llm_judge": "judge LLM",
    "realtime": "support bot realtime voice",
    "audio_analyst": "audio analyst",
}
INTERFACE_LABELS = {
    "stt": "STT",
    "tts": "TTS",
    "llm": "LLM",
    "realtime": "RealtimeVoice",
    "audio_analyst": "AudioAnalyst",
}
_SEVERITY = {"ok": 0, "off": 1, "loading": 2, "down": 3}
NONE_ADAPTER = "none"


@dataclass
class Health:
    role: str
    name: str
    model: str
    state: str  # ok, loading, down, off
    detail: str
    elapsed_s: float | None = None
    usual_s: float | None = None


def adapter_id(config: dict[str, Any]) -> str:
    model = config.get("model") or config.get("name")
    return f"{config['type']}:{model}" if model else str(config["type"])


def _uses_port(config: dict[str, Any], port: int) -> bool:
    mentions = re.compile(rf":{port}\b")
    return any(
        (isinstance(value, str) and mentions.search(value)) or value == port
        for value in config.values()
    )


def service_for(profile: Profile, role: str) -> LocalService | None:
    """The server behind a role. Profiles link them through the port in the adapter's URL."""
    config = profile.roles.get(role)
    if config is None:
        return None
    return next((s for s in profile.services if _uses_port(config, s.port)), None)


def read_profile_files(directory: Path) -> Iterator[tuple[Path, Profile]]:
    """Every profile that loads. One bad file shouldn't hide the ones that work."""
    for path in sorted(directory.glob("*.yaml")):
        try:
            yield path, Profile.model_validate(yaml.safe_load(path.read_text()))
        except Exception as exc:
            logger.error("skipping unreadable profile %s: %s", path, exc)


class ProfileManager:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        supervisor: ServiceSupervisor,
        *,
        stack_factory: Callable[[Profile], ProviderStack] = build_stack,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.supervisor = supervisor
        self.stack_factory = stack_factory
        self.items: dict[str, Profile] = {}
        self.paths: dict[str, Path] = {}
        self._stacks: dict[str, ProviderStack] = {}
        self._held: list[Profile] = []  # one entry per piece of work using a profile's servers
        self._released = asyncio.Event()
        self.reload()

    def reload(self) -> None:
        self.items, self.paths, self._stacks = {}, {}, {}
        for path, profile in read_profile_files(self.settings.profiles_dir):
            self.items[profile.id] = profile
            self.paths[profile.id] = path

    @property
    def active_id(self) -> str | None:
        stored = self.catalog.setting("active_profile")
        if stored in self.items:
            return stored
        if self.settings.default_profile in self.items:
            return self.settings.default_profile
        # Sorted by file name, an example cloud stack would otherwise win by alphabet.
        tested = [profile.id for profile in self.items.values() if not profile.untested]
        return next(iter(tested or self.items), None)

    def require(self, profile_id: str | None) -> Profile:
        profile = self.items.get(profile_id or "")
        if profile is None:
            raise Invalid(f"unknown profile {profile_id!r}")
        return profile

    def resolve(self, profile_id: str | None) -> Profile:
        """The named profile or the active one, for routes that need models."""
        profile_id = profile_id or self.active_id
        if profile_id is None:
            raise ModelNotReady("profile", reason="no model profile in config/profiles")
        return self.require(profile_id)

    def stack(self, profile: Profile) -> ProviderStack:
        if profile.id not in self._stacks:
            # An adapter can fail any way it likes, such as a missing optional package.
            try:
                self._stacks[profile.id] = self.stack_factory(profile)
            except Exception as exc:
                reason = f"can't build profile {profile.id}: {exc}"
                raise ModelNotReady("profile", profile.id, reason=reason) from exc
        return self._stacks[profile.id]

    def benchmark(self, profile: Profile) -> dict[str, Any] | None:
        measured = self.catalog.setting(f"benchmark:{profile.id}")
        if measured:
            return measured
        declared = profile.benchmark
        if "bot_reply_p50_s" in declared:
            return {"botReplyP50S": declared["bot_reply_p50_s"], "at": str(declared.get("at", ""))}
        return None

    def record_benchmark(self, profile: Profile, reply_p50_s: float) -> None:
        self.catalog.set_setting(
            f"benchmark:{profile.id}", {"botReplyP50S": reply_p50_s, "at": timestamp()}
        )

    # Readiness -----------------------------------------------------------------------------------

    @property
    def in_use(self) -> bool:
        return bool(self._held)

    def _clashing(self, profile: Profile) -> Profile | None:
        """A profile in use whose servers this one would have to replace."""
        return next(
            (held for held in self._held if held.id != profile.id and _share_a_slot(held, profile)),
            None,
        )

    async def require_ready(
        self, profile: Profile, roles: Iterable[str], *, hold: bool = False
    ) -> ProviderStack:
        """The stack, or `model_not_ready` naming the first role still loading. Doesn't wait, and
        asks for a lazy server to start so that a retry can succeed. With `hold`, the caller keeps
        these servers until it calls `release`."""
        roles = list(roles)
        other = self._clashing(profile)
        if other is not None:
            role = roles[0]
            reason = f"model servers are in use by profile {other.id}"
            raise ModelNotReady(role, _model_name(profile.roles.get(role, {})), reason=reason)
        if hold:
            self._held.append(profile)
        try:
            services = [service_for(profile, role) for role in roles]
            await self.supervisor.reconcile(service for service in services if service)
            statuses = self.supervisor.statuses()
            for role, service in zip(roles, services, strict=True):
                status = statuses[service.name] if service else None
                if status is not None and status.state != "ok":
                    raise ModelNotReady(
                        role,
                        _model_name(profile.roles.get(role, {})),
                        elapsed_s=status.elapsed_s,
                        usual_s=status.usual_s,
                        reason=f"{ROLE_LABELS.get(role, role)} is {status.state}: {status.detail}",
                    )
            return self.stack(profile)
        except BaseException:
            if hold:
                self.release(profile)
            raise

    async def ready(self, profile: Profile, timeout_s: float) -> ProviderStack:
        """Hold the profile's servers for one piece of queued work and wait for them. The caller
        must call `release`."""
        await self.hold(profile, timeout_s)
        try:
            return self.stack(profile)
        except BaseException:
            self.release(profile)
            raise

    async def hold(self, profile: Profile, timeout_s: float) -> None:
        """Wait for any profile sharing a server slot to finish, then claim the slots. Replacing a
        server under a running call would break it, so the swap waits for its turn."""
        deadline = time.monotonic() + timeout_s
        while (other := self._clashing(profile)) is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                reason = f"model servers are in use by profile {other.id}"
                raise ModelNotReady("profile", profile.id, reason=reason)
            self._released.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._released.wait(), left)
        self._held.append(profile)
        try:
            await self.wait_for_servers(profile, max(deadline - time.monotonic(), 0.0))
        except BaseException:
            self.release(profile)
            raise

    def release(self, profile: Profile) -> None:
        position = next(i for i, held in enumerate(self._held) if held.id == profile.id)
        del self._held[position]
        self._released.set()

    async def wait_for_servers(self, profile: Profile, timeout_s: float) -> None:
        await self.supervisor.reconcile(profile.services)
        if not await self.supervisor.wait_ready(profile.services, timeout_s):
            raise ModelNotReady(
                "profile", profile.id, reason=f"model servers not ready after {timeout_s:.0f} s"
            )

    # Health --------------------------------------------------------------------------------------

    def health(self, profile: Profile | None = None) -> list[Health]:
        profile = profile or self.items.get(self.active_id or "")
        if profile is None:
            detail = "no model profile in config/profiles"
            return [Health("profile", "model profile", "", "down", detail)]
        statuses = self.supervisor.statuses()
        return [self._role_health(profile, role, statuses) for role in ROLE_INTERFACES]

    def _role_health(
        self, profile: Profile, role: str, statuses: dict[str, ServiceStatus]
    ) -> Health:
        name = ROLE_LABELS[role]
        config = profile.roles.get(role)
        if config is None:
            return Health(role, name, "", "off", "not configured")
        model = _model_name(config)
        service = service_for(profile, role)
        if service is None:
            return Health(role, name, model, "ok", "no local server to watch")
        status = statuses.get(service.name)
        if status is None:
            return Health(role, name, model, "off", "not started")
        if status.service != service:
            return Health(role, name, model, "off", "another profile's server is on this slot")
        return Health(
            role, name, model, status.state, status.detail, status.elapsed_s, status.usual_s
        )

    def is_ready(self) -> bool:
        items = self.health()
        return (
            all(item.state == "ok" or item.role in OPTIONAL_ROLES for item in items)
            and items[0].role != "profile"
        )

    def overall_state(self, profile: Profile) -> str:
        required = [h for h in self.health(profile) if h.role not in OPTIONAL_ROLES]
        return max((h.state for h in required), key=_SEVERITY.__getitem__, default="off")

    # Changes -------------------------------------------------------------------------------------

    async def activate(self, profile_id: str) -> Profile:
        profile = self.require(profile_id)
        self.catalog.set_setting("active_profile", profile.id)
        await self.supervisor.apply(profile.services, start=True)
        return profile

    def role_options(self, role: str) -> list[dict[str, Any]]:
        """What each role could use: every adapter any profile configures for it."""
        options: dict[str, dict[str, Any]] = {}
        if role in OPTIONAL_ROLES:
            options[NONE_ADAPTER] = {
                "id": NONE_ADAPTER,
                "label": "none",
                "untested": False,
                "license": "",
            }
        for profile in self.items.values():
            config = profile.roles.get(role)
            if config is None:
                continue
            option_id = adapter_id(config)
            options.setdefault(
                option_id,
                {
                    "id": option_id,
                    "label": _model_name(config),
                    "untested": profile.untested,
                    "license": str(config.get("license", "")),
                },
            )
        return list(options.values())

    async def set_role(self, role: str, adapter: str) -> Profile:
        """Point a role of the active profile at another adapter and write the YAML."""
        if role not in ROLE_INTERFACES:
            raise NotFound(f"unknown role {role!r}")
        profile = self.resolve(None)
        path = self.paths[profile.id]
        data = yaml.safe_load(path.read_text())
        if adapter == NONE_ADAPTER:
            if role not in OPTIONAL_ROLES:
                raise Conflict(f"{role} is required")
            data["roles"].pop(role, None)
        else:
            source = self._profile_with_adapter(role, adapter)
            if source is None:
                raise Invalid(f"no profile configures adapter {adapter!r} for {role}")
            data["roles"][role] = source.roles[role]
            service = service_for(source, role)
            names = {s["name"] for s in data.get("services", [])}
            if service is not None and service.name not in names:
                data.setdefault("services", []).append(service.model_dump())
        data["services"] = _services_still_used(data)
        path.write_text(_leading_comments(path.read_text()) + yaml.safe_dump(data, sort_keys=False))
        self.reload()
        profile = self.require(profile.id)
        await self.supervisor.apply(profile.services, start=True)
        return profile

    def _profile_with_adapter(self, role: str, adapter: str) -> Profile | None:
        for profile in self.items.values():
            config = profile.roles.get(role)
            if config is not None and adapter_id(config) == adapter:
                return profile
        return None


def _leading_comments(text: str) -> str:
    """PyYAML drops comments, and these files keep their license terms in the header."""
    lines = []
    for line in text.splitlines(keepends=True):
        if not line.startswith("#"):
            break
        lines.append(line)
    return "".join(lines)


def _share_a_slot(first: Profile, second: Profile) -> bool:
    """Whether two profiles want different servers on one name or port."""
    return any(
        (a.name == b.name or a.port == b.port) and a != b
        for a in first.services
        for b in second.services
    )


def _model_name(config: dict[str, Any]) -> str:
    return str(config.get("model") or config.get("name") or config.get("type") or "")


def _services_still_used(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Drop servers no role points at, so a swapped-out model stops holding memory."""
    roles = data["roles"].values()
    return [s for s in data.get("services", []) if any(_uses_port(c, s["port"]) for c in roles)]
