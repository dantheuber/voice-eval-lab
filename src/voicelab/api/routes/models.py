"""Models and settings: profiles, roles, health, servers, thresholds, AWS."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter

from voicelab.api import mapping
from voicelab.api.schemas import ActiveProfileBody, RoleBody, Svc, ThresholdsBody
from voicelab.core.profiles import OPTIONAL_ROLES, ROLE_INTERFACES, Profile
from voicelab.services import authoring, system
from voicelab.services.container import THRESHOLDS_KEY
from voicelab.services.errors import Conflict, NotFound
from voicelab.services.profiles import (
    INTERFACE_LABELS,
    NONE_ADAPTER,
    ROLE_LABELS,
    adapter_id,
    service_for,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _profile_json(svc: Svc, profile: Profile) -> dict[str, Any]:
    root = svc.settings.profiles_dir.parent.parent
    path = svc.profiles.paths[profile.id]
    return {
        "id": profile.id,
        "label": profile.label,
        "description": profile.description,
        "configPath": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "health": svc.profiles.overall_state(profile),
        "liveCapable": profile.live_capable,
        "benchmark": svc.profiles.benchmark(profile),
        "untested": profile.untested,
    }


def _refuse_while_busy(svc: Svc, doing: str) -> None:
    if svc.worker.is_busy() or svc.talks.live:
        raise Conflict(f"can't {doing} while a run or mic call is in progress")


@router.get("/profiles")
def list_profiles(svc: Svc) -> dict[str, Any]:
    return {
        "items": [_profile_json(svc, profile) for profile in svc.profiles.items.values()],
        "activeId": svc.profiles.active_id,
    }


@router.put("/profiles/active")
async def set_active_profile(body: ActiveProfileBody, svc: Svc) -> dict[str, Any]:
    _refuse_while_busy(svc, "change profile")
    return _profile_json(svc, await svc.profiles.activate(body.id))


def _role_json(svc: Svc, profile: Profile, role: str) -> dict[str, Any]:
    config = profile.roles.get(role)
    health = next(item for item in svc.profiles.health(profile) if item.role == role)
    return {
        "role": role,
        "label": ROLE_LABELS[role],
        "interface": INTERFACE_LABELS[ROLE_INTERFACES[role]],
        "adapter": adapter_id(config) if config else NONE_ADAPTER,
        "options": svc.profiles.role_options(role),
        "health": mapping.to_api(health),
    }


@router.get("/roles")
def list_roles(svc: Svc) -> dict[str, Any]:
    profile = svc.profiles.items.get(svc.profiles.active_id or "")
    if profile is None:
        return {"items": []}
    roles = [r for r in ROLE_INTERFACES if r in profile.roles or r in OPTIONAL_ROLES]
    return {"items": [_role_json(svc, profile, role) for role in roles]}


@router.put("/roles/{role}")
async def set_role(role: str, body: RoleBody, svc: Svc) -> dict[str, Any]:
    _refuse_while_busy(svc, "change a role")
    profile = await svc.profiles.set_role(role, body.adapter)
    return _role_json(svc, profile, role)


@router.get("/health")
def health(svc: Svc) -> dict[str, Any]:
    return {"items": mapping.to_api(svc.profiles.health()), "ready": svc.profiles.is_ready()}


@router.post("/servers/restart-all")
async def restart_all(svc: Svc) -> dict[str, bool]:
    for name in svc.supervisor.statuses():
        svc.spawn(svc.supervisor.restart(name))
    return {"accepted": True}


@router.post("/servers/{role}/restart")
async def restart_server(role: str, svc: Svc) -> dict[str, bool]:
    profile = svc.profiles.resolve(None)
    service = service_for(profile, role)
    if service is None or service.name not in svc.supervisor.statuses():
        raise NotFound(f"no server behind role {role!r}")
    svc.spawn(svc.supervisor.restart(service.name))
    return {"accepted": True}


@router.post("/benchmark")
async def benchmark(svc: Svc) -> dict[str, bool]:
    profile = svc.profiles.resolve(None)
    stack = svc.profiles.stack(profile)

    async def measure() -> None:
        svc.profiles.record_benchmark(profile, await authoring.benchmark_reply(stack))

    svc.spawn(measure())
    return {"accepted": True}


@router.get("/system")
async def system_info(svc: Svc) -> dict[str, Any]:
    settings = svc.settings
    profile = svc.profiles.items.get(svc.profiles.active_id or "")
    statuses = svc.supervisor.statuses()
    first_role: dict[str, str] = {}
    for role in ROLE_INTERFACES:
        service = service_for(profile, role) if profile else None
        if service is not None:
            first_role.setdefault(service.name, role)
    address = await asyncio.to_thread(system.lan_ip)
    return {
        "gpu": await asyncio.to_thread(system.gpu_info),
        "servers": [
            {
                "role": first_role.get(name, name),
                "kind": name,
                "port": status.port,
                "pid": status.pid,
            }
            for name, status in statuses.items()
        ],
        "urls": {
            "local": f"http://127.0.0.1:{settings.http_port}",
            "lan": f"https://{address or settings.http_host}:{settings.https_port}",
        },
        "cert": "self-signed",
    }


@router.get("/thresholds")
def get_thresholds(svc: Svc) -> dict[str, Any]:
    return svc.thresholds()


@router.put("/thresholds")
def set_thresholds(body: ThresholdsBody, svc: Svc) -> dict[str, Any]:
    stored = body.model_dump(by_alias=True, exclude={"audio_listening_mode"})
    svc.catalog.set_setting(THRESHOLDS_KEY, stored)
    return svc.thresholds()


@router.get("/aws")
def aws() -> dict[str, Any]:
    return system.aws_status()


@router.post("/aws/test")
async def aws_test() -> dict[str, Any]:
    return await asyncio.to_thread(system.aws_test)
