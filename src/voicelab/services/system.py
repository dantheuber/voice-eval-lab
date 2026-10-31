"""What the settings screen shows about the machine: GPU memory, LAN address, and AWS status."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

_GIB = 1024**3


def lan_ip() -> str | None:
    """The address other machines reach this one on. A UDP connect sends nothing; it only makes
    the kernel pick the outbound interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            address = probe.getsockname()[0]
        except OSError:
            return None
    return None if address.startswith("127.") else address


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0


def gpu_info(drm_root: Path = Path("/sys/class/drm")) -> dict[str, Any]:
    """Memory on the amdgpu device. It's an iGPU, so most of it is GTT rather than VRAM."""
    best: dict[str, Any] = {"device": "unknown", "memoryUsedGb": 0.0, "memoryTotalGb": 0.0}
    for card in sorted(drm_root.glob("card[0-9]")):
        device = card / "device"
        total = _read_int(device / "mem_info_vram_total") + _read_int(device / "mem_info_gtt_total")
        if total / _GIB <= best["memoryTotalGb"]:
            continue
        used = _read_int(device / "mem_info_vram_used") + _read_int(device / "mem_info_gtt_used")
        name = device / "product_name"
        best = {
            "device": name.read_text().strip() if name.exists() else card.name,
            "memoryUsedGb": round(used / _GIB, 1),
            "memoryTotalGb": round(total / _GIB, 1),
        }
    return best


# These go to the network or run a program, so they wait for the first real use.
_FIRST_USE_PROVIDERS = ("iam-role", "container-role", "custom-process")
_NO_BOTO3 = "boto3 isn't installed: uv sync --extra aws"
_NO_CREDENTIALS = (
    "no AWS credentials were found. Set AWS_PROFILE or AWS keys in the environment before "
    "starting. Instance and container roles aren't looked at, since that takes a network request."
)


def aws_credentials_problem() -> str | None:
    """Why boto3 couldn't use the credentials selected here, or None if it could.

    This walks boto3's own provider chain, so it agrees with boto3 about AWS_PROFILE and about what
    counts as complete. It only reads local config. A role or SSO profile is taken on trust, since
    only a request to AWS could prove one works.
    """
    try:
        import boto3  # noqa: F401  (the adapters need it, not just botocore)
        from botocore.credentials import (
            DeferredRefreshableCredentials,
            create_credential_resolver,
        )
        from botocore.exceptions import BotoCoreError
        from botocore.session import Session
    except ImportError:
        return _NO_BOTO3
    session = Session()
    try:
        selected = session.get_scoped_config()
        resolver = create_credential_resolver(session)
        for provider in _FIRST_USE_PROVIDERS:
            resolver.remove(provider)
        found = resolver.load_credentials()
    except BotoCoreError as exc:  # names a profile or a variable, never a value
        return str(exc)
    if found is None:
        return None if selected.get("credential_process") else _NO_CREDENTIALS
    if isinstance(found, DeferredRefreshableCredentials) or (found.access_key and found.secret_key):
        return None
    return f"AWS credentials in {found.method} are incomplete: a key or secret is empty"


def aws_status() -> dict[str, Any]:
    """Only whether credentials are usable. Their values never leave this module's checks."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    problem = aws_credentials_problem()
    return {
        "region": region,
        "credentials": "not_found" if problem else "found",
        "problem": problem,
        "testedAgainstLive": False,
    }


def aws_test() -> dict[str, Any]:
    try:
        import boto3
    except ImportError:
        return {"ok": False, "message": _NO_BOTO3}
    try:
        identity = boto3.client("sts").get_caller_identity()
    except Exception as exc:
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "message": f"signed in as account {identity['Account']}"}
