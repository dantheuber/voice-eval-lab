"""What the settings screen shows about the machine: GPU memory, LAN address, and AWS status."""

from __future__ import annotations

import configparser
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


def _credentials_file() -> Path:
    default = Path.home() / ".aws" / "credentials"
    return Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE", default))


def _has_credentials() -> bool:
    """Whether boto3 would find any. A named profile in the file is not used unless AWS_PROFILE
    selects it, so it doesn't count. boto3's own chain would also ask the instance metadata
    service, which costs a timeout on every settings page."""
    if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
        return True
    parser = configparser.ConfigParser()
    try:
        parser.read(_credentials_file())
    except configparser.Error:
        return False
    return parser.has_section("default")


def aws_status() -> dict[str, Any]:
    """Only whether credentials exist. Their values never leave this function's checks."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return {
        "region": region,
        "credentials": "found" if _has_credentials() else "not_found",
        "testedAgainstLive": False,
    }


def aws_test() -> dict[str, Any]:
    try:
        import boto3
    except ImportError:
        return {"ok": False, "message": "boto3 isn't installed: uv sync --extra aws"}
    try:
        identity = boto3.client("sts").get_caller_identity()
    except Exception as exc:
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "message": f"signed in as account {identity['Account']}"}
