#!/usr/bin/env python3
"""Checks this machine for voice-eval-lab and says what to fix. It changes nothing.

Standard library only, so it runs before `uv sync`:

    python3 scripts/doctor.py [--profile local-default] [--http-port 18080] [--https-port 18443]

Exit code 1 means something needs fixing before `./run.sh` can work. Warnings don't count.
"""

from __future__ import annotations

import argparse
import ctypes
import grp
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "setup" / "manifest.tsv"
GB = 1_000_000_000
GIB = 1024**3

# GPU memory with the live stack, Whisper, and the judge all loaded, from the Memory table in
# docs/research/benchmarks-2026-10.md. Only local-realtime's total was measured as a whole. The
# other two are that table's per-model figures added up.
GPU_NEEDED_GB = {"local-default": 27.4, "local-fast": 22.7, "local-realtime": 29.8}
UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
AMD_VENDOR = 0x1002
CPU_DEVICE = 4


@dataclass
class Report:
    failures: int = 0
    warnings: int = 0

    def ok(self, text: str) -> None:
        print(f"ok    {text}")

    def warn(self, text: str, fix: str = "") -> None:
        self.warnings += 1
        print(f"warn  {text}")
        if fix:
            print(f"      {fix}")

    def fail(self, text: str, fix: str = "") -> None:
        self.failures += 1
        print(f"FAIL  {text}")
        if fix:
            print(f"      {fix}")


def vulkan_probe() -> None:
    """Runs in a child process, because a broken driver can take the whole process down."""

    class InstanceInfo(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_int),
            ("pNext", ctypes.c_void_p),
            ("flags", ctypes.c_uint32),
            ("pApplicationInfo", ctypes.c_void_p),
            ("enabledLayerCount", ctypes.c_uint32),
            ("ppEnabledLayerNames", ctypes.c_void_p),
            ("enabledExtensionCount", ctypes.c_uint32),
            ("ppEnabledExtensionNames", ctypes.c_void_p),
        ]

    vulkan = ctypes.CDLL("libvulkan.so.1")
    instance = ctypes.c_void_p()
    info = InstanceInfo(sType=1)  # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    result = vulkan.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance))
    if result != 0:
        raise SystemExit(f"vkCreateInstance returned {result}")
    count = ctypes.c_uint32()
    vulkan.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
    handles = (ctypes.c_void_p * count.value)()
    vulkan.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles)
    devices = []
    for handle in handles:
        # VkPhysicalDeviceProperties is 824 bytes, and these fields sit at fixed offsets.
        props = ctypes.create_string_buffer(1024)
        vulkan.vkGetPhysicalDeviceProperties(ctypes.c_void_p(handle), props)
        api, _driver, vendor, _device, kind = (
            int.from_bytes(props.raw[start : start + 4], "little") for start in range(0, 20, 4)
        )
        version = f"{(api >> 22) & 0x7F}.{(api >> 12) & 0x3FF}"
        name = props.raw[20:276].split(b"\0", 1)[0].decode(errors="replace")
        devices.append({"name": name, "vendor": vendor, "kind": kind, "vulkan": version})
    print(json.dumps(devices))


def check_system(report: Report) -> None:
    machine = f"{platform.system()} {platform.machine()}"
    libc = " ".join(platform.libc_ver())
    if machine == "Linux x86_64":
        report.ok(f"{machine}, kernel {platform.release()}, {libc}")
    else:
        report.fail(f"{machine}", "The binaries and wheels setup installs are Linux x86-64 only.")


def check_vulkan(report: Report) -> None:
    probe = subprocess.run(
        [sys.executable, "-I", __file__, "--vulkan-probe"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        detail = (probe.stderr.strip().splitlines() or [f"exit code {probe.returncode}"])[-1]
        report.fail(
            f"Vulkan: could not list devices ({detail})",
            "Install your distro's Vulkan loader and Mesa driver, such as libvulkan1 and "
            "mesa-vulkan-drivers. That step needs an administrator.",
        )
        return
    devices = json.loads(probe.stdout)
    real = [device for device in devices if device["kind"] != CPU_DEVICE]
    if not real:
        report.fail(
            "Vulkan: only a software renderer is available",
            "Models would run on the CPU, far below the benchmark numbers. Check the GPU driver.",
        )
        return
    first = real[0]
    text = f"Vulkan {first['vulkan']}: {first['name']}"
    if first["vendor"] == AMD_VENDOR:
        report.ok(text)
    else:
        report.warn(f"{text}. Only RADV on a Radeon 8060S has been tested.")


def check_render_group(report: Report) -> None:
    nodes = sorted(Path("/dev/dri").glob("renderD*"))
    if not nodes:
        report.fail("No /dev/dri/renderD* device", "This machine shows no GPU to programs.")
        return
    usable = [node for node in nodes if os.access(node, os.R_OK | os.W_OK)]
    group = grp.getgrgid(nodes[0].stat().st_gid).gr_name
    if usable:
        member = "in" if nodes[0].stat().st_gid in os.getgroups() else "not in, but access works,"
        report.ok(f"{usable[0]} is readable and writable, you are {member} group {group}")
    else:
        report.fail(
            f"Can't open {nodes[0]}, which belongs to group {group}",
            f"An administrator must run: usermod -aG {group} {os.environ.get('USER', 'you')}. "
            "Then log out and back in.",
        )


def _sysfs_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0


def check_memory(report: Report, profile: str) -> None:
    info = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line
    )
    total = int(info["MemTotal"].split()[0]) * 1024 / GIB
    available = int(info["MemAvailable"].split()[0]) * 1024 / GIB
    needed = GPU_NEEDED_GB.get(profile)
    usage = f"{available:.0f} GiB of {total:.0f} GiB memory available"
    if needed is None:
        report.ok(usage)
    elif available >= needed:
        report.ok(f"{usage}, {profile} needs about {needed} GB with everything loaded")
    else:
        report.warn(
            f"{usage}, {profile} needs about {needed} GB with everything loaded",
            "If its own servers are already up, that memory is counted as used here.",
        )
    gpu_total = max(
        (
            _sysfs_int(card / "device/mem_info_vram_total")
            + _sysfs_int(card / "device/mem_info_gtt_total")
            for card in Path("/sys/class/drm").glob("card[0-9]")
        ),
        default=0,
    )
    gpu_gb = gpu_total / GIB
    if not gpu_total:
        report.warn("GPU memory size is not readable from sysfs, which is amdgpu only")
    elif needed is not None and gpu_gb < needed:
        report.fail(
            f"The GPU can address {gpu_gb:.0f} GiB, {profile} needs about {needed} GB",
            "On an integrated GPU, raise the GTT size or the BIOS carveout.",
        )
    else:
        report.ok(f"GPU can address {gpu_gb:.0f} GiB")


def _listeners() -> dict[int, tuple[str, int] | None]:
    """Port to (name, pid). None when another user's process holds it."""
    if shutil.which("ss") is None:
        return {}
    listing = subprocess.run(["ss", "-H", "-ltnp"], capture_output=True, text=True).stdout
    found: dict[int, tuple[str, int] | None] = {}
    for line in listing.splitlines():
        columns = line.split()
        if len(columns) < 4:
            continue
        port = int(columns[3].rsplit(":", 1)[-1])
        owner = re.search(r'users:\(\("([^"]+)",pid=(\d+)', line)
        found[port] = (owner[1], int(owner[2])) if owner else None
    return found


def _profile_file(profile: str) -> Path | None:
    for path in sorted((ROOT / "config" / "profiles").glob("*.yaml")):
        if re.search(rf'^id: *"?{re.escape(profile)}"?\s*$', path.read_text(), re.M):
            return path
    return None


def check_ports(report: Report, profile_text: str, http_port: int, https_port: int) -> None:
    if shutil.which("ss") is None:
        report.warn("ss is missing, so ports were not checked")
        return
    wanted = {http_port: "dashboard", https_port: "LAN dashboard"}
    names = re.findall(r"^  - name: *(\S+)\n    port: *(\d+)", profile_text, re.M)
    wanted |= {int(port): name for name, port in names}
    listeners = _listeners()
    free = [port for port in wanted if port not in listeners]
    if free:
        report.ok("free ports: " + ", ".join(str(port) for port in sorted(free)))
    for port in sorted(set(wanted) & set(listeners)):
        owner = listeners[port]
        if owner is None:
            report.fail(f"port {port} ({wanted[port]}) is held by another user's process")
            continue
        name, pid = owner
        try:
            ours = Path(f"/proc/{pid}/cwd").resolve() == ROOT
        except OSError:
            ours = False
        text = f"port {port} ({wanted[port]}) is held by {name} (pid {pid})"
        if ours:
            report.warn(f"{text}, from this checkout. run.sh will refuse until it stops")
        else:
            report.fail(text)


def check_model_files(report: Report, profile: str, profile_text: str) -> None:
    missing_required: list[tuple[str, int]] = []
    missing_on_demand: list[tuple[str, int]] = []
    count = 0
    for line in MANIFEST.read_text().splitlines():
        if line.startswith("#"):
            continue
        profiles, need, path, size, *_ = line.split("\t")
        if profile not in profiles.split(","):
            continue
        count += 1
        target = ROOT / path
        if not target.exists() or target.stat().st_size != int(size):
            (missing_required if need == "required" else missing_on_demand).append(
                (path, int(size))
            )
    setup = f"Fix: scripts/setup.sh --profile {profile}"
    if count and not missing_required:
        report.ok(f"{count - len(missing_on_demand)} of {count} model files for {profile}")
    voices = [size for path, size in missing_required if "/voices/" in path]
    if voices:
        report.fail(f"{len(voices)} Phonon voices missing, {sum(voices) / GB:.3f} GB", setup)
    for path, size in missing_required:
        if "/voices/" not in path:
            report.fail(f"missing {path}, {size / GB:.3f} GB", setup)
    if missing_on_demand:
        names = ", ".join(Path(path).stem for path, _ in missing_on_demand)
        report.warn(f"Pocket voices download on first use: {names}", setup)
    paths = set(re.findall(r"\.runtime/[^\s\",}]+", profile_text))
    absent = sorted(path for path in paths if not (ROOT / path).exists())
    for path in absent:
        report.fail(f"profile names {path}, which is not here", setup)


def _version(tool: str) -> str:
    return subprocess.run([tool, "--version"], capture_output=True, text=True).stdout.strip()


def check_tools(report: Report) -> None:
    uv = shutil.which("uv")
    if uv:
        report.ok(_version(uv))
    else:
        report.fail("uv is missing", f"It installs without sudo: {UV_INSTALL}")
    for tool in ("curl", "git"):
        if shutil.which(tool) is None:
            report.fail(f"{tool} is missing")
    built = (ROOT / "web" / "dist" / "index.html").exists()
    for tool in ("node", "pnpm"):
        path = shutil.which(tool)
        if path:
            report.ok(f"{tool} {_version(path).removeprefix('v')}")
        elif built:
            report.warn(f"{tool} is missing. web/dist exists, so this only blocks a rebuild")
        else:
            report.fail(f"{tool} is missing, and the dashboard is not built")


def check_disk(report: Report) -> None:
    free = shutil.disk_usage(ROOT).free / GB
    text = f"{free:.0f} GB free on the disk holding this checkout"
    if free < 10:
        report.warn(f"{text}. Recordings land in data/, and a full setup needs about 31 GB")
    else:
        report.ok(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="local-default")
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--https-port", type=int, default=18443)
    parser.add_argument("--vulkan-probe", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.vulkan_probe:
        vulkan_probe()
        return 0

    profile_file = _profile_file(arguments.profile)
    if profile_file is None:
        print(f"no profile named {arguments.profile!r} in config/profiles", file=sys.stderr)
        return 2
    profile_text = profile_file.read_text()

    report = Report()
    print(f"voice-eval-lab doctor, profile {arguments.profile}")
    check_system(report)
    check_vulkan(report)
    check_render_group(report)
    check_memory(report, arguments.profile)
    check_disk(report)
    check_ports(report, profile_text, arguments.http_port, arguments.https_port)
    check_model_files(report, arguments.profile, profile_text)
    check_tools(report)

    if report.failures:
        print(f"{report.failures} problem(s) to fix, {report.warnings} warning(s).")
        return 1
    print(f"No blocking problems, {report.warnings} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
