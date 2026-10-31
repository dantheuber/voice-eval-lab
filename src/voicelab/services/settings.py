"""Where the backend keeps things and how it listens. Environment variables override defaults."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]

ServiceStart = Literal["startup", "lazy"]


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    profiles_dir: Path = field(default_factory=lambda: REPO_ROOT / "config" / "profiles")
    web_dist: Path = field(default_factory=lambda: REPO_ROOT / "web" / "dist")
    certs_dir: Path = field(default_factory=lambda: REPO_ROOT / "certs")
    http_host: str = "127.0.0.1"
    http_port: int = 18080
    https_host: str = "0.0.0.0"
    https_port: int = 18443
    service_start: ServiceStart = "startup"  # "lazy" waits until something needs a model
    default_profile: str = "local-default"
    reviewer: str = field(default_factory=getpass.getuser)  # no login, so decisions carry this
    model_wait_s: float = 900.0  # how long a queued call waits for its model servers
    talk_connect_timeout_s: float = 120.0  # a session the browser never connected
    talk_max_s: float = 7200.0  # a tab that vanished must not hold back the run queue forever
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024  # 2 hours of 48 kHz stereo WAV is 1.3 GiB
    extra_hosts: tuple[str, ...] = ()  # names this server is also reached by, such as a proxy's

    @property
    def db_path(self) -> Path:
        return self.data_dir / "voicelab.sqlite3"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls()
        settings.data_dir = _path("VOICELAB_DATA_DIR", settings.data_dir)
        settings.profiles_dir = _path("VOICELAB_PROFILES_DIR", settings.profiles_dir)
        settings.web_dist = _path("VOICELAB_WEB_DIST", settings.web_dist)
        settings.certs_dir = _path("VOICELAB_CERTS_DIR", settings.certs_dir)
        extra_hosts = os.environ.get("VOICELAB_EXTRA_HOSTS", "")
        settings.extra_hosts = tuple(
            name.strip() for name in extra_hosts.split(",") if name.strip()
        )
        start = os.environ.get("VOICELAB_SERVICE_START")
        if start in ("startup", "lazy"):
            settings.service_start = start
        return settings
