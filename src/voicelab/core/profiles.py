"""Model profiles: which adapter fills each role.

A profile is a YAML file under `config/profiles/`. Changing provider means editing or selecting a
profile, never engine code. See `config/profiles/bedrock.example.yaml` for a cloud stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from voicelab.providers.base import LLM, STT, TTS, AudioAnalyst, RealtimeVoice, build

ROLE_INTERFACES = {
    "stt_live": "stt",  # what each bot hears during a call
    "stt_eval": "stt",  # independent post-call transcript; must not be the live engine
    "tts_bot": "tts",
    "tts_caller": "tts",
    "llm_support": "llm",
    "llm_caller": "llm",
    "llm_judge": "llm",
    "realtime": "realtime",  # optional speech-to-speech alternative to the cascade
    "audio_analyst": "audio_analyst",  # optional, advisory
}
OPTIONAL_ROLES = {"realtime", "audio_analyst"}


class LocalService(BaseModel):
    """A model server the platform starts and supervises for this profile."""

    name: str
    command: list[str]
    port: int
    health_path: str = "/health"
    env: dict[str, str] = Field(default_factory=dict)
    startup_timeout_s: float = 180.0


class Profile(BaseModel):
    id: str
    label: str
    description: str = ""
    live_capable: bool = False  # benchmarks showed bot-to-bot can run in real time
    untested: bool = False  # adapters here have never run against the real service
    support_engine: Literal["cascade", "realtime"] = "cascade"  # what plays the bot in live calls
    max_calls: int | None = None  # at once. Set when a server here has only so many sessions
    roles: dict[str, dict[str, Any]]
    services: list[LocalService] = Field(default_factory=list)
    benchmark: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ProviderStack:
    """Built adapters for one profile. Optional roles are None when the profile omits them."""

    profile: Profile
    stt_live: STT
    stt_eval: STT
    tts_bot: TTS
    tts_caller: TTS
    llm_support: LLM
    llm_caller: LLM
    llm_judge: LLM
    realtime: RealtimeVoice | None = None
    audio_analyst: AudioAnalyst | None = None


def default_profiles_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "profiles"


def load_profiles(directory: Path | None = None) -> dict[str, Profile]:
    directory = directory or default_profiles_dir()
    profiles = {}
    for path in sorted(directory.glob("*.yaml")):
        profile = Profile.model_validate(yaml.safe_load(path.read_text()))
        profiles[profile.id] = profile
    return profiles


def build_stack(profile: Profile) -> ProviderStack:
    """Construct every adapter a profile names. Raises if a required role is missing."""
    import voicelab.providers.adapters  # noqa: F401  (registers adapters)

    built: dict[str, Any] = {}
    for role, interface in ROLE_INTERFACES.items():
        config = profile.roles.get(role)
        if config is None:
            if role not in OPTIONAL_ROLES:
                raise ValueError(f"profile {profile.id!r} has no adapter for role {role!r}")
            continue
        built[role] = build(interface, config)  # type: ignore[arg-type]
    return ProviderStack(profile=profile, **built)
