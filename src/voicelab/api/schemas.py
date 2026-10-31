"""Request bodies. Responses are built in `mapping`."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import Depends
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from starlette.requests import HTTPConnection

from voicelab.core.models import HumanVerdict, Verdict
from voicelab.services.container import Services


def _services(connection: HTTPConnection) -> Services:
    return connection.app.state.services


Svc = Annotated[Services, Depends(_services)]


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class FindingDecision(ApiModel):
    human_verdict: HumanVerdict | None = None
    human_note: str = ""


class CallVerdict(ApiModel):
    verdict: Verdict
    note: str = ""


class Reevaluate(ApiModel):
    evaluators: list[str] | None = None


class Pair(ApiModel):
    scenario_id: str
    persona_id: str


class RunBody(ApiModel):
    pairs: list[Pair]
    reps: int = 1
    profile: str | None = None
    mode: str = "auto"
    line: str | dict[str, Any] = "persona"
    concurrency: int = 1
    label: str | None = None


class PresetBody(ApiModel):
    name: str = Field(min_length=1)
    pairs: list[Pair]
    reps: int = 1
    profile: str | None = None
    mode: str = "auto"
    line: str | dict[str, Any] = "persona"
    concurrency: int = 1


class TalkBody(ApiModel):
    scenario_id: str | None = None
    persona_id: str | None = None
    line: dict[str, Any] | None = None
    record: bool = True
    profile: str | None = None


class OfferBody(BaseModel):
    """snake_case on purpose: this is Pipecat's client talking."""

    sdp: str
    type: str
    pc_id: str | None = None
    restart_pc: bool = False


class IceBody(BaseModel):
    pc_id: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class TtsPreview(ApiModel):
    text: str = Field(min_length=1, max_length=1000)
    voice: dict[str, Any]
    line: dict[str, Any] | None = None


class DraftBody(ApiModel):
    description: str = Field(min_length=1)
    customer_id: str | None = None
    base_scenario_id: str | None = None


class DryRunBody(ApiModel):
    persona_id: str | None = None


class ChannelMapBody(ApiModel):
    channel_map: Literal["caller_left", "caller_right", "mono"]


class ActiveProfileBody(ApiModel):
    id: str


class RoleBody(ApiModel):
    adapter: str


class ThresholdsBody(ApiModel):
    dead_air_s: float = Field(gt=0)
    barge_stop_target_s: float = Field(gt=0)
    reply_latency_warn_s: float = Field(gt=0)
    mos_baseline: float = Field(ge=1, le=5)
    audio_listening_mode: str = "advisory"  # read-only: accepted and ignored
