"""Fakes for the backend's injected seams: runner, talk factory, stack, and eval runner."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from voicelab.api.app import create_app
from voicelab.core.audio import AudioClip
from voicelab.core.models import (
    CallArtifact,
    CallStatus,
    EvalResult,
    EvaluatorRun,
    Finding,
    Speaker,
    ToolCallEvent,
    Utterance,
    Verdict,
)
from voicelab.core.profiles import Profile, ProviderStack
from voicelab.engine.contract import BotCallRequest, Emit, TalkRequest
from voicelab.providers.base import LLMDelta, LLMResponse, Transcript, VoiceInfo
from voicelab.services.container import Services, build_services
from voicelab.services.evaluation import EvalJob, Sink
from voicelab.services.settings import Settings
from voicelab.store.artifacts import ArtifactStore

RATE = 16000


def tone(duration_s: float, hz: float = 220.0, *, rate: int = RATE) -> AudioClip:
    t = np.arange(round(duration_s * rate)) / rate
    return AudioClip.from_samples(0.3 * np.sin(2 * np.pi * hz * t), rate)


def bursts(spans: list[tuple[float, float]], duration_s: float) -> AudioClip:
    samples = np.zeros(round(duration_s * RATE), dtype=np.float32)
    for start, end in spans:
        first, last = round(start * RATE), round(end * RATE)
        t = np.arange(last - first) / RATE
        samples[first:last] = 0.3 * np.sin(2 * np.pi * 220 * t)
    return AudioClip.from_samples(samples, RATE)


def write_recording(store: ArtifactStore, artifact: CallArtifact) -> CallArtifact:
    artifact.duration_s = 4.0
    artifact.timeline = [
        Utterance(speaker=Speaker.BOT, start_s=0.0, end_s=1.0, intended_text="Hello there."),
        Utterance(speaker=Speaker.CALLER, start_s=1.5, end_s=3.0, heard_text="Hi."),
        ToolCallEvent(at_s=2.0, name="find_customer", arguments={"billing_zip": "97214"}),
    ]
    store.write_call(artifact, bursts([(1.5, 3.0)], 4.0), bursts([(0.0, 1.0)], 4.0))
    return artifact


class FakeRunner:
    """A bot call that finishes at once, or blocks until released."""

    def __init__(self) -> None:
        self.requests: list[BotCallRequest] = []
        self.hold = False
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.fail_with: Exception | None = None
        self.status = CallStatus.EVALUATING
        self.error: str | None = None
        self.stacks: list[ProviderStack] = []

    async def __call__(
        self, request: BotCallRequest, *, stack: ProviderStack, store: ArtifactStore, emit: Emit
    ) -> CallArtifact:
        self.requests.append(request)
        self.stacks.append(stack)
        self.started.set()
        await emit({"type": "callStarted", "call": {"id": request.call_id}})
        await emit(
            {
                "type": "partial",
                "callId": request.call_id,
                "speaker": "bot",
                "text": "Hel",
                "atS": 0.5,
            }
        )
        await emit(
            {
                "type": "turn",
                "callId": request.call_id,
                "utterance": Utterance(speaker=Speaker.BOT, start_s=0.0, end_s=1.0),
            }
        )
        if self.hold:
            await self.release.wait()
        if self.fail_with is not None:
            raise self.fail_with
        artifact = store.read_artifact(request.call_id)
        if self.status == CallStatus.FAILED:
            artifact.status = CallStatus.FAILED
            artifact.error = self.error
            store.write_artifact(artifact)
            return artifact
        return write_recording(store, artifact)


class FakeEvalRunner:
    """Writes a scripted result, reporting through `sink` the way a real run does."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self.verdicts: list[Verdict] = []
        self.jobs: list[EvalJob] = []
        self.hold = False
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.fail_with: Exception | None = None
        self.findings: list[Finding] = []
        self.more_evaluators: list[EvaluatorRun] = []
        self.reply_p50_s: float | None = 1.4
        self.overlap_count: int = 1

    async def __call__(self, job: EvalJob, sink: Sink) -> EvalResult:
        self.jobs.append(job)
        self.started.set()
        if self.hold:
            await self.release.wait()
        if self.fail_with is not None:
            raise self.fail_with
        verdict = self.verdicts.pop(0) if self.verdicts else Verdict.PASS
        findings = [f.model_copy(update={"id": Finding(evaluator="x", category="x", title="x").id})
                    for f in self.findings]  # fmt: skip
        timing = EvaluatorRun(
            name="timing",
            status="ok",
            metrics={
                "response_latency": {"median_s": self.reply_p50_s, "p95_s": 2.9},
                "overlap": {"count": self.overlap_count},
                "dead_air": {"count_over_threshold": 1},
                "barge_in": {"median_reaction_s": 0.4},
            },
        )
        validity = EvaluatorRun(name="caller_validity", status="ok", reason="never gave the zip")
        result = EvalResult(
            call_id=job.call_id,
            verdict=verdict,
            summary=f"Verdict: {verdict.value}.",
            findings=findings,
            evaluators=[timing, validity, *self.more_evaluators],
        )
        for run in self.more_evaluators:
            sink({"evaluator": run.model_dump(mode="json")})
        sink({"evaluator": timing.model_dump(mode="json")})
        for finding in findings:
            sink({"finding": finding.model_dump(mode="json")})
        self.store.write_eval(result)
        return result


@dataclass
class FakeSession:
    call_id: str
    store: ArtifactStore
    emit: Emit
    offers: list[tuple[str, str]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    ended: int = 0

    async def handle_offer(self, sdp: str, sdp_type: str) -> dict[str, str]:
        self.offers.append((sdp, sdp_type))
        return {"sdp": f"answer:{sdp}", "type": "answer"}

    async def add_ice_candidates(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates.extend(candidates)

    async def end(self) -> CallArtifact:
        self.ended += 1
        return write_recording(self.store, self.store.read_artifact(self.call_id))


class FakeTalkFactory:
    def __init__(self) -> None:
        self.requests: list[TalkRequest] = []
        self.sessions: list[FakeSession] = []
        self.fail_with: Exception | None = None

    async def __call__(
        self, request: TalkRequest, *, stack: ProviderStack, store: ArtifactStore, emit: Emit
    ) -> FakeSession:
        if self.fail_with is not None:
            raise self.fail_with
        self.requests.append(request)
        session = FakeSession(request.call_id, store, emit)
        self.sessions.append(session)
        return session


class FakeTTS:
    name = "fake-tts"
    sample_rate = RATE

    def __init__(self) -> None:
        self.said: list[tuple[str, str]] = []

    async def voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="default_male_1", label="Default male 1", language="en-US")]

    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        self.said.append((text, voice_id))
        return tone(0.5)

    async def stream(self, text: str, *, voice_id: str, style: str | None = None):
        yield tone(0.1)


class FakeSTT:
    name = "fake-stt"

    async def transcribe(self, audio, *, language="en", hints=None) -> Transcript:
        return Transcript(text="I was charged twice")


class FakeLLM:
    """Answers with `reply`, which may be a list of replies for successive calls."""

    def __init__(self, name: str = "fake-llm", reply: Any = "Thanks for calling.") -> None:
        self.name = name
        self.reply = reply
        self.prompts: list[str] = []

    async def chat(self, messages, **kwargs) -> LLMResponse:
        self.prompts.append(messages[-1].content)
        reply = self.reply.pop(0) if isinstance(self.reply, list) else self.reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, model=self.name)

    async def stream_chat(self, messages, **kwargs):
        yield LLMDelta(text="Sure.", final=LLMResponse(text="Sure.", model=self.name))


def fake_stack(profile: Profile) -> ProviderStack:
    return ProviderStack(
        profile=profile,
        stt_live=FakeSTT(),
        stt_eval=FakeSTT(),
        tts_bot=FakeTTS(),
        tts_caller=FakeTTS(),
        llm_support=FakeLLM("support-llm"),
        llm_caller=FakeLLM("caller-llm"),
        llm_judge=FakeLLM("judge-llm"),
    )


PROFILE_YAML = """\
id: {id}
label: {label}
description: For tests.
live_capable: {live}
roles:
  stt_live: {{type: fake, model: stt}}
  stt_eval: {{type: fake, model: stt}}
  tts_bot: {{type: fake, model: tts}}
  tts_caller: {{type: fake, model: tts}}
  llm_support: {{type: fake, model: {llm}}}
  llm_caller: {{type: fake, model: llm}}
  llm_judge: {{type: fake, model: llm}}
"""


def write_profiles(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "local-default.yaml").write_text(
        PROFILE_YAML.format(id="local-default", label="Local", live="false", llm="llm")
    )
    (directory / "live.yaml").write_text(
        PROFILE_YAML.format(id="live", label="Live", live="true", llm="big-llm")
    )


@dataclass
class Rig:
    """A running app with its fakes."""

    client: TestClient
    services: Services
    runner: FakeRunner
    evals: FakeEvalRunner
    talk: FakeTalkFactory
    settings: Settings

    def run_on_loop(self, work):
        """Run a coroutine on the app's loop, where its asyncio objects live."""
        return self.client.portal.call(lambda: work())


def build_rig(
    tmp_path: Path,
    *,
    seed: bool = True,
    dist: Path | None = None,
    raise_server_exceptions: bool = True,
) -> Rig:
    profiles_dir = tmp_path / "profiles"
    if not profiles_dir.exists():
        write_profiles(profiles_dir)
    settings = Settings(
        data_dir=tmp_path / "data",
        profiles_dir=profiles_dir,
        web_dist=dist or tmp_path / "no-dist",
        certs_dir=tmp_path / "certs",
        service_start="lazy",
        reviewer="tester",
        extra_hosts=("testserver", "voicelab.lan", "voicebox", "192.0.2.7"),
    )
    runner = FakeRunner()
    talk = FakeTalkFactory()
    store = ArtifactStore(settings.data_dir)
    evals = FakeEvalRunner(store)
    services = build_services(
        settings,
        bot_runner=runner,
        talk_factory=talk,
        stack_factory=fake_stack,
        eval_runner=evals,
    )
    app = create_app(services, seed=seed)
    client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
    return Rig(client, services, runner, evals, talk, settings)
