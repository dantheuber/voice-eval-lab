"""Live microphone calls.

While a session is open it holds the `LiveGate`, which stops new bot calls and evals from starting.
A browser can vanish without saying goodbye, so a watchdog ends sessions that never connected or
that outlive any real call. Otherwise one closed tab would stall the run queue forever.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from voicelab.core.models import CallArtifact, CallMode, CallStatus, LineConditions, new_id, utcnow
from voicelab.core.profiles import Profile
from voicelab.engine.contract import TalkFactory, TalkRequest, TalkSession
from voicelab.services.errors import Conflict, Invalid, NotFound
from voicelab.services.evaluation import EvalService
from voicelab.services.events import EventHub, LiveGate
from voicelab.services.profiles import ProfileManager
from voicelab.services.runs import fill_from_stub
from voicelab.services.settings import Settings
from voicelab.store.artifacts import ArtifactStore
from voicelab.store.calls import CallIndex, used_untested_provider
from voicelab.store.catalog import Catalog
from voicelab.store.sqlite import timestamp

logger = logging.getLogger(__name__)

TALK_ROLES = ("llm_support", "stt_live", "tts_bot")
_WATCH_S = 5.0
_KEPT_SESSIONS = 32  # ended ones stay readable for the page that was on them


@dataclass
class ActiveTalk:
    session_id: str
    call_id: str
    session: TalkSession
    record: bool
    started_at: datetime
    opened: float
    profile: Profile
    state: str = "connecting"  # connecting, live, ended
    duration_s: float | None = None


class TalkManager:
    def __init__(
        self,
        *,
        settings: Settings,
        store: ArtifactStore,
        catalog: Catalog,
        calls: CallIndex,
        hub: EventHub,
        gate: LiveGate,
        evals: EvalService,
        profiles: ProfileManager,
        factory: Callable[[], TalkFactory],
    ) -> None:
        self.settings = settings
        self.store = store
        self.catalog = catalog
        self.calls = calls
        self.hub = hub
        self.gate = gate
        self.evals = evals
        self.profiles = profiles
        self.factory = factory
        self._sessions: dict[str, ActiveTalk] = {}
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._watch(), name="talk-watchdog")

    async def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        for talk in list(self._sessions.values()):
            if talk.state != "ended":
                await self._end_quietly(talk)

    @property
    def live(self) -> bool:
        return any(talk.state != "ended" for talk in self._sessions.values())

    async def create(
        self,
        *,
        scenario_id: str | None,
        persona_id: str | None,
        line: LineConditions | None,
        record: bool,
        profile_id: str | None,
    ) -> tuple[ActiveTalk, CallArtifact]:
        if self.live or self.gate.active:
            raise Conflict("a mic call is already live")
        profile = self.profiles.resolve(profile_id)
        factory = self.factory()
        scenario = self._scenario(scenario_id)
        persona = self._persona(persona_id)
        # Held for the whole call: nothing may swap a server out from under a person talking.
        stack = await self.profiles.require_ready(profile, TALK_ROLES, hold=True)
        artifact = CallArtifact(
            mode=CallMode.HUMAN,
            status=CallStatus.RUNNING,
            persona_id=persona_id,
            scenario_id=scenario_id,
            persona_snapshot=persona,
            scenario_snapshot=scenario,
            profile=profile.id,
            started_at=utcnow(),
        )
        session_id = new_id("talk")
        self.store.write_artifact(artifact)
        self.calls.add(
            id=artifact.id,
            scenario_id=scenario_id,
            scenario_title=scenario.title if scenario else None,
            persona_id=persona_id,
            persona_name=persona.name if persona else None,
            mode=CallMode.HUMAN.value,
            status="running",
            profile=profile.id,
            timing_is_real=1,
            untested=int(profile.untested),
            started_at=timestamp(artifact.started_at),
        )
        self.gate.hold(session_id)
        request = TalkRequest(artifact.id, persona=persona, scenario=scenario, line=line)

        async def emit(message: dict[str, Any]) -> None:
            self._on_session_message(session_id, message)

        try:
            session = await factory(request, stack=stack, store=self.store, emit=emit)
        except BaseException as exc:
            self.gate.release(session_id)
            self.profiles.release(profile)
            self._mark_failed(artifact.id, f"{type(exc).__name__}: {exc}")
            raise
        talk = ActiveTalk(
            session_id, artifact.id, session, record, artifact.started_at, time.monotonic(), profile
        )
        self._sessions[session_id] = talk
        self._forget_old()
        return talk, artifact

    def get(self, session_id: str) -> ActiveTalk:
        talk = self._sessions.get(session_id)
        if talk is None:
            raise NotFound(f"no talk session {session_id}")
        return talk

    def duration_s(self, talk: ActiveTalk) -> float:
        if talk.duration_s is not None:
            return talk.duration_s
        return round(time.monotonic() - talk.opened, 2)

    async def offer(
        self, session_id: str, sdp: str, sdp_type: str, pc_id: str | None
    ) -> dict[str, str]:
        talk = self.get(session_id)
        if talk.state == "ended":
            raise Conflict("this call has ended")
        answer = await talk.session.handle_offer(sdp, sdp_type)
        talk.state = "live"
        return {**answer, "pc_id": answer.get("pc_id") or pc_id or session_id}

    async def add_ice_candidates(self, session_id: str, candidates: list[dict[str, Any]]) -> None:
        """Trickle ICE. The contract's session has no such method, so it's optional there."""
        talk = self.get(session_id)
        add = getattr(talk.session, "add_ice_candidates", None)
        if add is not None and talk.state != "ended":
            await add(candidates)

    async def end(self, session_id: str) -> str:
        talk = self.get(session_id)
        if talk.state != "ended":
            await self._hang_up(talk)
        return talk.call_id

    async def _hang_up(self, talk: ActiveTalk) -> None:
        # Marked first, so a second press of End doesn't hang up twice.
        talk.state = "ended"
        talk.duration_s = self.duration_s(talk)
        try:
            artifact = await talk.session.end()
        except Exception as exc:
            logger.exception("talk session %s failed to end", talk.session_id)
            self._mark_failed(talk.call_id, f"{type(exc).__name__}: {exc}")
            return
        finally:
            self.gate.release(talk.session_id)
            self.profiles.release(talk.profile)
        stub = self.store.read_artifact(talk.call_id)
        artifact = fill_from_stub(artifact, stub)
        talk.duration_s = artifact.duration_s or talk.duration_s
        if not talk.record:
            self._discard(talk.call_id)
            return
        if artifact.status == CallStatus.FAILED or artifact.error:
            self.store.write_artifact(artifact)
            self._mark_failed(talk.call_id, artifact.error or "call failed")
            return
        self.store.write_artifact(artifact)
        self.calls.update(
            talk.call_id,
            duration_s=artifact.duration_s,
            timing_is_real=int(artifact.timing_is_real),
            untested=int(talk.profile.untested or used_untested_provider(artifact)),
        )
        self.evals.submit(talk.call_id)

    def _discard(self, call_id: str) -> None:
        """`record: false` means the person didn't want this kept."""
        self.calls.delete(call_id)
        shutil.rmtree(self.store.paths(call_id).dir, ignore_errors=True)

    def _mark_failed(self, call_id: str, error: str) -> None:
        artifact = self.store.read_artifact(call_id)
        artifact.status = CallStatus.FAILED
        artifact.error = error
        self.store.write_artifact(artifact)
        self.calls.update(call_id, status="failed", error=error, finished_at=timestamp())
        self.hub.publish(f"call:{call_id}", {"type": "status", "status": "failed"})
        self.hub.publish(f"call:{call_id}", {"type": "error", "message": error})

    def _on_session_message(self, session_id: str, message: dict[str, Any]) -> None:
        """The engine reports a dropped peer here. Anything else already reached the browser over
        the data channel."""
        talk = self._sessions.get(session_id)
        if talk is None or talk.state == "ended":
            return
        if message.get("type") == "disconnected":
            asyncio.get_running_loop().create_task(self._end_quietly(talk))

    async def _end_quietly(self, talk: ActiveTalk) -> None:
        try:
            await self._hang_up(talk)
        except Exception:
            logger.exception("could not end talk session %s", talk.session_id)

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(_WATCH_S)
            await self.sweep()

    async def sweep(self) -> None:
        now = time.monotonic()
        for talk in list(self._sessions.values()):
            age = now - talk.opened
            stale = (talk.state == "connecting" and age > self.settings.talk_connect_timeout_s) or (
                talk.state == "live" and age > self.settings.talk_max_s
            )
            if stale:
                logger.warning("ending abandoned talk session %s", talk.session_id)
                await self._end_quietly(talk)

    def _forget_old(self) -> None:
        ended = [t for t in self._sessions.values() if t.state == "ended"]
        for talk in ended[: max(0, len(ended) - _KEPT_SESSIONS)]:
            del self._sessions[talk.session_id]

    def _scenario(self, scenario_id: str | None) -> Any:
        if scenario_id is None:
            return None
        found = self.catalog.scenario(scenario_id)
        if found is None:
            raise Invalid(f"no scenario {scenario_id}")
        return found[0]

    def _persona(self, persona_id: str | None) -> Any:
        if persona_id is None:
            return None
        persona = self.catalog.persona(persona_id)
        if persona is None:
            raise Invalid(f"no persona {persona_id}")
        return persona
