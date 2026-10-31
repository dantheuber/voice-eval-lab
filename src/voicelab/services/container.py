"""Builds the backend's services and owns their startup and shutdown.

The API and CLI both use this. Runner, talk factory, provider stack, and eval runner are parameters
so tests can pass fakes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from voicelab.api.mapping import to_api
from voicelab.core.profiles import Profile, ProviderStack, build_stack
from voicelab.engine.contract import BotCallRunner, TalkFactory
from voicelab.eval.config import EvalConfig
from voicelab.services.errors import EngineUnavailable
from voicelab.services.evaluation import EvalRunner, EvalService, SubprocessEvalRunner
from voicelab.services.events import EventHub, LiveGate
from voicelab.services.profiles import ProfileManager
from voicelab.services.runs import RunWorker
from voicelab.services.settings import Settings
from voicelab.services.supervisor import ServiceSupervisor
from voicelab.services.talk import TalkManager
from voicelab.services.uploads import UploadService
from voicelab.store.artifacts import ArtifactStore
from voicelab.store.calls import CallIndex
from voicelab.store.catalog import Catalog
from voicelab.store.runs import RunStore
from voicelab.store.sqlite import Database

logger = logging.getLogger(__name__)

THRESHOLDS_KEY = "thresholds"


def _load_engine(name: str) -> Any:
    """The engine lands separately, so its absence must not stop the dashboard from starting."""
    try:
        from voicelab.engine import service
    except ImportError as exc:
        raise EngineUnavailable(
            f"voicelab.engine.service can't be imported ({exc}), so calls can't run"
        ) from exc
    try:
        return getattr(service, name)
    except AttributeError as exc:
        raise EngineUnavailable(f"voicelab.engine.service has no {name}") from exc


def default_thresholds() -> dict[str, Any]:
    """These come from the evaluators' own defaults, so the screen shows what eval really uses."""
    config = EvalConfig()
    return {
        "deadAirS": config.timing.dead_air_s,
        "bargeStopTargetS": config.timing.barge_in_reaction_s,
        "replyLatencyWarnS": config.timing.slow_response_s,
        "mosBaseline": config.audio_quality.low_mos,
        "audioListeningMode": "advisory",
    }


def eval_config(thresholds: dict[str, Any]) -> EvalConfig:
    config = EvalConfig()
    config.timing.dead_air_s = thresholds["deadAirS"]
    config.timing.barge_in_reaction_s = thresholds["bargeStopTargetS"]
    config.timing.slow_response_s = thresholds["replyLatencyWarnS"]
    config.audio_quality.low_mos = thresholds["mosBaseline"]
    return config


@dataclass
class Services:
    settings: Settings
    db: Database
    store: ArtifactStore
    catalog: Catalog
    calls: CallIndex
    runs: RunStore
    hub: EventHub
    gate: LiveGate
    supervisor: ServiceSupervisor
    profiles: ProfileManager
    evals: EvalService
    worker: RunWorker
    talks: TalkManager
    uploads: UploadService
    _background: set[asyncio.Task[Any]] = field(default_factory=set)

    def spawn(self, work: Coroutine[Any, Any, Any]) -> None:
        """Run work the client doesn't wait for. A failure lands in the log."""
        task = asyncio.create_task(work)
        self._background.add(task)

        def finished(done: asyncio.Task[Any]) -> None:
            self._background.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.error("background task failed", exc_info=done.exception())

        task.add_done_callback(finished)

    def thresholds(self) -> dict[str, Any]:
        return {**default_thresholds(), **(self.catalog.setting(THRESHOLDS_KEY) or {})}

    async def start(self, *, seed: bool = True) -> None:
        if seed:
            added = self.catalog.seed()
            if any(added.values()):
                logger.info("seeded %s", added)
        self._check_engine()
        active = self.profiles.items.get(self.profiles.active_id or "")
        if active is not None:
            start_now = self.settings.service_start == "startup"
            await self.supervisor.apply(active.services, start=start_now)
        self.evals.start()
        self.worker.start()
        self.talks.start()

    def _check_engine(self) -> None:
        for need, feature in (
            (self.worker.runner, "Bot calls"),
            (self.talks.factory, "Mic calls"),
        ):
            try:
                need()
            except EngineUnavailable as exc:
                logger.error("%s are disabled: %s", feature, exc.message)

    async def stop(self) -> None:
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)
        await self.talks.shutdown()
        await self.worker.shutdown()
        await self.evals.stop()
        await self.supervisor.stop_all()
        self.db.close()


def build_services(
    settings: Settings | None = None,
    *,
    bot_runner: BotCallRunner | None = None,
    talk_factory: TalkFactory | None = None,
    stack_factory: Callable[[Profile], ProviderStack] = build_stack,
    eval_runner: EvalRunner | None = None,
    supervisor: ServiceSupervisor | None = None,
) -> Services:
    settings = settings or Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path)
    store = ArtifactStore(settings.data_dir)
    catalog = Catalog(db)
    calls = CallIndex(db)
    runs = RunStore(db)
    hub = EventHub()
    gate = LiveGate()
    holder: dict[str, Services] = {}

    def publish_health() -> None:
        # The supervisor exists before the profiles it reports on.
        if "services" in holder:
            items = to_api(holder["services"].profiles.health())
            hub.publish("health", {"type": "health", "items": items})

    supervisor = supervisor or ServiceSupervisor(settings.log_dir)
    supervisor.on_change = publish_health
    profiles = ProfileManager(settings, catalog, supervisor, stack_factory=stack_factory)

    def thresholds() -> EvalConfig:
        return eval_config({**default_thresholds(), **(catalog.setting(THRESHOLDS_KEY) or {})})

    evals = EvalService(
        store=store,
        calls=calls,
        hub=hub,
        gate=gate,
        runner=eval_runner or SubprocessEvalRunner(settings, store),
        config=thresholds,
        default_profile=lambda: profiles.active_id,
    )
    worker = RunWorker(
        store=store,
        catalog=catalog,
        calls=calls,
        runs=runs,
        hub=hub,
        gate=gate,
        evals=evals,
        profiles=profiles,
        runner=(lambda: bot_runner) if bot_runner else (lambda: _load_engine("run_bot_call")),
        model_wait_s=settings.model_wait_s,
    )
    talks = TalkManager(
        settings=settings,
        store=store,
        catalog=catalog,
        calls=calls,
        hub=hub,
        gate=gate,
        evals=evals,
        profiles=profiles,
        factory=(
            (lambda: talk_factory)
            if talk_factory
            else (lambda: _load_engine("create_talk_session"))
        ),
    )
    services = Services(
        settings=settings,
        db=db,
        store=store,
        catalog=catalog,
        calls=calls,
        runs=runs,
        hub=hub,
        gate=gate,
        supervisor=supervisor,
        profiles=profiles,
        evals=evals,
        worker=worker,
        talks=talks,
        uploads=UploadService(store, calls, catalog),
    )
    holder["services"] = services
    return services
