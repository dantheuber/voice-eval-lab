"""The run queue: expands a run into calls, runs them, then has them evaluated.

A concurrency slot is held through a call's evaluation as well as the call. Evaluation uses the
same GPU, and running the next live call beside it would distort that call's latency numbers.
Calls live in SQLite, so a restart picks the queue up where it stopped. A mic call has priority:
nothing new starts while one is live, and what is already running finishes.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voicelab.api.mapping import call_summary, run_summary, stream_message
from voicelab.core.models import (
    CallArtifact,
    CallMode,
    CallStatus,
    LineConditions,
    new_id,
    utcnow,
)
from voicelab.engine.contract import BotCallRequest, BotCallRunner, Emit
from voicelab.services.errors import Conflict, Invalid, NotFound
from voicelab.services.evaluation import EvalService
from voicelab.services.events import EventHub, LiveGate
from voicelab.services.profiles import ProfileManager
from voicelab.store.artifacts import ArtifactStore
from voicelab.store.calls import CallIndex, used_untested_provider
from voicelab.store.catalog import Catalog
from voicelab.store.runs import RunStore
from voicelab.store.sqlite import timestamp

logger = logging.getLogger(__name__)

STOPPED = "stopped"
INTERRUPTED = "interrupted by restart"


@dataclass
class RunPlan:
    pairs: list[tuple[str, str]]  # (scenario_id, persona_id)
    reps: int
    profile: str
    mode: str  # auto, live_bots, turn_based
    line: LineConditions | None = None  # None uses each persona's line
    concurrency: int = 1
    label: str | None = None


def fill_from_stub(artifact: CallArtifact, stub: CallArtifact) -> CallArtifact:
    """The stub holds what the API knows about the call, and the engine may not repeat all of it."""
    artifact.batch_id = artifact.batch_id or stub.batch_id
    artifact.persona_id = artifact.persona_id or stub.persona_id
    artifact.scenario_id = artifact.scenario_id or stub.scenario_id
    artifact.persona_snapshot = artifact.persona_snapshot or stub.persona_snapshot
    artifact.scenario_snapshot = artifact.scenario_snapshot or stub.scenario_snapshot
    artifact.profile = artifact.profile or stub.profile
    artifact.started_at = artifact.started_at or stub.started_at
    return artifact


class RunWorker:
    def __init__(
        self,
        *,
        store: ArtifactStore,
        catalog: Catalog,
        calls: CallIndex,
        runs: RunStore,
        hub: EventHub,
        gate: LiveGate,
        evals: EvalService,
        profiles: ProfileManager,
        runner: Callable[[], BotCallRunner],
        model_wait_s: float,
    ) -> None:
        self.store = store
        self.catalog = catalog
        self.calls = calls
        self.runs = runs
        self.hub = hub
        self.gate = gate
        self.evals = evals
        self.profiles = profiles
        self.runner = runner
        self.model_wait_s = model_wait_s
        self.evals.on_settled = self._on_evaluated
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}  # call id to its slot
        self._on_the_line: set[str] = set()  # slots still in the call, not yet in its evaluation
        self._stopping: set[str] = set()
        self._announced: set[str] = set()

    # Lifecycle -----------------------------------------------------------------------------------

    def start(self) -> None:
        self.recover()
        self._task = asyncio.create_task(self._loop(), name="run-queue")
        self._wake.set()

    async def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        loop_task = [self._task] if self._task else []
        await asyncio.gather(*tasks, *loop_task, return_exceptions=True)

    def recover(self) -> None:
        """Pick up after a restart. Running runs continue and paused ones stay paused. A call that
        was on the line is lost, since its audio died with the process. A call that was being
        evaluated still has its recording, so it goes back in the eval queue."""
        for row in self.calls.with_status("running"):
            self._recover_call(row["id"], lambda call_id: self._fail(call_id, INTERRUPTED))
        for row in self.calls.with_status("evaluating"):
            self._recover_call(row["id"], self.evals.submit)
        for run in self.runs.unfinished():
            self._settle(run["id"])

    def _recover_call(self, call_id: str, action: Callable[[str], Any]) -> None:
        try:
            action(call_id)
        except FileNotFoundError:  # someone removed the call's directory while we were down
            self.calls.update(call_id, status="failed", error="call files are missing")

    # Commands ------------------------------------------------------------------------------------

    def create(self, plan: RunPlan) -> tuple[str, CallMode]:
        if not plan.pairs:
            raise Invalid("select at least one scenario and persona pair")
        if plan.reps < 1 or plan.concurrency < 1:
            raise Invalid("reps and concurrency must be at least 1")
        profile = self.profiles.require(plan.profile)
        mode = self._effective_mode(plan.mode, profile.live_capable)
        scenarios = {s: self._scenario(s) for s, _ in plan.pairs}
        personas = {p: self._persona(p) for _, p in plan.pairs}
        run_id = new_id("batch")
        calls: list[dict[str, Any]] = []
        try:
            for rep in range(1, plan.reps + 1):
                for scenario_id, persona_id in plan.pairs:
                    artifact = CallArtifact(
                        mode=mode,
                        timing_is_real=mode == CallMode.LIVE_BOTS,
                        batch_id=run_id,
                        persona_id=persona_id,
                        scenario_id=scenario_id,
                        persona_snapshot=personas[persona_id],
                        scenario_snapshot=scenarios[scenario_id],
                        profile=profile.id,
                    )
                    calls.append(
                        {
                            "id": artifact.id,
                            "batch_id": run_id,
                            "rep": rep,
                            "seed": rep,
                            "scenario_id": scenario_id,
                            "scenario_title": scenarios[scenario_id].title,
                            "persona_id": persona_id,
                            "persona_name": personas[persona_id].name,
                            "mode": mode.value,
                            "status": "queued",
                            "profile": profile.id,
                            "timing_is_real": int(artifact.timing_is_real),
                            "untested": int(profile.untested),
                        }
                    )
                    self.store.write_artifact(artifact)
            # All in one transaction, after the files exist. The scheduler sees the whole run or
            # none of it, so a wake-up can't start a run that still looks short and close it.
            with self.calls.db.transaction():
                self.runs.add(
                    id=run_id,
                    label=plan.label or _default_label(plan),
                    profile=profile.id,
                    mode=mode.value,
                    status="queued",
                    started_at=timestamp(),
                    reps=plan.reps,
                    concurrency=min(plan.concurrency, profile.max_calls or plan.concurrency),
                    pairs=[{"scenario_id": s, "persona_id": p} for s, p in plan.pairs],
                    line=plan.line.model_dump() if plan.line else None,
                    untested=int(profile.untested),
                )
                for columns in calls:
                    self.calls.add(**columns)
        except BaseException:
            for columns in calls:
                shutil.rmtree(self.store.paths(columns["id"]).dir, ignore_errors=True)
            raise
        return run_id, mode

    async def submit(self, plan: RunPlan) -> tuple[str, CallMode]:
        """Queue a run. Writing its calls is file work, so it stays off the loop a mic call may be
        using, and the wake-up has to happen on the loop because asyncio events aren't thread-safe.
        """
        created = await asyncio.to_thread(self.create, plan)
        self.wake()
        return created

    def wake(self) -> None:
        self._wake.set()

    def pause(self, run_id: str) -> None:
        run = self._run(run_id)
        if run["status"] not in ("queued", "running"):
            raise Conflict(f"run is {run['status']}, not active")
        self.runs.update(run_id, status="paused")
        self._publish_run(run_id)

    def resume(self, run_id: str) -> None:
        run = self._run(run_id)
        if run["status"] != "paused":
            raise Conflict(f"run is {run['status']}, not paused")
        self.runs.update(run_id, status="running")
        self._publish_run(run_id)
        self._wake.set()

    def stop(self, run_id: str) -> None:
        """Cut calls on the line and drop queued ones. A call already recorded is still evaluated,
        since that costs nothing the run hasn't paid for and the result is real."""
        run = self._run(run_id)
        if run["status"] not in ("queued", "running", "paused"):
            raise Conflict(f"run is already {run['status']}")
        self.runs.update(run_id, status="stopped", ended_at=timestamp())
        for row in self.calls.in_batch(run_id):
            if row["status"] == "queued":
                self._fail(row["id"], STOPPED)
            elif row["id"] in self._on_the_line:
                self._stopping.add(row["id"])
                self._tasks[row["id"]].cancel()
        self._settle(run_id)

    def is_busy(self) -> bool:
        return bool(self.runs.active()) or bool(self._tasks)

    def summary(self, run_id: str) -> dict[str, Any]:
        return run_summary(
            self._run(run_id),
            self.calls.batch_progress(run_id),
            self.calls.mean_call_seconds(run_id),
        )

    # Scheduling ----------------------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            await self.gate.wait_clear()
            try:
                self._schedule()
            except Exception:  # one bad call must not end the queue for every later run
                logger.exception("run scheduler failed")

    def _schedule(self) -> None:
        """Fill free slots from the oldest active run. A later run waits for it, so two runs don't
        share one GPU."""
        for run in self.runs.active():
            rows = self.calls.in_batch(run["id"])
            queued = [row for row in rows if row["status"] == "queued"]
            room = run["concurrency"] - len(self._tasks)
            for row in queued[: max(room, 0)]:
                try:
                    self._start(run, row)
                except FileNotFoundError:
                    self.calls.update(row["id"], status="failed", error="call files are missing")
                    self._settle(run["id"])
            if any(row["status"] in ("queued", "running", "evaluating") for row in rows):
                return

    def _start(self, run: dict[str, Any], row: dict[str, Any]) -> None:
        # Status moves to running here, not in the task, so the next pass can't pick it twice.
        call_id = row["id"]
        self._begin(run, row)
        self._on_the_line.add(call_id)
        task = asyncio.create_task(self._drive(run, row), name=f"call-{call_id}")
        self._tasks[call_id] = task
        task.add_done_callback(lambda _: self._release(call_id, run["id"]))

    def _release(self, call_id: str, run_id: str) -> None:
        self._tasks.pop(call_id, None)
        self._on_the_line.discard(call_id)
        if call_id in self._stopping:  # cancelled before the task took its first step
            self._stopping.discard(call_id)
            self._fail(call_id, STOPPED)
            self._settle(run_id)
        self._wake.set()

    async def _drive(self, run: dict[str, Any], row: dict[str, Any]) -> None:
        if await self._call(run, row):
            await self.evals.submit(row["id"])

    async def _call(self, run: dict[str, Any], row: dict[str, Any]) -> bool:
        """Run one call. True when there's a recording to evaluate."""
        call_id, topic = row["id"], f"run:{run['id']}"
        stub = self.store.read_artifact(call_id)

        async def emit(message: dict[str, Any]) -> None:
            # This class announces call starts and ends, with the index's fuller summary.
            if message.get("type") not in ("callStarted", "callDone"):
                self.hub.publish(topic, stream_message(message, call_id))

        try:
            artifact = await self._run_on_models(run, row, stub, emit)
        except asyncio.CancelledError:
            if call_id not in self._stopping:
                raise  # shutting down: recovery marks it after the restart
            self._stopping.discard(call_id)
            self._fail(call_id, STOPPED)
            self._settle(run["id"])
            return False
        except Exception as exc:
            logger.exception("call %s failed", call_id)
            self._fail(call_id, f"{type(exc).__name__}: {exc}")
            self._settle(run["id"])
            return False
        finally:
            self._on_the_line.discard(call_id)
        artifact = fill_from_stub(artifact, stub)
        untested = bool(row["untested"]) or used_untested_provider(artifact)
        self.calls.update(call_id, untested=int(untested))
        if artifact.status == CallStatus.FAILED or artifact.error:
            self.store.write_artifact(artifact)
            self._fail(call_id, artifact.error or "call failed")
            self._settle(run["id"])
            return False
        artifact.status = CallStatus.EVALUATING
        self.store.write_artifact(artifact)
        self.calls.update(
            call_id,
            status="evaluating",
            duration_s=artifact.duration_s,
            timing_is_real=int(artifact.timing_is_real),
        )
        self._publish_call("callDone", call_id)
        self._publish_run(run["id"])
        return True

    async def _run_on_models(
        self, run: dict[str, Any], row: dict[str, Any], stub: CallArtifact, emit: Emit
    ) -> CallArtifact:
        runner = self.runner()
        profile = self.profiles.require(run["profile"])
        stack = await self.profiles.ready(profile, self.model_wait_s)
        try:
            # A mic call may have started while the models loaded, and it goes first. Nothing
            # awaits between this and the runner, so one can't slip in again.
            await self.gate.wait_clear()
            request = BotCallRequest(
                call_id=row["id"],
                mode=stub.mode,
                persona=stub.persona_snapshot,
                scenario=stub.scenario_snapshot,
                batch_id=run["id"],
                seed=row["seed"],
                line=LineConditions.model_validate(run["line"]) if run["line"] else None,
            )
            return await runner(request, stack=stack, store=self.store, emit=emit)
        finally:
            self.profiles.release(profile)

    def _begin(self, run: dict[str, Any], row: dict[str, Any]) -> None:
        call_id = row["id"]
        started = utcnow()
        artifact = self.store.read_artifact(call_id)
        artifact.status = CallStatus.RUNNING
        artifact.started_at = started
        self.store.write_artifact(artifact)
        self.calls.update(call_id, status="running", started_at=timestamp(started))
        self.hub.publish(f"call:{call_id}", {"type": "status", "status": "running"})
        if self._run(run["id"])["status"] == "queued":
            self.runs.update(run["id"], status="running")
        self._publish_call("callStarted", call_id)
        self._publish_run(run["id"])

    def _fail(self, call_id: str, error: str) -> None:
        artifact = self.store.read_artifact(call_id)
        artifact.status = CallStatus.FAILED
        artifact.error = error
        artifact.ended_at = artifact.ended_at or utcnow()
        self.store.write_artifact(artifact)
        self.calls.update(
            call_id,
            status="failed",
            error=error,
            duration_s=artifact.duration_s,
            finished_at=timestamp(),
        )
        topic = f"call:{call_id}"
        self.hub.publish(topic, {"type": "status", "status": "failed"})
        self.hub.publish(topic, {"type": "error", "message": error})
        self._publish_call("callDone", call_id)

    def _on_evaluated(self, call_id: str) -> None:
        row = self.calls.get(call_id)
        if row is None or row["batch_id"] is None:
            return
        self._publish_call("callEvaluated", call_id)
        self._settle(row["batch_id"])

    def _settle(self, run_id: str) -> None:
        """Publish progress, and close the run when nothing is left in it."""
        run = self._run(run_id)
        self._publish_run(run_id)
        progress = self.calls.batch_progress(run_id)
        if progress["queued"] or progress["running"]:
            return
        if run["status"] in ("queued", "running", "paused"):
            self.runs.update(run_id, status="done", ended_at=timestamp())
        if run_id not in self._announced:
            self._announced.add(run_id)
            self.hub.publish(f"run:{run_id}", {"type": "runDone", "run": self.summary(run_id)})

    # Lookups and messages ------------------------------------------------------------------------

    def _publish_run(self, run_id: str) -> None:
        self.hub.publish(f"run:{run_id}", {"type": "run", "run": self.summary(run_id)})

    def _publish_call(self, kind: str, call_id: str) -> None:
        row = self.calls.get(call_id)
        if row is not None and row["batch_id"] is not None:
            self.hub.publish(f"run:{row['batch_id']}", {"type": kind, "call": call_summary(row)})

    def _run(self, run_id: str) -> dict[str, Any]:
        run = self.runs.get(run_id)
        if run is None:
            raise NotFound(f"no run {run_id}")
        return run

    def _scenario(self, scenario_id: str) -> Any:
        found = self.catalog.scenario(scenario_id)
        if found is None:
            raise Invalid(f"no scenario {scenario_id}")
        return found[0]

    def _persona(self, persona_id: str) -> Any:
        persona = self.catalog.persona(persona_id)
        if persona is None:
            raise Invalid(f"no persona {persona_id}")
        return persona

    @staticmethod
    def _effective_mode(mode: str, live_capable: bool) -> CallMode:
        if mode == "auto":
            return CallMode.LIVE_BOTS if live_capable else CallMode.TURN_BASED
        if mode in (CallMode.LIVE_BOTS, CallMode.TURN_BASED):
            return CallMode(mode)
        raise Invalid(f"unknown mode {mode!r}")


def _default_label(plan: RunPlan) -> str:
    scenarios = len({s for s, _ in plan.pairs})
    personas = len({p for _, p in plan.pairs})
    return f"{scenarios} scenarios x {personas} personas x {plan.reps}"
