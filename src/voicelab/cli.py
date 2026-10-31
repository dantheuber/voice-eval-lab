"""`voicelab` command line: serve, seed, run one call, evaluate one call."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Iterator

import uvicorn

from voicelab.api.app import create_app
from voicelab.services.certs import ensure_certificate
from voicelab.services.container import build_services
from voicelab.services.errors import ServiceError
from voicelab.services.runs import RunPlan
from voicelab.services.settings import Settings
from voicelab.services.system import lan_ip

logger = logging.getLogger("voicelab")


class _Server(uvicorn.Server):
    """Two servers share one process, so signals are handled once, outside both of them."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


async def _serve(settings: Settings) -> None:
    app = create_app(build_services(settings))
    cert, key = ensure_certificate(settings.certs_dir, lan_ip())
    plain = uvicorn.Config(
        app, host=settings.http_host, port=settings.http_port, lifespan="off", log_level="info"
    )
    secure = uvicorn.Config(
        app,
        host=settings.https_host,
        port=settings.https_port,
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
        lifespan="off",
        log_level="info",
    )
    servers = [_Server(plain), _Server(secure)]

    def request_stop() -> None:
        for server in servers:
            server.should_exit = True

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, request_stop)

    # Lifespan runs here instead of inside uvicorn, or each server would start the services.
    async with app.router.lifespan_context(app):
        address = lan_ip() or "this machine's LAN address"
        logger.info(
            "dashboard at http://%s:%s and https://%s:%s",
            settings.http_host,
            settings.http_port,
            address,
            settings.https_port,
        )
        await asyncio.gather(*(server.serve() for server in servers))


async def _seed(settings: Settings) -> None:
    services = build_services(settings)
    print(services.catalog.seed())
    services.db.close()


async def _run_call(arguments: argparse.Namespace, settings: Settings) -> int:
    services = build_services(settings)
    await services.start()
    try:
        services.worker.runner()  # a missing engine should say so before anything queues
        profile = arguments.profile or services.profiles.active_id
        if profile is None:
            print("no model profile in config/profiles", file=sys.stderr)
            return 1
        plan = RunPlan(
            pairs=[(arguments.scenario, arguments.persona)],
            reps=arguments.reps,
            profile=profile,
            mode=arguments.mode,
        )
        run_id, _ = await services.worker.submit(plan)
        print(f"run {run_id}", flush=True)
        with services.hub.subscribe(f"run:{run_id}") as subscription:
            while True:
                message = await subscription.get()
                if message["type"] == "callStarted":
                    print(f"started {message['call']['id']}", flush=True)
                elif message["type"] == "callEvaluated":
                    call = message["call"]
                    print(f"{call['id']}: {call['status']} {call['verdict']}", flush=True)
                elif message["type"] == "callDone" and message["call"]["status"] == "failed":
                    row = services.calls.get(message["call"]["id"])
                    print(f"{row['id']}: failed: {row['error']}", file=sys.stderr, flush=True)
                elif message["type"] == "runDone":
                    return 0
    finally:
        await services.stop()


async def _evaluate(call_id: str, settings: Settings) -> int:
    services = build_services(settings)
    await services.start(seed=False)
    try:
        if services.calls.get(call_id) is None:
            print(f"no call {call_id}", file=sys.stderr)
            return 1
        result = await services.evals.submit(call_id)
        if result is None:
            row = services.calls.get(call_id)
            print(f"evaluation failed: {row['error']}", file=sys.stderr)
            return 1
        print(result.summary)
        return 0
    finally:
        await services.stop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicelab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="start the API and dashboard")
    serve.add_argument("--http-port", type=int)
    serve.add_argument("--https-port", type=int)
    serve.add_argument(
        "--services", choices=["startup", "lazy"], help="when to start model servers"
    )

    commands.add_parser("seed", help="load personas and scenarios from seeds/")

    run = commands.add_parser("run", help="run a bot call and evaluate it")
    run.add_argument("scenario", help="scenario id")
    run.add_argument("persona", help="persona id")
    run.add_argument("--profile")
    run.add_argument("--mode", choices=["auto", "live_bots", "turn_based"], default="auto")
    run.add_argument("--reps", type=int, default=1)

    evaluate = commands.add_parser("eval", help="evaluate a stored call")
    evaluate.add_argument("call_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    try:
        match arguments.command:
            case "serve":
                settings.http_port = arguments.http_port or settings.http_port
                settings.https_port = arguments.https_port or settings.https_port
                settings.service_start = arguments.services or settings.service_start
                asyncio.run(_serve(settings))
                return 0
            case "seed":
                asyncio.run(_seed(settings))
                return 0
            case "run":
                return asyncio.run(_run_call(arguments, settings))
            case "eval":
                return asyncio.run(_evaluate(arguments.call_id, settings))
    except ServiceError as exc:
        print(exc.message, file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
