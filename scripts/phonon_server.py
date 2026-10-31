#!/usr/bin/env python
"""Serves Phonon TTS over HTTP, since `ptts` has no server of its own.

Runs under the bench venv, which has `ptts`, so the app itself loads no model:

    .runtime/venvs/bench/bin/python scripts/phonon_server.py --port 18103

POST /tts with {"text": ..., "voice": ...} streams 16-bit mono PCM at the rate in X-Sample-Rate.
Phonon generates at about 10x real time on CPU, so one lock doesn't hold anyone up for long.
"""

from __future__ import annotations

import argparse
import asyncio
import threading
from collections.abc import AsyncIterator, Callable

import numpy as np
import ptts
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

RATE = 24000

app = FastAPI()
engine: ptts.TTS | None = None
one_at_a_time = threading.Lock()


class Request(BaseModel):
    text: str
    voice: str = "Easton"


def _generate(text: str, voice: str, stop: threading.Event, send: Callable[[bytes | None], None]):
    """Runs on a worker thread. A client that hangs up must not leave the lock held."""
    try:
        with one_at_a_time:
            if stop.is_set():
                return
            with engine.stream(text, voice=voice) as audio:
                for chunk in audio:
                    if stop.is_set():
                        return
                    send((np.clip(np.asarray(chunk), -1, 1) * 32767).astype("<i2").tobytes())
    finally:
        send(None)


async def _pcm(text: str, voice: str) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    stop = threading.Event()

    def send(item: bytes | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    worker = loop.run_in_executor(None, _generate, text, voice, stop, send)
    try:
        while (item := await queue.get()) is not None:
            yield item
    finally:
        # Runs when the client disconnects too, which is every barge-in.
        stop.set()
        await asyncio.shield(worker)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tts")
async def tts(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _pcm(request.text, request.voice),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(RATE)},
    )


def main() -> None:
    global engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18103)
    parser.add_argument("--quant", default="q8")
    args = parser.parse_args()
    engine = ptts.TTS("hf://gradium/phonon-31m-en", lang="en", quant=args.quant)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
