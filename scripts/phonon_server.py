#!/usr/bin/env python
"""Serves Phonon TTS over HTTP, since `ptts` has no server of its own.

Runs under the bench venv, which has `ptts`, so the app itself loads no model:

    .runtime/venvs/bench/bin/python scripts/phonon_server.py --port 18103

POST /tts with {"text": ..., "voice": ...} streams 16-bit mono PCM at the rate in X-Sample-Rate.
Phonon generates at about 10x real time on CPU, so one lock doesn't hold anyone up for long.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterator

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


def _pcm(text: str, voice: str) -> Iterator[bytes]:
    with one_at_a_time, engine.stream(text, voice=voice) as audio:
        for chunk in audio:
            yield (np.clip(np.asarray(chunk), -1, 1) * 32767).astype("<i2").tobytes()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tts")
def tts(request: Request) -> StreamingResponse:
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
