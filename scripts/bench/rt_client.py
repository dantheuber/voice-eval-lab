"""Minimal /v1/realtime client on raw websockets. Keeps a timestamped log of every server event."""
from __future__ import annotations

import asyncio
import base64
import json
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import websockets

CHUNK_MS = 100


def read_pcm16(path: str, rate: int = 16000) -> np.ndarray:
    with wave.open(path, "rb") as f:
        assert f.getsampwidth() == 2 and f.getframerate() == rate and f.getnchannels() == 1, path
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")


def write_pcm16(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.astype("<i2").tobytes())


@dataclass
class Turn:
    """What came back for one response."""
    t_end_input: float = 0.0
    t_first_text: float | None = None
    t_first_audio: float | None = None
    t_first_call: float | None = None
    t_done: float | None = None
    text: str = ""
    audio: list = field(default_factory=list)
    audio_arrivals: list = field(default_factory=list)  # (t, n_samples)
    calls: list = field(default_factory=list)
    status: str | None = None
    usage: dict | None = None

    def pcm(self) -> np.ndarray:
        return np.concatenate(self.audio) if self.audio else np.zeros(0, dtype="<i2")

    def metrics(self, rate: int = 24000) -> dict:
        audio_s = sum(len(a) for a in self.audio) / rate
        gen = (self.t_done - self.t_first_audio) if self.t_done and self.t_first_audio else None
        # playback lead: if audio plays from its first delta, the lowest margin of audio received ahead of the play head
        lead = None
        if self.audio_arrivals:
            t0 = self.audio_arrivals[0][0]
            got = self.audio_arrivals[0][1]
            lead = 1e9
            for t, n in self.audio_arrivals[1:]:
                lead = min(lead, got / rate - (t - t0))
                got += n
        return {
            "first_audio_ms": 1000 * (self.t_first_audio - self.t_end_input) if self.t_first_audio else None,
            "first_text_ms": 1000 * (self.t_first_text - self.t_end_input) if self.t_first_text else None,
            "first_call_ms": 1000 * (self.t_first_call - self.t_end_input) if self.t_first_call else None,
            "total_ms": 1000 * (self.t_done - self.t_end_input) if self.t_done else None,
            "audio_s": audio_s,
            "gen_rtf": audio_s / gen if gen else None,
            "min_lead_s": lead,
            "text": self.text,
            "calls": self.calls,
            "status": self.status,
            "usage": self.usage,
        }


class Session:
    """One websocket session. A reader task routes events; `log` holds every event in arrival order."""

    def __init__(self, url: str, keep_payloads: bool = False):
        self.url = url.replace("http://", "ws://") + "/v1/realtime"
        self.keep_payloads = keep_payloads
        self.log: list[dict] = []
        self.errors: list[dict] = []
        self.session: dict = {}
        self.turn: Turn | None = None
        self.on_audio = None  # optional callback(pcm, t) per audio delta
        self._done = asyncio.Event()
        self._updated = asyncio.Event()
        self._created = asyncio.Event()
        self.t0 = time.perf_counter()

    async def __aenter__(self) -> "Session":
        self.ws = await websockets.connect(self.url, max_size=None, ping_interval=None)
        self.reader = asyncio.create_task(self._read())
        await asyncio.wait_for(self._created.wait(), 30)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.ws.close()
        self.reader.cancel()

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                t = time.perf_counter()
                ev = json.loads(raw)
                self._note(t, ev)
                self._route(t, ev)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            return

    def _note(self, t: float, ev: dict) -> None:
        rec = {"t": round(t - self.t0, 4), "type": ev.get("type")}
        for key in ("delta", "audio"):
            if key in ev and ev.get("type", "").endswith("audio.delta"):
                rec["audio_b64_len"] = len(ev[key])
        if self.keep_payloads or not ev.get("type", "").endswith("audio.delta"):
            rec["event"] = {k: v for k, v in ev.items() if not (k == "delta" and ev.get("type", "").endswith("audio.delta"))}
        self.log.append(rec)

    def _route(self, t: float, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "session.created":
            self.session = ev["session"]
            self._created.set()
        elif kind == "session.updated":
            self.session = ev["session"]
            self._updated.set()
        elif kind == "error":
            self.errors.append(ev["error"])
        turn = self.turn
        if turn is None:
            return
        if kind == "response.output_text.delta":
            turn.t_first_text = turn.t_first_text or t
            turn.text += ev["delta"]
        elif kind == "response.output_audio.delta":
            turn.t_first_audio = turn.t_first_audio or t
            pcm = np.frombuffer(base64.b64decode(ev["delta"]), dtype="<i2")
            turn.audio.append(pcm)
            turn.audio_arrivals.append((t, len(pcm)))
            if self.on_audio:
                self.on_audio(pcm, t)
        elif kind == "response.output_item.added" and ev["item"]["type"] == "function_call":
            turn.t_first_call = turn.t_first_call or t
        elif kind == "response.done":
            r = ev["response"]
            turn.t_done = t
            turn.status = r.get("status")
            turn.usage = r.get("usage")
            turn.calls = [{"name": o["name"], "call_id": o["call_id"], "arguments": o["arguments"]}
                          for o in r.get("output", []) if o["type"] == "function_call"]
            self._done.set()

    async def send(self, **ev) -> None:
        await self.ws.send(json.dumps(ev))

    async def update(self, **session) -> None:
        self._updated.clear()
        await self.send(type="session.update", session=session)
        await asyncio.wait_for(self._updated.wait(), 30)

    async def stream_audio(self, pcm: np.ndarray, realtime: bool = True, rate: int = 16000) -> float:
        """Sends PCM in CHUNK_MS pieces, paced to wall clock if realtime. Returns the time the last chunk left."""
        step = rate * CHUNK_MS // 1000
        start = time.perf_counter()
        for i, j in enumerate(range(0, len(pcm), step)):
            chunk = pcm[j:j + step]
            await self.send(type="input_audio_buffer.append", audio=base64.b64encode(chunk.tobytes()).decode())
            if realtime:
                target = start + (i + 1) * CHUNK_MS / 1000
                await asyncio.sleep(max(0, target - time.perf_counter()))
        return time.perf_counter()

    async def respond(self, **response) -> Turn:
        """Sends response.create now and returns the Turn at once; await done() for the end."""
        self.turn = Turn(t_end_input=time.perf_counter())
        self._done.clear()
        await self.send(type="response.create", response=response) if response else await self.send(type="response.create")
        return self.turn

    async def done(self, timeout: float = 120) -> Turn:
        await asyncio.wait_for(self._done.wait(), timeout)
        return self.turn

    async def user_text(self, text: str) -> None:
        await self.send(type="conversation.item.create",
                        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})

    async def tool_output(self, call_id: str, output) -> None:
        await self.send(type="conversation.item.create",
                        item={"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)})
