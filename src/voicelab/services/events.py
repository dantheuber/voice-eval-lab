"""In-process pub/sub for WebSockets, and the gate that gives a live mic call priority."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from types import TracebackType
from typing import Any

_QUEUE_SIZE = 1024


class Subscription:
    def __init__(self, hub: EventHub, topic: str) -> None:
        self._hub = hub
        self._topic = topic
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_SIZE)

    async def get(self) -> dict[str, Any]:
        return await self.queue.get()

    def __enter__(self) -> Subscription:
        self._hub._subscribers[self._topic].add(self)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._hub._subscribers[self._topic].discard(self)


class EventHub:
    """Fan-out on the event loop. A stalled browser loses old messages, never holds up a call."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[Subscription]] = defaultdict(set)

    def subscribe(self, topic: str) -> Subscription:
        return Subscription(self, topic)

    def publish(self, topic: str, message: dict[str, Any]) -> None:
        for subscription in list(self._subscribers[topic]):
            if subscription.queue.full():
                subscription.queue.get_nowait()
            subscription.queue.put_nowait(message)


class LiveGate:
    """Held while a person is on a mic call. New bot calls and evals wait for it to clear."""

    def __init__(self) -> None:
        self._holders: set[str] = set()
        self._clear = asyncio.Event()
        self._clear.set()

    @property
    def active(self) -> bool:
        return bool(self._holders)

    def hold(self, key: str) -> None:
        self._holders.add(key)
        self._clear.clear()

    def release(self, key: str) -> None:
        self._holders.discard(key)
        if not self._holders:
            self._clear.set()

    async def wait_clear(self) -> None:
        while self._holders:  # a new mic call can start before a waiter gets to run
            await self._clear.wait()
