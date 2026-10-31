"""A shared httpx client for adapters, one per model server."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

# No idle connections. A connection the server closes leaves that port in TIME_WAIT for 60 s, and
# nemo-speech can't bind through that, so every swap or restart of a server sat "down" for a minute.
# With none idle the client closes first, and TIME_WAIT lands on its own throwaway port. Reconnecting
# on loopback costs far less than a millisecond.
NO_IDLE_CONNECTIONS = httpx.Limits(max_keepalive_connections=0)


class PerLoopClient:
    """An `httpx.AsyncClient` remade per event loop, since its connections belong to one loop."""

    def __init__(self, **options: Any) -> None:
        self._options = options
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = httpx.AsyncClient(limits=NO_IDLE_CONNECTIONS, **self._options)
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
