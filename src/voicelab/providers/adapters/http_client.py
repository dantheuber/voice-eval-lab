"""A shared httpx client for adapters. Each model server gets one, for connection reuse."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class PerLoopClient:
    """An `httpx.AsyncClient` remade per event loop, since its connections belong to one loop."""

    def __init__(self, **options: Any) -> None:
        self._options = options
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = httpx.AsyncClient(**self._options)
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
