"""Who may talk to this server, and how much they may send.

There is no login and the HTTPS port listens on the LAN, so any page open in the user's browser
can address this server. CORS doesn't help: it only stops a page reading a response, and by then a
form POST has already run. These middlewares run before routing, and they cover WebSocket
handshakes too, which have no same-origin rule at all.

Host is checked on every request. DNS rebinding points an attacker's name at this machine, which
makes the browser treat the two as one origin, so such a request carries no Origin worth checking.
It does still carry the attacker's name in Host.

Origin is checked whenever it's present, whatever the method.

A request with no Origin needs its own reasoning, because curl and the CLI legitimately send none:

- Browsers attach Origin to every request except GET and HEAD, form posts included. A page that
  hides its origin (a sandboxed frame, `Referrer-Policy: no-referrer`) sends `Origin: null`, which
  isn't on the list. So a POST, PUT, PATCH, or DELETE with no Origin didn't come from a page in a
  current browser.
- Browsers always send Origin on a WebSocket handshake, so the same holds there.
- A browser old enough to omit Origin may still send `Sec-Fetch-Site`, which is refused here when
  it says the request crossed sites.
- A cross-site GET does arrive with no Origin, from an image tag or a link. That's allowed: no GET
  here changes anything, the page can't read the response, and links to the dashboard must work.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from voicelab.services.settings import Settings

logger = logging.getLogger(__name__)

VITE_DEV_PORT = 5173
MAX_JSON_BYTES = 16 * 1024 * 1024  # far above any scenario, far below what could hurt
_DEFAULT_PORTS = {"http": 80, "https": 443}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_CROSSED_SITES = {"cross-site", "same-site"}


def error_response(
    status: int, code: str, message: str, detail: dict[str, Any] | None = None
) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "detail": detail or {}}}
    return JSONResponse(body, status_code=status)


def _authority(scheme: str, host: str, port: int) -> str:
    # Browsers leave a default port out of both headers.
    return host if _DEFAULT_PORTS[scheme] == port else f"{host}:{port}"


@dataclass(frozen=True)
class TrustedSites:
    hosts: frozenset[str]
    origins: frozenset[str]

    def refusal(
        self, *, method: str | None, host: str | None, origin: str | None, fetch_site: str | None
    ) -> str | None:
        """Why this request may not proceed, or None. `method` is None for a WebSocket."""
        if (host or "").lower() not in self.hosts:
            return f"unknown Host {host!r}"
        if origin is not None:
            return None if origin.lower() in self.origins else f"untrusted Origin {origin!r}"
        if method not in _SAFE_METHODS and fetch_site in _CROSSED_SITES:
            return f"cross-site request with no Origin ({fetch_site})"
        return None


def trusted_sites(settings: Settings, lan_ip: str | None, hostname: str | None) -> TrustedSites:
    names = {"localhost", "127.0.0.1"}
    names.update(name.lower() for name in (lan_ip, hostname) if name)
    extras = {name.lower() for name in settings.extra_hosts}
    listeners = (("http", settings.http_port), ("https", settings.https_port))
    hosts: set[str] = set(extras)
    origins = {f"http://{name}:{VITE_DEV_PORT}" for name in ("localhost", "127.0.0.1")}
    for name in names | extras:
        for scheme, port in listeners:
            authority = _authority(scheme, name, port)
            hosts.add(authority)
            origins.add(f"{scheme}://{authority}")
    origins.update(f"{scheme}://{name}" for name in extras for scheme, _ in listeners)
    return TrustedSites(frozenset(hosts), frozenset(origins))


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope["headers"]:
        if key == name:
            return value.decode("latin-1")
    return None


class SameSiteOnly:
    def __init__(self, app: ASGIApp, sites: TrustedSites) -> None:
        self.app = app
        self.sites = sites

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        reason = self.sites.refusal(
            method=scope.get("method"),
            host=_header(scope, b"host"),
            origin=_header(scope, b"origin"),
            fetch_site=_header(scope, b"sec-fetch-site"),
        )
        if reason is None:
            await self.app(scope, receive, send)
            return
        logger.warning("refused %s: %s", scope["path"], reason)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await error_response(403, "invalid", reason)(scope, receive, send)


class BodyLimit:
    """Caps a request body as it arrives.

    The form parser spools file parts to disk before a route runs, so a check inside the route is
    too late. Counting here covers chunked bodies, understated lengths, and any number of parts.
    """

    def __init__(self, app: ASGIApp, limit_for: Callable[[str], int]) -> None:
        self.app = app
        self.limit_for = limit_for

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope["path"])
        message = f"request body is larger than the {limit} byte limit"
        declared = _header(scope, b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await error_response(413, "invalid", message)(scope, receive, send)
            return
        seen = 0

        async def counted() -> dict[str, Any]:
            nonlocal seen
            event = await receive()
            if event["type"] == "http.request":
                seen += len(event.get("body", b""))
                if seen > limit:
                    # Raised inside the parser's read, so nothing past the limit reaches disk.
                    raise HTTPException(413, message)
            return event

        await self.app(scope, counted, send)
