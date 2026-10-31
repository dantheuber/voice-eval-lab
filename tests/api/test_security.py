"""Cross-site protection and body limits. Requests carry the real Host a browser would send."""

from __future__ import annotations

import json
from typing import Any

import pytest
from starlette.websockets import WebSocketDisconnect

from voicelab.api.guards import trusted_sites
from voicelab.services.settings import Settings

HOST = {"host": "127.0.0.1:18080"}
EVIL = "https://evil.example"


def post(client, path: str, **headers: str):
    return client.post(path, headers={**HOST, **headers})


def test_cross_site_mutation_is_refused_before_it_runs(client, rig):
    response = post(client, "/api/servers/restart-all", origin=EVIL)
    assert response.status_code == 403
    assert response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/benchmark"),
        ("POST", "/api/runs/batch_x/stop"),
        ("PUT", "/api/profiles/active"),
        ("DELETE", "/api/calls/call_x"),
        ("PATCH", "/api/talk/sessions/talk_x/offer"),
    ],
)
def test_every_mutating_method_checks_the_origin(client, method, path):
    response = client.request(method, path, headers={**HOST, "origin": EVIL})
    assert response.status_code == 403


def test_opaque_origin_is_refused(client):
    """A sandboxed frame or a no-referrer form sends `Origin: null`."""
    assert post(client, "/api/benchmark", origin="null").status_code == 403


def test_cross_site_websocket_is_refused(client):
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/health", headers={**HOST, "origin": EVIL}),
    ):
        pass


def test_cross_site_read_is_refused(client):
    response = client.get("/api/calls", headers={**HOST, "origin": EVIL})
    assert response.status_code == 403


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:18080", "127.0.0.1:9999", "localhost:80", ""]
)
def test_dns_rebinding_host_is_refused(client, host):
    """A rebound name reaches this server with the attacker's name in Host and no Origin at all."""
    assert client.get("/api/health", headers={"host": host}).status_code == 403


def test_rebinding_websocket_is_refused(client):
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/health", headers={"host": "evil.example:18080"}),
    ):
        pass


def test_own_origins_pass(client, rig):
    for origin, host in (
        ("http://127.0.0.1:18080", "127.0.0.1:18080"),
        ("http://localhost:18080", "localhost:18080"),
        ("https://127.0.0.1:18443", "127.0.0.1:18443"),
        ("https://192.0.2.7:18443", "192.0.2.7:18443"),
        ("https://voicebox:18443", "voicebox:18443"),
    ):
        response = client.get("/api/health", headers={"host": host, "origin": origin})
        assert response.status_code == 200, origin


def test_vite_dev_origin_passes_through_its_proxy(client):
    """The proxy's changeOrigin rewrites Host to the backend and leaves Origin as the dev page."""
    headers = {**HOST, "origin": "http://localhost:5173"}
    assert client.post("/api/benchmark", headers=headers).status_code == 200


def test_wrong_scheme_or_port_is_not_our_origin(client):
    for origin in ("http://127.0.0.1:18443", "https://127.0.0.1:18080", "http://127.0.0.1:1"):
        assert post(client, "/api/benchmark", origin=origin).status_code == 403


def test_origin_and_host_compare_without_case(client):
    response = client.get(
        "/api/health", headers={"host": "LOCALHOST:18080", "origin": "HTTP://LocalHost:18080"}
    )
    assert response.status_code == 200


def test_curl_style_mutation_without_origin_passes(client):
    assert post(client, "/api/benchmark").status_code == 200


def test_mutation_a_browser_marked_cross_site_is_refused_without_origin(client):
    """An old browser that omits Origin still sends Fetch Metadata over HTTPS and to localhost."""
    assert post(client, "/api/benchmark", **{"sec-fetch-site": "cross-site"}).status_code == 403
    assert post(client, "/api/benchmark", **{"sec-fetch-site": "same-site"}).status_code == 403
    assert post(client, "/api/benchmark", **{"sec-fetch-site": "same-origin"}).status_code == 200


def test_following_a_link_to_the_dashboard_is_allowed(client):
    """A cross-site navigation is a GET with no Origin, and that is how bookmarks and links work."""
    response = client.get("/api/health", headers={**HOST, "sec-fetch-site": "cross-site"})
    assert response.status_code == 200


def test_websocket_tool_without_origin_passes(client):
    """Browsers always send Origin on a WebSocket, so its absence means a command line tool."""
    with client.websocket_connect("/ws/health", headers=HOST) as socket:
        assert socket.receive_json()["type"] == "health"


def test_a_configured_extra_host_is_trusted(client):
    headers = {"host": "voicelab.lan", "origin": "https://voicelab.lan"}
    assert client.get("/api/health", headers=headers).status_code == 200


def test_the_allowlist_is_built_from_the_servers_own_names():
    settings = Settings(extra_hosts=("Proxy.Lan",))
    sites = trusted_sites(settings, "192.168.0.75", "Framework")
    assert {
        "localhost:18080",
        "127.0.0.1:18443",
        "192.168.0.75:18443",
        "framework:18443",
        "proxy.lan",
    } <= sites.hosts
    assert {
        "http://127.0.0.1:18080",
        "https://localhost:18443",
        "https://192.168.0.75:18443",
        "https://framework:18443",
        "http://localhost:5173",
        "https://proxy.lan",
    } <= sites.origins
    assert "https://192.168.0.75:5173" not in sites.origins


def test_a_default_port_is_left_out_as_browsers_leave_it_out():
    sites = trusted_sites(Settings(http_port=80, https_port=443), None, None)
    assert "http://localhost" in sites.origins
    assert "https://127.0.0.1" in sites.origins
    assert "localhost" in sites.hosts


def test_no_lan_address_still_works():
    sites = trusted_sites(Settings(), None, None)
    assert "None:18080" not in sites.hosts


def test_extra_hosts_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("VOICELAB_EXTRA_HOSTS", " a.lan, b.lan ,")
    assert Settings.from_env().extra_hosts == ("a.lan", "b.lan")


# Body limits -------------------------------------------------------------------------------------

BOUNDARY = "xBOUNDARYx"


def _part(name: str, filename: str, body: bytes) -> bytes:
    head = (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"; '
        f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'
    )
    return head.encode() + body + b"\r\n"


class Sent:
    """The response of a raw ASGI call."""

    def __init__(self) -> None:
        self.status: int | None = None
        self.body = b""

    async def __call__(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")


def _scope(path: str, headers: list[tuple[bytes, bytes]]) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "client": ("127.0.0.1", 5000),
        "server": ("127.0.0.1", 18080),
        "headers": [(b"host", b"127.0.0.1:18080"), *headers],
    }


def _stream(rig, path: str, chunks: list[bytes], extra=()) -> tuple[Sent, list[int]]:
    """Feed a body in chunks, as a chunked request arrives, and record how far the app read."""
    consumed = [0]
    remaining = list(chunks)
    sent = Sent()
    headers = [(b"content-type", f"multipart/form-data; boundary={BOUNDARY}".encode()), *extra]

    async def receive() -> dict[str, Any]:
        consumed[0] += 1
        body = remaining.pop(0) if remaining else b""
        return {"type": "http.request", "body": body, "more_body": bool(remaining)}

    async def call() -> None:
        await rig.client.app(_scope(path, headers), receive, sent)

    rig.run_on_loop(call)
    return sent, consumed


def test_oversize_chunked_upload_stops_being_read_at_the_limit(rig):
    rig.settings.max_upload_bytes = 4 * 1024 * 1024
    megabyte = b"x" * (1024 * 1024)
    chunks = [_part("file", "a.wav", megabyte), *[megabyte] * 60]
    sent, consumed = _stream(rig, "/api/uploads", chunks)
    assert sent.status == 413
    assert json.loads(sent.body)["error"]["code"] == "invalid"
    assert consumed[0] < 10  # it stopped at about 4 MB, not after all 60


def test_extra_file_parts_count_toward_the_limit(rig):
    rig.settings.max_upload_bytes = 2 * 1024 * 1024
    megabyte = b"x" * (1024 * 1024)
    parts = [_part(f"extra{i}", f"{i}.wav", megabyte) for i in range(8)]
    sent, consumed = _stream(rig, "/api/uploads", parts)
    assert sent.status == 413
    assert consumed[0] < 6


def test_declared_oversize_upload_is_refused_without_reading(rig):
    rig.settings.max_upload_bytes = 1024
    extra = [(b"content-length", b"999999999")]
    sent, consumed = _stream(rig, "/api/uploads", [b"x"], extra)
    assert sent.status == 413
    assert consumed[0] == 0


def test_oversize_json_body_is_refused(rig):
    megabyte = b"x" * (1024 * 1024)
    sent, consumed = _stream(rig, "/api/personas", [megabyte] * 80)
    assert sent.status == 413
    assert consumed[0] < 40
