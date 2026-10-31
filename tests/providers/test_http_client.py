import asyncio
import socket

from voicelab.providers.adapters.http_client import PerLoopClient


async def test_a_new_server_can_bind_the_port_right_after_a_request():
    """nemo-speech binds without SO_REUSEADDR, so a TIME_WAIT on its port costs it a minute."""

    async def answer_then_wait(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        await reader.read()  # as a keep-alive server does, until the client goes away
        writer.close()

    server = await asyncio.start_server(answer_then_wait, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PerLoopClient(trust_env=False)

    response = await client.get().get(f"http://127.0.0.1:{port}/health")
    assert response.status_code == 200
    await asyncio.sleep(0.05)
    server.close()
    await server.wait_closed()

    with socket.socket() as next_server:
        next_server.bind(("127.0.0.1", port))
    await client.aclose()
