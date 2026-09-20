import asyncio

import pytest
from aiohttp import web

from vln_mujoco.vln_client import VlnClient


class NullLogger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass


@pytest.mark.asyncio
async def test_vln_login_next_and_result(unused_tcp_port: int) -> None:
    async def websocket(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            request_data = message.json()
            if request_data["action"] == "login":
                await socket.send_json({"data": {"rc": 0}})
            elif request_data["action"] == "next":
                await socket.send_json(
                    {
                        "data": {
                            "rc": 0,
                            "seq": request_data["data"]["seq"],
                            "actions": [0.6, 0.1, 0.2],
                            "stop": False,
                            "visible": True,
                            "pointing": {},
                        }
                    }
                )
            elif request_data["action"] == "reset":
                break
        return socket

    app = web.Application()
    app.router.add_get("/ws", websocket)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    client = VlnClient(NullLogger(), f"ws://127.0.0.1:{unused_tcp_port}/ws")
    try:
        client.start("go to the refrigerator")
        for _ in range(100):
            if client.snapshot().connected:
                break
            await asyncio.sleep(0.02)
        assert client.snapshot().connected
        assert client.offer_frame(123, b"jpeg")
        results = []
        for _ in range(100):
            results = client.take_results()
            if results:
                break
            await asyncio.sleep(0.02)
        assert results[0].sequence == 1
        assert results[0].waypoints == ((0.6, 0.1, 0.2),)
        assert results[0].visible is True
    finally:
        client.close()
        await runner.cleanup()
