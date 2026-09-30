import asyncio
import struct
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from vln_mujoco.streaming import MAX_IN_FLIGHT, stream_camera


@pytest.mark.asyncio
async def test_stream_pushes_without_round_trip_and_bounds_backlog(unused_tcp_port):
    class Camera:
        stamp = 0

        def display_frame(self):
            self.stamp += 1
            return SimpleNamespace(stamp_ns=self.stamp, jpeg=str(self.stamp).encode())

    camera = Camera()
    app = web.Application()
    app.router.add_get('/camera/ws', lambda request: stream_camera(request, camera))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', unused_tcp_port).start()
    try:
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(f'http://127.0.0.1:{unused_tcp_port}/camera/ws') as ws:
                for index in range(MAX_IN_FLIGHT):
                    data = await ws.receive_bytes(timeout=2)
                    assert struct.unpack('>I', data[:4])[0] == index + 1
                previous = int(data[4:])
                with pytest.raises(asyncio.TimeoutError):
                    await ws.receive_bytes(timeout=0.2)
                await ws.send_json({'ack': MAX_IN_FLIGHT})
                data = await ws.receive_bytes(timeout=2)
                assert int(data[4:]) > previous + 1  # stale camera images were discarded
    finally:
        await runner.cleanup()
