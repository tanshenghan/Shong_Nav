"""Continuous JPEG WebSocket with a bounded number of in-flight frames."""

import asyncio
import contextlib
import json
import struct

from aiohttp import WSMsgType, web

STREAM_FPS = 20
MAX_IN_FLIGHT = 12


async def stream_camera(request, simulation, sockets=None):
    socket = web.WebSocketResponse(heartbeat=15, max_msg_size=1024)
    await socket.prepare(request)
    if sockets is not None:
        sockets.add(socket)
    view = request.query.get("view", "first")
    sent = acknowledged = 0

    async def receive():
        nonlocal acknowledged
        async for message in socket:
            if message.type != WSMsgType.TEXT:
                continue
            try:
                ack = json.loads(message.data).get("ack")
                if isinstance(ack, int) and not isinstance(ack, bool):
                    acknowledged = max(acknowledged, min(sent, ack))
            except (ValueError, AttributeError):
                continue

    receiver = asyncio.create_task(receive())
    stamp = None
    try:
        while not socket.closed and not receiver.done():
            frame = (simulation.third_person_frame() if view == "third"
                     else simulation.display_frame())
            if frame is not None and frame.stamp_ns != stamp and sent - acknowledged < MAX_IN_FLIGHT:
                sent += 1
                # Uint32 sequence + JPEG. Acknowledgements bound network backlog;
                # when full, skip frames and resume with the newest camera image.
                await asyncio.wait_for(socket.send_bytes(struct.pack(">I", sent) + frame.jpeg), 2)
                stamp = frame.stamp_ns
            await asyncio.sleep(1 / STREAM_FPS)
    except (ConnectionError, asyncio.TimeoutError):
        pass
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver
        await socket.close()
        if sockets is not None:
            sockets.discard(socket)
    return socket
