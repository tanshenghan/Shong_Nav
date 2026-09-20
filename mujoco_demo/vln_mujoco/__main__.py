from __future__ import annotations

import argparse

from aiohttp import web

from .server import VlnMujocoServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vln_mujoco simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--vln-server", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    server = VlnMujocoServer(default_vln_server=args.vln_server)
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"vln_mujoco ready: http://{display_host}:{args.port}", flush=True)
    web.run_app(server.app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
