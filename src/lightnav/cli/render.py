"""``lightnav-render``: render recorded episode directories to mp4.

Examples:
    lightnav-render output/episodes                 # every episode under a tree
    lightnav-render <episode_dir> [<episode_dir> ...]
    lightnav-render output/episodes --fps 15 --height 1080 --overwrite
    lightnav-render output/episodes --forward-offset 0 --no-pointing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lightnav.viz.render import DEFAULT_WAYPOINT_DT_S, TRAJ_WIDTH_FRAC
from lightnav.viz.render_episode import DEFAULT_OUT_NAME, find_episode_dirs, render_episode_dir


def _forward_offset(value: str) -> str | float:
    if value.strip().lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--forward-offset expects 'auto' or a number of metres, got {value!r}"
        ) from e


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lightnav-render",
        description="Render recorded episodes (trajectory ribbon, pointing markers, HUD) to mp4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="+", type=Path,
                    help="episode directories, or trees containing them")
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                    help=f"output filename inside each episode dir (default: {DEFAULT_OUT_NAME})")
    ap.add_argument("--fps", type=int, default=None,
                    help="override fps (default: the manifest's video_fps)")
    ap.add_argument("--timeline", choices=("realtime", "per_step"), default=None,
                    help="override the timebase (default: the manifest's video_timeline)")
    ap.add_argument("--dt", type=float, default=None,
                    help="seconds per waypoint step for the HUD velocity readout "
                         f"(default: the manifest's waypoint_dt_s, else {DEFAULT_WAYPOINT_DT_S})")
    ap.add_argument("--traj-width", type=float, default=TRAJ_WIDTH_FRAC,
                    help="ribbon span at the frame's bottom edge as a fraction of frame "
                         f"width (default: {TRAJ_WIDTH_FRAC})")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="skip episodes shorter than this many steps (default: 0 = keep all)")
    ap.add_argument("--height", type=int, default=0,
                    help="resample every frame to this height before drawing, aspect "
                         "preserved (e.g. 1080). 0 = keep the recording's own size.")
    ap.add_argument("--forward-offset", type=_forward_offset, default="auto",
                    help="metres to push the chunk away from the camera before projecting, "
                         "added to the manifest's overlay_forward_offset. 'auto' (default) "
                         "uses the frame's bottom-edge depth so near waypoints stay visible; "
                         "0 draws the path exactly where it is.")
    ap.add_argument("--no-pointing", action="store_true",
                    help="drop the apos/opos pixel markers")
    ap.add_argument("--no-hud", action="store_true",
                    help="trajectory + pointing only, no telemetry overlay")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dirs = find_episode_dirs(args.paths)
    if not dirs:
        print("no episodes found (looked for directories containing actions.json / actions.jsonl)",
              file=sys.stderr)
        return 1

    print(f"{len(dirs)} episode(s)\n")
    ok = 0
    for d in dirs:
        try:
            ok += bool(render_episode_dir(
                d,
                out_name=args.out_name,
                fps=args.fps,
                timeline=args.timeline,
                dt_s=args.dt,
                height=args.height,
                forward_offset=args.forward_offset,
                hud=not args.no_hud,
                pointing=not args.no_pointing,
                traj_width=args.traj_width,
                overwrite=args.overwrite,
                min_steps=args.min_steps,
            ))
        except ImportError as e:
            print(f"  ! {e}", file=sys.stderr)
            return 1
        except Exception as e:  # noqa: BLE001 -- one bad episode must not stop the batch
            print(f"  ! {d}: {e}", file=sys.stderr)
    print(f"\n{ok}/{len(dirs)} episode(s) rendered")
    return 0 if ok == len(dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
