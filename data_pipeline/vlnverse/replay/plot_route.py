"""Draw a metric overhead route review from a canonical grid and planned poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planned-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    from PIL import Image, ImageDraw, ImageFont

    metadata = json.loads((args.planned_dir / "manifest.json").read_text())
    grid = OccupancyGrid.from_npz(metadata["map_path"])
    with np.load(args.planned_dir / "trajectory.npz", allow_pickle=False) as values:
        planned = values["positions_xyz"][:, :2]
    with np.load(args.planned_dir / "source_poses.npz", allow_pickle=False) as values:
        original = values["positions_xyz"][:, :2]
    with np.load(args.planned_dir / "planned_curve.npz", allow_pickle=False) as values:
        astar = values["astar_xy"]
    c, s = np.cos(grid.grid_yaw_rad), np.sin(grid.grid_yaw_rad)
    rotation = np.array([[c, -s], [s, c]])

    def local(points):
        return (points - grid.origin_xy_m) @ rotation

    original, planned, astar = map(local, [original, planned, astar])
    keys = metadata["route_alignment"]["waypoint_indices"]
    route = np.vstack([original, planned])
    bounds = [route.min(axis=0) - 1.2, route.max(axis=0) + 1.2]
    turn = original[keys[-3]:]
    zoom = [turn.min(axis=0) - .65, turn.max(axis=0) + .65]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)
    canvas = Image.new("RGB", (1400, 950), (18, 24, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 15), f"{metadata['episode_id']} | original route and A* curved route", font=font, fill="white")
    draw.text((25, 48), "Orange: official positions   Blue: curved reference   Grey: A* grid path   Dots: ordered checkpoints",
              font=small, fill=(215, 225, 235))
    raster = np.where(grid.occupied, 65, 232).astype(np.uint8)
    image = Image.fromarray(np.repeat(raster[:, :, None], 3, axis=2))
    # Two independent metric panels; crop/resize here is raster-map plotting.
    for box, low, high, label in [((25, 105, 1375, 505), *bounds, "Full route"),
                                  ((25, 555, 1100, 915), *zoom, "Window / utility turn detail")]:
        left, top, right, bottom = box
        scale = min((right - left) / (high[0] - low[0]), (bottom - top) / (high[1] - low[1]))
        width, height = int((high[0] - low[0]) * scale), int((high[1] - low[1]) * scale)
        crop = image.crop((low[0] / grid.resolution_m, low[1] / grid.resolution_m,
                           high[0] / grid.resolution_m, high[1] / grid.resolution_m))
        crop = crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(crop, (left, top))

        def pixels(points):
            xy = (points - low) * scale
            return [(float(left + x), float(top + height - y)) for x, y in xy]

        # Clip all paths through the panel so the zoom cannot draw into titles.
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ink = ImageDraw.Draw(layer)
        for path, color, line_width in [(astar, (145, 145, 145), 2),
                                         (original, (235, 139, 30), 4), (planned, (0, 125, 245), 4)]:
            pts = [(x - left, y - top) for x, y in pixels(path)]
            ink.line(pts, fill=(*color, 255), width=line_width)
        for index in keys:
            x, y = pixels(original[index:index+1])[0]
            x, y = x - left, y - top
            ink.ellipse((x-5, y-5, x+5, y+5), fill=(225, 55, 75, 255), outline="white")
            ink.text((x+7, y-20), str(index), font=small, fill=(170, 25, 45, 255))
        ink.line([(20, height - 20), (20 + scale, height - 20)], fill=(0, 0, 0, 255), width=3)
        ink.text((20, height - 43), "1 m", font=small, fill=(0, 0, 0, 255))
        canvas.paste(layer, (left, top), layer)
        draw.text((left, top - 25), label, font=small, fill="white")
    draw.text((1125, 590), "Same start / goal", font=small, fill="white")
    draw.text((1125, 620), "0.30 m envelope", font=small, fill="white")
    draw.text((1125, 650), "2D map validation", font=small, fill="white")
    draw.text((1125, 680), "Synthetic reference", font=small, fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
