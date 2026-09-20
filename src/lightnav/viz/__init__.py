"""Visualisation: trajectory ribbon + pointing + HUD overlay, episode recording, video.

Importing this package needs only numpy and Pillow. Functions that draw or encode
video import ``cv2`` / ``imageio`` lazily and raise ImportError pointing at
``pip install 'lightnav[video]'`` when they are missing.
"""

from lightnav.viz.projection import bottom_edge_depth, project_waypoints_to_image
from lightnav.viz.recorder import ConnectionRecorder, EpisodeRecorder
from lightnav.viz.render import (
    DEFAULT_WAYPOINT_DT_S,
    TRAJ_WIDTH_FRAC,
    body_velocity,
    draw_pointing_soft,
    draw_scifi_hud,
    draw_traj_ribbon,
    normalize_instruction,
    pointing_points,
    render_frame,
)
from lightnav.viz.render_episode import (
    DEFAULT_OUT_NAME,
    DEFAULT_VIDEO_FPS,
    find_episode_dirs,
    load_manifest,
    load_records,
    render_episode_dir,
)
from lightnav.viz.video import (
    MAX_STEP_REPEATS,
    decode_rgb_bytes,
    encode_jpeg_bytes,
    open_video_writer,
    pad_to_even_dimensions,
    step_repeats,
    upscale_to_height,
)

__all__ = [
    "DEFAULT_OUT_NAME",
    "DEFAULT_VIDEO_FPS",
    "DEFAULT_WAYPOINT_DT_S",
    "MAX_STEP_REPEATS",
    "TRAJ_WIDTH_FRAC",
    "ConnectionRecorder",
    "EpisodeRecorder",
    "body_velocity",
    "bottom_edge_depth",
    "decode_rgb_bytes",
    "draw_pointing_soft",
    "draw_scifi_hud",
    "draw_traj_ribbon",
    "encode_jpeg_bytes",
    "find_episode_dirs",
    "load_manifest",
    "load_records",
    "normalize_instruction",
    "open_video_writer",
    "pad_to_even_dimensions",
    "pointing_points",
    "project_waypoints_to_image",
    "render_episode_dir",
    "render_frame",
    "step_repeats",
    "upscale_to_height",
]
