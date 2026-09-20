"""Read official RGB rows associated with the already verified source episode."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def load_episode_rgb(path, source_parquet, frame_count):
    """Require the official episode directory identity, not just equal counts."""
    path, parquet = Path(path).resolve(strict=True), Path(source_parquet).resolve(strict=True)
    root = parquet.parent.parent.parent
    expected = root / "videos/chunk-000/observation.images.rgb/rgb.npy"
    if parquet.parent.parent.name != "data" or path != expected:
        raise ValueError("official RGB must belong to the same episode directory as the source Parquet")
    frames = np.load(path, allow_pickle=False, mmap_mode="r")
    if (frames.ndim != 4 or frames.shape[0] != frame_count or frames.shape[-1] != 3
            or min(frames.shape) < 1 or frames.dtype != np.uint8):
        raise ValueError("official RGB must be uint8 [source_frames, height, width, 3]")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return frames, dict(path=str(path), sha256=digest, shape=list(frames.shape),
                       association="same official episode root as verified Parquet; one RGB row per source pose",
                       camera_calibration_verified=False, renderer="official published RGB",
                       display_fit="preserve aspect ratio, letterbox without cropping",
                       pixel_interpolation="spatial resizing only; no temporal interpolation")
