"""Video-writing primitives: timebase, frame padding, encoder, and frame codecs.

``cv2`` and ``imageio`` belong to the ``video`` extra and are imported lazily inside
the functions that need them, so this module imports with numpy + Pillow alone.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np

# Ceiling on how many encoded frames one step may occupy on the realtime timebase,
# so a long client stall does not turn a single step into minutes of video.
MAX_STEP_REPEATS = 20

_INSTALL_HINT = "pip install 'lightnav[video]'"


def _optional_import(module: str):
    """Import an optional dependency or raise a clear ImportError naming the extra."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"lightnav.viz needs the optional module {module!r}; install it with "
            f"{_INSTALL_HINT}"
        ) from e


def step_repeats(step_dt_ms, fps: int, realtime: bool) -> int:
    """Number of frames a step earns at ``fps``.

    On the realtime timebase a step is repeated ``round(step_dt_ms * fps / 1000)``
    times, clamped to ``[1, MAX_STEP_REPEATS]``, so a stall reads as a stall. With
    ``realtime=False`` every step is exactly one frame. Missing or non-positive
    durations yield one frame.
    """
    if not realtime:
        return 1
    try:
        dt = float(step_dt_ms) if step_dt_ms is not None else 0.0
    except (TypeError, ValueError):
        return 1
    if not (dt > 0.0):
        return 1
    if not math.isfinite(dt):
        return MAX_STEP_REPEATS
    return max(1, min(int(round(dt * fps / 1000.0)), MAX_STEP_REPEATS))


def pad_to_even_dimensions(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return ``(frame, pad_right, pad_bottom)`` with even width and height.

    yuv420p needs even dimensions. Odd sides gain one replicated edge row/column
    instead of being resized, so the overlay geometry is preserved.
    """
    height, width = frame.shape[:2]
    padding_bottom = height % 2
    padding_right = width % 2
    if not padding_bottom and not padding_right:
        return frame, 0, 0
    padding = [(0, padding_bottom), (0, padding_right)] + [
        (0, 0) for _ in frame.shape[2:]
    ]
    return np.pad(frame, padding, mode="edge"), padding_right, padding_bottom


def open_video_writer(path: str | Path, fps: int):
    """Open an H.264 / yuv420p writer (imageio + ffmpeg) for RGB uint8 frames.

    ``macro_block_size=2`` keeps imageio from silently rescaling frames whose sides
    are not multiples of 16. Requires the ``video`` extra.
    """
    imageio = _optional_import("imageio.v2")
    _optional_import("imageio_ffmpeg")
    return imageio.get_writer(
        str(path),
        fps=int(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
    )


def upscale_to_height(rgb: np.ndarray, target_h: int) -> np.ndarray:
    """Resample so height equals ``target_h`` with the aspect ratio preserved.

    Width is rounded to an even number for the encoder. Lanczos when enlarging,
    area averaging when shrinking. ``target_h <= 0`` returns the input unchanged.
    Requires the ``video`` extra.
    """
    h, w = rgb.shape[:2]
    if target_h <= 0 or h == target_h:
        return rgb
    cv2 = _optional_import("cv2")
    target_w = int(round(w * target_h / h))
    target_w += target_w & 1
    interp = cv2.INTER_LANCZOS4 if target_h > h else cv2.INTER_AREA
    return cv2.resize(rgb, (target_w, target_h), interpolation=interp)


def decode_rgb_bytes(data: bytes) -> np.ndarray:
    """Decode an encoded image (JPEG/PNG) to an HWC uint8 RGB array via Pillow."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return np.array(image.convert("RGB"), dtype=np.uint8)


def encode_jpeg_bytes(rgb: np.ndarray, quality: int = 95) -> bytes:
    """Encode an HWC uint8 RGB array as JPEG bytes via Pillow."""
    from PIL import Image

    arr = np.ascontiguousarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()
