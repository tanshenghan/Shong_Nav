"""Frame preprocessing: RGB uint8 frames -> model-space video tensors."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def resize_video_tensor(video: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize a ``[C,H,W]`` or ``[T,C,H,W]`` float tensor to ``target_size``."""
    if video.numel() == 0:
        return video
    target_h, target_w = target_size
    if video.ndim == 3:
        _, h, w = video.shape
        if h == target_h and w == target_w:
            return video
        return F.interpolate(
            video.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        ).squeeze(0)
    _, _, h, w = video.shape
    if h == target_h and w == target_w:
        return video
    return F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)


def choose_video_size(
    frame_hw: tuple[int, int], base_video_size: tuple[int, int], *, multiple: int = 32
) -> tuple[int, int]:
    """Model frame size that keeps the source aspect ratio at the checkpoint's pixel budget.

    Both sides are multiples of ``multiple`` (patch size x merge size, 32 for Qwen3-VL) and
    the area is as close as possible to ``base_video_size``'s, so the vision-token count
    per frame stays what the checkpoint was trained with while a 4:3 camera is no longer
    stretched to 16:9. A source that already has the checkpoint's aspect ratio maps to
    ``base_video_size`` itself.
    """
    src_h, src_w = (max(1, int(v)) for v in frame_hw)
    base_h, base_w = (int(v) for v in base_video_size)
    if (src_h, src_w) == (base_h, base_w):
        return base_h, base_w
    aspect = src_w / src_h
    area = float(base_h * base_w)
    ideal_h = (area / aspect) ** 0.5
    best: tuple[float, float, tuple[int, int]] | None = None
    lo = max(multiple, int(ideal_h // multiple - 2) * multiple)
    for h in range(lo, int(ideal_h // multiple + 3) * multiple + 1, multiple):
        w = max(multiple, int(round(h * aspect / multiple)) * multiple)
        score = (abs(h * w - area), abs(w / h - aspect), (h, w))
        if best is None or score < best:
            best = score
    assert best is not None
    return best[2]


def rgb_frame_to_model_tensor(rgb_frame: np.ndarray, video_size: tuple[int, int]) -> torch.Tensor:
    """Convert one HWC RGB uint8 frame to the checkpoint's ``[C,H,W]`` tensor in ``[-1, 1]``.

    This is the single authority for the spatial contract: whatever a client
    sends, every session of one checkpoint must produce the same ``video_size``
    grid, or the ViT token count and batching assumptions drift per session.
    The check is two integer comparisons, so it runs on every frame; errors
    report shapes only, never pixels.
    """
    shape = tuple(int(dim) for dim in rgb_frame.shape)
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"Model Frame requires an HWC RGB Source Frame, got shape {shape}")
    t = torch.from_numpy(rgb_frame.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
    t = resize_video_tensor(t, video_size)
    spatial = tuple(int(dim) for dim in t.shape[-2:])
    if t.ndim != 3 or spatial != tuple(int(dim) for dim in video_size):
        raise ValueError(
            f"Model Frame shape {tuple(int(dim) for dim in t.shape)} does not match "
            f"the checkpoint video_size {tuple(int(dim) for dim in video_size)}"
        )
    return t * 2.0 - 1.0
