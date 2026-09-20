"""Qwen3-VL processor extension used for navigation/tracking inference.

Extends the stock ``Qwen3VLProcessor`` with:
- Per-video spatial pooling (``video_pool_spatial_list`` / ``video_pool_mode_list``),
  either before the ViT (``pre_vit``: pixel patches are pooled) or after it
  (``post_vit``: ViT embeddings are pooled, see :func:`post_vit_spatial_pool`).
- fps default taken from ``video_processor.fps`` instead of a hardcoded 24.
- ``temporal_patch_size``-aware ``<X.X seconds>`` timestamp merging, with an optional
  relative-time mode (``VLN_TIMESTAMP_RELATIVE=1``).
- A selective patchify path that only processes ViT-cache-miss tubelets.

The placeholder expansion and timestamp rendering here define the model input and must
stay byte-identical to the training-side processor.
"""

import logging
import math
import os
from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

logger = logging.getLogger(__name__)
_warned: set[str] = set()


def warn_once(msg: str) -> None:
    """Log ``msg`` at WARNING level the first time it is seen in this process."""
    if msg in _warned:
        return
    _warned.add(msg)
    logger.warning(msg)


def _pool_video_tokens(
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
    video_indices: list[int],
    spatial_factor: int,
    mode: str,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool selected videos on spatial merge-block grid while preserving block-major token order.
    """
    if spatial_factor <= 1 or not video_indices:
        return pixel_values_videos, video_grid_thw
    if mode not in {"avg", "max"}:
        raise ValueError(f"Unsupported video_pool_mode: {mode!r}. Expected 'avg' or 'max'.")
    if merge_size <= 0:
        raise ValueError(f"merge_size must be > 0, got {merge_size}")

    if not torch.is_tensor(video_grid_thw):
        video_grid_thw = torch.as_tensor(video_grid_thw, device=pixel_values_videos.device)
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
        raise ValueError(f"video_grid_thw must have shape [num_videos, 3], got {tuple(video_grid_thw.shape)}")
    if pixel_values_videos.ndim != 2:
        raise ValueError(
            f"pixel_values_videos must have shape [sum(t*h*w), d_patch], got {tuple(pixel_values_videos.shape)}"
        )

    num_videos = int(video_grid_thw.shape[0])
    selected = {int(i) for i in video_indices}
    for i in selected:
        if i < 0 or i >= num_videos:
            raise ValueError(f"video index {i} out of range for num_videos={num_videos}")

    split_sizes = [int(row[0].item() * row[1].item() * row[2].item()) for row in video_grid_thw]
    expected = sum(split_sizes)
    if expected != int(pixel_values_videos.shape[0]):
        raise ValueError(
            "pixel_values_videos and video_grid_thw are inconsistent: "
            f"sum(t*h*w)={expected}, pixel_values_videos.shape[0]={int(pixel_values_videos.shape[0])}"
        )

    video_blocks = torch.split(pixel_values_videos, split_sizes, dim=0)
    pooled_blocks: list[torch.Tensor] = []
    pooled_grids: list[list[int]] = []
    for vid_idx, (block, row) in enumerate(zip(video_blocks, video_grid_thw)):
        t, h, w = (int(row[0].item()), int(row[1].item()), int(row[2].item()))
        if vid_idx not in selected:
            pooled_blocks.append(block)
            pooled_grids.append([t, h, w])
            continue

        if h % merge_size != 0 or w % merge_size != 0:
            raise ValueError(
                f"Video[{vid_idx}] invalid grid ({t},{h},{w}) for merge_size={merge_size}: "
                "h and w must be divisible by merge_size."
            )

        merged_h = h // merge_size
        merged_w = w // merge_size
        if merged_h % spatial_factor != 0 or merged_w % spatial_factor != 0:
            raise ValueError(
                f"Video[{vid_idx}] invalid grid ({t},{h},{w}) for spatial_factor={spatial_factor}, "
                f"merge_size={merge_size}: (h//merge_size) and (w//merge_size) must be divisible by spatial_factor."
            )

        patch_dim = int(block.shape[-1])
        x = block.contiguous().view(t, merged_h, merged_w, merge_size * merge_size, patch_dim)
        x = x.view(t, merged_h, merged_w, merge_size * merge_size * patch_dim).permute(0, 3, 1, 2).contiguous()
        if mode == "avg":
            x = F.avg_pool2d(x, kernel_size=spatial_factor, stride=spatial_factor)
        else:
            x = F.max_pool2d(x, kernel_size=spatial_factor, stride=spatial_factor)

        pooled_merged_h = merged_h // spatial_factor
        pooled_merged_w = merged_w // spatial_factor
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(t, pooled_merged_h, pooled_merged_w, merge_size * merge_size, patch_dim)
        x = x.reshape(t * (h // spatial_factor) * (w // spatial_factor), patch_dim)

        pooled_blocks.append(x)
        pooled_grids.append([t, h // spatial_factor, w // spatial_factor])

    pooled_pixel_values = torch.cat(pooled_blocks, dim=0)
    pooled_grid = torch.tensor(pooled_grids, dtype=video_grid_thw.dtype, device=video_grid_thw.device)
    return pooled_pixel_values, pooled_grid


def compute_pooled_grid_per_video(
    video_grid_thw: torch.Tensor,
    pool_factors: torch.Tensor,
    merge_size: int,
) -> torch.Tensor:
    """Pooled ``grid_thw`` for tokenization in post-ViT pooling mode.

    Each row of ``video_grid_thw`` is pooled by its own ``pool_factors[i]``: the merged
    grid ``(h // merge_size, w // merge_size)`` is reduced by ``ceil(. / factor)`` and
    expressed back in pre-merge units. Rows with factor ``<=1`` are returned unchanged.
    The pixel values are untouched; the actual pooling happens after the ViT forward.
    """
    if pool_factors.shape[0] != video_grid_thw.shape[0]:
        raise ValueError(
            f"pool_factors length {pool_factors.shape[0]} != num videos {video_grid_thw.shape[0]}"
        )

    pooled_grids = []
    for row, factor in zip(video_grid_thw, pool_factors.tolist()):
        t, h, w = int(row[0].item()), int(row[1].item()), int(row[2].item())
        if factor <= 1:
            pooled_grids.append([t, h, w])
            continue
        merged_h, merged_w = h // merge_size, w // merge_size
        target_h = math.ceil(merged_h / factor)
        target_w = math.ceil(merged_w / factor)
        pooled_grids.append([t, target_h * merge_size, target_w * merge_size])
    return torch.tensor(pooled_grids, dtype=video_grid_thw.dtype, device=video_grid_thw.device)


def post_vit_spatial_pool(
    video_embeds: torch.Tensor,
    video_grid_thw: torch.Tensor,
    spatial_factor: "int | torch.Tensor | list[int]",
    merge_size: int,
    deepstack_embeds: "list[torch.Tensor] | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]":
    """Post-ViT spatial pooling using adaptive_avg_pool2d.

    Operates on ViT merger output (already spatially merged), reducing spatial
    dimensions by approximately ``spatial_factor`` in each direction.

    Args:
        video_embeds: Flat ``[sum(T_i * H'_i * W'_i), dim]`` where
            ``H' = H / merge_size, W' = W / merge_size``.
        video_grid_thw: ``[num_videos, 3]`` in **pre-merge** space ``(T, H, W)``.
        spatial_factor: Desired reduction factor -- either a single int applied
            to every video, or a per-video sequence/tensor of length
            ``num_videos``. Factor ``<=1`` leaves that video unchanged.
        merge_size: The model's ``spatial_merge_size`` (typically 2).
        deepstack_embeds: Optional list of deepstack tensors with the same
            spatial layout as *video_embeds*.

    Returns:
        ``(pooled_embeds, pooled_grid_thw, pooled_deepstack)``
    """
    n_videos = int(video_grid_thw.shape[0])
    if isinstance(spatial_factor, (int,)) or (
        isinstance(spatial_factor, torch.Tensor) and spatial_factor.dim() == 0
    ):
        factors = [int(spatial_factor)] * n_videos
    else:
        factors = [int(f) for f in (
            spatial_factor.tolist() if isinstance(spatial_factor, torch.Tensor) else spatial_factor
        )]
    if len(factors) != n_videos:
        raise ValueError(
            f"spatial_factor length {len(factors)} != num videos {n_videos}"
        )
    if all(f <= 1 for f in factors):
        return video_embeds, video_grid_thw, deepstack_embeds

    # Split flat embeddings by video using pre-merge grid dimensions
    split_sizes: list[int] = []
    for row in video_grid_thw:
        t, h, w = int(row[0].item()), int(row[1].item()), int(row[2].item())
        merged_h, merged_w = h // merge_size, w // merge_size
        split_sizes.append(t * merged_h * merged_w)

    video_blocks = torch.split(video_embeds, split_sizes, dim=0)
    ds_block_lists = (
        [torch.split(ds, split_sizes, dim=0) for ds in deepstack_embeds]
        if deepstack_embeds
        else None
    )

    pooled_blocks: list[torch.Tensor] = []
    pooled_ds_blocks: list[list[torch.Tensor]] = (
        [[] for _ in deepstack_embeds] if deepstack_embeds else []
    )
    pooled_grids: list[list[int]] = []

    for vid_idx, (block, row) in enumerate(zip(video_blocks, video_grid_thw)):
        t, h, w = int(row[0].item()), int(row[1].item()), int(row[2].item())
        merged_h, merged_w = h // merge_size, w // merge_size
        dim = block.shape[-1]
        factor = factors[vid_idx]

        if factor <= 1:
            pooled_blocks.append(block)
            pooled_grids.append([t, h, w])
            if ds_block_lists:
                for layer_idx, ds_per_vid in enumerate(ds_block_lists):
                    pooled_ds_blocks[layer_idx].append(ds_per_vid[vid_idx])
            continue

        target_h = math.ceil(merged_h / factor)
        target_w = math.ceil(merged_w / factor)

        # [T, H', W', dim] -> [T, dim, H', W'] -> pool -> [T, dim, tH, tW]
        x = block.view(t, merged_h, merged_w, dim).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, output_size=(target_h, target_w))
        x = x.permute(0, 2, 3, 1).reshape(-1, dim)
        pooled_blocks.append(x)

        # Grid in pre-merge space so downstream h // merge_size == target_h
        pooled_grids.append([t, target_h * merge_size, target_w * merge_size])

        # Pool deepstack features identically
        if ds_block_lists:
            for layer_idx, ds_per_vid in enumerate(ds_block_lists):
                ds = ds_per_vid[vid_idx]
                ds_dim = ds.shape[-1]
                ds_x = ds.view(t, merged_h, merged_w, ds_dim).permute(0, 3, 1, 2)
                ds_x = F.adaptive_avg_pool2d(ds_x, output_size=(target_h, target_w))
                ds_x = ds_x.permute(0, 2, 3, 1).reshape(-1, ds_dim)
                pooled_ds_blocks[layer_idx].append(ds_x)

    pooled_embeds = torch.cat(pooled_blocks, dim=0)
    pooled_grid = torch.tensor(
        pooled_grids, dtype=video_grid_thw.dtype, device=video_grid_thw.device,
    )

    pooled_deepstack: list[torch.Tensor] | None = None
    if ds_block_lists:
        pooled_deepstack = [torch.cat(layer, dim=0) for layer in pooled_ds_blocks]
    else:
        pooled_deepstack = deepstack_embeds

    return pooled_embeds, pooled_grid, pooled_deepstack


class VLNQwen3VLProcessor(Qwen3VLProcessor):
    """
    Navigation-specific extension of Qwen3VLProcessor.

    Adds per-video spatial pooling via video_pool_spatial_list / video_pool_mode_list
    kwargs in __call__, and fixes fps default / temporal merge size for timestamp
    calculation.
    """

    def _calculate_timestamps(self, indices, video_fps, merge_size: int = 2):
        """Per-temporal-patch timestamps rendered as ``<X.X seconds>`` prefixes.

        Default (absolute) is byte-identical to the parent / every existing
        checkpoint: ``t = idx / fps`` (seconds since episode start).

        ``VLN_TIMESTAMP_RELATIVE=1`` switches to time-before-current
        ``t = (newest_idx - idx) / fps`` so the current frame anchors near 0 and
        the wording no longer drifts with episode length. This changes the learned
        input, so it MUST match between training and inference; the model loader
        bridges the checkpoint's recorded mode into this env var. The
        ``<X.X seconds>`` render format is unchanged either way.
        """
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        if os.environ.get("VLN_TIMESTAMP_RELATIVE", "0").lower() in ("1", "true", "yes"):
            # Reference the GLOBAL current frame (set by __call__ across all
            # segments); fall back to this segment's newest if unset.
            ref = getattr(self, "_ts_global_ref", None)
            if ref is None:
                ref = max(indices)
            secs = [(ref - idx) / video_fps for idx in indices]
        else:
            secs = [idx / video_fps for idx in indices]
        # Average first/last frame within each temporal patch (merge_size=2).
        return [(secs[i] + secs[i + merge_size - 1]) / 2 for i in range(0, len(secs), merge_size)]

    def _selective_video_process(self, videos, selective_mask, videos_kwargs):
        """ViT-cache fast path: patchify ONLY cache-MISS tubelets.

        ``selective_mask[s][u]`` is True when segment ``s`` tubelet ``u`` is
        already in the engine's ViT cache, so its ViT embed will be reused and
        its pixels are never read. Returns the same dict shape as ``video_processor``
        (FULL pre-pool ``video_grid_thw`` + passthrough ``video_metadata``) but with
        ONLY the miss tubelets' pixel rows, in segment -> tubelet order; the ViT cache
        detects the miss-only layout by row count and maps misses in that same order.
        The downstream pooling / timestamp / input_ids path is unchanged.

        The real video_processor is used for the miss frames so the patch layout is
        byte-identical to full processing.
        """
        tp = int(getattr(self.video_processor, "temporal_patch_size", 2))
        miss_clips: list = []
        plan: list = []  # per segment: list[bool] (is_hit per tubelet)
        for vid, seg_mask in zip(videos, selective_mask):
            plan.append(list(seg_mask))
            # seg_mask is ceil(frames/tp) long (one entry per tubelet, INCLUDING
            # the padded odd-trailing-frame tubelet). For that last tubelet the
            # slice yields fewer than tp frames; self.video_processor pads it to a
            # full tubelet (same `-T % tp` padding as the full path), so the grid t
            # below (== len(seg_mask) == ceil) matches the full processor exactly.
            for u, is_hit in enumerate(seg_mask):
                if not is_hit:
                    miss_clips.append(vid[u * tp : (u + 1) * tp])

        # Degenerate: nothing new this step (shouldn't happen -- current tier is
        # always fresh) -> fall back to full processing for safety.
        if not miss_clips:
            return self.video_processor(videos=videos, **videos_kwargs)

        kw = {k: v for k, v in videos_kwargs.items() if k != "video_metadata"}
        miss_out = self.video_processor(videos=miss_clips, **kw)
        miss_pixels = miss_out["pixel_values_videos"]
        miss_grid = miss_out["video_grid_thw"]
        h, w = int(miss_grid[0][1]), int(miss_grid[0][2])

        grid_rows: list = [[len(seg_mask), h, w] for seg_mask in plan]  # full window, t=#tubelets
        video_grid_thw = torch.tensor(grid_rows, dtype=torch.long)

        # Ship ONLY miss-tubelet pixels: grid_thw stays full-window (token layout /
        # positions unchanged) while hit tubelets are filled from cached embeds.
        out = {"pixel_values_videos": miss_pixels, "video_grid_thw": video_grid_thw}
        if "video_metadata" in videos_kwargs:
            out["video_metadata"] = videos_kwargs["video_metadata"]
        return out

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        pool_spatial_list = kwargs.pop("video_pool_spatial_list", None)
        pool_mode_list = kwargs.pop("video_pool_mode_list", None)
        pool_stage = kwargs.pop("video_pool_stage", "pre_vit")
        selective_tubelet_mask = kwargs.pop("selective_tubelet_mask", None)

        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            if selective_tubelet_mask is not None:
                videos_inputs = self._selective_video_process(
                    videos, selective_tubelet_mask, output_kwargs["videos_kwargs"]
                )
            else:
                videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if video_grid_thw is not None:
                n_video = int(video_grid_thw.shape[0])

                if pool_spatial_list is None:
                    pool_spatial_values = [1] * n_video
                else:
                    try:
                        pool_spatial_values = [int(x) for x in pool_spatial_list]
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "video_pool_spatial_list must be a list of ints >= 1 when videos are provided"
                        ) from exc

                if pool_mode_list is None:
                    pool_mode_values = ["avg"] * n_video
                else:
                    pool_mode_values = [str(x) for x in pool_mode_list]

                if len(pool_spatial_values) != n_video:
                    raise ValueError(
                        f"video_pool_spatial_list length {len(pool_spatial_values)} != num_videos {n_video}"
                    )
                if len(pool_mode_values) != n_video:
                    raise ValueError(f"video_pool_mode_list length {len(pool_mode_values)} != num_videos {n_video}")

                for idx, spatial in enumerate(pool_spatial_values):
                    if spatial < 1:
                        raise ValueError(f"video_pool_spatial_list[{idx}] must be >= 1, got {spatial}")
                for idx, mode in enumerate(pool_mode_values):
                    if mode not in {"avg", "max"}:
                        raise ValueError(f"video_pool_mode_list[{idx}] must be 'avg' or 'max', got {mode!r}")

                pool_indices = [idx for idx, spatial in enumerate(pool_spatial_values) if spatial > 1]
                if pool_indices and pool_stage == "pre_vit":
                    # Pre-ViT pooling: reduce pixel_values before ViT
                    merge_length = self.video_processor.merge_size**2
                    before_tokens = int((video_grid_thw.prod(dim=1) // merge_length).sum().item())

                    grouped_indices = defaultdict(list)
                    for idx in pool_indices:
                        grouped_indices[(pool_spatial_values[idx], pool_mode_values[idx])].append(idx)

                    pooled_pixels = videos_inputs["pixel_values_videos"]
                    pooled_grid = video_grid_thw
                    for (spatial_factor, mode), indices in sorted(
                        grouped_indices.items(),
                        key=lambda x: (x[0][0], x[0][1]),
                    ):
                        pooled_pixels, pooled_grid = _pool_video_tokens(
                            pooled_pixels,
                            pooled_grid,
                            video_indices=indices,
                            spatial_factor=spatial_factor,
                            mode=mode,
                            merge_size=self.video_processor.merge_size,
                        )

                    videos_inputs["pixel_values_videos"] = pooled_pixels
                    videos_inputs["video_grid_thw"] = pooled_grid
                    video_grid_thw = pooled_grid
                    after_tokens = int((video_grid_thw.prod(dim=1) // merge_length).sum().item())

                    log_key = (
                        "pre_vit",
                        tuple(pool_spatial_values),
                        tuple(pool_mode_values),
                        tuple(pool_indices),
                    )
                    if getattr(self, "_video_pool_logged_key", None) != log_key:
                        logger.info(
                            "Qwen3VL pre-ViT video pooling: "
                            f"policies={list(zip(pool_spatial_values, pool_mode_values))}, "
                            f"indices={pool_indices}, video tokens {before_tokens}->{after_tokens}"
                        )
                        self._video_pool_logged_key = log_key

                elif pool_stage == "post_vit":
                    # Post-ViT pooling: pixel_values_videos stays at full
                    # resolution for ViT. We record (1) the original grid so
                    # the patched ``get_video_features`` can feed it to ViT
                    # and (2) per-video pool factors so the post-ViT pool step
                    # can pool only the videos that should be pooled. Emitted
                    # UNCONDITIONALLY (even when no video needs pooling) so the
                    # grid fed to the ViT always travels with pixel_values_videos.
                    merge_size = self.video_processor.merge_size
                    merge_length = merge_size**2
                    before_tokens = int((video_grid_thw.prod(dim=1) // merge_length).sum().item())

                    factors = torch.tensor(pool_spatial_values, dtype=torch.int64)
                    pooled_grid = compute_pooled_grid_per_video(video_grid_thw, factors, merge_size)

                    videos_inputs["_original_video_grid_thw"] = video_grid_thw.clone()
                    videos_inputs["_post_vit_pool_factors"] = factors
                    videos_inputs["video_grid_thw"] = pooled_grid
                    video_grid_thw = pooled_grid
                    after_tokens = int((video_grid_thw.prod(dim=1) // merge_length).sum().item())

                    log_key = (
                        "post_vit",
                        tuple(pool_spatial_values),
                        tuple(pool_mode_values),
                        tuple(pool_indices),
                    )
                    if getattr(self, "_video_pool_logged_key", None) != log_key:
                        logger.info(
                            "Qwen3VL post-ViT video pooling: "
                            f"per-video factors={pool_spatial_values}, "
                            f"video tokens {before_tokens}->{after_tokens}"
                        )
                        self._video_pool_logged_key = log_key
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            # Global current frame = max abs index across ALL segments. Relative
            # timestamps must reference the true current frame, but
            # _calculate_timestamps runs per-segment and only sees one segment's
            # indices -- so compute the global reference here for it to use.
            self._ts_global_ref = max(
                (int(i) for m in video_metadata
                 if getattr(m, "frames_indices", None) is not None for i in m.frames_indices),
                default=None,
            )
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        default_fps = getattr(self.video_processor, "fps", 24)
                        warn_once(
                            "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            f"Defaulting to `fps={default_fps}` (from video_processor.fps). "
                            "Please provide `video_metadata` for more accurate results."
                        )
                        metadata.fps = default_fps

                    temporal_merge = getattr(self.video_processor, "temporal_patch_size", 2)
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        temporal_merge,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)


def enable_post_vit_pool(model, spatial_factor: int, merge_size: int = 2):
    """Monkey-patch a Qwen3VL model to apply post-ViT spatial pooling.

    Qwen3VL has two levels:
      - ``Qwen3VLForConditionalGeneration`` (outer, CausalLM head)
      - ``Qwen3VLForConditionalGeneration.model``: ``Qwen3VLModel`` (inner;
        owns ``visual`` and calls ``self.get_video_features`` from its own
        ``forward``)

    The inner ``Qwen3VLModel.forward`` calls ``self.get_video_features`` where
    ``self`` is the inner module -- so the patch MUST be applied there. The
    outer method just delegates to the inner one; patching it alone is a no-op
    at runtime because the outer ``forward`` never calls its own
    ``get_video_features``.

    The caller should set ``_post_vit_original_grid`` / ``_post_vit_pool_factors`` on
    the object returned by :func:`get_post_vit_target` (the inner model) before each
    forward pass.
    """
    if spatial_factor <= 1:
        return

    # Resolve the inner module that actually runs get_video_features at forward
    target = get_post_vit_target(model)

    _orig_get_video_features = target.get_video_features

    def _patched_get_video_features(pixel_values_videos, video_grid_thw=None, **kwargs):
        original_grid = getattr(target, "_post_vit_original_grid", None)
        vit_grid = original_grid if original_grid is not None else video_grid_thw
        # The grid used by the ViT drives cu_seqlens for flash-attn varlen,
        # which requires CUDA tensors. Collated _original_video_grid_thw lives
        # on CPU; move it to match pixel_values_videos.
        if vit_grid is not None and vit_grid.device != pixel_values_videos.device:
            vit_grid = vit_grid.to(pixel_values_videos.device)

        # get_video_features returns a BaseModelOutputWithDeepstackFeatures. transformers
        # 5.8 sets `pooler_output` to a TUPLE of per-video flat tensors (split by the
        # original grid; the model forward then torch.cat's them back); some forks keep
        # it FLAT [sum_tokens, dim]. Handle both: concatenate to flat for pooling, then
        # return in the same shape the caller's forward expects.
        outputs = _orig_get_video_features(pixel_values_videos, vit_grid, return_dict=True)
        pooler = outputs.pooler_output
        was_tuple = isinstance(pooler, (tuple, list))
        video_embeds = torch.cat(list(pooler), dim=0) if was_tuple else pooler
        deepstack = outputs.deepstack_features

        # Per-video pool factors if provided by the data pipeline; otherwise
        # fall back to the single spatial_factor this patch was set up with.
        factors = getattr(target, "_post_vit_pool_factors", None)
        sf = factors if factors is not None else spatial_factor

        video_embeds, _, deepstack = post_vit_spatial_pool(
            video_embeds,
            vit_grid,
            spatial_factor=sf,
            merge_size=merge_size,
            deepstack_embeds=deepstack,
        )
        # Stock forward does `torch.cat(pooler_output, dim=0)`; a 1-tuple of the full
        # pooled flat tensor reassembles to exactly the pooled embeds in video order.
        outputs.pooler_output = (video_embeds,) if was_tuple else video_embeds
        outputs.deepstack_features = deepstack
        return outputs

    target.get_video_features = _patched_get_video_features
    logger.info(
        f"Post-ViT spatial pooling enabled on {type(target).__name__} "
        f"(factor={spatial_factor}, merge={merge_size})"
    )


def get_post_vit_target(model):
    """Resolve the module that owns the real ``get_video_features``.

    For Qwen3VL, this is ``model.model`` (the inner ``Qwen3VLModel``). Falls
    back to ``model`` if no such attribute exists so other model layouts still work.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "get_video_features"):
        return inner
    return model


__all__ = [
    "VLNQwen3VLProcessor",
    "_pool_video_tokens",
    "compute_pooled_grid_per_video",
    "post_vit_spatial_pool",
    "enable_post_vit_pool",
    "get_post_vit_target",
    "warn_once",
]
