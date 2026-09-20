"""Turn an inference sample dict into Qwen3-VL model inputs.

``Qwen3VLDataProcessor.process_sample`` renders the chat template for
``sample["conversations"]``, feeds ``sample["video_segments"]`` through
:class:`~lightnav.processing.VLNQwen3VLProcessor` with per-segment pooling
settings, and returns ``input_ids`` / ``attention_mask`` / ``labels`` /
``position_ids`` / ``pixel_values_videos`` / ``video_grid_thw`` (+ vision masks).

Labels are ``IGNORE_INDEX`` (-100) everywhere except the assistant turn: the engine uses
the first non-ignored label as the answer boundary and cuts the prompt there before
generation, so the mechanism is kept exactly as at training time.
"""

import logging
import re
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import torch

from lightnav.eval_config import DEFAULT_VIDEO_SIZE

IGNORE_INDEX = -100
VALID_POOL_MODES = {"avg", "max"}
logger = logging.getLogger(__name__)


def _parse_video_segments(
    item: Dict[str, Any],
    default_pool_spatial: int = 1,
    default_pool_mode: str = "avg",
) -> List[Dict[str, Any]]:
    """
    Parse and normalize sample["video_segments"].

    Only the `video_segments` contract is accepted.
    """
    if default_pool_spatial < 1:
        raise ValueError(f"default_pool_spatial must be >= 1, got {default_pool_spatial}")
    if default_pool_mode not in VALID_POOL_MODES:
        raise ValueError(f"default_pool_mode must be one of {VALID_POOL_MODES}, got {default_pool_mode!r}")

    raw_segments = item.get("video_segments")
    if not isinstance(raw_segments, list) or len(raw_segments) == 0:
        raise ValueError("sample['video_segments'] must be a non-empty list")

    segments: List[Dict[str, Any]] = []
    for seg_idx, raw_seg in enumerate(raw_segments):
        if not isinstance(raw_seg, dict):
            raise ValueError(f"video_segments[{seg_idx}] must be a dict, got {type(raw_seg)}")
        if "video" not in raw_seg:
            raise ValueError(f"video_segments[{seg_idx}] missing required key: 'video'")

        frame_indices_raw = raw_seg.get("frame_indices")
        frame_indices: Optional[List[int]] = None
        if frame_indices_raw is not None:
            if not isinstance(frame_indices_raw, Sequence) or isinstance(frame_indices_raw, (str, bytes)):
                raise ValueError(f"video_segments[{seg_idx}].frame_indices must be a sequence of ints or None")
            frame_indices = [int(x) for x in frame_indices_raw]
            if len(frame_indices) == 0:
                raise ValueError(f"video_segments[{seg_idx}].frame_indices must not be empty")

        total_frames_raw = raw_seg.get("total_frames")
        total_frames: Optional[int] = None
        if total_frames_raw is not None:
            total_frames = int(total_frames_raw)
            if total_frames <= 0:
                raise ValueError(f"video_segments[{seg_idx}].total_frames must be > 0, got {total_frames}")

        spatial_raw = raw_seg.get("pool_spatial", default_pool_spatial)
        if spatial_raw is None:
            spatial_raw = default_pool_spatial
        pool_spatial = int(spatial_raw)
        if pool_spatial < 1:
            raise ValueError(f"video_segments[{seg_idx}].pool_spatial must be >= 1, got {pool_spatial}")

        mode_raw = raw_seg.get("pool_mode", default_pool_mode)
        if mode_raw is None:
            mode_raw = default_pool_mode
        pool_mode = str(mode_raw)
        if pool_mode not in VALID_POOL_MODES:
            raise ValueError(
                f"video_segments[{seg_idx}].pool_mode must be one of {VALID_POOL_MODES}, got {pool_mode!r}"
            )

        segments.append(
            {
                "video": raw_seg["video"],
                "frame_indices": frame_indices,
                "total_frames": total_frames,
                "pool_spatial": pool_spatial,
                "pool_mode": pool_mode,
            }
        )
    return segments


def _build_messages(item: Dict[str, Any], videos: List[Any]) -> List[Dict[str, Any]]:
    """Build Qwen-style messages from conversations and ordered segment videos."""
    if "conversations" not in item:
        raise ValueError("sample is missing required key: 'conversations'")
    if not isinstance(item["conversations"], list) or len(item["conversations"]) == 0:
        raise ValueError("sample['conversations'] must be a non-empty list")

    video_pool = [{"type": "video", "video": vid} for vid in videos]
    messages: List[Dict[str, Any]] = []
    for turn_idx, turn in enumerate(item["conversations"]):
        if not isinstance(turn, dict):
            raise ValueError(f"conversations[{turn_idx}] must be a dict")
        if "from" not in turn or "value" not in turn:
            raise ValueError(f"conversations[{turn_idx}] must contain keys 'from' and 'value'")

        role = "user" if turn["from"] == "human" else "assistant"
        text = str(turn["value"])

        if role == "user":
            content = []
            parts = re.split(r"(<video>)", text)
            for part in parts:
                if part == "<video>":
                    if not video_pool:
                        raise ValueError("More <video> placeholders than video segments")
                    content.append(video_pool.pop(0))
                elif part.strip():
                    content.append({"type": "text", "text": part.strip()})
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if video_pool:
        raise ValueError(f"{len(video_pool)} unused video segments; placeholder count must equal segment count")
    return messages


def _update_processor_settings(
    processor,
    video_fps: int = 24,
    temporal_patch_size: int = 2,
):
    """Update shared processor runtime settings (no global pooling state mutation).

    Frames arrive pre-resized and pre-rescaled, so sampling / resizing / rescaling in
    the video processor are switched off.
    """
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "fps"):
            vp.fps = video_fps
        if hasattr(vp, "temporal_patch_size"):
            vp.temporal_patch_size = temporal_patch_size
        if hasattr(vp, "do_sample_frames"):
            vp.do_sample_frames = False
        if hasattr(vp, "do_resize"):
            vp.do_resize = False
        if hasattr(vp, "do_rescale"):
            vp.do_rescale = False

        vp_size = getattr(vp, "size", None)
        max_pixels = (
            vp_size.get("longest_edge") if isinstance(vp_size, dict) else getattr(vp_size, "longest_edge", None)
        )
        settings_key = (
            getattr(vp, "do_resize", None),
            getattr(vp, "do_rescale", None),
            getattr(vp, "do_sample_frames", None),
            getattr(vp, "fps", None),
            getattr(vp, "temporal_patch_size", None),
            max_pixels,
        )
        logged_keys = getattr(processor, "_vln_video_settings_log_keys", None)
        if logged_keys is None:
            logged_keys = set()
            processor._vln_video_settings_log_keys = logged_keys
        if settings_key not in logged_keys:
            logger.info(
                "Configured video processor: "
                f"do_resize={getattr(vp, 'do_resize', None)}, "
                f"do_rescale={getattr(vp, 'do_rescale', None)}, "
                f"do_sample_frames={getattr(vp, 'do_sample_frames', None)}, "
                f"fps={getattr(vp, 'fps', None)}, "
                f"temporal_patch_size={getattr(vp, 'temporal_patch_size', None)}, "
                f"max_pixels={max_pixels}"
            )
            logged_keys.add(settings_key)
    return processor


class Qwen3VLDataProcessor:
    """
    Sample -> model-input converter for Qwen3-VL.

    Converts samples with conversations and `video_segments` to model inputs:
    - input_ids: Tokenized text with vision placeholders
    - labels: Same as input_ids but IGNORE_INDEX outside the assistant turn
    - attention_mask: All ones
    - position_ids: 3D position IDs for RoPE
    - pixel_values_videos: Processed video tensors
    - video_grid_thw: Video grid dimensions
    - image_mask, video_mask: Masks for vision tokens
    """

    def __init__(
        self,
        processor,
        model_max_length: int = 4096,
        video_fps: int = 24,
        temporal_patch_size: int = 2,
        video_pool_enable: bool = False,
        video_pool_spatial: int = 2,
        video_pool_mode: str = "avg",
        video_pool_stage: str = "pre_vit",
        video_size: Optional[tuple] = None,
        position_id_func=None,
    ):
        if position_id_func is None:
            raise ValueError(
                "position_id_func is required. Pass the model's mrope position-id "
                "function so position_ids are computed by the same code path the model uses."
            )
        self._position_id_func = position_id_func
        self.processor = processor
        self.video_fps = video_fps
        self.temporal_patch_size = temporal_patch_size
        # Pooling policy is carried by sample["video_segments"]. The constructor
        # args are kept for interface compatibility, but do not mutate
        # processor-level pooling state from them.
        self.video_pool_enable = bool(video_pool_enable)
        self.video_pool_spatial = int(video_pool_spatial)
        self.video_pool_mode = str(video_pool_mode)
        self.video_pool_stage = str(video_pool_stage)
        self.segment_default_pool_spatial = 1
        self.segment_default_pool_mode = "avg"
        self.video_size = tuple(video_size) if video_size is not None else DEFAULT_VIDEO_SIZE

        self._sync_processor_settings()
        self.tokenizer = processor.tokenizer
        self.model_max_length = model_max_length

        # Get token IDs from tokenizer (not hardcoded)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
        self.assistant_token_id = assistant_tokens[0] if assistant_tokens else None

    def _sync_processor_settings(self) -> None:
        """Ensure the shared processor carries this checkpoint's runtime video settings."""
        self.processor = _update_processor_settings(
            self.processor,
            self.video_fps,
            self.temporal_patch_size,
        )

    def _compute_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute labels: only assistant responses are kept, the rest is IGNORE_INDEX."""
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_ids_flat = input_ids[0].tolist()
        length = len(input_ids_flat)

        if self.assistant_token_id is None:
            return labels

        pos = 0
        while pos < length:
            if input_ids_flat[pos] == self.assistant_token_id:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < length and input_ids_flat[ans_end] != self.im_end_token_id:
                    ans_end += 1
                if ans_end < length:
                    labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                    pos = ans_end
            pos += 1

        return labels

    def process_sample(
        self,
        sample: Dict[str, Any],
        add_generation_prompt: bool = False,
        validate_video_shapes: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single sample into model-ready format.

        Required sample keys:
        - conversations: list of dialog turns
        - video_segments: non-empty list of segment dicts
        - video_fps: float

        ``validate_video_shapes`` is accepted for interface compatibility and ignored:
        frames are resized to the checkpoint's ``video_size`` before they get here.
        """
        self._sync_processor_settings()

        # When the caller has pre-normalized video frames, temporarily disable
        # normalize in video_processor to avoid redundant work.
        skip_normalize = sample.get("_skip_normalize", False)
        vp = getattr(self.processor, "video_processor", None)
        if skip_normalize and vp is not None:
            _orig_do_normalize = getattr(vp, "do_normalize", True)
            vp.do_normalize = False

        segments = _parse_video_segments(
            sample,
            default_pool_spatial=self.segment_default_pool_spatial,
            default_pool_mode=self.segment_default_pool_mode,
        )

        vp = getattr(self.processor, "video_processor", None)
        if vp is not None and getattr(vp, "do_sample_frames", False):
            for seg_idx, seg in enumerate(segments):
                if seg["frame_indices"] is not None:
                    raise ValueError(
                        "video_segments.frame_indices requires video_processor.do_sample_frames=False. "
                        f"Found frame_indices in segment {seg_idx} while do_sample_frames=True."
                    )

        videos = [seg["video"] for seg in segments]
        messages = _build_messages(sample, videos)
        video_pool_spatial_list = [int(seg["pool_spatial"]) for seg in segments]
        video_pool_mode_list = [str(seg["pool_mode"]) for seg in segments]

        # SlowFast: feed our ABSOLUTE frame_indices so the <t.t seconds> prefixes
        # reflect true elapsed time + sparse stride. Without this the processor
        # defaults to frames_indices=range(T) (dense), corrupting sparse-tier time.
        # None for legacy samples -> processor builds default metadata (unchanged).
        video_metadata = None
        if sample.get("slowfast_abs_frame_indices"):
            from transformers.video_utils import VideoMetadata

            video_metadata = [
                VideoMetadata(
                    total_num_frames=int(seg["total_frames"]),
                    fps=self.video_fps,
                    frames_indices=[int(x) for x in seg["frame_indices"]],
                )
                for seg in segments
            ]
        # ViT-cache fast path: per-tubelet hit mask so the processor patchifies only
        # cache-MISS tubelets (hit tubelets' pixels are discarded by the ViT cache).
        # Keys must match the engine's ViT-cache keys: (f0, f1, grid_h, grid_w).
        selective_tubelet_mask = None
        cached_keys = sample.get("_vit_cached_keys")
        # Pre-ViT pooling operates on the full pixel rows of every tubelet, so it
        # cannot consume the miss-only rows the selective path emits; leave the
        # mask off for pooled pre_vit segments (the ViT cache still works: it
        # detects full-window rows by their count and skips cache hits itself).
        pre_vit_pooled = self.video_pool_stage == "pre_vit" and any(
            int(seg.get("pool_spatial") or 1) > 1 for seg in segments
        )
        if (
            cached_keys is not None
            and not pre_vit_pooled
            and all(seg["frame_indices"] is not None for seg in segments)
        ):
            vp_patch = int(getattr(vp, "patch_size", 16)) if vp is not None else 16
            # Grid of the frames actually in this sample (== video_size unless the
            # session chose an aspect-preserving size), never the configured default.
            frame_h, frame_w = (int(v) for v in segments[0]["video"].shape[-2:])
            gh = frame_h // vp_patch
            gw = frame_w // vp_patch
            tp = int(getattr(vp, "temporal_patch_size", 2)) if vp is not None else 2
            selective_tubelet_mask = []
            for seg in segments:
                fi = [int(x) for x in seg["frame_indices"]]
                # Tubelet count is ceil(frames / temporal_patch_size): the video
                # processor PADS an odd trailing frame into a full tubelet, so a
                # segment with an odd frame count still emits the padded tubelet.
                # Enumerate by ceil (not floor) and CLAMP the frame indices the
                # same way the ViT-cache path keys them (f0=fi[min(2u,n-1)],
                # f1=fi[min(2u+1,n-1)]) so the padded tubelet's hit key matches and
                # the selective grid t == the full processor's grid t (ceil).
                n = len(fi)
                num_tubelets = -(-n // tp) if n else 0  # ceil(n/tp)
                seg_hits = [
                    (fi[min(tp * u, n - 1)], fi[min(tp * u + 1, n - 1)], gh, gw) in cached_keys
                    for u in range(num_tubelets)
                ]
                selective_tubelet_mask.append(seg_hits)
        if selective_tubelet_mask is not None and video_metadata is None:
            # Non-SlowFast samples must keep the processor's DEFAULT timestamps
            # (frame positions within the window, ``range(T)`` per segment):
            # that is what the checkpoint saw at training time. Only the
            # per-tubelet mask needs metadata to be passed explicitly, so hand
            # over window-relative indices, never the absolute frame ids.
            from transformers.video_utils import VideoMetadata

            video_metadata = [
                VideoMetadata(
                    total_num_frames=int(seg["total_frames"]),
                    fps=self.video_fps,
                    frames_indices=list(range(len(seg["frame_indices"]))),
                )
                for seg in segments
            ]

        # Two-step processing: render template first, then call __call__
        # directly so video_pool_spatial_list kwargs reach the processor
        # without being consumed by apply_chat_template's kwargs routing.
        rendered_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
        )

        # Only pass video_metadata for SlowFast; legacy path stays byte-identical.
        metadata_kwargs = {"video_metadata": video_metadata} if video_metadata is not None else {}
        if selective_tubelet_mask is not None:
            metadata_kwargs["selective_tubelet_mask"] = selective_tubelet_mask
        result = self.processor(
            text=rendered_text,
            videos=videos,
            video_pool_spatial_list=video_pool_spatial_list,
            video_pool_mode_list=video_pool_mode_list,
            video_pool_stage=self.video_pool_stage,
            return_tensors="pt",
            **metadata_kwargs,
        )

        # Restore normalize setting
        if skip_normalize and vp is not None:
            vp.do_normalize = _orig_do_normalize

        input_ids = result["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)
        result["input_ids"] = input_ids

        result["labels"] = self._compute_labels(input_ids)

        grid_thw = result.get("image_grid_thw")
        if grid_thw is not None and not isinstance(grid_thw, Sequence):
            grid_thw = [grid_thw]

        video_grid_thw = result.get("video_grid_thw")
        if video_grid_thw is not None and not isinstance(video_grid_thw, Sequence):
            video_grid_thw = [video_grid_thw]

        cat_image_grid_thw = torch.cat(grid_thw, dim=0) if grid_thw else None
        cat_video_grid_thw = torch.cat(video_grid_thw, dim=0) if video_grid_thw else None

        # Qwen3VL encodes temporal position via timestamp tokens in the input
        # sequence, so second_per_grid_ts is not needed here. transformers 5.8's
        # Qwen3VL get_rope_index requires mm_token_type_ids (image==1, video==2,
        # text==0) to locate vision spans for 3D mrope.
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        mm_token_type_ids[input_ids == self.video_token_id] = 2
        pid_result = self._position_id_func(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=cat_image_grid_thw,
            video_grid_thw=cat_video_grid_thw,
        )
        result["position_ids"] = pid_result["position_ids"]

        result["attention_mask"] = torch.ones_like(input_ids, dtype=torch.bool)

        result["input_ids"] = result["input_ids"].squeeze(0)
        result["attention_mask"] = result["attention_mask"].squeeze(0)
        result["labels"] = result["labels"].squeeze(0)
        # get_rope_index returns [3, batch, seq_len]; batch dim is at index 1
        result["position_ids"] = result["position_ids"].squeeze(1)
        # transformers 5.8's Qwen3-VL processor emits `mm_token_type_ids` ([1, seq]),
        # which is not needed at inference (image/video masks are derived below).
        result.pop("mm_token_type_ids", None)
        # pixel_values_videos / video_grid_thw: processor may return with
        # leading batch dim [1, L, C] / [1, N, 3]; squeeze to [L, C] / [N, 3]
        # since process_sample always handles a single sample.
        pv = result.get("pixel_values_videos")
        if pv is not None and pv.ndim == 3 and pv.shape[0] == 1:
            result["pixel_values_videos"] = pv.squeeze(0)
        vgt = result.get("video_grid_thw")
        if vgt is not None and vgt.ndim == 3 and vgt.shape[0] == 1:
            result["video_grid_thw"] = vgt.squeeze(0)

        if add_generation_prompt:
            prompt_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            if prompt_tokens:
                added_ids = torch.tensor(prompt_tokens)
                result["input_ids"] = torch.cat([result["input_ids"], added_ids])
                result["attention_mask"] = torch.cat(
                    [result["attention_mask"], torch.ones_like(added_ids, dtype=torch.bool)]
                )
                result["labels"] = torch.cat([result["labels"], torch.full_like(added_ids, IGNORE_INDEX)])

        result = self._apply_max_length(result)

        input_mask = result["labels"] == IGNORE_INDEX
        result["image_mask"] = (result["input_ids"] == self.image_token_id) & input_mask
        result["video_mask"] = (result["input_ids"] == self.video_token_id) & input_mask

        return result

    def _apply_max_length(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Truncate sequences to max length."""
        if self.model_max_length is None:
            return data

        max_len = self.model_max_length
        if data["input_ids"].size(0) <= max_len:
            return data

        data["input_ids"] = data["input_ids"][:max_len]
        data["attention_mask"] = data["attention_mask"][:max_len]
        data["labels"] = data["labels"][:max_len]
        pos = data["position_ids"]
        if pos.dim() == 1:
            data["position_ids"] = pos[:max_len]
        else:
            # Qwen3-VL position ids are typically (3, seq_len): truncate on seq dim.
            data["position_ids"] = pos[..., :max_len]

        pos_len = data["position_ids"].size(-1) if data["position_ids"].dim() > 1 else data["position_ids"].size(0)
        if pos_len != data["input_ids"].size(0):
            raise ValueError(
                f"position_ids length ({pos_len}) does not match input_ids length ({data['input_ids'].size(0)}) "
                f"after truncation."
            )

        last_token = data["input_ids"][-1].item()
        if last_token in (self.image_token_id, self.video_token_id):
            raise ValueError(f"Truncation cut off vision token at position {max_len}")

        return data
