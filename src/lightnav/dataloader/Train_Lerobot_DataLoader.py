"""LightNav training: image-backed LeRobot v3 R2R -> SlowFast -> BetaTrainBatch.

Reads selected Parquet columns/row groups directly: no full-dataset Arrow copy,
depth loading, LeRobot policy dependencies, or model weights in loader workers.
The supported export stores each episode in one data file (possibly multiple
row groups). All indices and masks are checked before making a training sample.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lightnav.inference.beta_model import BetaTrainBatch, uses_frozen_continuous_backbone
from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor
from lightnav.inference.model import _resolve_processing_params, resolve_model_paths
from lightnav.prompts import (
    CONTINUOUS_TRAJ_PROMPT_TEMPLATE,
    UNIFIED_TRAJ_PROMPT_TEMPLATE,
    build_video_block,
    to_rvq_prompt,
)
from lightnav.slowfast import slowfast_segments, validate_slowfast_tiers
from lightnav.training.stop_labels import stop_target_for_row, validate_stop_supervision

RGB_KEY = "observation.images.rgb"
TARGET_COLUMNS = [
    "index", "episode_index", "frame_index", "task_index", "source.episode_id", "source.scene_id",
    "action.apos.token_id", "action.apos.token", "action.apos.valid_mask",
    "action.opos.token_id", "action.opos.token", "action.opos.valid_mask",
    "action.rvq.codes", "action.rvq.tokens", "action.rvq.valid_mask",
    "action.fm.clean_se2_diff", "action.fm.valid_mask", "action.fm.future_frame_ids",
    "action.explicit_stop", "action.stop_loss_valid",
]
FM_TARGET_COLUMNS = [name for name in TARGET_COLUMNS if not name.startswith(
    ("action.apos.", "action.opos.", "action.rvq.", "action.explicit_stop", "action.stop_loss_valid")
)]
FM_STOP_TARGET_COLUMNS = [*FM_TARGET_COLUMNS, "action.explicit_stop", "action.stop_loss_valid"]
EPISODE_COLUMNS = [
    "episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
    "dataset_from_index", "dataset_to_index",
]


def _json(path: Path):
    return json.loads(path.read_text())


def _arrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Run bash scripts/setup_betanav_train.sh and use .venv-train/bin/python") from exc
    return pq


def _scalar(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar or one-element LeRobot feature")
    return array.item()


def _put_lru(cache, key, value, capacity):
    if capacity:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
    return value


@dataclass(frozen=True)
class LightNavTrainingSpec:
    model_path: str
    processor_path: str
    params: dict[str, Any]
    rvq_path: str
    training_mode: str = "ce_fm"

    @classmethod
    def from_checkpoint(cls, model_path: str | Path, *, training_mode: str = "ce_fm"):
        if training_mode not in {"ce_fm", "fm_only", "fm_stop"}:
            raise ValueError("training_mode must be ce_fm, fm_only or fm_stop")
        path = str(Path(model_path).resolve())
        params = _resolve_processing_params(path, task_type="vlnce")
        if not params.get("slowfast_tiers"):
            raise ValueError("Train_Lerobot_DataLoader requires checkpoint SlowFast tiers")
        params["slowfast_tiers"] = validate_slowfast_tiers(params["slowfast_tiers"])
        config_dir, _, processor_dir = resolve_model_paths(path)
        if uses_frozen_continuous_backbone(training_mode):
            params.pop("action_tokenizer", None)
            return cls(path, processor_dir, params, "", training_mode)
        action = params.get("action_tokenizer") or {}
        if action.get("method") != "rvq" or int(params["predict_horizon"]) != 10:
            raise ValueError("this dataset requires an RVQ checkpoint with horizon=10")
        bundle = Path(action.get("bundle_path", "action_tokenizer"))
        if not bundle.is_absolute():
            bundle = Path(config_dir) / bundle
        return cls(path, processor_dir, params, str(bundle.resolve()))


@dataclass(frozen=True)
class _Episode:
    index: int
    start: int
    stop: int
    path: str
    instruction: str

    @property
    def length(self):
        return self.stop - self.start


def _check_codebooks(data_root: Path, model_root: Path) -> dict:
    """Check the actual decoder assets, not just the number of token classes."""
    data_manifest = _json(data_root / "manifest.json")
    if data_manifest.get("levels") != [256, 256, 256] or data_manifest.get("horizon") != 10:
        raise ValueError("expected the LightNav 3x256, horizon-10 RVQ bundle")
    if data_manifest.get("representation") != "se2_diff":
        raise ValueError("expected an SE(2)-increment RVQ bundle")
    files = ["manifest.json", *data_manifest["codebook_files"],
             data_manifest["jacobian_weights_file"], data_manifest["alpha_file"]]
    for name in files:
        # These are small decoder assets; distance matrices are not needed here.
        if Path(name).name != name:
            raise ValueError("RVQ asset names must be simple filenames")
        a, b = data_root / name, model_root / name
        if not b.is_file() or hashlib.sha256(a.read_bytes()).digest() != hashlib.sha256(b.read_bytes()).digest():
            raise ValueError(f"dataset/checkpoint RVQ asset mismatch: {name}")
    return data_manifest


class LeRobotFrameDataset(Dataset):
    """Map-style episode/frame sampling, preserving every independent loss mask.

    Dataset indices enumerate selected episodes' frames. episode_indices, when
    supplied, are original LeRobot episode_index values, not source episode IDs.
    Only selected historical RGBs are decoded. A bounded row-group cache may
    contain compressed bytes of other rows; those observations never enter a sample.
    """

    def __init__(
        self, root: str | Path, spec: LightNavTrainingSpec, *,
        episode_indices: Sequence[int] | None = None,
        cache_row_groups: int = 4, cache_frames: int = 64,
        fm_horizon: int = 30,
        stop_supervision: dict | None = None,
    ):
        if type(fm_horizon) is not int or not 1 <= fm_horizon <= 30:
            raise ValueError("fm_horizon must be an integer in [1,30] for this dataset")
        self.fm_horizon = fm_horizon
        self.root = Path(root).resolve()
        self.spec = spec
        self.stop_episodes = None
        if spec.training_mode == "fm_stop":
            validate_stop_supervision(stop_supervision)
            self.stop_episodes = stop_supervision["episodes"]
        self.info = _json(self.root / "meta/info.json")
        if self.info.get("codebase_version") != "v3.0" or self.info.get("fps") != 1:
            raise ValueError("expected this project's frame-index-timed LeRobot v3.0 export (fps=1)")
        features = self.info["features"]
        self.target_columns = {"fm_only": FM_TARGET_COLUMNS, "fm_stop": FM_STOP_TARGET_COLUMNS}.get(
            spec.training_mode, TARGET_COLUMNS
        )
        missing = set(self.target_columns + [RGB_KEY]) - features.keys()
        if missing:
            raise ValueError(f"dataset lacks Beta training fields: {sorted(missing)}")
        if features[RGB_KEY]["dtype"] != "image":
            raise ValueError("only embedded image-backed LeRobot exports are supported")
        if features["action.fm.clean_se2_diff"]["shape"] != [30, 3] or features["action.fm.valid_mask"]["shape"] != [30]:
            raise ValueError("expected per-row FM GT [30,3] and mask [30]")
        self.raw_rgb_shape = tuple(features[RGB_KEY]["shape"])
        if len(self.raw_rgb_shape) != 3 or self.raw_rgb_shape[-1] != 3:
            raise ValueError("RGB feature must be HWC with three channels")
        self.rvq_manifest = None if uses_frozen_continuous_backbone(spec.training_mode) else _check_codebooks(
            self.root / "assets/action_tokenizer", Path(spec.rvq_path)
        )
        if cache_row_groups < 1 or cache_frames < 0:
            raise ValueError("cache_row_groups must be positive; cache_frames must be nonnegative")
        self.cache_row_groups, self.cache_frames = cache_row_groups, cache_frames
        self._reset_caches()

        pq = _arrow()
        all_episodes = []
        for meta_path in sorted((self.root / "meta/episodes").rglob("*.parquet")):
            for row in pq.read_table(meta_path, columns=EPISODE_COLUMNS, use_threads=False).to_pylist():
                relative = self.info["data_path"].format(
                    chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
                )
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root) or not path.is_file():
                    raise ValueError(f"invalid/missing episode data path: {relative}")
                if len(row["tasks"]) != 1:
                    raise ValueError("each R2R episode must have exactly one navigation instruction")
                ep = _Episode(int(row["episode_index"]), int(row["dataset_from_index"]),
                              int(row["dataset_to_index"]), str(path), row["tasks"][0])
                if ep.length != row["length"] or ep.length <= 0:
                    raise ValueError("inconsistent episode lengths")
                all_episodes.append(ep)
        all_episodes.sort(key=lambda e: e.index)
        offset = 0
        for index, ep in enumerate(all_episodes):
            if ep.index != index or ep.start != offset:
                raise ValueError("episode indices/frame ranges must be complete and contiguous")
            offset = ep.stop
        if len(all_episodes) != self.info["total_episodes"] or offset != self.info["total_frames"]:
            raise ValueError("episode metadata does not cover info.json totals")
        selected = list(range(len(all_episodes))) if episode_indices is None else list(episode_indices)
        if not selected or len(set(selected)) != len(selected) or any(
            not isinstance(i, int) or i < 0 or i >= len(all_episodes) for i in selected
        ):
            raise ValueError("episode_indices must be unique existing LeRobot episode indices")
        self.episodes = [all_episodes[i] for i in selected]
        if self.stop_episodes is not None:
            for ep in self.episodes:
                if str(ep.index) not in self.stop_episodes or self.stop_episodes[str(ep.index)]["length"] != ep.length:
                    raise ValueError("STOP audit does not cover the selected episode/frame ranges")
        self._ends = np.cumsum([ep.length for ep in self.episodes]).tolist()

    def _reset_caches(self):
        self._files, self._tables, self._frames = OrderedDict(), OrderedDict(), OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_files=OrderedDict(), _tables=OrderedDict(), _frames=OrderedDict())
        return state

    def __len__(self):
        return self._ends[-1]

    def locate(self, index: int) -> tuple[_Episode, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        slot = bisect.bisect_right(self._ends, index)
        return self.episodes[slot], index - (self._ends[slot - 1] if slot else 0)

    def _file(self, path):
        if path in self._files:
            self._files.move_to_end(path)
            return self._files[path]
        file = _arrow().ParquetFile(path)
        index_column = file.schema.names.index("index")
        ranges = []
        for group in range(file.num_row_groups):
            rg = file.metadata.row_group(group)
            stats = rg.column(index_column).statistics
            if stats is None or not stats.has_min_max or stats.max - stats.min + 1 != rg.num_rows:
                raise ValueError("data row groups need contiguous global-index statistics")
            ranges.append((int(stats.min), int(stats.max) + 1))
        return _put_lru(self._files, path, (file, ranges), self.cache_row_groups)

    def _row(self, ep: _Episode, frame: int, *, rgb=False):
        if not 0 <= frame < ep.length:
            raise IndexError("frame access would cross an episode boundary")
        global_index = ep.start + frame
        file, ranges = self._file(ep.path)
        for group, (start, stop) in enumerate(ranges):
            if start <= global_index < stop:
                break
        else:
            raise ValueError("episode spans another data file; this R2R reader requires one file per episode")
        key = (ep.path, group, rgb)
        if key in self._tables:
            table = self._tables[key]
            self._tables.move_to_end(key)
        else:
            columns = ["index", "episode_index", "frame_index", RGB_KEY] if rgb else self.target_columns
            table = file.read_row_group(group, columns=columns, use_threads=False)
            _put_lru(self._tables, key, table, self.cache_row_groups)
        row = table.slice(global_index - start, 1).to_pylist()[0]
        if row["index"] != global_index or row["episode_index"] != ep.index or row["frame_index"] != frame:
            raise ValueError("Parquet row ordering or episode/frame identity mismatch")
        return row

    def _image(self, ep, frame):
        key = ep.start + frame
        if key in self._frames:
            self._frames.move_to_end(key)
            return self._frames[key]
        saved = self._row(ep, frame, rgb=True)[RGB_KEY]
        if not saved.get("bytes"):
            raise ValueError("RGB must contain embedded image bytes, not an external capture path")
        with Image.open(io.BytesIO(saved["bytes"])) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape != self.raw_rgb_shape:
            raise ValueError("decoded RGB shape disagrees with dataset metadata")
        tensor = rgb_frame_to_model_tensor(rgb, tuple(self.spec.params["video_size"]))
        return _put_lru(self._frames, key, tensor, self.cache_frames)

    def _targets(self, ep, t, row):
        clean = np.asarray(row["action.fm.clean_se2_diff"], dtype=np.float32)
        valid = np.asarray(row["action.fm.valid_mask"], dtype=bool)
        future = t + np.arange(1, 31)
        expected = future < ep.length
        if clean.shape != (30, 3) or valid.shape != (30,) or not np.array_equal(valid, expected):
            raise ValueError("FM horizon/mask disagrees with actual future episode frames")
        if not np.isfinite(clean[valid]).all() or not np.array_equal(
            row["action.fm.future_frame_ids"], np.where(expected, future, -1)
        ):
            raise ValueError("invalid FM target values or future frame IDs")
        if uses_frozen_continuous_backbone(self.spec.training_mode):
            return [], [], torch.from_numpy(clean[:self.fm_horizon].copy()), torch.from_numpy(valid[:self.fm_horizon].copy())
        tokens, supervised = [], []
        for channel in ("apos", "opos"):
            token_id = int(_scalar(row[f"action.{channel}.token_id"]))
            pointing_valid = bool(_scalar(row[f"action.{channel}.valid_mask"]))
            if not 0 <= token_id <= 1296 or (channel == "apos" and pointing_valid and token_id == 0):
                raise ValueError("pointing GT must follow spatial/zero-token rules, with no apos STOP")
            token = f"<{channel}_{token_id}>"
            if row[f"action.{channel}.token"] != token:
                raise ValueError("pointing token string disagrees with its channel ID")
            tokens.append(token)
            supervised.append(pointing_valid)

        codes = np.asarray(row["action.rvq.codes"], dtype=np.int64)
        rvq_valid = bool(_scalar(row["action.rvq.valid_mask"]))
        explicit_stop = bool(_scalar(row["action.explicit_stop"]))
        stop_valid = bool(_scalar(row["action.stop_loss_valid"]))
        if codes.shape != (3,) or stop_valid != explicit_stop:
            raise ValueError("RVQ code shape or explicit STOP metadata is invalid")
        if explicit_stop and (t != ep.length - 1 or not rvq_valid or valid.any()
                              or codes.tolist() != self.rvq_manifest["stop"]["codes"]):
            raise ValueError("explicit STOP must use the canonical terminal RVQ tuple")
        if rvq_valid != (bool(expected[:10].all()) or explicit_stop):
            raise ValueError("RVQ validity disagrees with a complete 10-step chunk / explicit STOP")
        if rvq_valid:
            if np.any(codes < 0) or np.any(codes >= 256):
                raise ValueError("valid RVQ codes must be in [0,255]")
            if not explicit_stop and codes[0] == self.rvq_manifest["stop"]["l0"]:
                raise ValueError("ordinary movement cannot use the reserved STOP L0 code")
            rvq_tokens = [f"<act_l{i}_{code}>" for i, code in enumerate(codes)]
            if row["action.rvq.tokens"] != "".join(rvq_tokens):
                raise ValueError("RVQ token string disagrees with its codes")
        else:
            if np.any(codes != -1) or row["action.rvq.tokens"] != "":
                raise ValueError("masked RVQ chunks must retain their original -1/empty storage")
            # Legal INPUT placeholders, not invented GT. All three CE labels are ignored.
            placeholder_l0 = 0 if self.rvq_manifest["stop"]["l0"] != 0 else 1
            rvq_tokens = [f"<act_l0_{placeholder_l0}>", "<act_l1_0>", "<act_l2_0>"]
        tokens.extend(rvq_tokens)
        supervised.extend([rvq_valid] * 3)
        return tokens, supervised, torch.from_numpy(clean[:self.fm_horizon].copy()), torch.from_numpy(valid[:self.fm_horizon].copy())

    def __getitem__(self, index):
        ep, t = self.locate(int(index))
        row = self._row(ep, t)
        tokens, supervised, clean, valid = self._targets(ep, t, row)
        stop_target, stop_valid = None, None
        if self.stop_episodes is not None:
            stop_target, stop_valid = stop_target_for_row(self.stop_episodes[str(ep.index)], t, row)
        layout = slowfast_segments(t, t + 1, self.spec.params["slowfast_tiers"])
        selected = sorted({f for segment in layout for f in segment["frame_ids"]})
        if not selected or min(selected) < 0 or max(selected) > t:
            raise ValueError("SlowFast layout contains a future/out-of-episode frame")
        frames = {f: self._image(ep, f) for f in selected}
        segments = [{
            "video": torch.stack([frames[f] for f in s["frame_ids"]]),
            "frame_indices": list(s["frame_ids"]), "total_frames": t + 1,
            "pool_spatial": s["pool_spatial"], "pool_mode": s["pool_mode"],
        } for s in layout]
        template = (CONTINUOUS_TRAJ_PROMPT_TEMPLATE if uses_frozen_continuous_backbone(self.spec.training_mode)
                    else to_rvq_prompt(UNIFIED_TRAJ_PROMPT_TEMPLATE))
        text = template.format(
            task=ep.instruction, videos=build_video_block(len(segments))
        )
        return {
            "sample": {
                "video_segments": segments,
                "conversations": [{"from": "human", "value": text},
                                  {"from": "gpt", "value": "".join(tokens)}],
                "video_fps": self.spec.params["video_fps"],
                "slowfast_abs_frame_indices": True, "_skip_normalize": True,
            },
            "tokens": tokens, "token_valid": supervised,
            "fm_clean": clean, "fm_valid": valid,
            "stop_target": stop_target, "stop_valid_mask": stop_valid,
            "info": {
                "dataset_index": int(index), "global_frame_index": ep.start + t,
                "episode_index": ep.index, "source_episode_id": row["source.episode_id"],
                "source_scene_id": row["source.scene_id"], "frame_index": t,
                "instruction": ep.instruction, "target_text": "".join(tokens),
                "token_valid": supervised, "explicit_stop": bool(_scalar(row["action.explicit_stop"]))
                if "action.explicit_stop" in row else None,
                "fm_horizon": self.fm_horizon, "fm_valid_steps": int(valid.sum()), "history": layout,
            },
        }


@dataclass
class TrainLerobotBatch(BetaTrainBatch):
    """A BetaTrainBatch with review metadata (never passed into the VLM)."""

    sample_info: list[dict[str, Any]]

    def pin_memory(self):
        self.model_inputs = {k: v.pin_memory() if isinstance(v, Tensor) else v
                             for k, v in self.model_inputs.items()}
        for name in ("labels", "answer_start", "fm_clean_se2_diff", "fm_valid_mask"):
            setattr(self, name, getattr(self, name).pin_memory())
        for name in ("stop_target", "stop_valid_mask"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, value.pin_memory())
        return self


class BetaLerobotCollator:
    """Lazy per-worker CPU processor; never holds a model or ModelBundle."""

    def __init__(self, spec: LightNavTrainingSpec):
        self.spec = spec
        self._processor = None

    def __getstate__(self):
        return {"spec": self.spec, "_processor": None}

    def _get_processor(self):
        if self._processor is None:
            from lightnav.inference.model import (
                _build_data_processor,
                _position_id_func_from_config,
            )
            from lightnav.processing import VLNQwen3VLProcessor

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            if self.spec.params.get("timestamp_relative"):
                os.environ.setdefault("VLN_TIMESTAMP_RELATIVE", "1")
            processor = VLNQwen3VLProcessor.from_pretrained(
                self.spec.processor_path, padding_side="left", local_files_only=True
            )
            # Disable silent truncation; a shortened answer would invalidate the FM boundary.
            self._processor = _build_data_processor(
                processor, {**self.spec.params, "max_seq_len": None},
                _position_id_func_from_config(self.spec.model_path),
            )
        return self._processor

    @staticmethod
    def _token_id(tokenizer, token):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != token:
            raise ValueError(f"checkpoint does not encode {token} as one exact vocabulary token")
        return int(ids[0])

    def _process(self, item):
        processor = self._get_processor()
        tokenizer = processor.tokenizer
        result = processor.process_sample(item["sample"])
        ids = result["input_ids"]
        if ids.ndim != 1 or len(ids) > self.spec.params["max_seq_len"]:
            raise ValueError("training sequence exceeds checkpoint max_seq_len; refusing to truncate")
        values = ids.tolist()
        header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        starts = [i + len(header) for i in range(len(values) - len(header) + 1)
                  if values[i:i + len(header)] == header]
        if not header or not starts:
            raise ValueError("cannot locate the assistant answer boundary in the chat template")
        start = starts[-1]
        if uses_frozen_continuous_backbone(self.spec.training_mode):
            # Empty assistant turn is a processor scaffold only. Drop its EOS/newline;
            # the VLM sees exactly the observation/instruction prefix and answer header.
            if item["tokens"] or item["token_valid"]:
                raise ValueError("frozen continuous training cannot contain action token targets")
            result["input_ids"] = ids[:start]
            return result, torch.full_like(ids[:start], -100), start
        target_ids = [self._token_id(tokenizer, token) for token in item["tokens"]]
        end_id = self._token_id(tokenizer, "<|im_end|>")
        if values[start:start + 6] != target_ids + [end_id]:
            raise ValueError("assistant answer was truncated or differs from the five-token contract")
        labels = torch.full_like(ids, -100)
        for offset, valid in enumerate(item["token_valid"]):
            if valid:
                labels[start + offset] = ids[start + offset]
        labels[start + 5] = end_id  # answer termination is supervised independently of RVQ validity
        return result, labels, start

    def __call__(self, items) -> TrainLerobotBatch:
        if not items:
            raise ValueError("cannot collate an empty batch")
        processed = [self._process(item) for item in items]
        length = max(len(result["input_ids"]) for result, _, _ in processed)
        pad_id = self._get_processor().tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("checkpoint tokenizer needs a pad_token_id")
        ids = torch.full((len(items), length), pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids, dtype=torch.bool)
        starts = []
        for i, (result, row_labels, start) in enumerate(processed):
            n = len(result["input_ids"])
            ids[i, -n:] = result["input_ids"]
            labels[i, -n:] = row_labels
            attention[i, -n:] = True
            starts.append(length - n + start)
        inputs = {"input_ids": ids, "attention_mask": attention}
        # Visual patches/grids follow sample then segment order. They are concatenated,
        # never padded or stacked along a fake image batch dimension. Native HF mRoPE
        # is reconstructed by BetaModel; processor position_ids are intentionally omitted.
        for key in ("pixel_values_videos", "video_grid_thw", "_original_video_grid_thw"):
            values = [result.get(key) for result, _, _ in processed]
            if any(value is not None for value in values):
                if any(value is None for value in values):
                    raise ValueError(f"inconsistent per-sample visual metadata: {key}")
                inputs[key] = torch.cat(values, dim=0)
        factors = [result.get("_post_vit_pool_factors") for result, _, _ in processed]
        if any(value is not None for value in factors):
            if any(value is None for value in factors):
                raise ValueError("inconsistent per-sample post-ViT pool factors")
            inputs["_post_vit_pool_factors"] = [int(f) for values in factors for f in values]
        stop_fields = {}
        if self.spec.training_mode == "fm_stop":
            if any(item.get("stop_target") is None or item.get("stop_valid_mask") is None for item in items):
                raise ValueError("fm_stop batch requires independent audited STOP targets")
            stop_fields = dict(stop_target=torch.tensor([item["stop_target"] for item in items], dtype=torch.float32),
                               stop_valid_mask=torch.tensor([item["stop_valid_mask"] for item in items], dtype=torch.bool))
        return TrainLerobotBatch(
            model_inputs=inputs, labels=labels, answer_start=torch.tensor(starts),
            fm_clean_se2_diff=torch.stack([item["fm_clean"] for item in items]),
            fm_valid_mask=torch.stack([item["fm_valid"] for item in items]),
            sample_info=[item["info"] for item in items],
            **stop_fields,
        )


class Train_Lerobot_DataLoader(DataLoader):
    """Training-ready DataLoader. Iteration yields TrainLerobotBatch/BetaTrainBatch.

    Use set_epoch(epoch) before each epoch; pass rank/world_size explicitly for
    DDP. This object does not create a process group or choose a validation split.
    """

    def __init__(
        self, dataset_root: str | Path, model_path: str | Path, *, batch_size: int = 2,
        num_workers: int = 0, shuffle: bool = True, seed: int = 0,
        episode_indices: Sequence[int] | None = None, drop_last: bool = False,
        pin_memory: bool = False, prefetch_factor: int = 1,
        cache_row_groups: int = 4, cache_frames: int = 64,
        rank: int = 0, world_size: int = 1,
        training_mode: str = "ce_fm",
        fm_horizon: int = 30,
        stop_supervision: dict | None = None,
    ):
        if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
            raise ValueError("invalid batch_size, num_workers or prefetch_factor")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0,world_size)")
        spec = LightNavTrainingSpec.from_checkpoint(model_path, training_mode=training_mode)
        dataset = LeRobotFrameDataset(dataset_root, spec, episode_indices=episode_indices,
                                      cache_row_groups=cache_row_groups, cache_frames=cache_frames,
                                      fm_horizon=fm_horizon, stop_supervision=stop_supervision)
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        generator = torch.Generator().manual_seed(seed)
        worker_options = {}
        if num_workers:
            worker_options = dict(multiprocessing_context="spawn", persistent_workers=True,
                                  prefetch_factor=prefetch_factor)
        super().__init__(
            dataset, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle if sampler is None else False, sampler=sampler,
            collate_fn=BetaLerobotCollator(spec), drop_last=drop_last,
            pin_memory=pin_memory, generator=generator, **worker_options,
        )
        self.spec, self.seed = spec, seed

    def set_epoch(self, epoch: int):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.generator.manual_seed(self.seed + epoch)
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def describe_batch(batch: TrainLerobotBatch):
    return {
        "samples": batch.sample_info,
        "model_inputs": {k: list(v.shape) if isinstance(v, Tensor) else v
                         for k, v in batch.model_inputs.items()},
        "labels_shape": list(batch.labels.shape), "answer_start": batch.answer_start.tolist(),
        "ce_tokens_per_sample": (batch.labels != -100).sum(dim=1).tolist(),
        "fm_shape": list(batch.fm_clean_se2_diff.shape),
        "fm_valid_steps": batch.fm_valid_mask.sum(dim=1).tolist(),
        "fm_first_three": batch.fm_clean_se2_diff[:, :3].tolist(),
        "stop_target": batch.stop_target.tolist() if batch.stop_target is not None else None,
        "stop_valid_mask": batch.stop_valid_mask.tolist() if batch.stop_valid_mask is not None else None,
    }


def _tiny_model_check(batch, spec):
    """Real HF Qwen3-VL with tiny random weights; an interface check, not model training."""
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    from lightnav.inference.beta_model import BetaModel, BetaModelConfig
    from lightnav.processing import enable_post_vit_pool, get_post_vit_target

    config = AutoConfig.from_pretrained(spec.model_path, local_files_only=True)
    for k, v in dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                      "mrope_section": [1, 1, 2]}).items():
        setattr(config.text_config, k, v)
    for k, v in dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                     out_hidden_size=32, deepstack_visual_indexes=[0, 1]).items():
        setattr(config.vision_config, k, v)
    torch.manual_seed(0)
    vlm = Qwen3VLForConditionalGeneration(config)
    if spec.params["pool_enable"] and spec.params["pool_stage"] == "post_vit":
        enable_post_vit_pool(vlm, spatial_factor=spec.params["pool_spatial"])
        vlm._post_vit_target = get_post_vit_target(vlm)
    model = BetaModel(vlm, BetaModelConfig(action_scale=(1., 1., 1.), hidden_size=32,
                                         horizon=batch.fm_clean_se2_diff.shape[1], training_mode=spec.training_mode,
                                         stop_head_type="last_prefix_mlp_v1" if spec.training_mode == "fm_stop" else "none",
                                         num_heads=4, num_layers=2, intermediate_size=64))
    out = model(batch)
    if not torch.isfinite(out.loss):
        raise AssertionError("non-finite model loss")
    out.fm_loss.backward()
    isolated = all(p.grad is None for p in vlm.parameters())
    projection_trains = model.expert.context_proj.weight.grad is not None
    if out.fm_step_count and (not isolated or not projection_trains):
        raise AssertionError("FM gradient isolation check failed")
    return dict(ce_loss=float(out.ce_loss.detach()), fm_loss=float(out.fm_loss.detach()),
                fm_backbone_isolated=isolated, fm_projection_receives_gradient=projection_trains,
                pretrained_weights_loaded=False, device="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fm-horizon", type=int, default=30, help="consecutive future actions, 1–30")
    parser.add_argument("--training-mode", choices=["ce_fm", "fm_only", "fm_stop"], default="ce_fm")
    parser.add_argument("--stop-manifest", type=Path, help="audited training manifest required for fm_stop")
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--sample-indices", type=int, nargs="+",
                        help="Inspect these dataset-local frame indices instead of the first batch")
    parser.add_argument("--tiny-model-check", action="store_true")
    parser.add_argument("--report", type=Path, help="New small JSON report; never overwrites a report")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("report already exists")
    if args.sample_indices and args.num_workers:
        parser.error("--sample-indices inspection uses the main process; use --num-workers 0")
    torch.set_num_threads(1)
    stop_supervision = None
    if args.training_mode == "fm_stop":
        from lightnav.training.stop_labels import verify_stop_supervision

        if args.stop_manifest is None:
            parser.error("fm_stop requires --stop-manifest from prepare_data --with-stop-labels")
        manifest = _json(args.stop_manifest)
        if Path(manifest["dataset_root"]).resolve() != args.dataset.resolve():
            parser.error("STOP manifest points at a different dataset")
        stop_supervision = manifest.get("stop_supervision")
        verify_stop_supervision(args.dataset, stop_supervision)
    loader = Train_Lerobot_DataLoader(args.dataset, args.model_path, batch_size=args.batch_size,
                                     num_workers=args.num_workers, shuffle=False,
                                     fm_horizon=args.fm_horizon, training_mode=args.training_mode,
                                     episode_indices=args.episode_indices, stop_supervision=stop_supervision)
    if args.sample_indices:
        batch = loader.collate_fn([loader.dataset[i] for i in args.sample_indices])
    else:
        batch = next(iter(loader))
    result = dict(dataset=str(args.dataset.resolve()), model_path=str(args.model_path.resolve()),
                  selected_episodes=len(loader.dataset.episodes), selected_frames=len(loader.dataset),
                  batch=describe_batch(batch))
    if args.tiny_model_check:
        result["tiny_model_check"] = _tiny_model_check(batch, loader.spec)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
