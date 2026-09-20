"""Checkpoint-side ``eval_config.json``: loading and inference parameter resolution.

A checkpoint directory (or one of its parents) may carry an ``eval_config.json`` that
records the data-processing parameters the model was trained with, so inference picks
them up automatically instead of relying on CLI flags.

Schema (version 1)::

    {
        "version": 1,
        "common": {
            "max_seq_len": 8192,
            "video_size": [224, 320],
            "pool_enable": true,
            "pool_spatial": 2,
            "pool_mode": "avg",
            "pool_stage": "pre_vit"
        },
        "tasks": {
            "vlnce": {
                "num_history_frames": 64,
                "predict_horizon": 10,
                "video_fps": 4,
                "traj_vocab_path": "/path/to/traj_vocab",
                "traj_vocab_K": 256,
                "action_tokenizer": {"method": "flat" | "rvq", "bundle_path": "..."},
                "slowfast_tiers": [ {tier dicts, see lightnav.slowfast} ],
                "prompt_style": "unified_traj",
                "timestamp_relative": false
            },
            "trackvla": { ... }
        }
    }

``common`` holds model-wide settings; ``tasks`` is keyed by benchmark family
(``"vlnce"`` for navigation, ``"trackvla"`` for person tracking) and overrides
``common`` key by key. Only the keys listed in :data:`INFERENCE_FALLBACK_DEFAULTS` reach
the model bundle; everything else in the file is ignored.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

EVAL_CONFIG_FILENAME = "eval_config.json"

DEFAULT_VIDEO_SIZE: tuple[int, int] = (224, 320)

# Inference resolution fallback when caller arg and eval_config are absent. This dict
# also acts as the whitelist of eval_config keys that reach the bundle.
INFERENCE_FALLBACK_DEFAULTS: dict[str, Any] = {
    "num_history_frames": 16,
    "predict_horizon": 1,
    "video_fps": 4,
    "video_size": DEFAULT_VIDEO_SIZE,
    "max_seq_len": 8192,
    "pool_enable": None,  # None -> derived from (pool_spatial > 1) for back-compat
    "pool_spatial": 1,
    "pool_mode": "avg",
    "pool_stage": "pre_vit",
    "slowfast_tiers": None,  # None -> legacy dense window (per-tier list when enabled)
    "prompt_style": None,  # None -> per-task templates; "unified_traj" for SlowFast
    "timestamp_relative": False,  # False -> absolute timestamps (seconds since episode start)
    # RVQ action tokenizer block {method, bundle_path}. Must be in this whitelist or
    # resolve_inference_params drops it -> action_method silently falls back to "flat" and
    # the rvq prompt swap (to_rvq_prompt) never fires. None -> flat (back-compat).
    "action_tokenizer": None,
}


def normalize_value(key: str, value: Any) -> Any:
    """Normalize schema value shape for downstream use (``video_size`` list -> tuple)."""
    if key == "video_size" and isinstance(value, list):
        return tuple(value)
    return value


def resolve_inference_params(
    caller_kwargs: Mapping[str, Any],
    cfg_params: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve inference params with priority:

    1) caller kwargs (value is not None)
    2) eval config params
    3) INFERENCE_FALLBACK_DEFAULTS

    Returns ``(resolved, from_config)`` where ``from_config`` lists the ``key=value``
    pairs that came from the eval config.
    """
    resolved: dict[str, Any] = {}
    from_config: list[str] = []
    fell_back: list[str] = []

    for key, fallback in INFERENCE_FALLBACK_DEFAULTS.items():
        caller_val = caller_kwargs.get(key)
        if caller_val is not None:
            resolved[key] = normalize_value(key, caller_val)
            continue

        if key in cfg_params:
            cfg_val = normalize_value(key, cfg_params[key])
            resolved[key] = cfg_val
            from_config.append(f"{key}={cfg_val}")
            continue

        resolved[key] = fallback
        fell_back.append(f"{key}={fallback}")

    # Back-compat: old eval_config.json has no explicit pool_enable; derive it
    # from pool_spatial so old checkpoints keep the correct behavior.
    if resolved.get("pool_enable") is None:
        resolved["pool_enable"] = int(resolved.get("pool_spatial", 1)) > 1

    # A history-window mismatch between training and inference silently destroys
    # accuracy, so flag the fallback loudly.
    if "num_history_frames" in [f.split("=")[0] for f in fell_back]:
        warnings.warn(
            "[eval] num_history_frames not found in eval_config.json or caller args -- "
            f"falling back to default {INFERENCE_FALLBACK_DEFAULTS['num_history_frames']}. "
            "If training used a different window size this will silently destroy accuracy. "
            "Check that the checkpoint's eval_config.json has tasks.<task>.num_history_frames set.",
            stacklevel=2,
        )

    return resolved, from_config


def find_eval_config(model_path: str) -> Optional[Path]:
    """Locate ``eval_config.json`` in ``model_path`` or up to four parent directories.

    This handles both ``model_path = <run root>`` (the file sits right there) and
    ``model_path = <run root>/checkpoints/global_step_N/hf_ckpt``.
    """
    p = Path(model_path).expanduser().resolve()
    for _ in range(5):
        candidate = p / EVAL_CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return None


def load_eval_config(model_path: str) -> Optional[Dict[str, Any]]:
    """Parsed ``eval_config.json`` found by :func:`find_eval_config`, or None."""
    candidate = find_eval_config(model_path)
    if candidate is None:
        return None
    with open(candidate) as f:
        return json.load(f)


def resolve_asset_path(model_path: str, value: str) -> Path:
    """Resolve a path stored in ``eval_config.json`` (e.g. ``action_tokenizer.bundle_path``).

    Absolute paths are returned as they are. Relative paths are taken relative to the
    directory holding ``eval_config.json`` (falling back to ``model_path`` itself), so a
    checkpoint directory can ship its action decoder alongside the weights and reference
    it as ``"action_tokenizer/vlnce"``.
    """
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    cfg = find_eval_config(model_path)
    base = cfg.parent if cfg is not None else Path(model_path).expanduser().resolve()
    candidate = base / p
    if candidate.exists():
        return candidate
    alt = Path(model_path).expanduser().resolve() / p
    return alt if alt.exists() else candidate


def get_task_params(
    config: Dict[str, Any],
    task_type: str,
) -> Dict[str, Any]:
    """
    Merge common + task-specific parameters into a flat dict.

    ``config["common"]`` is copied first, then updated with ``config["tasks"][task_type]``
    (missing sections are treated as empty).
    """
    result = dict(config.get("common", {}))
    task = config.get("tasks", {}).get(task_type, {})
    result.update(task)
    return result
