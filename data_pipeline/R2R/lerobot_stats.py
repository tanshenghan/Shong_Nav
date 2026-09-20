"""Mask-aware statistics adapter for the pinned LeRobot 0.4.4 writer.

Only statistics are adapted. The official writer still owns schemas, Parquet,
image embedding, task/episode indexing and finalization. Stored GT is untouched.
"""

from contextlib import contextmanager

import numpy as np


def numeric_stats(key, values, episode):
    values = np.asarray(values)
    mask = None
    if key == "observation.depth_m":
        mask = np.asarray(episode["observation.depth_valid"], dtype=bool)
    elif key == "observation.depth_valid":
        values = values.reshape(-1, 1)
    elif key.startswith("action.fm.") and not key.endswith("valid_mask"):
        mask = np.asarray(episode["action.fm.valid_mask"], dtype=bool)
    elif key.startswith("action.rvq.") and not key.endswith("valid_mask"):
        mask = np.asarray(episode["action.rvq.valid_mask"], dtype=bool).reshape(-1)
    elif key.startswith(("action.apos.", "action.opos.")):
        channel, field = key.rsplit(".", 1)
        if field in ("pixel", "world", "pixel_rc", "grid_rc", "grid_id"):
            mask = np.asarray(episode[channel + ".token_id"]).reshape(-1) > 0
    if mask is not None:
        values = values[mask]
    if values.ndim == 1:
        values = values[:, None]
    # Use a common [forward,left,yaw] scale across valid horizon steps.
    if key == "action.rvq.reconstruction":
        values = values.reshape(-1, 3)
    values = values.astype(np.float64)
    values = values[np.isfinite(values).reshape(len(values), -1).all(1)] if len(values) else values
    shape = values.shape[1:]
    if not len(values):
        return {"min": np.zeros(shape), "max": np.zeros(shape), "mean": np.zeros(shape),
                "std": np.ones(shape), "count": np.array([0])}
    return {"min": values.min(0), "max": values.max(0), "mean": values.mean(0),
            "std": values.std(0), "count": np.array([len(values)])}


def aggregate_masked_stats(groups):
    """Combine episode/shard statistics while excluding empty supervision groups."""
    from lerobot.datasets.compute_stats import aggregate_stats

    result = {}
    for key in sorted(set().union(*(g.keys() for g in groups))):
        present = [g[key] for g in groups if key in g]
        nonempty = [g for g in present if np.asarray(g["count"]).sum() > 0]
        result[key] = aggregate_stats([{key: g} for g in nonempty])[key] if nonempty else present[0]
    return result


@contextmanager
def masked_writer_statistics(module):
    """Scope the adapter to one sequential export; never edit installed packages."""
    from lerobot.datasets.compute_stats import compute_episode_stats

    def compute(episode, features, **kwargs):
        images = {k: v for k, v in episode.items() if features[k]["dtype"] in ("image", "video")}
        result = compute_episode_stats(images, {k: features[k] for k in images}, **kwargs) if images else {}
        for key, values in episode.items():
            if features[key]["dtype"] not in ("image", "video", "string"):
                result[key] = numeric_stats(key, values, episode)
        return result

    old_compute, old_aggregate = module.compute_episode_stats, module.aggregate_stats
    module.compute_episode_stats, module.aggregate_stats = compute, aggregate_masked_stats
    try:
        yield
    finally:
        module.compute_episode_stats, module.aggregate_stats = old_compute, old_aggregate
