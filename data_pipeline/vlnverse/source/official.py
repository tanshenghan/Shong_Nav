"""Read official stepwise poses without inventing motion from discrete actions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _finite_array(value, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass
class SourceEpisode:
    """Ordered world-frame, z-up robot poses; source time units remain unverified."""

    episode_id: str
    positions_xyz: np.ndarray
    yaw_rad: np.ndarray
    instruction: str = ""
    provenance: dict = field(default_factory=dict)
    source_timestamps: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        self.positions_xyz = _finite_array(self.positions_xyz, "positions_xyz")
        if self.positions_xyz.ndim != 2 or self.positions_xyz.shape[1] != 3:
            raise ValueError("positions_xyz must have shape (N, 3)")
        count = len(self.positions_xyz)
        if not count:
            raise ValueError("source episode must contain at least one pose")
        self.yaw_rad = _finite_array(self.yaw_rad, "yaw_rad")
        if self.yaw_rad.shape != (count,):
            raise ValueError("yaw_rad must have shape (N,) matching positions_xyz")
        if self.source_timestamps is not None:
            self.source_timestamps = _finite_array(self.source_timestamps, "source_timestamps")
            if self.source_timestamps.shape != (count,):
                raise ValueError("source_timestamps must have shape (N,) matching positions_xyz")
        if not isinstance(self.instruction, str) or not isinstance(self.provenance, dict):
            raise ValueError("instruction must be a string and provenance must be a dict")


def _column(values: list, name: str, width: int | None = None) -> np.ndarray:
    decoded = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name}[{index}] is not valid JSON") from exc
        decoded.append(value)
    result = _finite_array(decoded, name)
    expected = (len(values),) if width is None else (len(values), width)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_metadata(path: Path, episode_index: int | None, episode_ids: set[str]) -> dict:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metadata JSON at {path}:{number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"metadata row at {path}:{number} must be an object")
        rows.append(row)
    # Official per-trajectory metadata uses an episode_id such as "0_0" rather
    # than the per-Parquet episode_index. Match that identity to the source path;
    # a single unrelated metadata row is not sufficient evidence of association.
    if len(rows) == 1 and "episode_index" not in rows[0]:
        if rows[0].get("episode_id") in episode_ids:
            return rows[0]
        raise ValueError("metadata episode_id does not match the source episode identity")
    matching = [row for row in rows if row.get("episode_index") == episode_index]
    if len(matching) != 1:
        raise ValueError(f"metadata must contain exactly one row for episode_index={episode_index}")
    return matching[0]


def load_official_episode(
    parquet_path: str | Path,
    *,
    episode_id: str | None = None,
    metadata_path: str | Path | None = None,
) -> SourceEpisode:
    """Load one official Parquet episode, optionally with its episodes.jsonl metadata.

    Requires positions, yaw, and an explicit ordered frame/step index. Neither action
    labels nor source timestamps are used to infer missing poses or elapsed seconds.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading official Parquet requires pyarrow in the pipeline environment") from exc

    path = Path(parquet_path).expanduser().resolve(strict=True)
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    required = {"observation.robot_position", "observation.robot_yaw"}
    missing = required - names
    if missing:
        raise ValueError(f"missing official robot pose columns: {sorted(missing)}")
    order_names = [name for name in ("frame_index", "observation.step") if name in names]
    if not order_names:
        raise ValueError("source requires frame_index or observation.step to verify pose order")
    optional = {"observation.robot_orientation", "timestamp", "episode_index"}
    columns = parquet.read(columns=sorted(required | set(order_names) | (optional & names)))
    data = columns.to_pydict()
    if not columns.num_rows:
        raise ValueError("source episode must contain at least one pose")
    for name in order_names:
        order = _column(data[name], name)
        if (order < 0).any() or (order != np.floor(order)).any() or (np.diff(order) != 1).any():
            raise ValueError(f"{name} must be unique contiguous integers in original file order")

    positions = _column(data["observation.robot_position"], "observation.robot_position", 3)
    yaw = _column(data["observation.robot_yaw"], "observation.robot_yaw")
    if "observation.robot_orientation" in data:
        quaternions = _column(data["observation.robot_orientation"], "observation.robot_orientation", 4)
        norms = np.linalg.norm(quaternions, axis=1)
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-4):
            raise ValueError("observation.robot_orientation must contain unit wxyz quaternions")
        w, x, y, z = (quaternions / norms[:, None]).T
        quaternion_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        difference = np.arctan2(np.sin(quaternion_yaw - yaw), np.cos(quaternion_yaw - yaw))
        if (np.abs(difference) > 1e-4).any():
            raise ValueError("robot_yaw disagrees with z-up wxyz robot_orientation")

    match = re.fullmatch(r"episode_(\d+)", path.stem)
    episode_index = int(match.group(1)) if match else None
    if "episode_index" in data:
        indices = _column(data["episode_index"], "episode_index")
        if (indices < 0).any() or (indices != np.floor(indices)).any() or (indices != indices[0]).any():
            raise ValueError("Parquet must contain exactly one nonnegative integer episode_index")
        if episode_index is not None and episode_index != int(indices[0]):
            raise ValueError("episode_index disagrees with the Parquet filename")
        episode_index = int(indices[0])

    root = path.parent.parent.parent if path.parent.parent.name == "data" else None
    if episode_id is None:
        episode_id = f"{root.parent.name}/{root.name}" if root is not None else path.stem
    metadata = Path(metadata_path).expanduser().resolve(strict=True) if metadata_path else None
    if metadata is None and root is not None:
        candidate = root / "meta" / "episodes.jsonl"
        metadata = candidate.resolve() if candidate.is_file() else None
    provenance = {
        "source_format": "vlnverse_official_parquet",
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "source_order_columns": order_names,
        "source_coordinate_system": "world_xyz_z_up",
        "source_timestamp_semantics": "unverified_source_values_not_seconds",
        "finish_status": None,
    }
    instruction = ""
    if metadata is not None:
        episode_ids = {episode_id, path.stem}
        if root is not None:
            episode_ids.update((root.name, f"{root.parent.name}/{root.name}"))
        row = _read_metadata(metadata, episode_index, episode_ids)
        provenance.update({
            "metadata_path": str(metadata),
            "metadata_sha256": _sha256(metadata),
            "source_metadata": row,
            "finish_status": row.get("finish_status"),
        })
        instruction = row.get("instruction_text", row.get("instruction", ""))
        if not instruction and isinstance(row.get("tasks"), list) and len(row["tasks"]) == 1:
            instruction = row["tasks"][0]
        if not isinstance(instruction, str):
            raise ValueError("metadata instruction (or sole task) must be a string")
    timestamps = _column(data["timestamp"], "timestamp") if "timestamp" in data else None
    return SourceEpisode(episode_id, positions, yaw, instruction, provenance, timestamps)
