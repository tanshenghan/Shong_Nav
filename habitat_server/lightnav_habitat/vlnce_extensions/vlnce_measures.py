# Adapted from VLN-CE (https://github.com/jacobkrantz/VLN-CE), MIT License,
# Copyright (c) 2020 Jacob Krantz. See THIRD_PARTY_NOTICES.md.
"""VLN-CE evaluation measures for habitat-lab: PathLength, OracleSuccess, StepsTaken, NDTW.

habitat-lab 0.3.x ships none of these for navigation tasks; the environment wrappers inject
them into ``habitat.task.measurements`` at runtime.
"""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal
from numpy import ndarray

if TYPE_CHECKING:
    from omegaconf import DictConfig

    Config = DictConfig
else:
    Config = Any


def _get_config_attr(config: Any, attr_name: str, default: Any = None) -> Any:
    """Read a config attribute in lower_case (OmegaConf) or UPPER_CASE (yacs) form."""
    if hasattr(config, attr_name.lower()):
        return getattr(config, attr_name.lower())
    if hasattr(config, attr_name.upper()):
        return getattr(config, attr_name.upper())
    return default


# DTW backends are optional dependencies; NDTW needs at least one of them.
try:
    from dtw import dtw

    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    dtw = None

try:
    from fastdtw import fastdtw

    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False
    fastdtw = None


def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class PathLength(Measure):
    """Path Length (PL): sum of Euclidean distances between consecutive agent positions."""

    cls_uuid: str = "path_length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(current_position, self._previous_position)
        self._previous_position = current_position


@registry.register_measure
class OracleSuccess(Measure):
    """Oracle Success (OS): 1.0 once the agent has been within ``success_distance`` of the goal."""

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Config, **kwargs: Any):
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        success_distance = _get_config_attr(self._config, "success_distance", 3.0)
        self._metric = float(self._metric or d < success_distance)


@registry.register_measure
class StepsTaken(Measure):
    """Number of actions taken (STOP included)."""

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0


@registry.register_measure
class NDTW(Measure):
    """Normalized Dynamic Time Warping between the agent path and the reference path.

    ref: https://arxiv.org/abs/1907.05446. Requires ``fastdtw`` (preferred, ``fdtw: true``)
    or ``dtw-python``. Ground-truth paths come from ``gt_path`` (formatted with ``split``).
    """

    cls_uuid: str = "ndtw"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config

        use_fastdtw = _get_config_attr(config, "fdtw", True)
        if use_fastdtw and FASTDTW_AVAILABLE:
            self.dtw_func = fastdtw
        elif DTW_AVAILABLE:
            self.dtw_func = dtw
        else:
            raise ImportError(
                "NDTW measure requires either 'fastdtw' or 'dtw-python' package. "
                "Install with: pip install fastdtw dtw-python"
            )

        gt_path = _get_config_attr(config, "gt_path")
        split = _get_config_attr(config, "split")
        if gt_path:
            with gzip.open(gt_path.format(split=split), "rt") as f:
                self.gt_json = json.load(f)
        else:
            self.gt_json = {}

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations = []
        self.gt_locations = self.gt_json[episode.episode_id]["locations"]
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            # Rotation-only steps do not add a point to the agent path.
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(self.locations, self.gt_locations, dist=euclidean_distance)[0]

        success_distance = _get_config_attr(self._config, "success_distance", 3.0)
        nDTW = np.exp(-dtw_distance / (len(self.gt_locations) * success_distance))
        self._metric = nDTW
