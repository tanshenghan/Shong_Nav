"""ObjectNav (HM3D v1 / MP3D v1 / HM3D-OVON) Habitat environment wrapper.

Inherits the velocity-control, early-stop and termination logic from ``VLNCEEnv``.
Differences: the instruction is synthesized from ``episode.object_category``
("Find the chair."), success is measured to the nearest goal viewpoint
(``success_distance`` 0.1 m for HM3D v1, 0.25 m for OVON), there is no NDTW, and the
``velocity_control`` action is injected programmatically (the ObjectNav task config
does not declare one).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .vlnce import (
    VLNCEEnv,
    _apply_image_size,
    _disable_episode_shuffle,
    _inject_nav_measurements,
)

logger = logging.getLogger(__name__)


class ObjectNavEnv(VLNCEEnv):
    """Habitat ObjectNav task wrapper (see module docstring).

    Metrics in ``info``: success, spl, soft_spl, distance_to_goal, path_length,
    oracle_success, steps_taken, plus ``object_category`` and ``goal_positions``.
    """

    # Human-readable instruction templates per object category.
    _CATEGORY_TEMPLATES: Dict[str, str] = {
        "chair": "Find the chair.",
        "bed": "Find the bed.",
        "sofa": "Find the sofa.",
        "plant": "Find the potted plant.",
        "toilet": "Find the toilet.",
        "tv_monitor": "Find the TV monitor.",
    }

    def __init__(
        self,
        config_path: str,
        data_path: Optional[str] = None,
        scenes_dir: Optional[str] = None,
        split: Optional[str] = None,
        gpu_id: int = 0,
        image_size: Optional[Tuple[int, int]] = None,
        max_steps: int = 500,
        success_distance: float = 0.1,  # HM3D ObjectNav standard: 0.1 m to a viewpoint
        split_id: Optional[int] = None,
        split_num: Optional[int] = None,
        early_stop_rotation: int = 0,
        early_stop_steps: int = 0,
    ):
        # ``split=None`` keeps the yaml split (``val`` for HM3D / MP3D v1, ``val_unseen`` for OVON).
        self._object_category: str = ""
        super().__init__(
            config_path=config_path,
            data_path=data_path,
            scenes_dir=scenes_dir,
            split=split,
            gpu_id=gpu_id,
            image_size=image_size,
            max_steps=max_steps,
            success_distance=success_distance,
            split_id=split_id,
            split_num=split_num,
            early_stop_rotation=early_stop_rotation,
            early_stop_steps=early_stop_steps,
        )

    # -- config assembly --------------------------------------------------------

    def _create_habitat_env(self) -> Any:
        try:
            import habitat
            from habitat.config.read_write import read_write
        except ImportError as e:
            raise ImportError(f"habitat-lab not installed: {e}")
        from habitat.config.default_structured_configs import VelocityControlActionConfig
        from omegaconf import OmegaConf

        # Tolerant "ObjectNav-v1" loader (OVON episode fields / empty category maps) and the
        # PathLength/OracleSuccess/StepsTaken measures; must precede habitat.Env.
        from . import objectnav_extensions, vlnce_extensions  # noqa: F401

        logger.info(f"Loading ObjectNav config from: {self.config_path}")
        config = habitat.get_config(self.config_path)

        with read_write(config):
            self._apply_dataset_overrides(config)
            _disable_episode_shuffle(config)
            _apply_image_size(config, self.image_size)

            # Keep the ObjectNav built-ins (distance_to_goal to VIEW_POINTS, success, spl,
            # soft_spl); add path_length / oracle_success / steps_taken. No NDTW.
            _inject_nav_measurements(config, self.success_distance)

            # Inject velocity_control as a structured config so OmegaConf validates it.
            actions = config.habitat.task.actions
            OmegaConf.set_struct(actions, False)
            actions.velocity_control = OmegaConf.structured(
                VelocityControlActionConfig(
                    lin_vel_range=[0.0, 0.25],  # m/s   -> 0.25 m per step at dt=1 s
                    ang_vel_range=[-30.0, 30.0],  # deg/s -> 30 deg per step at dt=1 s
                    time_step=1.0,
                    min_abs_lin_speed=0.025,
                    min_abs_ang_speed=1.0,
                )
            )
            OmegaConf.set_struct(actions, True)

            self._cache_velocity_control_config(config)

        self.split = str(config.habitat.dataset.split)
        logger.info(
            f"ObjectNav config loaded - task: {config.habitat.task.type}, "
            f"dataset: {config.habitat.dataset.type}, split: {self.split}"
        )

        env = habitat.Env(config=config)

        try:
            total_eps = len(env._dataset.episodes)
            categories: Dict[str, int] = {}
            for ep in env._dataset.episodes:
                cat = getattr(ep, "object_category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            cat_str = ", ".join(f"{k}={v}" for k, v in sorted(categories.items()))
            logger.info(f"Dataset loaded: {total_eps} episodes. Categories: {cat_str}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not summarize dataset: {e}")

        self._apply_episode_split(env)
        return env

    # -- observations -----------------------------------------------------------

    def _compute_progress(self) -> float:
        """Progress against the viewpoint distance reported by the DistanceToGoal measure."""
        progress = 0.0
        env = self._habitat_env
        if env is None:
            return progress
        try:
            dtg = float(env.get_metrics().get("distance_to_goal", float("inf")))
            start = self._start_geodesic_distance
            if np.isfinite(dtg) and np.isfinite(start) and start > 0:
                progress = float(np.clip((start - dtg) / start, 0.0, 1.0))
        except Exception:
            pass
        return progress

    def _extract_task_obs(self, habitat_obs: Dict) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}

        # Instruction from the current episode's object_category (valid after reset()).
        category = ""
        if self._habitat_env is not None:
            episode = getattr(self._habitat_env, "current_episode", None)
            if episode is not None:
                category = getattr(episode, "object_category", "")

        if category:
            instruction_text = self._CATEGORY_TEMPLATES.get(
                category,
                f"Find the {category.replace('_', ' ')}.",
            )
            self._object_category = category
        else:
            instruction_text = self._current_instruction  # keep the previous one on error

        obs["instruction"] = {"text": instruction_text}
        self._current_instruction = instruction_text
        obs["goal_distance"] = np.array([float("inf")], dtype=np.float32)
        obs["progress"] = np.array([self._compute_progress()], dtype=np.float32)
        return obs

    # -- reset / info -------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset and re-baseline the progress start distance to the nearest viewpoint.

        ``VLNCEEnv.reset`` measures the start distance to ``goals[0].position`` (object
        centre). For ObjectNav the reference is the distance to the nearest viewpoint, which
        is what Habitat's DistanceToGoal reports in ``info["distance_to_goal"]``.
        """
        obs, info = super().reset(seed=seed, options=options)

        if "distance_to_goal" in info:
            dtg = float(info["distance_to_goal"])
            if np.isfinite(dtg) and dtg > 0.0:
                self._start_geodesic_distance = dtg

        if self._object_category:
            info["object_category"] = self._object_category

        return obs, info

    def _compute_info(self, habitat_obs: Dict) -> Dict[str, Any]:
        info = super()._compute_info(habitat_obs)
        if self._object_category:
            info["object_category"] = self._object_category
        # ObjectNav episodes target every instance of the category: one goal per instance.
        if hasattr(self._habitat_env, "current_episode"):
            episode = self._habitat_env.current_episode
            if getattr(episode, "goals", None):
                info["goal_positions"] = [[float(x) for x in g.position] for g in episode.goals]
        return info
