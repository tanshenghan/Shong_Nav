"""VLN-CE (R2R / RxR) Habitat environment wrapper with velocity control and early-stop logic."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseHabitatEnv
from .constants import ACTION_NAMES

logger = logging.getLogger(__name__)


def _action_label(action: Any) -> str:
    """Human-readable label for a discrete-int or velocity-dict action."""
    if isinstance(action, (int, np.integer)):
        return ACTION_NAMES.get(int(action), str(int(action)))
    if isinstance(action, dict):
        args = action.get("action_args", {}) or {}
        lin = args.get("linear_velocity", float("nan"))
        ang = args.get("angular_velocity", float("nan"))
        return f"VEL(lin={float(lin):+.2f},ang={float(ang):+.2f})"
    return str(action)


# --------------------------------------------------------------------------------------
# Habitat config helpers shared by VLNCEEnv and ObjectNavEnv (call inside ``read_write``).
# --------------------------------------------------------------------------------------


def _disable_episode_shuffle(config: Any) -> None:
    """Pin the EpisodeIterator order (habitat defaults to shuffle=True)."""
    from omegaconf import OmegaConf

    env_cfg = config.habitat.environment
    if hasattr(env_cfg, "iterator_options"):
        OmegaConf.set_struct(env_cfg.iterator_options, False)
        env_cfg.iterator_options.shuffle = False
        OmegaConf.set_struct(env_cfg.iterator_options, True)


def _apply_image_size(config: Any, image_size: Optional[Tuple[int, int]]) -> None:
    """Override the rgb/depth sensor resolution of every agent; no-op when ``image_size`` is None."""
    if image_size is None:
        return
    if not hasattr(config.habitat.simulator, "agents"):
        return
    for agent_name in config.habitat.simulator.agents:
        agent_cfg = config.habitat.simulator.agents[agent_name]
        if not hasattr(agent_cfg, "sim_sensors"):
            continue
        for sensor_name in ("rgb_sensor", "depth_sensor"):
            if hasattr(agent_cfg.sim_sensors, sensor_name):
                sensor = getattr(agent_cfg.sim_sensors, sensor_name)
                sensor.height = image_size[0]
                sensor.width = image_size[1]


def _inject_nav_measurements(config: Any, success_distance: float) -> None:
    """Set ``success.success_distance`` and add path_length / oracle_success / steps_taken."""
    from habitat.config.default_structured_configs import MeasurementConfig
    from omegaconf import OmegaConf

    measurements = config.habitat.task.measurements
    OmegaConf.set_struct(measurements, False)
    measurements.success.success_distance = success_distance
    measurements.path_length = OmegaConf.structured(MeasurementConfig(type="PathLength"))
    measurements.oracle_success = OmegaConf.structured(MeasurementConfig(type="OracleSuccess"))
    OmegaConf.set_struct(measurements.oracle_success, False)
    measurements.oracle_success.success_distance = success_distance
    OmegaConf.set_struct(measurements.oracle_success, True)
    measurements.steps_taken = OmegaConf.structured(MeasurementConfig(type="StepsTaken"))
    OmegaConf.set_struct(measurements, True)


def _inject_ndtw_measurement(config: Any, success_distance: float) -> None:
    """Add the NDTW measure; the ground-truth path file sits next to the episode file."""
    from habitat.config.default_structured_configs import MeasurementConfig
    from omegaconf import OmegaConf

    measurements = config.habitat.task.measurements
    data_path = config.habitat.dataset.data_path
    # R2R:  .../{split}/{split}.json.gz       -> .../{split}/{split}_gt.json.gz
    # RxR:  .../{split}/{split}_guide.json.gz -> .../{split}/{split}_guide_gt.json.gz
    gt_path = data_path.replace(".json.gz", "_gt.json.gz")
    OmegaConf.set_struct(measurements, False)
    measurements.ndtw = OmegaConf.structured(MeasurementConfig(type="NDTW"))
    OmegaConf.set_struct(measurements.ndtw, False)
    measurements.ndtw.gt_path = gt_path
    measurements.ndtw.split = config.habitat.dataset.split
    measurements.ndtw.success_distance = success_distance
    measurements.ndtw.fdtw = True
    OmegaConf.set_struct(measurements.ndtw, True)
    OmegaConf.set_struct(measurements, True)


class VLNCEEnv(BaseHabitatEnv):
    """Habitat VLN-CE task wrapper.

    Actions: an ``int`` in 0..3 (STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT) or a
    ``velocity_control`` dict with normalized ``linear_velocity`` / ``angular_velocity``
    in [-1, 1]. Observations: ``rgb`` (H, W, 3) uint8, ``depth`` (H, W) float32,
    ``instruction`` ``{"text": str}``, ``goal_distance`` (1,) float32, ``progress`` (1,)
    float32. ``info`` carries the Habitat metrics plus the velocity_control ranges the
    client needs to map waypoints to commands (see ``_compute_info``).
    """

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

    def __init__(
        self,
        config_path: str,
        data_path: Optional[str] = None,
        scenes_dir: Optional[str] = None,
        split: Optional[str] = "val_unseen",
        gpu_id: int = 0,
        image_size: Optional[Tuple[int, int]] = None,
        max_steps: int = 500,
        success_distance: float = 3.0,
        split_id: Optional[int] = None,
        split_num: Optional[int] = None,
        early_stop_rotation: int = 0,
        early_stop_steps: int = 0,
    ):
        """
        Args:
            config_path: Habitat config yaml.
            data_path: Dataset root; rewrites ``dataset.data_path`` to
                ``<root>/{split}/{split}.json.gz``. ``None`` keeps the yaml value.
            scenes_dir: Scene dataset directory override. ``None`` keeps the yaml value.
            split: Dataset split written to ``dataset.split`` (VLN-CE benchmark: ``val_unseen``).
                ``None`` keeps the yaml value.
            gpu_id: habitat-sim render device.
            image_size: Optional (height, width) sensor override; ``None`` keeps the yaml.
            max_steps: Steps after which ``truncated`` is reported.
            success_distance: Success radius in meters (VLN-CE standard: 3.0).
            split_id / split_num: Parallel-evaluation shard (scene-sorted slices).
            early_stop_rotation: Force STOP once ``distance_to_goal`` has been unchanged for
                more than this many consecutive steps. 0 disables.
            early_stop_steps: Force STOP once more than this many steps were taken. 0 disables.
        """
        self.data_path = data_path
        self.scenes_dir = scenes_dir
        self.split = split
        self.success_distance = success_distance
        self.habitat_time_step: Optional[float] = None
        self.lin_vel_range: Optional[Tuple[float, float]] = None
        self.ang_vel_range: Optional[Tuple[float, float]] = None
        self.min_abs_lin_speed: Optional[float] = None
        self.min_abs_ang_speed: Optional[float] = None

        # Early stop is applied by forcing a STOP action through Habitat so that
        # ``is_stop_called`` is set and Success / SPL are well defined.
        self._early_stop_rotation = early_stop_rotation
        self._early_stop_steps = early_stop_steps
        self._consecutive_no_progress: int = 0
        self._last_distance_to_goal: float = float("inf")

        self._current_episode: Any = None
        self._current_instruction = ""
        self._current_language = ""
        self._start_geodesic_distance = float("inf")

        # Per-episode diagnostic histories (used by termination_details).
        self._action_history: List[str] = []
        self._distance_history: List[float] = []
        self._no_progress_start_step: int = 0

        super().__init__(
            config_path=config_path,
            gpu_id=gpu_id,
            image_size=image_size,
            max_steps=max_steps,
            split_id=split_id,
            split_num=split_num,
        )

    # -- config assembly --------------------------------------------------------

    def _cache_velocity_control_config(self, config: Any) -> None:
        """Remember the velocity_control ranges so ``info`` can expose them to the client."""
        actions = config.habitat.task.actions
        vel_cfg = actions.get("velocity_control") if hasattr(actions, "get") else None
        if vel_cfg is None:
            self.habitat_time_step = None
            self.lin_vel_range = None
            self.ang_vel_range = None
            self.min_abs_lin_speed = None
            self.min_abs_ang_speed = None
            return

        self.habitat_time_step = float(vel_cfg.time_step)
        self.lin_vel_range = tuple(float(x) for x in vel_cfg.lin_vel_range)
        self.ang_vel_range = tuple(float(x) for x in vel_cfg.ang_vel_range)
        self.min_abs_lin_speed = float(vel_cfg.min_abs_lin_speed)
        self.min_abs_ang_speed = float(vel_cfg.min_abs_ang_speed)
        if len(self.lin_vel_range) != 2 or len(self.ang_vel_range) != 2:
            raise ValueError(
                "velocity_control lin_vel_range and ang_vel_range must each contain exactly two values"
            )

    def _is_stop_called(self) -> bool:
        task = getattr(self._habitat_env, "task", None)
        return bool(
            getattr(task, "is_stop_called", False)
            or getattr(self._habitat_env, "is_stop_called", False)
        )

    def _apply_dataset_overrides(self, config: Any) -> None:
        """gpu id, split, data_path / scenes_dir overrides (inside ``read_write``)."""
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = self.gpu_id
        if self.split is not None:
            config.habitat.dataset.split = self.split
        if self.data_path:
            config.habitat.dataset.data_path = f"{self.data_path}/{{split}}/{{split}}.json.gz"
        if self.scenes_dir:
            config.habitat.dataset.scenes_dir = self.scenes_dir

    def _apply_episode_split(self, env: Any) -> None:
        """Scene-sorted contiguous slicing for parallel evaluation (last shard takes the rest)."""
        if self.split_num is None or self.split_id is None:
            return
        all_episodes = list(env._dataset.episodes)
        all_episodes.sort(key=lambda ep: ep.scene_id)
        total_episodes = len(all_episodes)
        chunk_size = total_episodes // self.split_num
        start = self.split_id * chunk_size
        if self.split_id < self.split_num - 1:
            end = start + chunk_size
        else:
            end = total_episodes
        split_episodes = all_episodes[start:end]
        n_scenes = len({ep.scene_id for ep in split_episodes})
        # The property setter rebuilds the EpisodeIterator.
        env.episodes = split_episodes
        logger.info(
            f"[Parallel] Split {self.split_id}/{self.split_num}: "
            f"{len(split_episodes)}/{total_episodes} episodes, {n_scenes} scenes (scene-sorted)"
        )

    def _create_habitat_env(self) -> Any:
        try:
            import habitat
            from habitat.config.read_write import read_write
        except ImportError as e:
            raise ImportError(f"habitat-lab not installed: {e}")
        from omegaconf import OmegaConf

        # Registers the "VLN-CE-v1" dataset and the PathLength/OracleSuccess/StepsTaken/NDTW
        # measures; must happen before habitat.Env is built.
        from . import vlnce_extensions  # noqa: F401

        logger.info(f"Loading config from: {self.config_path}")
        config = habitat.get_config(self.config_path)

        with read_write(config):
            self._apply_dataset_overrides(config)

            # RxR-only default language filter (en-US + en-IN). habitat-lab's DatasetConfig
            # has no `languages` field, so it is added at runtime unless the yaml sets it.
            ds_cfg = config.habitat.dataset
            data_path_str = str(getattr(ds_cfg, "data_path", "") or "")
            is_rxr = "_guide.json.gz" in data_path_str or "RxR" in data_path_str
            languages_set = hasattr(ds_cfg, "languages") and getattr(ds_cfg, "languages", None)
            if is_rxr and not languages_set:
                OmegaConf.set_struct(ds_cfg, False)
                ds_cfg.languages = ["en-US", "en-IN"]
                OmegaConf.set_struct(ds_cfg, True)
                logger.info(
                    "RxR detected, defaulting to English-only language filter: "
                    "languages=['en-US', 'en-IN']. Override via habitat.dataset.languages "
                    "in the yaml."
                )

            _disable_episode_shuffle(config)
            _apply_image_size(config, self.image_size)
            _inject_nav_measurements(config, self.success_distance)
            _inject_ndtw_measurement(config, self.success_distance)
            self._cache_velocity_control_config(config)

        self.split = str(config.habitat.dataset.split)
        logger.info(
            f"Config loaded, dataset type: {config.habitat.dataset.type}, split: {self.split}"
        )

        env = habitat.Env(config=config)

        # Report the dataset size after the dataset-level filters (languages, ...) applied.
        try:
            total_eps = len(env._dataset.episodes)
            ds_lang = getattr(config.habitat.dataset, "languages", None)
            n_scenes = len({ep.scene_id for ep in env._dataset.episodes})
            lang_breakdown: Dict[str, int] = {}
            for ep in env._dataset.episodes:
                inst = getattr(ep, "instruction", None)
                lang = getattr(inst, "language", None) if inst is not None else None
                if lang:
                    lang_breakdown[lang] = lang_breakdown.get(lang, 0) + 1
            lang_summary = (
                ", ".join(f"{k}={v}" for k, v in sorted(lang_breakdown.items()))
                if lang_breakdown
                else "(no language metadata)"
            )
            logger.info(
                f"Dataset loaded: {total_eps} episodes across {n_scenes} scenes "
                f"(filter languages={list(ds_lang) if ds_lang else '(all)'}; "
                f"per-language: {lang_summary})"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not summarize dataset size: {e}")

        self._apply_episode_split(env)
        return env

    # -- observations -----------------------------------------------------------

    def _compute_progress(self) -> float:
        """Fraction of the start geodesic distance (to goals[0]) already covered, in [0, 1]."""
        progress = 0.0
        env = self._habitat_env
        if env is None:
            return progress
        try:
            episode = env.current_episode
            if getattr(episode, "goals", None):
                agent_pos = env.sim.get_agent_state().position.tolist()
                goal_pos = episode.goals[0].position
                current_distance = env.sim.geodesic_distance(agent_pos, goal_pos)
                if np.isfinite(current_distance) and self._start_geodesic_distance > 0:
                    progress = (
                        self._start_geodesic_distance - current_distance
                    ) / self._start_geodesic_distance
                    progress = max(0.0, min(1.0, progress))
        except Exception:
            pass
        return progress

    def _extract_task_obs(self, habitat_obs: Dict) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}

        instruction_data = habitat_obs.get("instruction", {})
        if isinstance(instruction_data, dict):
            instruction_text = instruction_data.get("text", "")
        else:
            instruction_text = str(instruction_data) if instruction_data else ""
        obs["instruction"] = {"text": instruction_text}
        self._current_instruction = instruction_text

        # Placeholder; reset()/step() overwrite it from the distance_to_goal metric.
        obs["goal_distance"] = np.array([float("inf")], dtype=np.float32)
        obs["progress"] = np.array([self._compute_progress()], dtype=np.float32)
        return obs

    # -- actions / reward / termination ----------------------------------------

    def _convert_action(self, action: Any) -> Any:
        """Pass-through for discrete int or velocity_control dict actions."""
        if isinstance(action, (int, np.integer)):
            return int(action)
        if isinstance(action, dict):
            if action.get("action") != "velocity_control":
                raise ValueError(f"Unsupported dict action name: {action.get('action')!r}")
            args = action.get("action_args")
            if (
                not isinstance(args, dict)
                or "linear_velocity" not in args
                or "angular_velocity" not in args
            ):
                raise ValueError(
                    f"velocity_control requires action_args with linear/angular velocity, got: {args!r}"
                )
            lin = float(args["linear_velocity"])
            ang = float(args["angular_velocity"])
            if not math.isfinite(lin) or not math.isfinite(ang):
                raise ValueError(
                    f"velocity_control velocities must be finite, got lin={lin!r}, ang={ang!r}"
                )
            return {
                "action": "velocity_control",
                "action_args": {
                    "linear_velocity": lin,
                    "angular_velocity": ang,
                },
            }
        raise TypeError(f"Unsupported action type: {type(action).__name__}")

    def _compute_reward(self, obs: Dict, info: Dict) -> float:
        """Sparse shaping reward (not used by the evaluation metrics)."""
        distance_to_goal = info.get("distance_to_goal", float("inf"))
        reward = -0.01
        if distance_to_goal < self.success_distance:
            reward += 10.0
        return reward

    def _check_terminated(self, obs: Dict, info: Dict, action: Any) -> bool:
        """True on a discrete STOP or when Habitat's ``is_stop_called`` fired.

        The velocity_control action sets ``is_stop_called`` when both speeds fall below
        the task's ``min_abs_*_speed`` thresholds, which Success / SPL require.
        """
        if isinstance(action, (int, np.integer)) and int(action) == self.STOP:
            return True
        return self._is_stop_called()

    # -- info -------------------------------------------------------------------

    def _compute_info(self, habitat_obs: Dict) -> Dict[str, Any]:
        info = super()._compute_info(habitat_obs)
        info["instruction"] = self._current_instruction
        if self.habitat_time_step is not None:
            info["habitat_time_step"] = self.habitat_time_step
        if self.lin_vel_range is not None:
            info["lin_vel_range"] = self.lin_vel_range
        if self.ang_vel_range is not None:
            info["ang_vel_range"] = self.ang_vel_range
        if self.min_abs_lin_speed is not None:
            info["min_abs_lin_speed"] = self.min_abs_lin_speed
        if self.min_abs_ang_speed is not None:
            info["min_abs_ang_speed"] = self.min_abs_ang_speed
        if self._current_language:
            info["language"] = self._current_language

        # Habitat's Success measure (is_stop_called AND distance < success_distance) is
        # kept as-is; never override info["success"].
        if "distance_to_goal" in info:
            info["goal_distance"] = info["distance_to_goal"]

        if hasattr(self._habitat_env, "current_episode"):
            episode = self._habitat_env.current_episode
            if getattr(episode, "reference_path", None):
                info["reference_path"] = episode.reference_path
            if getattr(episode, "goals", None):
                info["goal_position"] = episode.goals[0].position

        return info

    # -- reset / step -------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._consecutive_no_progress = 0
        self._last_distance_to_goal = float("inf")
        self._action_history = []
        self._distance_history = []
        self._no_progress_start_step = 0

        obs, info = super().reset(seed=seed, options=options)

        if hasattr(self._habitat_env, "current_episode"):
            self._current_episode = self._habitat_env.current_episode

            instruction = getattr(self._current_episode, "instruction", None)
            if instruction is not None:
                if hasattr(instruction, "instruction_text"):
                    self._current_instruction = instruction.instruction_text
                elif isinstance(instruction, dict):
                    self._current_instruction = instruction.get("text", "")
                # Expose the language for downstream filtering (RxR English-only eval).
                lang = getattr(instruction, "language", None)
                if lang is None and isinstance(instruction, dict):
                    lang = instruction.get("language")
                self._current_language = lang or ""

            # Baseline for the progress observation.
            try:
                if hasattr(self._habitat_env, "sim") and getattr(
                    self._current_episode, "goals", None
                ):
                    agent_pos = self._habitat_env.sim.get_agent_state().position.tolist()
                    goal_pos = self._current_episode.goals[0].position
                    self._start_geodesic_distance = self._habitat_env.sim.geodesic_distance(
                        agent_pos, goal_pos
                    )
                    if not np.isfinite(self._start_geodesic_distance):
                        self._start_geodesic_distance = float(
                            np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
                        )
            except Exception as e:
                logger.warning(f"Failed to compute start geodesic distance: {e}")
                self._start_geodesic_distance = float("inf")

        if "distance_to_goal" in info:
            obs["goal_distance"] = np.array([info["distance_to_goal"]], dtype=np.float32)

        # The language of the *new* episode is only known after super().reset().
        if self._current_language:
            info["language"] = self._current_language

        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute an action with the optional early-stop rules.

        Before forwarding the action: if ``early_stop_rotation`` > 0 and the no-progress
        streak exceeds it, or ``early_stop_steps`` > 0 and the step count exceeds it, the
        action is replaced by STOP so Habitat sets ``is_stop_called``. On the final step
        ``info`` gains ``termination_reason`` and ``termination_details``.
        """
        original_action = action
        early_stop_reason = None

        # A limit of 0 means the check is disabled.
        if (
            self._early_stop_rotation > 0
            and self._consecutive_no_progress > self._early_stop_rotation
        ):
            early_stop_reason = "early_stop_no_progress"
            logger.info(
                f"[Early stop] no_progress={self._consecutive_no_progress}/"
                f"{self._early_stop_rotation} -> forcing STOP"
            )
            action = self.STOP
        elif self._early_stop_steps > 0 and self._step_count > self._early_stop_steps:
            early_stop_reason = "early_stop_step_limit"
            logger.info(
                f"[Early stop] steps={self._step_count}/{self._early_stop_steps} -> forcing STOP"
            )
            action = self.STOP

        obs, reward, terminated, truncated, info = super().step(action)

        # Consecutive steps without any change of distance_to_goal (rotation in place or
        # blocked motion). Any translation resets the streak.
        dtg = info.get("distance_to_goal", float("inf"))
        if dtg != self._last_distance_to_goal:
            self._last_distance_to_goal = dtg
            self._consecutive_no_progress = 0
            self._no_progress_start_step = self._step_count + 1
        else:
            self._consecutive_no_progress += 1

        self._action_history.append(_action_label(original_action))
        self._distance_history.append(dtg)

        if terminated or truncated:
            # Discrete path: original int == STOP. Velocity path: is_stop_called fired.
            voluntary_stop = (
                isinstance(original_action, (int, np.integer)) and int(original_action) == self.STOP
            ) or self._is_stop_called()
            if early_stop_reason is not None:
                reason = early_stop_reason
            elif terminated and voluntary_stop:
                reason = "agent_stop"
            elif truncated:
                reason = "max_steps_truncated"
            else:
                reason = "unknown"

            action_counts: Dict[str, int] = {}
            for a in self._action_history:
                action_counts[a] = action_counts.get(a, 0) + 1

            dists = self._distance_history
            initial_dist = dists[0] if dists else float("inf")
            min_dist = min(dists) if dists else float("inf")
            min_dist_step = dists.index(min_dist) + 1 if dists else 0

            details: Dict[str, Any] = {
                # Free-form debug record of what happened on the termination step.
                "original_action": _action_label(original_action),
                "executed_action": _action_label(action),
                "early_stop_rotation_limit": self._early_stop_rotation,
                "early_stop_steps_limit": self._early_stop_steps,
                "total_steps": self._step_count,
                "action_counts": action_counts,
                "distance_to_goal_initial": round(initial_dist, 4),
                "distance_to_goal_final": round(dtg, 4),
                "distance_to_goal_min": round(min_dist, 4),
                "distance_to_goal_min_at_step": min_dist_step,
            }

            if reason == "early_stop_no_progress":
                streak_len = self._consecutive_no_progress
                streak_start = self._no_progress_start_step
                streak_actions = (
                    self._action_history[streak_start - 1 :] if streak_start > 0 else []
                )
                streak_action_counts: Dict[str, int] = {}
                for a in streak_actions:
                    streak_action_counts[a] = streak_action_counts.get(a, 0) + 1

                details["no_progress"] = {
                    "description": (
                        f"distance_to_goal stayed at {dtg:.4f}m for "
                        f"{streak_len} consecutive steps (threshold: "
                        f"{self._early_stop_rotation}), meaning the agent "
                        f"never executed FORWARD successfully during this period"
                    ),
                    "frozen_distance": round(dtg, 4),
                    "streak_length": streak_len,
                    "streak_start_step": streak_start,
                    "streak_end_step": self._step_count,
                    "streak_action_counts": streak_action_counts,
                    "streak_actions_last_10": list(streak_actions[-10:]),
                }
            elif reason == "early_stop_step_limit":
                details["step_limit"] = {
                    "description": (
                        f"Agent took {self._step_count} steps, exceeding the "
                        f"early_stop_steps limit of {self._early_stop_steps}. "
                        f"The agent was still {dtg:.4f}m from the goal."
                    ),
                    "last_10_actions": list(self._action_history[-10:]),
                    "last_10_distances": [round(d, 4) for d in self._distance_history[-10:]],
                }
            elif reason == "max_steps_truncated":
                details["max_steps"] = {
                    "description": (
                        f"Runner max_steps reached at step {self._step_count}. "
                        f"The agent was still {dtg:.4f}m from the goal."
                    ),
                    "last_10_actions": list(self._action_history[-10:]),
                    "last_10_distances": [round(d, 4) for d in self._distance_history[-10:]],
                }
            elif reason == "agent_stop":
                outcome = (
                    "SUCCESS - within success_distance."
                    if info.get("success", False)
                    else "FAIL - not close enough to goal."
                )
                details["agent_stop"] = {
                    "description": (
                        f"Agent voluntarily chose STOP at step {self._step_count}. "
                        f"Distance to goal: {dtg:.4f}m. {outcome}"
                    ),
                }

            info["termination_reason"] = reason
            info["termination_details"] = details

        if "distance_to_goal" in info:
            obs["goal_distance"] = np.array([info["distance_to_goal"]], dtype=np.float32)

        return obs, reward, terminated, truncated, info
