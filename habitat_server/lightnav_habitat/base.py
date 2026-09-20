"""Base class shared by the Habitat environment wrappers (RGB/depth extraction, reset/step plumbing)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Shape used for the zero image returned when the simulator exposes no RGB/depth
# sensor and no explicit image_size was requested (never hit with the shipped configs).
_FALLBACK_IMAGE_HW = (270, 480)


class BaseHabitatEnv(ABC):
    """Common lifecycle for a wrapped ``habitat.Env``.

    Subclasses implement ``_create_habitat_env``, ``_extract_task_obs``, ``_convert_action``,
    ``_compute_reward`` and ``_check_terminated``.
    """

    # Sensor uuids accepted as the RGB / depth observation.
    RGB_KEYS = ["rgb", "head_rgb"]
    DEPTH_KEYS = ["depth", "head_depth"]

    def __init__(
        self,
        config_path: str,
        gpu_id: int = 0,
        image_size: Optional[Tuple[int, int]] = None,
        max_steps: int = 500,
        split_id: Optional[int] = None,
        split_num: Optional[int] = None,
    ):
        """
        Args:
            config_path: Path to the Habitat (Hydra) config yaml.
            gpu_id: habitat-sim render device (``habitat_sim_v0.gpu_device_id``).
            image_size: Optional (height, width) override for the RGB/depth sensors.
                ``None`` keeps the sensor size from the yaml.
            max_steps: Steps after which ``truncated`` is reported. Must not exceed the
                yaml's ``environment.max_episode_steps``.
            split_id / split_num: Shard index / count for parallel evaluation
                (scene-sorted contiguous slices of the episode list). ``None`` = no sharding.
        """
        self.config_path = config_path
        self.gpu_id = gpu_id
        self.image_size = tuple(image_size) if image_size is not None else None
        self.max_steps = max_steps
        self.split_id = split_id
        self.split_num = split_num

        self._habitat_env: Any = None
        self._step_count = 0
        self._initialized = False
        self._obs_keys_logged = False

    # -- abstract hooks --------------------------------------------------------

    @abstractmethod
    def _create_habitat_env(self) -> Any:
        """Build and return the underlying ``habitat.Env``."""

    @abstractmethod
    def _extract_task_obs(self, habitat_obs: Dict) -> Dict[str, Any]:
        """Return task-specific observation entries."""

    @abstractmethod
    def _convert_action(self, action: Any) -> Any:
        """Convert a client action into the Habitat action format."""

    @abstractmethod
    def _compute_reward(self, obs: Dict, info: Dict) -> float:
        """Return the reward for the current step."""

    @abstractmethod
    def _check_terminated(self, obs: Dict, info: Dict, action: Any) -> bool:
        """Return True when the episode ended on this step."""

    # -- observation helpers ---------------------------------------------------

    def _fallback_hw(self) -> Tuple[int, int]:
        return self.image_size if self.image_size is not None else _FALLBACK_IMAGE_HW

    def _extract_rgb(self, habitat_obs: Dict) -> np.ndarray:
        """RGB image as (H, W, 3) uint8."""
        for key in self.RGB_KEYS:
            if key in habitat_obs:
                rgb = np.asarray(habitat_obs[key])
                if rgb.dtype in (np.float32, np.float64):
                    rgb = (rgb * 255).astype(np.uint8)
                return rgb

        logger.warning(f"No RGB found in keys: {self.RGB_KEYS}")
        return np.zeros((*self._fallback_hw(), 3), dtype=np.uint8)

    def _extract_depth(self, habitat_obs: Dict) -> np.ndarray:
        """Depth image as (H, W) float32."""
        for key in self.DEPTH_KEYS:
            if key in habitat_obs:
                depth = np.asarray(habitat_obs[key]).squeeze()
                return depth.astype(np.float32)

        return np.zeros(self._fallback_hw(), dtype=np.float32)

    def _build_observation(self, habitat_obs: Dict) -> Dict[str, Any]:
        if not self._obs_keys_logged:
            logger.info(f"Available habitat_obs keys: {list(habitat_obs.keys())}")
            self._obs_keys_logged = True

        obs: Dict[str, Any] = {
            "rgb": self._extract_rgb(habitat_obs),
            "depth": self._extract_depth(habitat_obs),
        }
        obs.update(self._extract_task_obs(habitat_obs))
        return obs

    def _compute_info(self, habitat_obs: Dict) -> Dict[str, Any]:
        """steps, episode_id, scene_id (+ raw_episode_id when a loader renumbered) and metrics."""
        info: Dict[str, Any] = {"steps": self._step_count}

        if hasattr(self._habitat_env, "current_episode"):
            episode = self._habitat_env.current_episode
            info["episode_id"] = getattr(episode, "episode_id", "unknown")
            info["scene_id"] = getattr(episode, "scene_id", "unknown")
            raw_episode_id = (getattr(episode, "info", None) or {}).get("raw_episode_id")
            if raw_episode_id is not None:
                info["raw_episode_id"] = raw_episode_id

        if hasattr(self._habitat_env, "get_metrics"):
            info.update(self._habitat_env.get_metrics())

        return info

    # -- lifecycle -------------------------------------------------------------

    def initialize(self) -> None:
        """Eagerly create the Habitat environment (idempotent)."""
        if not self._initialized:
            logger.info("Creating Habitat environment...")
            self._habitat_env = self._create_habitat_env()
            self._initialized = True
            logger.info("Habitat environment created")

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Start the next episode. ``seed`` / ``options`` are accepted for API compatibility."""
        self.initialize()
        self._step_count = 0

        habitat_obs = self._habitat_env.reset()
        obs = self._build_observation(habitat_obs)
        info = self._compute_info(habitat_obs)
        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one action; returns (obs, reward, terminated, truncated, info)."""
        self._step_count += 1

        habitat_action = self._convert_action(action)
        habitat_obs = self._habitat_env.step(habitat_action)
        obs = self._build_observation(habitat_obs)
        info = self._compute_info(habitat_obs)
        reward = self._compute_reward(obs, info)
        terminated = self._check_terminated(obs, info, action)
        truncated = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._habitat_env is not None:
            self._habitat_env.close()
            self._habitat_env = None
            self._initialized = False
            logger.info("Habitat environment closed")
