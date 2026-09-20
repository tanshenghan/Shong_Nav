"""BetaNav FM policy adapter for the Habitat velocity-control evaluator."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lightnav.habitat.policy import extract_instruction, select_action_waypoint
from lightnav.inference.samples import build_vln_traj_sample
from lightnav.tracking import TrackingAgent
from lightnav.velocity import first_waypoint_to_velocity_cmd
from lightnav.vln_utils import parse_rvq_action_tokens

logger = logging.getLogger(__name__)


@dataclass
class BetaFMRuntime:
    """Loaded BetaModel plus the small runtime settings used by the policy."""

    beta_model: Any
    bundle: Any
    checkpoint_path: Path
    checkpoint_step: int
    max_new_tokens: int = 64
    fm_steps: int | None = None
    fm_seed: int = 42
    aspect_mode: str = "stretch"
    stop_mode: str = "baseline"  # baseline | decode_continue | explicit_gate
    action_output: str = "fm"  # fm | token (trained BetaNav RVQ branch)
    fm_only_stop_mode: str = "max_steps"  # max_steps | rvq

    def reset_episode_state(self) -> None:
        """Beta HF inference currently has no episode-local engine cache."""


def _prompt_only_inputs(processed: dict[str, Any]) -> dict[str, Any]:
    """Trim the placeholder answer and batch one processed inference sample."""
    result = dict(processed)
    labels = result.pop("labels", None)
    answer_start: int | None = None
    if labels is not None:
        answer_tokens = (labels != -100).nonzero(as_tuple=False)
        if answer_tokens.numel():
            answer_start = int(answer_tokens[0].item())

    sequence_keys = ("input_ids", "attention_mask", "image_mask", "video_mask")
    for key in sequence_keys:
        value = result.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if answer_start is not None:
            value = value[:answer_start]
        if key in {"input_ids", "attention_mask"} and value.ndim == 1:
            value = value.unsqueeze(0)
        result[key] = value

    # BetaModel reconstructs native multimodal RoPE and deliberately ignores the
    # processor's position ids. Keep only its documented tensor/metadata contract.
    keep = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "position_ids",
        "mm_token_type_ids",
        "image_mask",
        "video_mask",
        "_original_video_grid_thw",
        "_post_vit_pool_factors",
        "_post_vit_pool_spatial",
    }
    return {key: value for key, value in result.items() if key in keep}


class BetaFMVLNCEPolicy:
    """Execute FM or trained RVQ trajectories with the same token STOP semantics."""

    def __init__(
        self,
        runtime: BetaFMRuntime,
        *,
        dt: float,
        lin_vel_range: tuple[float, float],
        ang_vel_range: tuple[float, float],
        num_history_frames: int,
        rvq_bundle: Any,
        min_abs_lin_speed: float = 0.01,
        min_abs_ang_speed: float = 0.5,
    ) -> None:
        if rvq_bundle is None:
            raise ValueError("Beta FM evaluation requires the checkpoint's RVQ action decoder")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.runtime = runtime
        self.engine = runtime
        self.model = runtime.beta_model
        self.bundle = runtime.bundle
        self.dt = float(dt)
        self.lin_vel_range = tuple(float(value) for value in lin_vel_range)
        self.ang_vel_range = tuple(float(value) for value in ang_vel_range)
        self.num_history_frames = int(num_history_frames)
        self.min_abs_lin_speed = float(min_abs_lin_speed)
        self.min_abs_ang_speed = float(min_abs_ang_speed)
        if runtime.action_output not in {"fm", "token"}:
            raise ValueError("Beta action_output must be fm or token")
        if runtime.action_output == "token" and runtime.stop_mode != "baseline":
            raise ValueError("Beta token evaluation requires baseline RVQ STOP semantics")
        if runtime.stop_mode not in {"baseline", "decode_continue", "explicit_gate"}:
            raise ValueError(f"unsupported Beta FM stop mode: {runtime.stop_mode!r}")
        self.agent = TrackingAgent(
            runtime,
            num_history_frames=self.num_history_frames,
            rvq_bundle=rvq_bundle,
        )
        self.instruction = ""
        self._step = 0
        self._clear_last()

    def _clear_last(self) -> None:
        self._last_raw_text = ""
        self._last_rvq_codes: list[int] | None = None
        self._last_waypoints: np.ndarray | None = None
        self._last_rvq_waypoints: np.ndarray | None = None
        self._last_se2_deltas: np.ndarray | None = None
        self._last_action_waypoint_index: int | None = None
        self._last_action_waypoint: np.ndarray | None = None
        self._last_stop = False
        self._decode_failed = False
        self._last_explicit_stop = False
        self._last_stop_source = ""
        self._last_stop_suppressed = False

    def reset(self, obs: dict[str, Any] | None = None) -> None:
        self.instruction = extract_instruction(obs) if obs is not None else ""
        self.agent.reset(self.instruction)
        self._step = 0
        self._clear_last()

    def observe(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        self.agent.observe(obs["rgb"])

    def stop_action(self) -> dict[str, Any]:
        return first_waypoint_to_velocity_cmd(
            np.zeros(3, dtype=np.float32),
            self.dt,
            self.lin_vel_range,
            self.ang_vel_range,
        )

    def _generator(self, info: dict[str, Any] | None) -> torch.Generator:
        info = info or {}
        identity = (
            f"{self.runtime.fm_seed}|{info.get('scene_id', '')}|"
            f"{info.get('episode_id', '')}|{self._step}"
        ).encode()
        seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % (2**63 - 1)
        device = self.model.expert.step_embedding.device
        return torch.Generator(device=device).manual_seed(seed)

    @staticmethod
    def _physical_speed(raw: float, limits: tuple[float, float]) -> float:
        return limits[0] + (float(raw) + 1.0) * 0.5 * (limits[1] - limits[0])

    def _command_would_stop(self, action: dict[str, Any]) -> bool:
        args = action["action_args"]
        lin = self._physical_speed(args["linear_velocity"], self.lin_vel_range)
        ang = self._physical_speed(args["angular_velocity"], self.ang_vel_range)
        return abs(lin) < self.min_abs_lin_speed and abs(ang) < self.min_abs_ang_speed

    def _explicit_gate_action(self, waypoints: np.ndarray) -> dict[str, Any]:
        """Choose a non-terminating FM command, or a small deterministic recovery turn."""
        for index, waypoint in enumerate(waypoints):
            action = first_waypoint_to_velocity_cmd(
                waypoint, self.dt, self.lin_vel_range, self.ang_vel_range
            )
            if not self._command_would_stop(action):
                self._last_action_waypoint_index = int(index)
                self._last_action_waypoint = np.asarray(waypoint, dtype=np.float32).copy()
                return action

        turn_hint = float(np.asarray(waypoints)[:, 1:3].sum())
        direction = -1.0 if turn_hint < 0.0 else 1.0
        recovery_dps = max(5.0, 2.0 * self.min_abs_ang_speed)
        recovery = np.array(
            [0.0, 0.0, direction * math.radians(recovery_dps * self.dt)], dtype=np.float32
        )
        self._last_action_waypoint_index = -1
        self._last_action_waypoint = recovery.copy()
        return first_waypoint_to_velocity_cmd(
            recovery, self.dt, self.lin_vel_range, self.ang_vel_range
        )

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        self._decode_failed = False
        self._last_explicit_stop = False
        self._last_stop = False
        self._last_stop_source = ""
        self._last_stop_suppressed = False
        self.agent.observe(obs["rgb"])
        sample = build_vln_traj_sample(
            self.agent._get_video_tensor(),
            self.agent.instruction,
            list(self.agent._history_frame_ids),
            self.bundle,
        )
        processed = self.bundle.data_processor.process_sample(
            sample,
            add_generation_prompt=True,
            validate_video_shapes=False,
        )
        prefix_inputs = _prompt_only_inputs(processed)
        try:
            prediction = self.model.predict(
                prefix_inputs,
                tokenizer=self.bundle.tokenizer,
                action_decoder=self.agent,
                max_new_tokens=int(self.runtime.max_new_tokens),
                num_steps=self.runtime.fm_steps,
                generator=self._generator(info) if self.runtime.action_output == "fm" else None,
                action_output=self.runtime.action_output,
                continue_on_decode_error=self.runtime.stop_mode != "baseline",
                explicit_stop_only=self.runtime.stop_mode == "explicit_gate",
            )
        except ValueError as exc:
            logger.warning("Beta token/STOP decode failure (%s); zero-velocity fallback", exc)
            self._decode_failed = True
            self._last_stop = True
            self._last_stop_source = "decode_failure"
            self._step += 1
            return self.stop_action()

        self._last_raw_text = prediction.texts[0]
        self._decode_failed = bool(getattr(prediction, "decode_failed", [False])[0])
        self._last_explicit_stop = bool(getattr(prediction, "explicit_stop", [False])[0])
        self._last_stop = bool(prediction.stop[0].item())
        try:
            self._last_rvq_codes = [int(code) for code in parse_rvq_action_tokens(self._last_raw_text)]
        except ValueError:
            self._last_rvq_codes = None
        self._last_rvq_waypoints = prediction.rvq_waypoints[0].float().cpu().numpy().copy()
        self._last_se2_deltas = prediction.se2_deltas[0].float().cpu().numpy().copy()
        self._last_waypoints = prediction.waypoints[0].float().cpu().numpy().copy()
        if not np.isfinite(self._last_waypoints).all():
            raise RuntimeError("Beta policy produced non-finite waypoints")

        if self._last_stop:
            self._last_stop_source = (
                "explicit_token_stop" if self._last_explicit_stop else "near_zero_rvq_stop"
            )
            self._last_action_waypoint_index = 0
            self._last_action_waypoint = np.zeros(3, dtype=np.float32)
            self._step += 1
            return self.stop_action()

        waypoint_index, waypoint = select_action_waypoint(self._last_waypoints)
        self._last_action_waypoint_index = int(waypoint_index)
        self._last_action_waypoint = np.asarray(waypoint, dtype=np.float32).copy()
        action = first_waypoint_to_velocity_cmd(
            waypoint,
            self.dt,
            self.lin_vel_range,
            self.ang_vel_range,
        )
        if self._command_would_stop(action):
            if self.runtime.stop_mode == "explicit_gate":
                self._last_stop_suppressed = True
                self._last_stop_source = "fm_zero_suppressed"
                action = self._explicit_gate_action(self._last_waypoints)
            else:
                self._last_stop = True
                self._last_stop_source = (
                    "fm_zero_command" if self.runtime.action_output == "fm" else "rvq_zero_command"
                )
        self._step += 1
        return action

    def get_info(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action_source": self.runtime.action_output,
            "raw_text": self._last_raw_text,
            "rvq_codes": self._last_rvq_codes,
            "cluster_id": self._last_rvq_codes[0] if self._last_rvq_codes else -1,
            "num_history_frames": self.num_history_frames,
            "stop": self._last_stop,
            "explicit_stop": self._last_explicit_stop,
            "stop_source": self._last_stop_source,
            "stop_suppressed": self._last_stop_suppressed,
            "decode_failed": self._decode_failed,
        }
        if self._last_waypoints is not None:
            result.update(
                predicted_traj=self._last_waypoints,
                rvq_predicted_traj=self._last_rvq_waypoints,
                action_waypoint_index=self._last_action_waypoint_index,
                action_waypoint=self._last_action_waypoint,
            )
        if self.runtime.action_output == "fm" and self._last_se2_deltas is not None:
            result["fm_se2_deltas"] = self._last_se2_deltas
        return result


class BetaFMOnlyVLNCEPolicy(BetaFMVLNCEPolicy):
    """FM-only motion with fixed-budget or frozen-LightNav RVQ baseline STOP."""

    def __init__(
        self, runtime: BetaFMRuntime, *, dt: float,
        lin_vel_range: tuple[float, float], ang_vel_range: tuple[float, float],
        num_history_frames: int, min_abs_lin_speed: float = 0.01,
        min_abs_ang_speed: float = 0.5, rvq_bundle: Any = None,
    ) -> None:
        from lightnav.inference.policies import NavigationPolicy

        if runtime.beta_model.config.training_mode != "fm_only":
            raise ValueError("continuous-only policy requires an fm_only checkpoint")
        if runtime.action_output != "fm" or runtime.stop_mode != "baseline":
            raise ValueError("fm_only requires action_output=fm and baseline configuration")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.runtime = self.engine = runtime
        self.model = runtime.beta_model
        self.bundle = runtime.bundle
        self.dt = float(dt)
        self.lin_vel_range = tuple(float(v) for v in lin_vel_range)
        self.ang_vel_range = tuple(float(v) for v in ang_vel_range)
        self.num_history_frames = int(num_history_frames)
        self.min_abs_lin_speed = float(min_abs_lin_speed)
        self.min_abs_ang_speed = float(min_abs_ang_speed)
        if runtime.fm_only_stop_mode not in {"max_steps", "rvq"}:
            raise ValueError("fm_only_stop_mode must be max_steps or rvq")
        if runtime.fm_only_stop_mode == "rvq":
            if rvq_bundle is None:
                raise ValueError("RVQ STOP requires the frozen LightNav decoder")
            self.agent = TrackingAgent(runtime, num_history_frames=num_history_frames,
                                       rvq_bundle=rvq_bundle)
        else:
            self.agent = NavigationPolicy(runtime, num_history_frames=num_history_frames)
        self.instruction = ""
        self._step = 0
        self._clear_last()

    @torch.no_grad()
    def _rvq_requests_stop(self) -> bool:
        # STOP uses the frozen VLM's original RVQ prompt. FM must be encoded
        # separately with its continuous training prompt; do not reuse this prefill.
        sample = build_vln_traj_sample(
            self.agent._get_video_tensor(), self.agent.instruction,
            list(self.agent._history_frame_ids), self.bundle,
        )
        processed = self.bundle.data_processor.process_sample(
            sample, add_generation_prompt=True, validate_video_shapes=False,
        )
        tokenizer = self.bundle.tokenizer
        with self.model._hf_inputs(_prompt_only_inputs(processed)) as inputs:
            sequences = self.model.vlm.generate(
                **inputs, use_cache=True, do_sample=False, num_beams=1,
                num_return_sequences=1, return_dict_in_generate=False,
                output_hidden_states=False, output_attentions=False,
                max_new_tokens=int(self.runtime.max_new_tokens), min_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            tokens = sequences[0, inputs["input_ids"].shape[1]:]
        self._last_raw_text = tokenizer.decode(tokens, skip_special_tokens=False)
        try:
            codes = parse_rvq_action_tokens(self._last_raw_text)
            self._last_rvq_codes = [int(code) for code in codes]
            decoded = self.agent.decode_waypoints(self._last_raw_text)[0]
            self._last_rvq_waypoints = np.asarray(decoded, dtype=np.float32).copy()
            self._last_explicit_stop = bool(self.agent.rvq.is_stop(codes))
        except ValueError as exc:
            logger.warning("Frozen LightNav RVQ decode failure (%s); baseline STOP", exc)
            self._decode_failed = True
            self._last_stop_source = "decode_failure"
            return True
        if not np.any(self._last_rvq_waypoints):
            self._last_stop_source = (
                "explicit_token_stop" if self._last_explicit_stop else "near_zero_rvq_stop"
            )
            return True
        return False

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        from lightnav.inference.samples import build_continuous_traj_sample

        self._clear_last()
        self.agent.observe(obs["rgb"])
        if self.runtime.fm_only_stop_mode == "rvq" and self._rvq_requests_stop():
            self._last_stop = True
            self._step += 1
            return self.stop_action()
        sample = build_continuous_traj_sample(
            self.agent._get_video_tensor(), self.agent.instruction,
            list(self.agent._history_frame_ids), self.bundle,
        )
        processed = self.bundle.data_processor.process_sample(
            sample, add_generation_prompt=True, validate_video_shapes=False,
        )
        context = self.model.encode_context(_prompt_only_inputs(processed))
        deltas, waypoints = self.model.sample_actions(
            context, num_steps=self.runtime.fm_steps, generator=self._generator(info),
        )
        self._last_se2_deltas = deltas[0].float().cpu().numpy().copy()
        self._last_waypoints = waypoints[0].float().cpu().numpy().copy()
        if not np.isfinite(self._last_waypoints).all():
            raise RuntimeError("FM-only produced non-finite waypoints")
        index, waypoint = select_action_waypoint(self._last_waypoints)
        self._last_action_waypoint_index = int(index)
        self._last_action_waypoint = np.asarray(waypoint, dtype=np.float32).copy()
        action = first_waypoint_to_velocity_cmd(
            waypoint, self.dt, self.lin_vel_range, self.ang_vel_range,
        )
        if self._command_would_stop(action):
            # Habitat treats near-zero velocity as STOP. Keep the requested fixed
            # budget protocol: try later predicted waypoints, then a recovery turn.
            if self.runtime.fm_only_stop_mode == "rvq":
                self._last_stop = True
                self._last_stop_source = "fm_zero_command"
            else:
                self._last_stop_suppressed = True
                self._last_stop_source = "fm_only_zero_command_suppressed"
                action = self._explicit_gate_action(self._last_waypoints)
                if self._command_would_stop(action):
                    raise RuntimeError("velocity limits cannot support a non-STOP recovery action")
        self._step += 1
        return action

    def get_info(self) -> dict[str, Any]:
        result = {
            "action_source": "fm_only", "stop_protocol": self.runtime.fm_only_stop_mode,
            "raw_text": self._last_raw_text, "rvq_codes": self._last_rvq_codes,
            "num_history_frames": self.num_history_frames,
            "stop": self._last_stop, "explicit_stop": self._last_explicit_stop,
            "decode_failed": self._decode_failed,
            "stop_source": self._last_stop_source,
            "stop_suppressed": self._last_stop_suppressed,
            "predicted_traj": self._last_waypoints,
            "fm_se2_deltas": self._last_se2_deltas,
            "action_waypoint_index": self._last_action_waypoint_index,
            "action_waypoint": self._last_action_waypoint,
        }
        if self.runtime.fm_only_stop_mode == "max_steps":
            result.pop("raw_text")
            result.pop("rvq_codes")
        return result
