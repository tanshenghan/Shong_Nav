"""Habitat evaluation runner: drive a remote Habitat env with the VLN-CE velocity policy.

Flow: build the inference engine -> connect to the env server -> loop over
episodes (``reset`` -> per-step ``policy.act`` -> ``env.step``) -> append one
result dict per episode to ``<output_dir>/results.jsonl`` -> print/write the
VLN-CE or ObjectNav summary. Decoding is greedy; no sampling knobs are set.
With ``save_video`` / ``record_dir`` each step's frame is also rendered with the
prediction (``lightnav.viz``) into ``<output_dir>/videos/`` and/or recorded raw.

The env, policy and engine constructors can be injected so the loop can be
exercised without a GPU or model.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lightnav.eval_config import get_task_params, load_eval_config, resolve_asset_path
from lightnav.habitat.policy import TrajVocabVLNCEPolicy, extract_instruction
from lightnav.habitat.results import (
    make_json_safe,
    print_objectnav_summary,
    print_vlnce_summary,
)
from lightnav.serving.protocol import pointing_payload
from lightnav.serving.token_budget import decode_token_budget, probe_grounding_tokens
from lightnav.tracking import load_centroids
from lightnav.traj_vocab import load_rvq_bundle
from lightnav.vln_utils import DEFAULT_TRAJ_HORIZON, DEFAULT_TRAJ_K

logger = logging.getLogger(__name__)

_CENTROIDS_FILENAME = "centroids_whole_chunk_K{K}_h{H}.npy"


@dataclass
class HabitatEvalConfig:
    """Settings for one evaluation run (mirrored by the ``lightnav-eval-habitat`` CLI)."""

    model_path: str
    server: str = "tcp://localhost:5555"
    backend: str = "vllm_local"
    action_output: str = "token"  # token | fm (BetaNav continuous expert)
    beta_checkpoint: str = ""  # checkpoint dir/model.pt or run dir with latest pointers
    fm_only_stop_mode: str | None = None  # default: learned for fm_stop, max_steps otherwise
    stop_threshold: float = 0.5  # learned StopHead score threshold
    stop_confirm_steps: int = 1  # consecutive observations above the learned threshold
    fm_seed: int = 42
    fm_steps: int | None = None  # None = checkpoint's configured integration steps
    fm_stop_mode: str = "baseline"  # baseline | decode_continue | explicit_gate
    episodes: int = -1  # <= 0: full split, stop at the first repeated episode key
    episode_manifest: str = ""  # optional JSON subset keyed by scene_id + episode_id
    oracle_stop_distance: float | None = None  # privileged diagnostic STOP radius
    max_steps: int = 500
    # Simulation-only: when the step budget is exhausted without the policy having
    # stopped, the LAST action of the budget is an explicit STOP (zero velocity) so the
    # episode ends by ``agent_stop`` and Success/SPL are measured at the final pose
    # instead of being zeroed by the step-limit truncation. Never applies to the
    # WebSocket serving path.
    force_stop_at_max_steps: bool = True
    output_dir: str = "output/habitat_eval"
    languages: list[str] | None = None  # RxR only, e.g. ["en-US", "en-IN"]
    traj_vocab_path: str | None = None  # flat vocab dir or centroids .npy
    K: int | None = None
    horizon: int | None = None
    action_tokenizer_bundle: str | None = None  # RVQ bundle dir
    gpu_memory_utilization: float = 0.65
    max_num_seqs: int = 1
    num_history_frames: int | None = None
    aspect_mode: str = "stretch"  # stretch | keep (see InferenceConfig)
    # Per-step decode cap. Greedy decoding stops at eos, so this only truncates; 64 is
    # the reference value and leaves room for any grounding prefix + action tokens.
    max_new_tokens: int = 64
    zmq_timeout_ms: int = 600000
    verbose: bool = False
    # Visualisation (docs/VISUALIZATION.md). `save_video` needs the `video` extra and writes
    # <output_dir>/videos/<episode_id>.mp4 with one frame per policy step; `record_dir`
    # additionally records the raw episodes in the layout `lightnav-render` reads.
    save_video: bool = False
    video_fps: int = 10
    hfov_deg: float = 120.0  # agent camera, as in habitat_server/configs/*.yaml
    cam_height: float = 0.88  # metres above the floor, as in the yamls
    waypoint_dt_s: float = 0.1  # seconds per waypoint row assumed by the HUD readout only
    record_dir: str = ""


# ---------------------------------------------------------------------------
# Action decoder resolution
# ---------------------------------------------------------------------------


def _manifest_horizon(bundle_dir: Path) -> int | None:
    manifest = bundle_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        horizon = json.loads(manifest.read_text()).get("horizon")
    except (OSError, ValueError):
        return None
    return int(horizon) if horizon else None


def _load_bundle(bundle_dir: Path, horizon: int | None):
    if horizon is None:
        horizon = _manifest_horizon(bundle_dir) or DEFAULT_TRAJ_HORIZON
    return load_rvq_bundle(bundle_dir, int(horizon), num_frames=0, load_cluster_ids=False)


def resolve_action_decoder(cfg: HabitatEvalConfig, bundle: Any) -> tuple[np.ndarray | None, Any]:
    """Return ``(centroids, rvq_bundle)`` with exactly one of them set.

    Resolution order:
      1. explicit ``cfg.action_tokenizer_bundle`` or ``cfg.traj_vocab_path`` (+ ``K``/``horizon``,
         defaulting to the checkpoint's ``eval_config.json`` ``tasks.vlnce`` values);
      2. the ``eval_config.json`` snapshot: a task with ``action_tokenizer.method == "rvq"``
         whose ``bundle_path`` exists, else a task whose ``traj_vocab_path`` holds the
         matching centroids file;
      3. sibling directories ``<model_path>/action_tokenizer`` / ``<model_path>/traj_vocab``.
    """
    saved = load_eval_config(cfg.model_path) or {}
    vlnce_params = get_task_params(saved, "vlnce") if saved else {}
    K = int(cfg.K) if cfg.K else int(vlnce_params.get("traj_vocab_K") or DEFAULT_TRAJ_K)
    horizon: int | None = int(cfg.horizon) if cfg.horizon else None
    if horizon is None and vlnce_params.get("predict_horizon"):
        horizon = int(vlnce_params["predict_horizon"])

    # 1. explicit
    if cfg.action_tokenizer_bundle and cfg.traj_vocab_path:
        raise ValueError("pass only one of --action_tokenizer_bundle and --traj_vocab_path")
    if cfg.action_tokenizer_bundle:
        bundle_dir = Path(cfg.action_tokenizer_bundle)
        print(f"[eval] RVQ bundle: {bundle_dir}", flush=True)
        return None, _load_bundle(bundle_dir, horizon)
    if cfg.traj_vocab_path:
        h = horizon or DEFAULT_TRAJ_HORIZON
        print(f"[eval] traj vocab centroids: {cfg.traj_vocab_path} (K={K}, H={h})", flush=True)
        return load_centroids(cfg.traj_vocab_path, K, h), None

    # 2. eval_config.json snapshot. Only the family the checkpoint declares is
    # considered (an RVQ checkpoint never falls back to flat centroids and vice
    # versa), so a mismatch surfaces here with the tried paths instead of after
    # the first env.reset().
    is_rvq = getattr(bundle, "action_method", "flat") == "rvq"
    tried: list[str] = []
    tasks = [t for t in (saved.get("tasks") or {}).values() if isinstance(t, dict)]
    for task in tasks if is_rvq else []:
        at = task.get("action_tokenizer") or {}
        if at.get("method") != "rvq":
            continue
        bundle_path = at.get("bundle_path")
        task_horizon = int(task.get("predict_horizon") or 0)
        if not bundle_path or not task_horizon:
            continue
        bundle_dir = resolve_asset_path(cfg.model_path, bundle_path)
        if (bundle_dir / "manifest.json").is_file():
            print(f"[eval] RVQ bundle from eval_config.json snapshot: {bundle_dir}", flush=True)
            return None, load_rvq_bundle(
                bundle_dir, task_horizon, num_frames=0, load_cluster_ids=False
            )
        tried.append(str(bundle_dir / "manifest.json"))
    for task in tasks if not is_rvq else []:
        vocab_path = task.get("traj_vocab_path")
        if not vocab_path:
            continue
        k = int(task.get("traj_vocab_K") or 0)
        h = int(task.get("predict_horizon") or 0)
        if not k or not h:
            continue
        candidate = resolve_asset_path(cfg.model_path, vocab_path) / _CENTROIDS_FILENAME.format(K=k, H=h)
        if candidate.is_file():
            print(
                f"[eval] traj vocab centroids from eval_config.json snapshot: {candidate}",
                flush=True,
            )
            return load_centroids(candidate, k, h), None
        tried.append(str(candidate))

    # 3. sibling dirs next to the checkpoint (same family rule)
    model_dir = Path(cfg.model_path)
    if is_rvq:
        for sibling_bundle in (model_dir / "action_tokenizer" / "vlnce", model_dir / "action_tokenizer"):
            if (sibling_bundle / "manifest.json").is_file():
                print(f"[eval] RVQ bundle: {sibling_bundle}", flush=True)
                return None, _load_bundle(sibling_bundle, horizon)
            tried.append(str(sibling_bundle / "manifest.json"))
    else:
        h = horizon or DEFAULT_TRAJ_HORIZON
        sibling_centroids = model_dir / "traj_vocab" / _CENTROIDS_FILENAME.format(K=K, H=h)
        if sibling_centroids.is_file():
            print(f"[eval] traj vocab centroids: {sibling_centroids}", flush=True)
            return load_centroids(sibling_centroids, K, h), None
        tried.append(str(sibling_centroids))

    method = getattr(bundle, "action_method", None)
    hint = (
        "--action_tokenizer_bundle <dir>"
        if method == "rvq"
        else f"--traj_vocab_path <dir-or-.npy> [--K {K} --horizon {h}]"
    )
    raise FileNotFoundError(
        "Could not resolve the action decoder for this checkpoint"
        + (f" (action_method={method!r})" if method else "")
        + f". Tried: {tried}. Pass {hint}."
    )


# ---------------------------------------------------------------------------
# Default factories (real engine / env / policy)
# ---------------------------------------------------------------------------


def _default_engine_factory(cfg: HabitatEvalConfig) -> tuple[Any, Any]:
    from lightnav.inference import InferenceConfig

    inf_cfg = InferenceConfig(
        model_path=cfg.model_path,
        backend=cfg.backend,
        max_new_tokens=cfg.max_new_tokens,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs,
        num_history_frames=cfg.num_history_frames,
        aspect_mode=cfg.aspect_mode,
    )
    if cfg.action_output not in {"token", "fm"}:
        raise ValueError("action_output must be token or fm")
    if cfg.action_output == "fm" or cfg.beta_checkpoint:
        if cfg.backend != "hf":
            raise ValueError(
                "BetaNav checkpoint inference requires --backend hf; BetaModel is not a vLLM backend"
            )
        if not cfg.beta_checkpoint:
            raise ValueError("--action_output fm requires --beta_checkpoint")
        from lightnav.habitat.beta_policy import BetaFMRuntime
        from lightnav.inference.beta_model import load_beta_checkpoint

        model, bundle, path, metadata = load_beta_checkpoint(
            inf_cfg, cfg.beta_checkpoint, task_type="vlnce"
        )
        step = int(metadata.get("step", -1))
        print(f"[eval] BetaNav checkpoint: {path} (step={step}, action_output={cfg.action_output})", flush=True)
        return BetaFMRuntime(
            beta_model=model,
            bundle=bundle,
            checkpoint_path=path,
            checkpoint_step=step,
            max_new_tokens=cfg.max_new_tokens,
            fm_steps=cfg.fm_steps,
            fm_seed=cfg.fm_seed,
            aspect_mode=cfg.aspect_mode,
            stop_mode=cfg.fm_stop_mode,
            action_output=cfg.action_output,
            fm_only_stop_mode=cfg.fm_only_stop_mode,
            stop_threshold=cfg.stop_threshold,
            stop_confirm_steps=cfg.stop_confirm_steps,
            checkpoint_metadata=metadata,
        ), bundle

    if cfg.action_output != "token":
        raise ValueError(f"action_output must be 'token' or 'fm', got {cfg.action_output!r}")
    from lightnav.inference import build_engine

    return build_engine(inf_cfg, task_type="vlnce", max_new_tokens=cfg.max_new_tokens)


def _default_env_factory(cfg: HabitatEvalConfig) -> Any:
    from lightnav.habitat.remote_env import RemoteEnvClient

    return RemoteEnvClient(cfg.server, timeout_ms=cfg.zmq_timeout_ms)


def _apply_decode_budget(engine: Any, bundle: Any, rvq_bundle: Any) -> None:
    """Raise ``engine.max_new_tokens`` to the checkpoint's per-step token budget if needed.

    The cap only truncates (greedy decoding stops at eos), so raising it never changes
    the decoded action; a cap below grounding prefix + action tokens would cut the last
    action token and turn every step into a decode failure. Engines without a
    tokenizer (test doubles) are left untouched.
    """
    tokenizer = getattr(bundle, "tokenizer", None)
    current = getattr(engine, "max_new_tokens", None)
    if tokenizer is None or current is None:
        return
    try:
        budget = decode_token_budget(probe_grounding_tokens(tokenizer), rvq_bundle)
    except Exception as exc:  # pragma: no cover - defensive; keep the configured cap
        logger.warning("could not probe the decode token budget (%s); keeping %s", exc, current)
        return
    if budget > int(current):
        print(
            f"[eval] raising max_new_tokens {current} -> {budget} (grounding prefix + action tokens)",
            flush=True,
        )
        engine.max_new_tokens = int(budget)


def build_velocity_policy(
    cfg: HabitatEvalConfig, engine: Any, bundle: Any, info: dict[str, Any]
) -> Any:
    """Build the policy from the velocity-control settings the env reports on reset."""
    missing = [k for k in ("habitat_time_step", "lin_vel_range", "ang_vel_range") if k not in info]
    if missing:
        raise KeyError(
            f"Habitat env did not expose velocity-control config keys: {missing}. "
            "The env server must report habitat_time_step / lin_vel_range / ang_vel_range in info."
        )
    model_config = getattr(getattr(engine, "beta_model", None), "config", None)
    training_mode = getattr(model_config, "training_mode", None)
    if training_mode in {"fm_only", "fm_stop"}:
        if cfg.action_output != "fm" or cfg.fm_stop_mode != "baseline":
            raise ValueError("continuous Beta evaluation requires FM output and baseline configuration")
        if (training_mode == "fm_only"
                and (cfg.oracle_stop_distance is not None or not cfg.force_stop_at_max_steps)):
            raise ValueError("fm_only evaluation uses forced STOP at max_steps, without oracle STOP")
        from lightnav.habitat.beta_policy import BetaFMOnlyVLNCEPolicy

        engine.fm_only_stop_mode = cfg.fm_only_stop_mode or (
            "learned" if training_mode == "fm_stop" else "max_steps"
        )
        engine.stop_threshold = cfg.stop_threshold
        engine.stop_confirm_steps = cfg.stop_confirm_steps
        stop_bundle = None
        if engine.fm_only_stop_mode == "rvq":
            centroids, stop_bundle = resolve_action_decoder(cfg, bundle)
            if centroids is not None or stop_bundle is None:
                raise ValueError("FM-only RVQ STOP requires an RVQ decoder")
            _apply_decode_budget(engine, bundle, stop_bundle)
        return BetaFMOnlyVLNCEPolicy(
            engine,
            dt=float(info["habitat_time_step"]),
            lin_vel_range=tuple(info["lin_vel_range"]),
            ang_vel_range=tuple(info["ang_vel_range"]),
            num_history_frames=int(cfg.num_history_frames or bundle.num_history_frames),
            rvq_bundle=stop_bundle,
            min_abs_lin_speed=float(info.get("min_abs_lin_speed", 0.01)),
            min_abs_ang_speed=float(info.get("min_abs_ang_speed", 0.5)),
        )
    centroids, rvq_bundle = resolve_action_decoder(cfg, bundle)
    _apply_decode_budget(engine, bundle, rvq_bundle)
    num_history_frames = (
        int(cfg.num_history_frames)
        if cfg.num_history_frames is not None
        else int(bundle.num_history_frames)
    )
    if cfg.action_output == "fm" or cfg.beta_checkpoint:
        if centroids is not None or rvq_bundle is None:
            raise ValueError("Beta FM evaluation requires an RVQ action-tokenizer bundle for STOP")
        from lightnav.habitat.beta_policy import BetaFMVLNCEPolicy

        return BetaFMVLNCEPolicy(
            engine,
            dt=float(info["habitat_time_step"]),
            lin_vel_range=tuple(float(x) for x in info["lin_vel_range"]),
            ang_vel_range=tuple(float(x) for x in info["ang_vel_range"]),
            num_history_frames=num_history_frames,
            rvq_bundle=rvq_bundle,
            min_abs_lin_speed=float(info.get("min_abs_lin_speed", 0.01)),
            min_abs_ang_speed=float(info.get("min_abs_ang_speed", 0.5)),
        )
    return TrajVocabVLNCEPolicy(
        engine,
        centroids,
        dt=float(info["habitat_time_step"]),
        lin_vel_range=tuple(float(x) for x in info["lin_vel_range"]),
        ang_vel_range=tuple(float(x) for x in info["ang_vel_range"]),
        num_history_frames=num_history_frames,
        rvq_bundle=rvq_bundle,
    )


# ---------------------------------------------------------------------------
# Visualisation: per-episode overlay videos + optional raw episode recording
# ---------------------------------------------------------------------------

_VIDEO_MODULES = ("cv2", "imageio", "imageio_ffmpeg")
_VIDEOS_SUBDIR = "videos"


def _require_video_deps() -> None:
    """Fail fast (before the engine loads) when ``save_video`` lacks the ``video`` extra."""
    missing = []
    for name in _VIDEO_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise ImportError(
            f"--save_video needs {', '.join(missing)} (missing); install the video extra: "
            "pip install 'lightnav[video]'"
        )


class _EpisodeVideo:
    """One per-episode mp4: opened on the first frame, renamed into place on ``close``.

    Frames are written under a temporary name so a crash never leaves a truncated
    file that looks finished. Every frame is padded to even dimensions (yuv420p).
    """

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.fps = int(fps)
        self._partial = path.with_name(path.stem + ".partial" + path.suffix)
        self._writer: Any = None
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        from lightnav.viz import open_video_writer, pad_to_even_dimensions

        frame, _pad_right, _pad_bottom = pad_to_even_dimensions(np.ascontiguousarray(frame))
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = open_video_writer(self._partial, self.fps)
        self._writer.append_data(frame)
        self.frames += 1

    def abort(self) -> None:
        """Discard the partial file after an encoding failure."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001 - the writer is already broken
                pass
            self._writer = None
        self._partial.unlink(missing_ok=True)

    def close(self) -> Path | None:
        """Finish the file; return its final path, or None when nothing was written."""
        if self._writer is None:
            return None
        try:
            self._writer.close()
        finally:
            self._writer = None
        if self.frames == 0:
            self._partial.unlink(missing_ok=True)
            return None
        os.replace(self._partial, self.path)
        return self.path


class _EvalVisualizer:
    """Renders the overlay for the observation the policy acted on, and records raw steps.

    Built only when ``cfg.save_video`` or ``cfg.record_dir`` is set, so ``lightnav.viz``
    is imported only then. The video timeline is ``per_step`` (one frame per policy
    step); the recorder gets the same frames JPEG-encoded plus the per-step record, in
    the layout ``lightnav-render`` reads.
    """

    def __init__(self, cfg: HabitatEvalConfig, output_dir: str) -> None:
        self.cfg = cfg
        self.videos_dir = Path(output_dir) / _VIDEOS_SUBDIR
        self._video: _EpisodeVideo | None = None
        self._video_failed = False
        self._last_waypoints: np.ndarray | None = None
        self._last_diagnostics: dict[str, Any] = {}
        self._recorder: Any = None
        self._conn: Any = None
        if cfg.save_video:
            _require_video_deps()
        from lightnav.viz import encode_jpeg_bytes, render_frame

        self._render_frame = render_frame
        self._encode_jpeg_bytes = encode_jpeg_bytes
        if cfg.record_dir:
            from lightnav.viz import EpisodeRecorder

            self._recorder = EpisodeRecorder(
                Path(cfg.record_dir),
                task="habitat",
                model_path=cfg.model_path,
                hfov_deg=float(cfg.hfov_deg),
                cam_height=float(cfg.cam_height),
                video_fps=int(cfg.video_fps),
                timeline="per_step",
                waypoint_dt_s=float(cfg.waypoint_dt_s),
            )
            self._conn = self._recorder.begin_connection(label="eval")

    @property
    def record_run_dir(self) -> Path | None:
        return getattr(self._recorder, "run_dir", None) if self._recorder is not None else None

    def begin_episode(self, episode_id: str) -> None:
        self._last_waypoints = None
        self._last_diagnostics = {}
        self._video_failed = False
        if self.cfg.save_video:
            self._video = _EpisodeVideo(self.videos_dir / f"{episode_id}.mp4", self.cfg.video_fps)
        if self._conn is not None:
            self._conn.begin_episode()

    def _render(
        self,
        rgb: np.ndarray,
        *,
        waypoints: np.ndarray | None,
        instruction: str,
        step: int,
        stop: bool,
        pointing: dict | None,
        diagnostics: dict[str, Any] | None = None,
    ) -> np.ndarray:
        frame = self._render_frame(
            rgb,
            waypoints=waypoints,
            instruction=instruction,
            step=step,
            step_fps=None,
            stop=stop,
            pointing=pointing,
            hfov_deg=self.cfg.hfov_deg,
            cam_height=self.cfg.cam_height,
            dt_s=self.cfg.waypoint_dt_s,
        )
        if diagnostics:
            frame = self._diagnostic_overlay(frame, diagnostics)
        return self._instruction_banner(frame, instruction)

    @staticmethod
    def _instruction_banner(frame: np.ndarray, instruction: str) -> np.ndarray:
        """Append the complete instruction below the camera image without truncation.

        This affects evaluation videos only. The policy input and recorded raw RGB
        retain the camera resolution, and every frame in an episode has the same size.
        """
        from PIL import Image, ImageDraw, ImageFont

        text = " ".join(str(instruction or "").split())
        if not text:
            return frame
        height, width = frame.shape[:2]
        padding = max(2, min(12, width // 40))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, height // 30))
        except OSError:
            font = ImageFont.load_default()
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        available = max(1, width - 2 * padding)
        lines: list[str] = []
        line = ""
        # Break at spaces where possible; character fallback preserves long words.
        for word in ("Instruction: " + text).split():
            candidate = f"{line} {word}" if line else word
            if measure.textlength(candidate, font=font) <= available:
                line = candidate
                continue
            if line:
                lines.append(line)
                line = ""
            for character in word:
                if line and measure.textlength(line + character, font=font) > available:
                    lines.append(line)
                    line = ""
                line += character
        if line:
            lines.append(line)
        top, bottom = font.getbbox("Ag")[1::2]
        line_height = max(16, bottom - top + 5)
        banner = Image.new("RGB", (width, len(lines) * line_height + 2 * padding), (8, 12, 20))
        draw = ImageDraw.Draw(banner)
        for index, line in enumerate(lines):
            draw.text((padding, padding + index * line_height - top), line,
                      font=font, fill=(235, 245, 255))
        return np.concatenate((frame, np.asarray(banner)), axis=0)

    @staticmethod
    def _diagnostic_overlay(frame: np.ndarray, diagnostics: dict[str, Any]) -> np.ndarray:
        """Add goal-distance and STOP provenance used by the failure analysis videos."""
        import cv2

        out = np.ascontiguousarray(frame.copy())
        height, width = out.shape[:2]
        scale = max(0.38, height / 720.0)
        line_height = max(15, int(round(22 * scale)))
        lines = [
            (
                f"DTG {float(diagnostics.get('distance_to_goal', float('nan'))):.2f}m  "
                f"MIN {float(diagnostics.get('min_distance', float('nan'))):.2f}m"
            ),
            (
                f"STOP {diagnostics.get('stop_source') or '-'}  "
                f"TOKEN {int(bool(diagnostics.get('explicit_stop')))}  "
                f"DECODE_FAIL {int(bool(diagnostics.get('decode_failed')))}"
            ),
            f"MODE {diagnostics.get('fm_stop_mode', 'baseline')}",
        ]
        if diagnostics.get("stop_score") is not None:
            threshold = diagnostics.get("stop_threshold")
            threshold_text = "-" if threshold is None else f"{float(threshold):.3f}"
            lines.insert(2, f"P(STOP) {float(diagnostics['stop_score']):.3f}  "
                         f"THRESHOLD {threshold_text}")
        panel_width = min(width - 8, max(300, int(round(460 * scale))))
        panel_height = line_height * len(lines) + 10
        x0, y0 = 4, max(4, height - panel_height - 4)
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_width, y0 + panel_height), (8, 8, 8), -1)
        cv2.addWeighted(overlay, 0.68, out, 0.32, 0.0, out)
        for index, line in enumerate(lines):
            cv2.putText(
                out,
                line,
                (x0 + 7, y0 + 7 + line_height * (index + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (235, 245, 255),
                max(1, int(round(scale))),
                cv2.LINE_AA,
            )
        return out

    def _write(self, frame: np.ndarray) -> None:
        if self._video is None or self._video_failed:
            return
        try:
            self._video.write(frame)
        except Exception as exc:
            # Diagnostics must not cost the evaluation: drop this episode's video and go on.
            self._video_failed = True
            logger.warning(
                "video write failed for %s (%s: %s); the episode has no video",
                self._video.path, type(exc).__name__, exc,
            )

    def step(
        self,
        rgb: np.ndarray,
        *,
        step: int,
        instruction: str,
        policy_info: dict[str, Any],
        latency_ms: float,
        diagnostics: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Overlay + record one policy step; ``rgb`` is the frame the policy acted on."""
        waypoints = policy_info.get("predicted_traj")  # None after a failed decode
        if waypoints is not None:
            waypoints = np.asarray(waypoints, dtype=np.float32)
        raw_text = str(policy_info.get("raw_text", "") or "")
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        pointing = pointing_payload(raw_text, width=width, height=height)
        stop = bool(
            policy_info.get(
                "stop", waypoints is not None and not np.any(np.abs(waypoints) > 1e-6)
            )
        )
        self._last_waypoints = waypoints
        self._last_diagnostics = dict(diagnostics or {})

        if self._video is not None:
            frame = self._render(
                rgb, waypoints=waypoints, instruction=instruction, step=step, stop=stop,
                pointing=pointing, diagnostics=diagnostics,
            )
            self._write(frame)

        if self._conn is not None:
            try:
                self._conn.record_step(
                    step=step,
                    seq=step,
                    image=self._encode_jpeg_bytes(rgb),
                    instruction=instruction,
                    waypoints=None if waypoints is None else waypoints.tolist(),
                    stop=bool(stop),
                    visible=None,
                    raw_text=raw_text,
                    latency_ms=float(latency_ms),
                    pointing=pointing,
                    **{
                        key: value
                        for key, value in (diagnostics or {}).items()
                        if key != "stop"
                    },
                    **extra,
                )
            except Exception as exc:
                logger.warning("episode recording failed at step %d: %s: %s", step, type(exc).__name__, exc)

    def final_frame(
        self,
        rgb: np.ndarray,
        *,
        step: int,
        instruction: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Append the terminal observation (with the last prediction) so the end is visible."""
        if self._video is None or self._video.frames == 0:
            return
        waypoints = self._last_waypoints
        final_diagnostics = diagnostics or self._last_diagnostics
        stop = bool(
            final_diagnostics.get(
                "stop", waypoints is not None and not np.any(np.abs(waypoints) > 1e-6)
            )
        )
        frame = self._render(
            rgb,
            waypoints=waypoints,
            instruction=instruction,
            step=step,
            stop=stop,
            pointing=None,
            diagnostics=final_diagnostics,
        )
        self._write(frame)

    def end_episode(self) -> str | None:
        """Close the episode; return the video path relative to ``output_dir`` if written."""
        video_rel: str | None = None
        if self._video is not None:
            written = None
            if self._video_failed:
                self._video.abort()
            else:
                try:
                    written = self._video.close()
                except Exception as exc:
                    logger.warning(
                        "could not finish %s: %s: %s", self._video.path, type(exc).__name__, exc
                    )
                    self._video.abort()
            if written is not None:
                video_rel = f"{_VIDEOS_SUBDIR}/{written.name}"
            self._video = None
        if self._conn is not None:
            try:
                self._conn.end_episode()
            except Exception as exc:
                logger.warning("could not finish the episode recording: %s: %s", type(exc).__name__, exc)
        return video_rel

    def close(self) -> None:
        self.end_episode()
        if self._recorder is not None:
            try:
                self._recorder.close()
            except Exception as exc:
                logger.warning("recorder shutdown failed: %s: %s", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


def _safe_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else float(default)
    try:
        return float(value[0])
    except Exception:
        try:
            return float(value)
        except Exception:
            return float(default)


def _format_action(action: Any) -> str:
    if isinstance(action, dict):
        args = action.get("action_args", {}) or {}
        lin = float(args.get("linear_velocity", float("nan")))
        ang = float(args.get("angular_velocity", float("nan")))
        return f"VEL(lin={lin:+.2f},ang={ang:+.2f})"
    return str(action)


def _load_episode_manifest(path: str) -> set[tuple[str, str]] | None:
    """Load a diagnostic episode subset keyed by exact ``(scene_id, episode_id)``."""
    if not path:
        return None
    manifest_path = Path(path).expanduser()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("episodes") if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ValueError("episode manifest must be a non-empty list or contain an episodes list")
    keys: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "scene_id" not in entry:
            raise ValueError(f"episode manifest entry {index} needs scene_id and episode_id")
        episode_id = entry.get("habitat_episode_id", entry.get("episode_id"))
        if episode_id is None:
            raise ValueError(f"episode manifest entry {index} needs scene_id and episode_id")
        key = (str(entry["scene_id"]), str(episode_id))
        if key in keys:
            raise ValueError(f"duplicate episode manifest key: {key!r}")
        keys.add(key)
    return keys


def run_habitat_eval(
    cfg: HabitatEvalConfig,
    *,
    env_factory: Callable[[HabitatEvalConfig], Any] | None = None,
    policy_factory: Callable[[HabitatEvalConfig, Any, Any, dict], Any] | None = None,
    engine_factory: Callable[[HabitatEvalConfig], tuple[Any, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the evaluation and return the per-episode result dicts.

    ``engine_factory(cfg) -> (engine, bundle)``, ``env_factory(cfg) -> env`` with
    ``reset()/step()/close()``, ``policy_factory(cfg, engine, bundle, info) -> policy``
    with ``reset(obs)/act(obs, info)/get_info()``. Each defaults to the real one.
    """
    engine_factory = engine_factory or _default_engine_factory
    env_factory = env_factory or _default_env_factory
    policy_factory = policy_factory or build_velocity_policy
    episode_filter = _load_episode_manifest(cfg.episode_manifest)
    if type(cfg.stop_confirm_steps) is not int or cfg.stop_confirm_steps < 1:
        raise ValueError("stop_confirm_steps must be a positive integer")
    if cfg.stop_confirm_steps > 1 and cfg.action_output != "fm":
        raise ValueError("stop_confirm_steps > 1 requires learned FM STOP")
    if cfg.oracle_stop_distance is not None and cfg.oracle_stop_distance <= 0:
        raise ValueError("oracle_stop_distance must be positive")

    output_dir = cfg.output_dir or "output/habitat_eval"
    os.makedirs(output_dir, exist_ok=True)
    results_jsonl = os.path.join(output_dir, "results.jsonl")

    # Before the engine: a missing video dependency should fail in seconds, not after
    # the weights are loaded.
    viz = _EvalVisualizer(cfg, output_dir) if (cfg.save_video or cfg.record_dir) else None

    engine, bundle = engine_factory(cfg)
    env = env_factory(cfg)

    sep = "=" * 60
    print(f"\n{sep}")
    print("Habitat Evaluation")
    print(sep)
    print(f"  Model:    {cfg.model_path}")
    print(f"  Server:   {cfg.server}")
    print(f"  Backend:  {cfg.backend}")
    print(f"  Action:   {cfg.action_output}")
    if cfg.beta_checkpoint:
        print(f"  Beta ckpt:{cfg.beta_checkpoint}")
        print(f"  FM stop:  {cfg.fm_stop_mode}")
    print(f"  Episodes: {cfg.episodes if cfg.episodes > 0 else 'full split'}")
    if episode_filter is not None:
        print(f"  Manifest: {cfg.episode_manifest} ({len(episode_filter)} selected globally)")
    if cfg.oracle_stop_distance is not None:
        print(f"  Oracle STOP: distance < {cfg.oracle_stop_distance:.3f}m (diagnostic only)")
    print(f"  Output:   {output_dir}")
    lang_filter = set(cfg.languages) if cfg.languages else None
    if lang_filter:
        print(f"  Language filter: {sorted(lang_filter)}")
    if cfg.save_video:
        print(f"  Videos:   {os.path.join(output_dir, _VIDEOS_SUBDIR)} ({cfg.video_fps} fps, one frame per step)")
    if viz is not None and viz.record_run_dir is not None:
        print(f"  Record:   {viz.record_run_dir}")
    print(f"{sep}\n", flush=True)

    results: list[dict[str, Any]] = []
    skipped_lang = 0
    skipped_manifest = 0
    seen: set[tuple[Any, Any]] = set()
    n_run = 0
    policy = None
    start_time = time.time()

    try:
        while True:
            obs, info = env.reset()

            # (scene_id, episode_id): episode ids are scene-local in HM3D ObjectNav, so
            # a repeat of the pair (not the id alone) marks the env cycling back to the start.
            habitat_ep_id = info.get("episode_id", "")
            habitat_scene_id = info.get("scene_id", "")
            ep_key = (habitat_scene_id, habitat_ep_id)
            if ep_key in seen:
                print(f"Full split completed: {len(seen)} unique episodes seen.", flush=True)
                break
            seen.add(ep_key)

            # Language-filtered episodes are skipped without stepping and do not
            # count toward `episodes` or the episode counter.
            if lang_filter and info.get("language", "") not in lang_filter:
                skipped_lang += 1
                continue
            if episode_filter is not None and (
                str(habitat_scene_id), str(habitat_ep_id)
            ) not in episode_filter:
                skipped_manifest += 1
                continue

            episode_id = f"episode_{n_run:03d}"
            print(
                f"Episode {n_run + 1} ({episode_id}) [{habitat_scene_id} / {habitat_ep_id}]...",
                flush=True,
            )

            instruction = extract_instruction(obs)
            if policy is None:
                policy = policy_factory(cfg, engine, bundle, info)
            policy.reset(obs)
            if viz is not None:
                viz.begin_episode(episode_id)

            final_success: Any = False
            forced_stop_used = False
            oracle_stop_used = False
            stop_source = ""
            policy_event_counts: dict[str, int] = {}
            min_distance_to_goal = float("inf")
            step = -1
            video_rel: str | None = None
            try:
                for step in range(cfg.max_steps):
                    if obs.get("rgb") is None:
                        break
                    t_act = time.monotonic()
                    forced_stop = bool(
                        cfg.force_stop_at_max_steps
                        and step == cfg.max_steps - 1
                        and hasattr(policy, "stop_action")
                    )
                    current_dtg = _safe_scalar(
                        info.get("distance_to_goal"), default=float("inf")
                    )
                    oracle_stop = bool(
                        cfg.oracle_stop_distance is not None
                        and current_dtg < cfg.oracle_stop_distance
                        and hasattr(policy, "stop_action")
                    )
                    if oracle_stop:
                        action = policy.stop_action()
                        oracle_stop_used = True
                        policy_info = {
                            "stop": True,
                            "explicit_stop": False,
                            "decode_failed": False,
                            "stop_source": "oracle_stop",
                        }
                    elif forced_stop:
                        # Step budget exhausted: end the episode with an explicit STOP.
                        action = policy.stop_action()
                        forced_stop_used = True
                        policy_info = {
                            "stop": True,
                            "explicit_stop": False,
                            "decode_failed": False,
                            "stop_source": "max_steps",
                        }
                    else:
                        action = policy.act(obs, info)
                        policy_info = (
                            policy.get_info() if hasattr(policy, "get_info") else {}
                        )
                    act_ms = (time.monotonic() - t_act) * 1000.0
                    if (forced_stop_used or oracle_stop_used) and (
                        getattr(engine, "fm_only_stop_mode", None) == "learned"
                    ):
                        policy_info.update(
                            stop_confirm_steps=cfg.stop_confirm_steps,
                            stop_confirmed=False,
                        )
                    event_source = str(policy_info.get("stop_source", "") or "")
                    if event_source:
                        policy_event_counts[event_source] = policy_event_counts.get(event_source, 0) + 1
                    if policy_info.get("decode_failed"):
                        policy_event_counts["decode_failure_steps"] = (
                            policy_event_counts.get("decode_failure_steps", 0) + 1
                        )
                    if policy_info.get("stop_suppressed"):
                        policy_event_counts["stop_suppressed_steps"] = (
                            policy_event_counts.get("stop_suppressed_steps", 0) + 1
                        )
                    if policy_info.get("stop_candidate"):
                        key = "stop_candidate_steps"
                        policy_event_counts[key] = policy_event_counts.get(key, 0) + 1
                        if not policy_info.get("stop_confirmed"):
                            key = "stop_pending_steps"
                            policy_event_counts[key] = policy_event_counts.get(key, 0) + 1
                    if cfg.verbose:
                        print(
                            f"    Step {step}: {_format_action(action)}"
                            f" raw={policy_info.get('raw_text', '')!r}"
                            f" traj_id={policy_info.get('cluster_id')}"
                            f" wp_idx={policy_info.get('action_waypoint_index')}",
                            flush=True,
                        )
                    if viz is not None:
                        # The overlay belongs to the frame the policy acted on, so it is
                        # rendered before env.step replaces the observation.
                        viz.step(
                            obs["rgb"],
                            step=step,
                            instruction=instruction,
                            policy_info=policy_info,
                            latency_ms=act_ms,
                            diagnostics={
                                "distance_to_goal": current_dtg,
                                "min_distance": min(min_distance_to_goal, current_dtg),
                                "stop": bool(policy_info.get("stop")),
                                "stop_source": event_source,
                                "stop_score": policy_info.get("stop_score"),
                                "stop_threshold": policy_info.get("stop_threshold"),
                                **{key: policy_info[key] for key in (
                                    "stop_candidate", "stop_confirmation_count",
                                    "stop_confirm_steps", "stop_confirmed",
                                ) if key in policy_info},
                                "explicit_stop": bool(policy_info.get("explicit_stop")),
                                "decode_failed": bool(policy_info.get("decode_failed")),
                                "stop_suppressed": bool(policy_info.get("stop_suppressed")),
                                "fm_stop_mode": cfg.fm_stop_mode,
                            },
                            episode_id=episode_id,
                            habitat_episode_id=str(habitat_ep_id),
                            scene_id=str(habitat_scene_id),
                        )
                    obs, _reward, terminated, truncated, info = env.step(action)

                    dtg = _safe_scalar(info.get("distance_to_goal"), default=float("inf"))
                    if dtg < min_distance_to_goal:
                        min_distance_to_goal = dtg

                    if terminated or truncated:
                        final_success = info.get("success", False)
                        if oracle_stop_used:
                            stop_source = "oracle_stop"
                        elif forced_stop_used:
                            stop_source = "max_steps"
                        elif event_source:
                            stop_source = event_source
                        elif terminated:
                            stop_source = "environment_stop"
                        else:
                            stop_source = "environment_truncated"
                        break
                if viz is not None and obs.get("rgb") is not None:
                    final_dtg = _safe_scalar(
                        info.get("distance_to_goal"), default=float("inf")
                    )
                    viz.final_frame(
                        obs["rgb"],
                        step=step + 1,
                        instruction=instruction,
                        diagnostics={
                            "distance_to_goal": final_dtg,
                            "min_distance": min(min_distance_to_goal, final_dtg),
                            "stop": True,
                            "stop_source": stop_source,
                            "stop_score": policy_info.get("stop_score"),
                            "stop_threshold": policy_info.get("stop_threshold"),
                            **{key: policy_info[key] for key in (
                                "stop_candidate", "stop_confirmation_count",
                                "stop_confirm_steps", "stop_confirmed",
                            ) if key in policy_info},
                            "explicit_stop": bool(policy_info.get("explicit_stop")),
                            "decode_failed": bool(policy_info.get("decode_failed")),
                            "stop_suppressed": bool(policy_info.get("stop_suppressed")),
                            "fm_stop_mode": cfg.fm_stop_mode,
                        },
                    )
            finally:
                if viz is not None:
                    video_rel = viz.end_episode()

            oracle_success_metric = info.get("oracle_success")
            if oracle_success_metric is None:
                fallback_success_distance = 0.1 if info.get("object_category") else 3.0
                oracle_success = min_distance_to_goal < fallback_success_distance
            else:
                oracle_success = _safe_scalar(oracle_success_metric) > 0.0

            result = {
                "episode_id": episode_id,
                "habitat_episode_id": habitat_ep_id,
                "raw_episode_id": str(info.get("raw_episode_id", "")),
                "scene_id": habitat_scene_id,
                "rollout_idx": 0,
                "success": final_success,
                "oracle_success": oracle_success,
                "spl": info.get("spl", 0.0),
                "ndtw": info.get("ndtw", 0.0),
                "soft_spl": float(info.get("soft_spl", 0.0)),
                "object_category": str(info.get("object_category", "")),
                "steps": step + 1,
                "final_distance": info.get("distance_to_goal", float("inf")),
                "min_distance": min_distance_to_goal,
                "instruction": instruction,
                "termination_reason": info.get("termination_reason", "unknown"),
                "forced_stop": forced_stop_used,
                "oracle_stop": oracle_stop_used,
                "stop_source": stop_source or "unknown",
                "policy_event_counts": policy_event_counts,
                "termination_details": info.get("termination_details", {}),
            }
            if policy_info.get("stop_score") is not None:
                result.update(
                    stop_score=policy_info["stop_score"],
                    stop_threshold=policy_info["stop_threshold"],
                    **{key: policy_info[key] for key in (
                        "stop_candidate", "stop_confirmation_count",
                        "stop_confirm_steps", "stop_confirmed",
                    ) if key in policy_info},
                )
            if policy_info.get("stop_confirm_steps") is not None:
                result.update(
                    stop_confirm_steps=policy_info["stop_confirm_steps"],
                    stop_confirmed=bool(policy_info.get("stop_confirmed")),
                )
            if video_rel is not None:
                result["video"] = video_rel
            results.append(result)
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(make_json_safe(result)) + "\n")
                f.flush()
                os.fsync(f.fileno())

            status = "OK" if result["success"] else "FAIL"
            print(
                f"  [{status}] success={bool(result['success'])}, "
                f"oracle={'True' if oracle_success else 'False'}, "
                f"min_dist={min_distance_to_goal:.2f}m, "
                f"steps={result['steps']}, "
                f"dist={float(result['final_distance']):.2f}m, "
                f"SPL={float(result['spl']):.3f}, "
                f"NDTW={float(result['ndtw']):.3f}",
                flush=True,
            )

            n_run += 1
            if cfg.episodes > 0 and n_run >= cfg.episodes:
                print(f"Reached --episodes {cfg.episodes}.", flush=True)
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if viz is not None:
            viz.close()
        env.close()

    elapsed = time.time() - start_time
    if skipped_lang > 0:
        print(f"\nSkipped {skipped_lang} episodes due to language filter.")
    if skipped_manifest > 0:
        print(f"Skipped {skipped_manifest} episodes outside the episode manifest.")

    extra_info = {
        "model": cfg.model_path,
        "backend": cfg.backend,
        "action_output": cfg.action_output,
    }
    if cfg.beta_checkpoint:
        extra_info.update(
            beta_checkpoint=str(getattr(engine, "checkpoint_path", cfg.beta_checkpoint)),
            beta_checkpoint_step=int(getattr(engine, "checkpoint_step", -1)),
            fm_seed=cfg.fm_seed,
            fm_steps=cfg.fm_steps,
            fm_stop_mode=cfg.fm_stop_mode,
        )
    beta_config = getattr(getattr(engine, "beta_model", None), "config", None)
    if getattr(beta_config, "training_mode", None) in {"fm_only", "fm_stop"}:
        extra_info.update(
            training_mode=beta_config.training_mode,
            stop_protocol=engine.fm_only_stop_mode, max_steps=cfg.max_steps,
        )
        if engine.fm_only_stop_mode == "learned":
            metadata = getattr(engine, "checkpoint_metadata", None) or {}
            extra_info.update(
                stop_threshold=cfg.stop_threshold,
                stop_confirm_steps=cfg.stop_confirm_steps,
                stop_head_type=getattr(beta_config, "stop_head_type", None),
                stop_positive_weight=getattr(beta_config, "stop_positive_weight", None),
                stop_training_data_sha256=metadata.get("contract", {}).get("data_sha256"),
            )
            if "stop_supervision" in metadata.get("contract", {}):
                extra_info["stop_supervision"] = metadata["contract"]["stop_supervision"]
    if cfg.episode_manifest:
        extra_info["episode_manifest"] = str(Path(cfg.episode_manifest).resolve())
    if cfg.oracle_stop_distance is not None:
        extra_info["oracle_stop_distance"] = cfg.oracle_stop_distance
    if any(r.get("object_category") for r in results):
        print_objectnav_summary(results, elapsed, output_dir, extra_info=extra_info)
    else:
        print_vlnce_summary(results, elapsed, output_dir, extra_info=extra_info)
    return results
