"""``lightnav-eval-habitat``: evaluate a checkpoint against a running Habitat env server.

Example::

    lightnav-eval-habitat --model_path /path/to/checkpoint \
        --server tcp://localhost:5555 --episodes -1 --output_dir output/r2r

See ``docs/EVAL_HABITAT.md`` for the full guide.
"""

from __future__ import annotations

import argparse
import os
import sys

from lightnav.habitat.runner import HabitatEvalConfig, run_habitat_eval

# Argparse defaults come from the dataclass so the CLI and the Python API cannot drift.
_DEFAULTS = HabitatEvalConfig(model_path="")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lightnav-eval-habitat",
        description="Run VLN-CE (R2R/RxR) or ObjectNav (HM3D/OVON) evaluation against a "
        "Habitat env server over ZMQ.",
    )
    p.add_argument("--model_path", required=True, help="Checkpoint directory.")
    p.add_argument("--server", default=_DEFAULTS.server, help="Habitat env server address.")
    p.add_argument("--backend", choices=["hf", "vllm_local"], default=_DEFAULTS.backend)
    p.add_argument(
        "--action_output",
        choices=["token", "fm"],
        default=_DEFAULTS.action_output,
        help="Use token/RVQ (base model or --beta_checkpoint) or the BetaNav FM trajectory. FM still "
        "uses the trained token branch for STOP and requires --backend hf.",
    )
    p.add_argument(
        "--beta_checkpoint",
        default=_DEFAULTS.beta_checkpoint,
        help="For token or fm output: model.pt, committed checkpoint directory, or BetaNav "
        "run directory whose latest committed pointer should be resolved.",
    )
    p.add_argument(
        "--fm_only_stop_mode",
        choices=["max_steps", "rvq"],
        default=_DEFAULTS.fm_only_stop_mode,
        help="FM-only checkpoints: fixed budget, or frozen LightNav RVQ baseline STOP.",
    )
    p.add_argument(
        "--fm_seed",
        type=int,
        default=_DEFAULTS.fm_seed,
        help="Deterministic per-episode/per-step FM noise seed.",
    )
    p.add_argument(
        "--fm_steps",
        type=int,
        default=_DEFAULTS.fm_steps,
        help="Override the FM Euler integration steps (default: checkpoint beta_config).",
    )
    p.add_argument(
        "--fm_stop_mode",
        choices=["baseline", "decode_continue", "explicit_gate"],
        default=_DEFAULTS.fm_stop_mode,
        help="Beta FM stopping ablation: reproduce the baseline, continue FM after malformed "
        "RVQ tokens, or allow only the canonical explicit STOP tuple to terminate.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=_DEFAULTS.episodes,
        help="Number of episodes to run; <= 0 runs the full split (until the env wraps).",
    )
    p.add_argument(
        "--episode_manifest",
        default=_DEFAULTS.episode_manifest,
        help="Optional JSON list (or object with episodes) selecting exact scene_id + "
        "habitat_episode_id pairs.",
    )
    p.add_argument(
        "--oracle_stop_distance",
        type=float,
        default=_DEFAULTS.oracle_stop_distance,
        help="Diagnostic only: issue STOP when the privileged goal distance is below this "
        "radius. Never use this as a deployable metric.",
    )
    p.add_argument(
        "--max_steps", type=int, default=_DEFAULTS.max_steps, help="Per-episode step cap."
    )
    p.add_argument(
        "--no_force_stop",
        action="store_true",
        help="Do not send an explicit STOP as the last action of the step budget; let the "
        "env truncate instead (default: force STOP at --max_steps).",
    )
    p.add_argument("--output_dir", default=_DEFAULTS.output_dir)
    p.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="RxR only: keep episodes whose info['language'] is in this list, e.g. en-US en-IN.",
    )
    p.add_argument(
        "--traj_vocab_path",
        default=None,
        help="Flat trajectory vocabulary: directory holding centroids_whole_chunk_K{K}_h{H}.npy "
        "or the .npy file itself.",
    )
    p.add_argument("--K", type=int, default=None, help="Vocabulary size for --traj_vocab_path.")
    p.add_argument("--horizon", type=int, default=None, help="Trajectory horizon H.")
    p.add_argument(
        "--action_tokenizer_bundle",
        default=None,
        help="RVQ action tokenizer bundle directory (env ACTION_TOKENIZER_BUNDLE is used when "
        "neither this flag nor --traj_vocab_path is given).",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=_DEFAULTS.gpu_memory_utilization,
        help="vLLM GPU memory fraction (vllm_local backend).",
    )
    p.add_argument("--max_num_seqs", type=int, default=_DEFAULTS.max_num_seqs)
    p.add_argument(
        "--num_history_frames",
        type=int,
        default=None,
        help="Override the checkpoint's history window (default: from eval_config.json).",
    )
    p.add_argument(
        "--aspect_mode",
        choices=["stretch", "keep"],
        default=_DEFAULTS.aspect_mode,
        help="stretch frames to the checkpoint's video_size (default) or keep the aspect ratio.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=_DEFAULTS.max_new_tokens,
        help="Per-step decode cap; raised automatically if the checkpoint's grounding prefix "
        "plus action tokens need more.",
    )
    p.add_argument("--zmq_timeout_ms", type=int, default=_DEFAULTS.zmq_timeout_ms)
    p.add_argument("--verbose", action="store_true", help="Print one line per step.")

    # Visualisation (docs/VISUALIZATION.md); needs `pip install 'lightnav[video]'`.
    viz = p.add_argument_group("visualisation")
    viz.add_argument(
        "--save_video",
        action="store_true",
        help="Write <output_dir>/videos/<episode_id>.mp4 per episode: the agent's RGB frames "
        "with the predicted trajectory, pointing markers and a HUD (one frame per step).",
    )
    viz.add_argument(
        "--video_fps",
        type=int,
        default=_DEFAULTS.video_fps,
        help="Playback frame rate of the saved videos (one policy step per frame).",
    )
    viz.add_argument(
        "--hfov_deg",
        type=float,
        default=_DEFAULTS.hfov_deg,
        help="Horizontal FOV of the agent camera, for the trajectory overlay (the shipped "
        "server yamls use 120).",
    )
    viz.add_argument(
        "--cam_height",
        type=float,
        default=_DEFAULTS.cam_height,
        help="Camera height above the floor in metres, for the trajectory overlay (0.88 in "
        "the shipped yamls).",
    )
    viz.add_argument(
        "--waypoint_dt_s",
        type=float,
        default=_DEFAULTS.waypoint_dt_s,
        help="Seconds per waypoint row assumed by the HUD velocity readout.",
    )
    viz.add_argument(
        "--record_dir",
        default=_DEFAULTS.record_dir,
        help="Also record the raw episodes (frames + per-step JSON) under this directory, in "
        "the layout `lightnav-render` reads. Empty (default) = off.",
    )
    return p


def config_from_args(args: argparse.Namespace) -> HabitatEvalConfig:
    bundle = args.action_tokenizer_bundle
    if bundle is None and not args.traj_vocab_path:
        bundle = os.environ.get("ACTION_TOKENIZER_BUNDLE") or None
    args.action_tokenizer_bundle = bundle
    return HabitatEvalConfig(
        model_path=args.model_path,
        server=args.server,
        backend=args.backend,
        action_output=args.action_output,
        beta_checkpoint=args.beta_checkpoint,
        fm_seed=args.fm_seed,
        fm_only_stop_mode=args.fm_only_stop_mode,
        fm_steps=args.fm_steps,
        fm_stop_mode=args.fm_stop_mode,
        episodes=args.episodes,
        episode_manifest=args.episode_manifest,
        oracle_stop_distance=args.oracle_stop_distance,
        max_steps=args.max_steps,
        force_stop_at_max_steps=not args.no_force_stop,
        output_dir=args.output_dir,
        languages=args.languages,
        traj_vocab_path=args.traj_vocab_path,
        K=args.K,
        horizon=args.horizon,
        action_tokenizer_bundle=args.action_tokenizer_bundle,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        num_history_frames=args.num_history_frames,
        aspect_mode=args.aspect_mode,
        max_new_tokens=args.max_new_tokens,
        zmq_timeout_ms=args.zmq_timeout_ms,
        verbose=args.verbose,
        save_video=args.save_video,
        video_fps=args.video_fps,
        hfov_deg=args.hfov_deg,
        cam_height=args.cam_height,
        waypoint_dt_s=args.waypoint_dt_s,
        record_dir=args.record_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_habitat_eval(config_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
