"""Read local R2R/RxR expert sources without Habitat or training dependencies."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

DATA_ROOTS = {
    "r2r": Path("/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed"),
    "rxr": Path("/mnt/cfs/data/habitat/datasets/RxR_VLNCE_v0"),
}
LANGUAGES = ("en-US", "en-IN", "hi-IN", "te-IN")
SPLITS = ("train", "val_seen", "val_unseen")


def rxr_replay_actions(actions):
    """RxR's native 30-degree turns become two actual LightNav 15-degree steps.

    Forward stays 0.25 m; STOP remains exactly one final action. All intermediate
    images/poses are rendered by Habitat, never interpolated from source images.
    """
    if not actions or actions[-1] != 0 or any(a not in (1, 2, 3) for a in actions[:-1]):
        raise ValueError("RxR expert actions must end with exactly one STOP and use only forward/left/right")
    return [action for action in actions for _ in range(2 if action in (2, 3) else 1)]


def source_paths(root, dataset="r2r", split="train", role="guide"):
    """Keep original split/role files separate; never manufacture test GT."""
    if dataset not in DATA_ROOTS or role not in ("guide", "follower"):
        raise ValueError("expected dataset r2r/rxr and role guide/follower")
    if split not in SPLITS:
        raise ValueError("only train/val_seen/val_unseen have supported expert GT; test_challenge is excluded")
    if dataset == "r2r" and (split != "train" or role != "guide"):
        raise ValueError("the existing R2R pipeline supports train only, without follower annotations")
    stem = split if dataset == "r2r" else f"{split}_{role}"
    root = Path(root)
    paths = (root / split / f"{stem}.json.gz", root / split / f"{stem}_gt.json.gz")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"source episodes and expert GT are both required: {path}")
    return paths


def load_source(root, dataset="r2r", split="train", role="guide", languages=None):
    """Return selected original episodes, their GT, and auditable selection metadata.

    Instruction text is never translated or deduplicated. Speech timing remains
    in the untouched source JSON; it is not used as the synthetic frame clock.
    """
    languages = sorted(set(languages or ["all"]))
    if "all" in languages and len(languages) != 1:
        raise ValueError("languages=all cannot be combined with individual languages")
    if set(languages) - {"all", *LANGUAGES}:
        raise ValueError(f"unsupported languages: {languages}")
    if dataset == "r2r" and languages != ["all"]:
        raise ValueError("language filtering is only supported for RxR")
    episode_path, gt_path = source_paths(root, dataset, split, role)
    with gzip.open(episode_path, "rt", encoding="utf-8") as stream:
        originals = json.load(stream)["episodes"]
    with gzip.open(gt_path, "rt", encoding="utf-8") as stream:
        gt = json.load(stream)
    selected, selected_gt, seen = [], {}, set()
    for original in originals:
        episode_id = str(original["episode_id"])
        if not episode_id or episode_id in (".", "..") or "/" in episode_id or "\\" in episode_id:
            raise ValueError(f"unsafe source episode ID: {episode_id!r}")
        if episode_id in seen:
            raise ValueError(f"duplicate source episode ID: {episode_id}")
        seen.add(episode_id)
        instruction = original["instruction"]
        language = instruction.get("language")
        if dataset == "rxr":
            if language not in LANGUAGES:
                raise ValueError(f"missing/unsupported RxR language: {episode_id}: {language}")
            if original.get("info", {}).get("role") != role:
                raise ValueError(f"RxR role does not match source file: {episode_id}")
        if languages != ["all"] and language not in languages:
            continue
        if episode_id not in gt or not {"actions", "locations"} <= gt[episode_id].keys():
            raise ValueError(f"missing expert actions/locations: {episode_id}")
        if not isinstance(instruction.get("instruction_text"), str) or not instruction["instruction_text"].strip():
            raise ValueError(f"empty instruction: {episode_id}")
        episode = dict(original)
        expert = gt[episode_id]
        if dataset == "rxr":
            replay_actions = rxr_replay_actions(expert["actions"])
            episode["source_metadata"] = dict(dataset=dataset, split=split, role=role, language=language,
                source_action_count=len(expert["actions"]), replay_frame_count=len(replay_actions),
                source_turn_angle_deg=30, replay_turn_angle_deg=15, forward_step_m=0.25,
                action_conversion="each source 30-degree turn becomes two rendered 15-degree turns",
                **{k: instruction[k] for k in ("instruction_id", "annotator_id", "edit_distance") if k in instruction})
            if "demonstration_id" in original.get("info", {}):
                episode["source_metadata"]["demonstration_id"] = original["info"]["demonstration_id"]
            expert = dict(expert, actions=replay_actions)
        selected_gt[episode_id] = expert
        selected.append(episode)
    if not selected:
        raise ValueError("source selection contains no episodes")
    metadata = dict(dataset=dataset, split=split, role=role if dataset == "rxr" else None,
        languages=languages, source_episode_count=len(originals), selected_episode_count=len(selected),
        language_counts=dict(sorted(Counter(e["instruction"].get("language", "unspecified") for e in selected).items())),
        source_frames=sum(len(gt[str(e["episode_id"])]["actions"]) for e in selected),
        replay_frames=sum(len(v["actions"]) for v in selected_gt.values()),
        episode_file=str(episode_path), gt_file=str(gt_path))
    if dataset == "rxr":
        metadata["motion_conversion"] = dict(source_turn_angle_deg=30, replay_turn_angle_deg=15,
                                              turns_per_source_turn=2, forward_step_m=0.25)
        metadata["navigation"] = dict(navmesh="rxr_in_memory", agent_height_m=.88,
                                       agent_radius_m=.18, allow_sliding=False,
                                       other_navmesh_settings="Habitat-Sim NavMeshSettings defaults")
    return selected, selected_gt, metadata
