# Adapted from habitat-lab's ObjectNavDatasetV1 (https://github.com/facebookresearch/habitat-lab),
# MIT License, Copyright (c) Meta Platforms, Inc. and its affiliates. See THIRD_PARTY_NOTICES.md.
"""Tolerant ``ObjectNav-v1`` dataset loader for HM3D ObjectNav v1 and HM3D-OVON episodes.

Some ObjectNav releases carry episode fields that habitat-lab 0.3.x's
``ObjectGoalNavEpisode`` does not accept (HM3D-OVON adds ``children_object_categories``
and ``additional_obj_config_paths``; other releases add ``scene_dataset_config``,
``is_thda``, ...), and OVON shards ship empty ``category_to_*_id`` maps. This loader strips
the unknown fields, synthesizes stable category ids when the maps are empty, and
re-registers under ``ObjectNav-v1`` so the stock task config picks it up. For HM3D
ObjectNav v1 it behaves like the stock loader (episode ids become ``str(index)`` within
each content shard, so episodes must be keyed on ``(scene_id, episode_id)``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)

# Episode keys that ObjectGoalNavEpisode in this habitat-lab version rejects (TypeError).
_UNSUPPORTED_EPISODE_KEYS = frozenset(
    [
        "is_thda",
        "scene_state",
        "scene_dataset",
        "reference_replay",
        "attempts",
        "scene_dataset_config",
        "children_object_categories",
        "additional_obj_config_paths",
    ]
)


def _filter_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the keys ObjectGoalNavEpisode does not accept."""
    return {k: v for k, v in ep.items() if k not in _UNSUPPORTED_EPISODE_KEYS}


def _deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
    g = ObjectGoal(**serialized_goal)
    for vidx, view in enumerate(g.view_points):
        view_location = ObjectViewLocation(**view)  # type: ignore[arg-type]
        view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore[arg-type]
        g.view_points[vidx] = view_location
    return g


@registry.register_dataset(name="ObjectNav-v1")
class TolerantObjectNavDatasetV1(ObjectNavDatasetV1):
    """``ObjectNavDatasetV1`` that tolerates extra episode fields and empty category maps."""

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category: Dict[str, Any] = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0]["object_category"]
            clean = _filter_episode(ep)
            ep_obj = ObjectGoalNavEpisode(**clean)

            goals_key = ep_obj.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep_obj.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category
        return dataset

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        # OVON shards ship these maps empty or omit them on content shards; make sure the
        # attributes exist so the consistency asserts below compare {} == {}.
        if not hasattr(self, "category_to_task_category_id"):
            self.category_to_task_category_id = {}
        if not hasattr(self, "category_to_scene_annotation_category_id"):
            self.category_to_scene_annotation_category_id = {}

        # Only overwrite when the shard carries a non-empty map, so ids synthesized from
        # episode categories accumulate across shards instead of being wiped.
        if deserialized.get("category_to_task_category_id"):
            self.category_to_task_category_id = deserialized["category_to_task_category_id"]

        if deserialized.get("category_to_scene_annotation_category_id"):
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if deserialized.get("category_to_mp3d_category_id"):
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )
        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        )

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [_deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            clean = _filter_episode(episode)
            ep_obj = ObjectGoalNavEpisode(**clean)
            # Renumbering discards the shard's original id; keep it in episode.info.
            if ep_obj.info is None:
                ep_obj.info = {}
            ep_obj.info["raw_episode_id"] = ep_obj.episode_id
            ep_obj.episode_id = str(i)

            # No fixed category table (OVON): synthesize stable, accumulating ids so
            # habitat's ObjectGoalSensor can size its observation space (max over ids).
            # No-op when the category is already in the map.
            _cat = ep_obj.object_category
            if _cat and _cat not in self.category_to_task_category_id:
                _cid = len(self.category_to_task_category_id)
                self.category_to_task_category_id[_cat] = _cid
                self.category_to_scene_annotation_category_id[_cat] = _cid

            if scenes_dir is not None:
                if ep_obj.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    ep_obj.scene_id = ep_obj.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                ep_obj.scene_id = os.path.join(scenes_dir, ep_obj.scene_id)

            ep_obj.goals = self.goals_by_category[ep_obj.goals_key]

            if ep_obj.shortest_paths is not None:
                for path in ep_obj.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {"action": point, "rotation": None, "position": None}
                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(ep_obj)  # type: ignore[attr-defined]
