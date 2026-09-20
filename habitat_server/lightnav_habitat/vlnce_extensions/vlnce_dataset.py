# Adapted from VLN-CE (https://github.com/jacobkrantz/VLN-CE), MIT License,
# Copyright (c) 2020 Jacob Krantz. See THIRD_PARTY_NOTICES.md.
"""Habitat dataset loader for VLN-CE episodes (R2R and RxR), registered as ``VLN-CE-v1``.

R2R ships one ``{split}.json.gz`` per split with integer episode ids and a bare
``instruction_text``; RxR ships ``{split}_guide.json.gz`` with string ids and extra
instruction metadata (``language``, ``annotator_id``, ``timed_instruction``, ...) that
habitat-lab's ``InstructionData`` rejects. This loader accepts both, normalizes episode
ids to strings (the NDTW ground-truth file is keyed by strings) and supports the optional
``languages`` filter.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, List, Optional, Union

import attr
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_EPISODES_MASK = "*"


def _get_config_attr(config: Any, attr_name: str, default: Any = None) -> Any:
    """Read a config attribute in lower_case (OmegaConf) or UPPER_CASE (yacs) form."""
    if hasattr(config, attr_name.lower()):
        return getattr(config, attr_name.lower())
    if hasattr(config, attr_name.upper()):
        return getattr(config, attr_name.upper())
    if hasattr(config, "get"):
        val = config.get(attr_name.lower()) or config.get(attr_name.upper())
        if val is not None:
            return val
    return default


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    """Instruction data with the optional RxR metadata fields."""

    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(default=None)
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    """VLN episode with goals, reference path and the extended instruction."""

    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(default=None, validator=not_none_validator)
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    """Unified VLN-CE loader for R2R and RxR (single episode file per split).

    The ``content_scenes``, ``languages`` and ``episodes_allowed`` filters are no-ops when
    the corresponding config value is the wildcard ``"*"`` (or absent).
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    def __init__(self, config: Optional[Any] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        data_path = _get_config_attr(config, "data_path")
        split = _get_config_attr(config, "split")
        scenes_dir = _get_config_attr(config, "scenes_dir")
        content_scenes = _get_config_attr(config, "content_scenes", ["*"])
        languages = _get_config_attr(config, "languages", ["*"])
        episodes_allowed = _get_config_attr(config, "episodes_allowed", ["*"])

        with gzip.open(data_path.format(split=split), "rt") as f:
            self.from_json(f.read(), scenes_dir=scenes_dir, split=split)

        if ALL_SCENES_MASK not in content_scenes:
            scenes_to_load = set(content_scenes)
            self.episodes = [
                e for e in self.episodes if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        # Language filter - meaningful only for multilingual data (RxR). For R2R every
        # episode's `language` is None, so the filter is only applied when configured.
        if ALL_LANGUAGES_MASK not in languages:
            languages_to_load = set(languages)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]

        if ALL_EPISODES_MASK not in episodes_allowed:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(episodes_allowed)
            self.episodes = [
                episode for episode in self.episodes if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, split: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        # R2R-only field; RxR JSONs don't ship one.
        if "instruction_vocab" in deserialized:
            self.instruction_vocab = VocabDict(
                word_list=deserialized["instruction_vocab"]["word_list"]
            )
        elif not hasattr(self, "instruction_vocab"):
            self.instruction_vocab = VocabDict(word_list=[])

        effective_split = split or _get_config_attr(self.config, "split")

        for episode_dict in deserialized["episodes"]:
            # R2R stores ints, RxR strings; normalize to str (idempotent).
            if "episode_id" in episode_dict:
                episode_dict["episode_id"] = str(episode_dict["episode_id"])
            if "trajectory_id" in episode_dict:
                episode_dict["trajectory_id"] = str(episode_dict["trajectory_id"])

            episode = VLNExtendedEpisode(**episode_dict)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            # ExtendedInstructionData is a superset of InstructionData (extra fields are
            # Optional), so one constructor covers R2R and RxR; fall back to the base class
            # only if the dict carries a field that breaks strict attr.s validation.
            try:
                episode.instruction = ExtendedInstructionData(**episode.instruction)
            except TypeError:
                episode.instruction = InstructionData(**episode.instruction)
            if effective_split is not None and hasattr(episode.instruction, "split"):
                episode.instruction.split = effective_split

            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> Optional[str]:
        """Return the instruction language, or ``None`` for R2R (no field)."""
        inst = getattr(episode, "instruction", None)
        if inst is None:
            return None
        return getattr(inst, "language", None)
