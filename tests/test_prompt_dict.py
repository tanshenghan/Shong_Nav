"""_build_prompt_dict: assemble a single vLLM prompt entry. CPU-only (no engine)."""

from __future__ import annotations

import torch

from lightnav.inference.engine import _build_prompt_dict


def test_text_only_when_no_embeds():
    d = _build_prompt_dict([1, 2, 3], None, None)
    assert d == {"prompt_token_ids": [1, 2, 3]}


def test_includes_multimodal_when_embeds_present():
    embeds = torch.zeros(4, 8)
    grid = torch.tensor([[1, 4, 4]])
    d = _build_prompt_dict([1, 2], embeds, grid)
    assert d["prompt_token_ids"] == [1, 2]
    mm = d["multi_modal_data"]["video"]
    assert mm["video_embeds"].device.type == "cpu"
    assert mm["video_grid_thw"].device.type == "cpu"
    assert torch.equal(mm["video_embeds"], embeds)
    assert torch.equal(mm["video_grid_thw"], grid)


def test_prompt_ids_are_copied_into_a_plain_list():
    ids = (5, 6, 7)
    d = _build_prompt_dict(ids, None, None)
    assert d["prompt_token_ids"] == [5, 6, 7]
    assert isinstance(d["prompt_token_ids"], list)
