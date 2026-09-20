"""Per-step decode token budget: grounding prefix (per checkpoint family) + action tokens.

Known layouts (grounding tokens a step emits before the action tokens):
    tracking, legacy           -> <tpos_k>                     1
    tracking + pointing        -> <opos_k>                     1
    nav/vln + pointing         -> <apos_A><opos_O>             2  (either may be omitted)
    vln, legacy flat           -> (none)                       0
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from lightnav.serving import token_budget, ws_server
from lightnav.serving.token_budget import (
    action_token_count,
    decode_token_budget,
    probe_grounding_tokens,
)


def test_flat_action_is_one_token_rvq_is_one_per_level():
    assert action_token_count(None) == 1
    assert action_token_count() == 1
    assert action_token_count(SimpleNamespace(levels=[256, 256, 256])) == 3


@pytest.mark.parametrize(
    ("grounding", "levels", "expected"),
    [
        (1, None, 2),  # legacy tracking, flat traj vocab
        (0, None, 1),  # legacy vln, flat traj vocab
        (1, [256, 256, 256], 4),  # tracking (tpos or opos) + 3-level RVQ
        (2, [256, 256, 256], 5),  # pointing nav: <apos><opos> + 3 act levels
        (2, [64, 64], 4),
        (0, [256, 256, 256], 3),
    ],
)
def test_budget_is_grounding_plus_action_tokens(grounding, levels, expected):
    rvq = SimpleNamespace(levels=levels) if levels else None

    assert decode_token_budget(grounding, rvq) == expected


def test_budget_rejects_a_negative_grounding_count():
    with pytest.raises(ValueError, match="grounding_tokens"):
        decode_token_budget(-1, None)


def test_budget_does_not_depend_on_the_task_string():
    """The prefix length is a property of the CHECKPOINT, not of the served task, so
    the signature takes it explicitly (a task-derived count truncates pointing ckpts)."""
    sig = inspect.signature(decode_token_budget)
    assert "task" not in sig.parameters
    assert "grounding_tokens" in sig.parameters


class _FakeTokenizer:
    """Reports only the token families a given checkpoint family carries."""

    unk_token_id = 0

    def __init__(self, present):
        self._present = set(present)

    def convert_tokens_to_ids(self, token):
        return 100 if token in self._present else self.unk_token_id


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["<apos_0>"], 2),  # dual-pointing nav ckpt
        (["<apos_0>", "<tpos_0>"], 3),  # mixed vocab -> upper bound, never truncating
        (["<tpos_0>"], 1),  # v2 tracking
        ([], 0),  # v1 / legacy flat
    ],
)
def test_grounding_prefix_is_probed_from_the_ckpt_vocab(present, expected):
    assert probe_grounding_tokens(_FakeTokenizer(present)) == expected


def test_probe_treats_a_none_id_as_absent():
    class NoneTokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return None

    assert probe_grounding_tokens(NoneTokenizer()) == 0


def test_probe_is_an_upper_bound_so_the_cap_can_never_truncate():
    """A ckpt trained on a mix carries every family but emits one per source. Summing
    what is PRESENT can only overshoot -- and overshooting costs one decode step,
    while undershooting truncates the action tokens."""
    rvq = SimpleNamespace(levels=[256, 256, 256])
    mixed = probe_grounding_tokens(_FakeTokenizer(["<apos_0>", "<tpos_0>"]))
    nav_actual, tracking_actual = 2, 1

    assert mixed >= max(nav_actual, tracking_actual)
    assert decode_token_budget(mixed, rvq) >= decode_token_budget(nav_actual, rvq)


def test_ws_server_reexports_the_budget_helpers():
    assert ws_server.action_token_count is token_budget.action_token_count
    assert ws_server.decode_token_budget is token_budget.decode_token_budget
    assert ws_server.probe_grounding_tokens is token_budget.probe_grounding_tokens
