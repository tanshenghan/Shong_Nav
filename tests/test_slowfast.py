"""SlowFast multi-tier history sampling: tier validation and the shared segment layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lightnav.slowfast import (
    DEFAULT_SLOWFAST_TIERS,
    DEFAULT_SLOWFAST_TIERS_SPAN,
    _tier_ages,
    slowfast_segments,
    slowfast_video_segments,
    validate_slowfast_tiers,
)

TIERS = validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS)
TIERS_SPAN = validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS_SPAN)


# -- validate_slowfast_tiers ----------------------------------------------------


def test_default_tiers_validate_and_normalize():
    norm = validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS)
    assert [t["name"] for t in norm] == ["current", "fast", "mid", "long"]
    assert all(t["pool_mode"] == "avg" for t in norm)  # default filled
    assert norm[2]["pair_stride"] == 6 and norm[3]["pair_stride"] == 12


def test_validate_returns_a_new_list_with_names_filled():
    tiers = [{"age_lo": 0, "age_hi": 1, "mode": "dense", "pool_spatial": 1}]
    norm = validate_slowfast_tiers(tiers)
    assert norm is not tiers
    assert norm[0]["name"] == "tier0"
    assert "name" not in tiers[0]


@pytest.mark.parametrize(
    "bad",
    [
        [],  # empty
        [{"age_lo": 0, "age_hi": 1, "mode": "weird", "pool_spatial": 1}],  # bad mode
        [{"age_lo": 5, "age_hi": 2, "mode": "dense", "pool_spatial": 1}],  # lo>hi
        [{"age_lo": -1, "age_hi": 2, "mode": "dense", "pool_spatial": 1}],  # negative age
        [{"age_lo": 0, "age_hi": 1, "mode": "dense", "pool_spatial": 0}],  # pool<1
        [  # overlapping ranges
            {"age_lo": 0, "age_hi": 5, "mode": "dense", "pool_spatial": 1},
            {"age_lo": 5, "age_hi": 9, "mode": "dense", "pool_spatial": 1},
        ],
        [{"age_lo": 0, "age_hi": 9, "mode": "burst", "pair_stride": 1, "pool_spatial": 1}],  # stride<2
    ],
)
def test_invalid_tiers_raise(bad):
    with pytest.raises(ValueError):
        validate_slowfast_tiers(bad)


# -- slowfast_segments: full / cap case -----------------------------------------


def test_segments_full_episode_layout_and_token_frames():
    cur = 300
    segs = slowfast_segments(cur, n_available=cur + 1, tiers=TIERS)
    # 4 segments, chronological (oldest first)
    assert [s["tier"] for s in segs] == [3, 2, 1, 0]  # long, mid, fast, current
    counts = [len(s["frame_ids"]) for s in segs]
    assert counts == [28, 24, 16, 2]
    pools = [s["pool_spatial"] for s in segs]
    assert pools == [4, 2, 2, 1]
    # every segment even (temporal_patch_size=2) and chronologically sorted
    for s in segs:
        assert len(s["frame_ids"]) % 2 == 0
        assert s["frame_ids"] == sorted(s["frame_ids"])
    # absolute frame ids: current tier ends at current_abs; global monotone across segments
    assert segs[-1]["frame_ids"][-1] == cur
    flat = [fid for s in segs for fid in s["frame_ids"]]
    assert flat == sorted(flat)
    assert min(flat) == cur - 247  # oldest kept frame


def test_total_visual_tokens_550_at_256x320():
    # token/temporal-step at 256x320 = (256/16/2)*(320/16/2) = 8*10 = 80; /pool^2
    cur = 300
    segs = slowfast_segments(cur, n_available=cur + 1, tiers=TIERS)
    tokens = 0
    for s in segs:
        steps = len(s["frame_ids"]) // 2  # temporal_patch_size=2
        tokens += steps * (80 // (s["pool_spatial"] ** 2))
    assert tokens == 550


# -- slowfast_segments: short episodes / graceful degradation -------------------


def test_short_episode_drops_far_tiers_keeps_current():
    # 10-frame episode, current at index 9
    segs = slowfast_segments(9, n_available=10, tiers=TIERS)
    tiers_present = {s["tier"] for s in segs}
    assert tiers_present == {0, 1}  # only current + fast; mid/long empty
    by_tier = {s["tier"]: s for s in segs}
    assert by_tier[0]["frame_ids"] == [8, 9]  # current
    assert by_tier[1]["frame_ids"] == [0, 1, 2, 3, 4, 5, 6, 7]  # fast covers rest


def test_dense_tier_pad_to_even():
    # current_abs=10 -> fast ages 2..10 = 9 frames (odd) -> padded to 10 by dup newest
    segs = slowfast_segments(10, n_available=11, tiers=TIERS)
    fast = next(s for s in segs if s["tier"] == 1)
    assert len(fast["frame_ids"]) == 10
    assert fast["frame_ids"][-1] == fast["frame_ids"][-2]  # duplicated newest


def test_burst_keeps_only_complete_pairs():
    # current_abs=19, n_available=20: mid pair (18,19) completable (need abs 1 and 0)
    segs = slowfast_segments(19, n_available=20, tiers=TIERS)
    mid = next((s for s in segs if s["tier"] == 2), None)
    assert mid is not None and mid["frame_ids"] == [0, 1]  # exactly one complete pair


def test_burst_drops_incomplete_pair():
    # current_abs=18, n_available=19: pair (18,19) needs age 19 -> abs -1 -> unavailable -> dropped
    segs = slowfast_segments(18, n_available=19, tiers=TIERS)
    assert all(s["tier"] != 2 for s in segs)  # mid empty
    assert all(s["tier"] != 3 for s in segs)  # long empty


def test_single_frame_episode_pads_current():
    segs = slowfast_segments(0, n_available=1, tiers=TIERS)
    assert len(segs) == 1 and segs[0]["tier"] == 0
    assert segs[0]["frame_ids"] == [0, 0]  # only current frame, padded to even


def test_segments_deterministic_for_mirror():
    a = slowfast_segments(123, n_available=124, tiers=TIERS)
    b = slowfast_segments(123, n_available=124, tiers=TIERS)
    assert a == b


def test_tier_ages_without_a_cap_is_the_fixed_pattern():
    fast = TIERS[1]
    assert _tier_ages(fast, None) == list(range(2, 18))
    assert _tier_ages(fast, 10) == list(range(2, 10))
    with pytest.raises(ValueError):
        _tier_ages(TIERS_SPAN[3], None)  # span needs the episode position


# -- span tier (adaptive-stride memory, anchors frame 0) -----------------------


def test_span_tier_validates():
    norm = validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS_SPAN)
    assert norm[3]["mode"] == "span" and norm[3]["num_pairs"] == 14
    assert norm[4]["mode"] == "anchor" and norm[4]["num_frames"] == 2


def test_span_num_pairs_validation():
    with pytest.raises(ValueError):
        validate_slowfast_tiers(
            [{"age_lo": 0, "age_hi": 10, "mode": "span", "num_pairs": 0, "pool_spatial": 1}]
        )


def test_anchor_num_frames_validation():
    with pytest.raises(ValueError):
        validate_slowfast_tiers(
            [{"age_lo": 0, "age_hi": 0, "mode": "anchor", "num_frames": 0, "pool_spatial": 1}]
        )


def test_frame_zero_always_in_input():
    # Hard requirement: frame 0 must be present for EVERY episode position/length
    # (the anchor tier guarantees it, regardless of span/burst stride boundaries).
    for cur in (0, 1, 2, 17, 18, 30, 89, 90, 95, 181, 300, 600):
        segs = slowfast_segments(cur, cur + 1, TIERS_SPAN)
        flat = [f for s in segs for f in s["frame_ids"]]
        assert 0 in flat, f"frame 0 missing at current_abs={cur}"


def test_no_frame_shared_across_segments():
    for cur in (30, 181, 600):
        segs = slowfast_segments(cur, cur + 1, TIERS_SPAN)
        owner: dict = {}
        for s in segs:
            for f in set(s["frame_ids"]):  # padding may repeat within a segment; that's fine
                assert f not in owner, f"frame {f} shared by tiers {owner.get(f)} and {s['tier']}"
                owner[f] = s["tier"]


def test_span_reserves_anchor_frames():
    # the long span must not include frames 0,1 (owned by the anchor tier)
    long_seg = next(s for s in slowfast_segments(600, 601, TIERS_SPAN) if s["tier"] == 3)
    assert 0 not in long_seg["frame_ids"] and 1 not in long_seg["frame_ids"]


def test_span_pairs_adjacent_and_spaced():
    ids = next(s for s in slowfast_segments(600, 601, TIERS_SPAN) if s["tier"] == 3)["frame_ids"]
    for k in range(0, len(ids), 2):
        assert ids[k + 1] - ids[k] == 1  # adjacent within pair (no far blend)
    starts = [ids[k] for k in range(0, len(ids), 2)]
    assert all(starts[i + 1] - starts[i] >= 2 for i in range(len(starts) - 1))  # pairs disjoint


def test_span_empty_when_unreachable_but_frame0_kept():
    # current too early for age_lo=90 -> span inactive, yet anchor still keeps frame 0
    segs = slowfast_segments(50, n_available=51, tiers=TIERS_SPAN)
    assert all(s["tier"] != 3 for s in segs)  # span (age_lo=90) inactive
    assert 0 in [f for s in segs for f in s["frame_ids"]]


def test_episode_start_current_keeps_pool1():
    # Anchor must NOT steal frames 0/1 from the dense 'current' tier at episode
    # start: the live observation has to stay at pool_spatial=1.
    for cur in (0, 1):
        segs = slowfast_segments(cur, cur + 1, TIERS_SPAN)
        current_seg = next(s for s in segs if s["tier"] == 0)
        assert cur in current_seg["frame_ids"]
        assert current_seg["pool_spatial"] == 1
        assert not any(s["tier"] == 4 for s in segs), f"anchor should be empty at cur={cur}"


def test_early_episode_dense_owns_start_frames():
    # While frames 0/1 are within dense reach (current+fast), they are emitted
    # at dense resolution and the anchor tier stays empty.
    segs = slowfast_segments(10, 11, TIERS_SPAN)
    owner = {f: s["tier"] for s in segs for f in s["frame_ids"]}
    assert owner[0] == 1 and owner[1] == 1  # fast tier (pool2)
    assert owner[10] == 0  # current tier (pool1)
    assert not any(s["tier"] == 4 for s in segs)


def test_anchor_takes_over_beyond_dense_reach():
    # Once frames 0/1 age out of dense reach (age > fast.age_hi=17), the anchor
    # tier must re-emit them so the episode start never drops out of the input.
    segs = slowfast_segments(30, 31, TIERS_SPAN)
    anchor_seg = next(s for s in segs if s["tier"] == 4)
    assert 0 in anchor_seg["frame_ids"] and 1 in anchor_seg["frame_ids"]
    dense_ids = {f for s in segs for f in s["frame_ids"] if s["tier"] in (0, 1)}
    assert 0 not in dense_ids and 1 not in dense_ids


def test_long_span_layout_is_stable_for_a_long_episode():
    """Pin the exact layout at one position so any drift in the span math is caught."""
    segs = slowfast_segments(600, 601, TIERS_SPAN)
    long_seg = next(s for s in segs if s["tier"] == 3)
    # 14 pair starts are requested (ages 90, 129, ..., 560, 599); the oldest pair
    # (ages 599/600 -> frames 1/0) collides with the anchor tier and is dropped
    # WHOLE, so 13 pairs survive.
    assert len(long_seg["frame_ids"]) == 2 * 13
    assert long_seg["frame_ids"][:2] == [600 - 561, 600 - 560]  # oldest surviving pair
    assert long_seg["frame_ids"][-2:] == [600 - 91, 600 - 90]  # newest pair (age_lo+1, age_lo)
    anchor = next(s for s in segs if s["tier"] == 4)
    assert anchor["frame_ids"] == [0, 1]


# -- slowfast_video_segments: index a full-episode tensor ---------------------------


def test_slowfast_video_segments_indexes_correct_frames_numpy():
    n = 300
    # frame i is a constant-i tile so we can read back which frames were selected
    ep = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1) * np.ones((n, 1, 2, 2), dtype=np.float32)
    cur = n - 1
    segs = slowfast_video_segments(ep, cur, cur + 1, TIERS_SPAN)
    assert len(segs) >= 1
    for s in segs:
        assert set(s) == {"video", "frame_indices", "total_frames", "pool_spatial", "pool_mode"}
        assert s["total_frames"] == n
        assert s["video"].shape[0] == len(s["frame_indices"])
        picked = s["video"][:, 0, 0, 0].tolist()
        assert picked == [float(i) for i in s["frame_indices"]]  # content == absolute index
    assert any(0 in s["frame_indices"] for s in segs)  # span anchors frame 0


def test_slowfast_video_segments_indexes_correct_frames_torch():
    n = 40
    ep = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1, 1) * torch.ones((n, 3, 2, 2))
    cur = n - 1
    segs = slowfast_video_segments(ep, cur, cur + 1, TIERS_SPAN)
    layout = slowfast_segments(cur, cur + 1, TIERS_SPAN)
    assert [s["frame_indices"] for s in segs] == [s["frame_ids"] for s in layout]
    assert [s["pool_spatial"] for s in segs] == [s["pool_spatial"] for s in layout]
    for s in segs:
        assert isinstance(s["video"], torch.Tensor)
        assert s["video"][:, 0, 0, 0].tolist() == [float(i) for i in s["frame_indices"]]
