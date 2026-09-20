"""VitTubeletCache: the per-session ViT tubelet cache (vLLM output format only).

The decomposed collect / apply / reassemble phases must produce exactly what the
one-shot ``get_embeddings`` does, and a warm cache must skip the ViT entirely.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lightnav.inference.engine import VLNInferenceEngine
from lightnav.inference.vit_cache import CachedTubeletEmbeds, VitTubeletCache


def _engine(slowfast_tiers=None) -> VLNInferenceEngine:
    bundle = SimpleNamespace(
        num_history_frames=64,
        slowfast_tiers=slowfast_tiers,
        video_size=(256, 320),
        processor=SimpleNamespace(video_processor=SimpleNamespace(patch_size=16)),
    )
    return VLNInferenceEngine(bundle=bundle, backend="vllm_local")


def _entry(value: float = 0.0) -> CachedTubeletEmbeds:
    return CachedTubeletEmbeds(base_embed=torch.full((1, 2), value), deepstack_embeds=[])


class _PositionIndependentViT:
    """A fake ViT whose per-tubelet output depends only on that tubelet's pixels.

    Output rows per tubelet = (h*w)//(merge*merge); each output row is the mean of
    that tubelet's pixel rows in a ``merge_area``-strided block.
    """

    dtype = torch.float32

    def __init__(self, merge_size: int = 2, out_dim: int = 5):
        self.merge_area = merge_size * merge_size
        self.out_dim = out_dim
        self.calls = 0

    def __call__(self, pixels, grid_thw):
        self.calls += 1
        rows = []
        off = 0
        for t, h, w in grid_thw.tolist():
            n_rows = int(t) * int(h) * int(w)
            block = pixels[off : off + n_rows]
            off += n_rows
            out_len = n_rows // self.merge_area
            merged = block.reshape(out_len, self.merge_area, -1).mean(dim=1)
            proj = merged.sum(dim=1, keepdim=True).repeat(1, self.out_dim)
            rows.append(proj.to(self.dtype))
        return torch.cat(rows, dim=0)


def _pixels(grid_thw, channels=3, seed=0):
    """Flat pixels for a grid (full-window layout)."""
    g = torch.Generator().manual_seed(seed)
    total = sum(int(t) * int(h) * int(w) for t, h, w in grid_thw)
    return torch.rand(total, channels, generator=g)


# -- low-level API ------------------------------------------------------------


def test_put_cached_keys_clear_and_lru_eviction():
    cache = VitTubeletCache(max_entries=2, store_device="cpu")
    cache.put((0, 1, 16, 20), _entry(1.0))
    cache.put((2, 3, 16, 20), _entry(2.0))
    assert cache.cached_keys() == {(0, 1, 16, 20), (2, 3, 16, 20)}

    cache.put((4, 5, 16, 20), _entry(3.0))  # evicts the least recently used (0, 1)
    assert cache.cached_keys() == {(2, 3, 16, 20), (4, 5, 16, 20)}

    cache.put((2, 3, 16, 20), _entry(4.0))  # refresh: (2, 3) becomes most recent
    cache.put((6, 7, 16, 20), _entry(5.0))  # now evicts (4, 5)
    assert cache.cached_keys() == {(2, 3, 16, 20), (6, 7, 16, 20)}
    assert float(cache._store[(2, 3, 16, 20)].base_embed[0, 0]) == 4.0

    cache.clear()
    assert cache.cached_keys() == set()


def test_max_entries_is_at_least_one():
    assert VitTubeletCache(max_entries=0, store_device="cpu").max_entries == 1


def test_store_device_env_escape_hatch(monkeypatch):
    monkeypatch.setenv("VLN_VIT_CACHE_CPU", "1")
    assert VitTubeletCache(max_entries=4).store_device == "cpu"
    monkeypatch.delenv("VLN_VIT_CACHE_CPU")
    assert VitTubeletCache(max_entries=4).store_device == "cuda"
    assert VitTubeletCache(max_entries=4, store_device="cpu").store_device == "cpu"


# -- engine helpers -------------------------------------------------------------


def test_new_vit_cache_returns_independent_caches():
    engine = _engine()

    a = engine.new_vit_cache()
    b = engine.new_vit_cache()
    a.put((0, 1, 16, 20), _entry())

    assert a.cached_keys() == {(0, 1, 16, 20)}
    assert b.cached_keys() == set()
    assert a.max_entries == engine._vit_cache_max_entries == 128  # max(32, 2 * history)


def test_slowfast_engine_uses_a_larger_cache():
    tiers = [{"age_lo": 0, "age_hi": 1, "mode": "dense", "pool_spatial": 1}]
    assert _engine(slowfast_tiers=tiers)._vit_cache_max_entries == 512


def test_reset_episode_state_clears_the_engine_cache():
    engine = _engine()
    engine._vit_cache.put((0, 1, 16, 20), _entry())
    engine.reset_episode_state()
    assert engine._vit_cache.cached_keys() == set()


def test_processor_cached_keys_adds_the_processor_grid_spelling():
    engine = _engine()
    cache = VitTubeletCache(max_entries=8, store_device="cpu")
    assert engine._processor_vit_cached_keys(cache) == set()

    cache.put((0, 1, 32, 40), _entry())
    keys = engine._processor_vit_cached_keys(cache)
    # 256x320 / patch 16 -> 16x20 grid; the raw key is kept alongside.
    assert keys == {(0, 1, 32, 40), (0, 1, 16, 20)}


# -- collect / apply / reassemble ------------------------------------------------


def test_collect_apply_reassemble_equals_get_embeddings():
    """collect_misses + apply_miss_embeds + reassemble == get_embeddings, byte-exact."""
    vit = _PositionIndependentViT()
    grid = [[2, 2, 4], [1, 2, 4]]  # 3 tubelets
    seg_fi = [[0, 1, 2, 3], [4, 5]]
    px = _pixels(grid, seed=1)

    # Reference: the public one-shot API.
    ref_cache = VitTubeletCache(max_entries=16, store_device="cpu")
    ref_out, ref_hit, ref_miss = ref_cache.get_embeddings(px, grid, seg_fi, vit=vit, merge_size=2)
    assert (ref_hit, ref_miss) == (0, 3)

    # Decomposed path.
    cache = VitTubeletCache(max_entries=16, store_device="cpu")
    plan = cache.collect_misses(px, grid, seg_fi, merge_size=2)
    assert (plan.hit, plan.miss) == (0, 3)  # cold: all miss
    assert plan.miss_grid_rows == [[1, 2, 4], [1, 2, 4], [1, 2, 4]]
    packed = vit(plan.miss_pixels, grid_thw=torch.tensor(plan.miss_grid_rows))
    cache.apply_miss_embeds(plan, packed)
    out = plan.reassemble(vit.dtype)

    assert torch.equal(out, ref_out)
    assert cache.cached_keys() == ref_cache.cached_keys() == {
        (0, 1, 2, 4),
        (2, 3, 2, 4),
        (4, 5, 2, 4),
    }
    for k in cache.cached_keys():
        assert torch.equal(cache._store[k].base_embed, ref_cache._store[k].base_embed)


def test_warm_cache_skips_the_vit_and_reproduces_the_window():
    vit = _PositionIndependentViT()
    grid = [[2, 2, 4], [1, 2, 4]]
    seg_fi = [[0, 1, 2, 3], [4, 5]]
    px = _pixels(grid, seed=2)

    cache = VitTubeletCache(max_entries=16, store_device="cpu")
    first, hit, miss = cache.get_embeddings(px, grid, seg_fi, vit=vit, merge_size=2)
    assert (hit, miss) == (0, 3) and vit.calls == 1

    second, hit, miss = cache.get_embeddings(px, grid, seg_fi, vit=vit, merge_size=2)
    assert (hit, miss) == (3, 0)
    assert vit.calls == 1  # all hits: no ViT forward
    assert torch.equal(first, second)

    plan = cache.collect_misses(px, grid, seg_fi, merge_size=2)
    assert plan.miss == 0 and plan.missing_units == [] and plan.miss_pixels is None
    assert torch.equal(plan.reassemble(vit.dtype), first)


def test_sliding_window_only_encodes_the_new_tubelet():
    """Step t+1 shares all but the newest pair with step t: exactly one miss."""
    vit = _PositionIndependentViT()
    cache = VitTubeletCache(max_entries=16, store_device="cpu")

    grid_t = [[2, 2, 4], [1, 2, 4]]
    px_t = _pixels(grid_t, seed=3)
    cache.get_embeddings(px_t, grid_t, [[0, 1, 2, 3], [4, 5]], vit=vit, merge_size=2)

    # Next step: frames 2..7 -> tubelets (2,3), (4,5) cached; (6,7) new.
    rows = 2 * 4
    px_next = torch.cat([px_t[rows : 3 * rows], _pixels([[1, 2, 4]], seed=4)], dim=0)
    out, hit, miss = cache.get_embeddings(
        px_next, grid_t, [[2, 3, 4, 5], [6, 7]], vit=vit, merge_size=2
    )
    assert (hit, miss) == (2, 1)
    assert vit.calls == 2
    assert out.shape == (3 * 2, vit.out_dim)
    # The reused tubelets come back byte-identical to the fresh encode.
    fresh = _PositionIndependentViT()
    ref, _, _ = VitTubeletCache(max_entries=16, store_device="cpu").get_embeddings(
        px_next, grid_t, [[2, 3, 4, 5], [6, 7]], vit=fresh, merge_size=2
    )
    assert torch.equal(out, ref)


def test_collect_misses_accepts_miss_only_pixel_rows():
    """The selective processor path hands over only the miss tubelets' pixel rows."""
    vit = _PositionIndependentViT()
    grid = [[2, 2, 4]]
    seg_fi = [[0, 1, 2, 3]]
    px = _pixels(grid, seed=5)
    cache = VitTubeletCache(max_entries=16, store_device="cpu")
    cache.get_embeddings(px, grid, seg_fi, vit=vit, merge_size=2)

    # Frames 2..5: tubelet (2,3) cached, (4,5) missing. Pass only the miss rows.
    new_rows = _pixels([[1, 2, 4]], seed=6)
    plan = cache.collect_misses(new_rows, grid, [[2, 3, 4, 5]], merge_size=2)
    assert (plan.hit, plan.miss) == (1, 1)
    assert torch.equal(plan.miss_pixels, new_rows)


def test_collect_misses_rejects_an_inconsistent_row_count():
    cache = VitTubeletCache(max_entries=16, store_device="cpu")
    with pytest.raises(ValueError, match="matches neither"):
        cache.collect_misses(torch.zeros(5, 3), [[2, 2, 4]], [[0, 1, 2, 3]], merge_size=2)


def test_apply_miss_embeds_is_a_no_op_when_nothing_is_missing():
    vit = _PositionIndependentViT()
    grid = [[1, 2, 4]]
    px = _pixels(grid, seed=7)
    cache = VitTubeletCache(max_entries=16, store_device="cpu")
    cache.get_embeddings(px, grid, [[0, 1]], vit=vit, merge_size=2)
    plan = cache.collect_misses(px, grid, [[0, 1]], merge_size=2)
    cache.apply_miss_embeds(plan, torch.zeros(0, vit.out_dim))
    assert cache.cached_keys() == {(0, 1, 2, 4)}
