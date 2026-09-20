"""Per-session LRU cache of ViT tubelet embeddings for sliding-window video inference.

Keys are ``(frame_idx_0, frame_idx_1, grid_h, grid_w)`` for ``temporal_patch_size=2``
tubelets, using absolute episode frame ids, so an embedding computed at one step
is reused verbatim at later steps that show the same frame pair. The cache only
affects speed: a hit returns exactly the tensor a fresh ViT forward would produce.

ViT call contract: ``vit(pixels, grid_thw=cpu LongTensor[N, 3]) -> Tensor[tokens, hidden]``
(the in-process vLLM visual tower, base + deepstack already concatenated along dim 1).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch  # noqa: F401


TubeletKey = tuple[int, int, int, int]


@dataclass
class CachedTubeletEmbeds:
    """Cached per-tubelet ViT output (stored on ``VitTubeletCache.store_device``)."""

    base_embed: "torch.Tensor"
    deepstack_embeds: list["torch.Tensor"]


class VitTubeletCache:
    """Fixed-size LRU cache for per-tubelet ViT embeddings with batched miss processing.

    Key format: ``(frame_idx_0, frame_idx_1, grid_h, grid_w)``.

    ``get_embeddings`` runs the full pipeline for one request: parse segments ->
    check cache -> one ViT forward over the misses -> populate -> reassemble the
    packed window in original tubelet order. ``collect_misses`` /
    ``apply_miss_embeds`` / ``MissPlan.reassemble`` are the same three phases
    exposed separately.
    """

    def __init__(self, max_entries: int = 64, store_device: str | None = None):
        self.max_entries = max(1, int(max_entries))
        # Cache embeds on GPU by default so per-step reassembly is a device-local
        # concat instead of a full-window CPU->GPU copy every step. VLN_VIT_CACHE_CPU=1
        # reverts to CPU storage as an escape hatch under tight GPU memory.
        if store_device is None:
            cpu = os.environ.get("VLN_VIT_CACHE_CPU", "0").lower() in ("1", "true", "yes")
            store_device = "cpu" if cpu else "cuda"
        self.store_device = store_device
        self._store: OrderedDict[TubeletKey, CachedTubeletEmbeds] = OrderedDict()

    # -- Low-level API ----------------------------------------------------

    def put(self, key: TubeletKey, value: CachedTubeletEmbeds) -> None:
        if key in self._store:
            self._store[key] = value
            self._store.move_to_end(key)
            return
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def cached_keys(self) -> set[TubeletKey]:
        """Snapshot of currently-cached tubelet keys.

        Used by the data path to skip pixel preprocessing for tubelets whose ViT
        embeds are already cached (their pixels would be discarded anyway).
        """
        return set(self._store.keys())

    def clear(self) -> None:
        self._store.clear()

    # -- High-level API ---------------------------------------------------

    def collect_misses(
        self,
        pixel_values: "torch.Tensor",
        grid_thw: list[list[int]],
        segment_frame_indices: list[list[int]],
        merge_size: int = 2,
    ) -> "MissPlan":
        """Phase 1: parse tubelets, check cache, assign pixels to misses.

        Does NOT run the ViT and does NOT populate the cache; it returns a
        :class:`MissPlan` describing exactly which tubelets are hits (already
        cached) and which are misses (need encoding), with each miss's pixel
        slice attached. The plan is consumed by :meth:`apply_miss_embeds` (to
        populate the cache after a ViT forward) and :meth:`MissPlan.reassemble`
        (to build the packed window).

        Raises ``ValueError`` if the ``pixel_values`` row count matches neither
        the miss-only nor the full-window layout.
        """

        merge_area = merge_size * merge_size
        device = pixel_values.device

        # 1. Parse segments into per-tubelet descriptors (key + full-window row
        #    span; pixels are assigned to misses in step 3, not sliced here).
        unit_desc: list[dict[str, Any]] = []
        offset = 0
        for seg_idx, row in enumerate(grid_thw):
            t, h, w = (int(x) for x in row)

            fi = list(segment_frame_indices[seg_idx]) if seg_idx < len(segment_frame_indices) else []
            if not fi:
                fi = list(range(t * 2))

            for u in range(t):
                f0 = fi[min(2 * u, len(fi) - 1)]
                f1 = fi[min(2 * u + 1, len(fi) - 1)]
                key: TubeletKey = (int(f0), int(f1), int(h), int(w))
                unit_desc.append(
                    {
                        "key": key,
                        "h": h,
                        "w": w,
                        "out_len": (h * w) // merge_area,
                        "full_off": offset,
                        "rows": h * w,
                    }
                )
                offset += h * w

        # 2. Check cache: hits vs misses.
        hit = 0
        miss = 0
        cached_map: dict[TubeletKey, CachedTubeletEmbeds] = {}
        missing_units: list[dict[str, Any]] = []
        for d in unit_desc:
            entry = self._store.get(d["key"])
            if entry is not None:
                hit += 1
                self._store.move_to_end(d["key"])
                cached_map[d["key"]] = entry
            else:
                miss += 1
                missing_units.append(d)

        # 3. Assign pixels to miss tubelets. pixel_values is either the full
        #    pre-pool window or miss-only rows (selective processor path); detect
        #    by row count. Miss order matches the data path's miss-clip order
        #    (both iterate segment->tubelet over the same cache snapshot, so they
        #    agree row-for-row).
        total_rows = offset
        miss_rows = sum(d["rows"] for d in missing_units)
        n_rows = int(pixel_values.shape[0])
        if missing_units and n_rows == miss_rows:
            off = 0
            for d in missing_units:
                d["pixels"] = pixel_values[off : off + d["rows"]]
                off += d["rows"]
        elif n_rows == total_rows:
            for d in missing_units:
                d["pixels"] = pixel_values[d["full_off"] : d["full_off"] + d["rows"]]
        elif missing_units:
            raise ValueError(
                f"pixel_values rows={n_rows} matches neither miss-only ({miss_rows}) "
                f"nor full-window ({total_rows})"
            )

        return MissPlan(
            cache=self,
            device=device,
            unit_desc=unit_desc,
            cached_map=cached_map,
            missing_units=missing_units,
            hit=hit,
            miss=miss,
        )

    def get_embeddings(
        self,
        pixel_values: "torch.Tensor",
        grid_thw: list[list[int]],
        segment_frame_indices: list[list[int]],
        vit: Any,
        merge_size: int = 2,
        grid_device: str | Any = "cpu",
    ) -> tuple["torch.Tensor", int, int]:
        """Compute packed video embeddings with tubelet-level caching.

        All cache misses are batched into a single ViT forward pass.

        Args:
            pixel_values: Flat ``[sum(Ti*Hi*Wi), C]`` tensor on the ViT device.
            grid_thw: ``[[T0, H0, W0], ...]`` per video segment (pre-merge grid).
            segment_frame_indices: Per-segment list of absolute frame indices.
            vit: ViT model (callable with ``(pixels, grid_thw=...)``).
            merge_size: Spatial merge size of the ViT (typically 2).
            grid_device: Device for the ``grid_thw`` tensor passed to the ViT.

        Returns:
            ``(packed_embeddings, num_hits, num_misses)``
        """
        import torch

        plan = self.collect_misses(
            pixel_values, grid_thw, segment_frame_indices, merge_size=merge_size,
        )

        if plan.missing_units:
            miss_grid = torch.tensor(plan.miss_grid_rows, dtype=torch.long, device=grid_device)
            with torch.no_grad():
                miss_out = vit(plan.miss_pixels, grid_thw=miss_grid)
            self.apply_miss_embeds(plan, miss_out)

        video_embeds_packed = plan.reassemble(vit.dtype)
        return video_embeds_packed, plan.hit, plan.miss

    def apply_miss_embeds(self, plan: "MissPlan", miss_out: "torch.Tensor") -> None:
        """Phase 2: split a ViT output across the plan's miss tubelets and populate the cache.

        ``miss_out`` must be the ViT output for EXACTLY this plan's miss tubelets,
        in ``plan.missing_units`` order. The split lengths are the per-tubelet
        ``out_len`` values.
        """
        import torch

        missing_units = plan.missing_units
        if not missing_units:
            return
        split_lens = [int(d["out_len"]) for d in missing_units]
        split_packed = torch.split(miss_out, split_lens, dim=0)
        for i, d in enumerate(missing_units):
            entry = CachedTubeletEmbeds(
                base_embed=split_packed[i].detach().to(self.store_device),
                deepstack_embeds=[],
            )
            self.put(d["key"], entry)
            plan.cached_map[d["key"]] = entry


@dataclass
class MissPlan:
    """Resolved hit/miss layout for one request's window (output of ``collect_misses``).

    Carries everything needed to populate the cache after a ViT forward and to
    reassemble the packed window in original tubelet order without re-deriving
    keys or order.
    """

    cache: "VitTubeletCache"
    device: Any
    unit_desc: list[dict[str, Any]]
    cached_map: dict[TubeletKey, CachedTubeletEmbeds]
    missing_units: list[dict[str, Any]]
    hit: int
    miss: int

    @property
    def miss_pixels(self) -> "torch.Tensor | None":
        """Concatenated miss pixels for this plan (None if all hits)."""
        import torch

        if not self.missing_units:
            return None
        return torch.cat([d["pixels"] for d in self.missing_units], dim=0)

    @property
    def miss_grid_rows(self) -> list[list[int]]:
        """``[[1, h, w], ...]`` per miss tubelet (the ViT grid for this plan's misses)."""
        return [[1, d["h"], d["w"]] for d in self.missing_units]

    def reassemble(self, dtype: Any) -> "torch.Tensor":
        """Phase 3: concat cached tubelet embeds in original order into the packed window.

        Every tubelet must be present in ``cached_map`` (all hits, or
        ``apply_miss_embeds`` already ran for the misses).
        """
        import torch

        base_parts: list[torch.Tensor] = []
        deep_parts: list[list[torch.Tensor]] = []
        for d in self.unit_desc:
            entry = self.cached_map[d["key"]]
            base_parts.append(entry.base_embed.to(device=self.device, dtype=dtype))
            if not deep_parts:
                deep_parts = [[] for _ in range(len(entry.deepstack_embeds))]
            for li, ds in enumerate(entry.deepstack_embeds):
                deep_parts[li].append(ds.to(device=self.device, dtype=dtype))

        base_embeds = torch.cat(base_parts, dim=0)
        deepstack_list = [torch.cat(parts, dim=0) for parts in deep_parts]
        return torch.cat([base_embeds] + deepstack_list, dim=1)
