"""Drop-in accelerated `merge_kv_cache` for the published IKOD configuration (identical outputs).

Profiling the published decoder (LLaVA-1.5, 128-token captions) put 60% of the wall clock inside
`merge_kv_cache` and only 32% inside the two model forwards. The merge cost is not the arithmetic:
at T=700 the bucket means take 15 ms and the gathers 5 ms of a 151 ms call. The remaining ~130 ms is
per-layer host synchronisation -- `region_idx[is_anchor]` is boolean-mask indexing, whose output size
is unknown to the host, so every layer forces a device-to-host copy; with keys and values that is 64
synchronisations per decoding step, each of which stalls behind whatever else the GPU is running.

This module computes every index for all layers in one batched pass (topk instead of a boolean mask,
so shapes are static and nothing synchronises) and keeps the frozen per-layer tensor operations
unchanged. Only the default published path is re-implemented (anchor_strategy='low',
merge_mode='mean'); anything else falls back to the frozen function.
"""
from __future__ import annotations

import torch

from . import kv_merge as KM


@torch.no_grad()
def merge_kv_cache_fast(kv_list, score, start: int, T: int, ratio: float, protect_recent: int = 1,
                        anchor_strategy: str = "low", merge_mode: str = "mean", append_slots: int = 0):
    num_layers = len(kv_list)
    region_len = T - start - protect_recent
    select_num = int(round(ratio * region_len))
    if (anchor_strategy != "low" or merge_mode != "mean" or select_num <= 0 or region_len <= 0
            or select_num >= region_len):
        return KM.merge_kv_cache(kv_list, score, start, T, ratio, protect_recent,
                                 anchor_strategy=anchor_strategy, merge_mode=merge_mode,
                                 append_slots=append_slots)
    dev = kv_list[0][0].device
    s = score[:num_layers, start:T - protect_recent]                       # [L, region]
    order = torch.argsort(s, dim=-1, stable=True)                          # ascending, ties by position
    anchor_local = torch.sort(order[:, :select_num], dim=-1).values         # [L, select], ascending
    pos = torch.arange(region_len, device=dev).expand(num_layers, region_len)
    right = torch.searchsorted(anchor_local, pos.contiguous()).clamp(max=select_num - 1)
    left = (right - 1).clamp(min=0)
    d_left = (pos - anchor_local.gather(1, left)).abs()
    d_right = (anchor_local.gather(1, right) - pos).abs()
    assign = torch.where(d_left <= d_right, left, right)                    # [L, region], monotone
    counts = torch.zeros(num_layers, select_num, device=dev, dtype=torch.float32)
    counts.scatter_add_(1, assign, torch.ones_like(assign, dtype=torch.float32))
    front = torch.arange(0, start, device=dev)
    tail = torch.arange(T - protect_recent, T, device=dev)
    anchor_idx_all = anchor_local + start                                   # [L, select] original positions
    kept_all_idx = torch.cat([front.expand(num_layers, start), anchor_idx_all,
                              tail.expand(num_layers, protect_recent)], dim=1)   # [L, kept]
    kept_len = kept_all_idx.shape[1]

    merged, kept_all = [], []
    for i, (k, v) in enumerate(kv_list):
        kept = kept_all_idx[i]
        a_local = anchor_local[i]
        # bucket means over the region, then write them at the anchors' slots (frozen semantics)
        w = torch.nn.functional.one_hot(assign[i], num_classes=select_num).transpose(0, 1).to(torch.float32)
        w = w / counts[i].unsqueeze(-1).clamp_min(1.0)
        kk_full = k.index_select(-2, kept)
        vv_full = v.index_select(-2, kept)
        reg = slice(start, start + select_num)
        kk_full[..., reg, :] = torch.matmul(w, k[..., start:T - protect_recent, :].to(torch.float32)).to(k.dtype)
        vv_full[..., reg, :] = torch.matmul(w, v[..., start:T - protect_recent, :].to(torch.float32)).to(v.dtype)
        if append_slots:
            kb = torch.empty((*kk_full.shape[:-2], kept_len + append_slots, kk_full.shape[-1]), dtype=k.dtype, device=dev)
            vb = torch.empty((*vv_full.shape[:-2], kept_len + append_slots, vv_full.shape[-1]), dtype=v.dtype, device=dev)
            kb[..., :kept_len, :].copy_(kk_full); vb[..., :kept_len, :].copy_(vv_full)
            kk_full, vv_full = kb, vb
        merged.append([kk_full, vv_full]); kept_all.append(kept)
        del w, a_local
    return merged, kept_all


def install() -> None:
    from . import decode as D
    D.merge_kv_cache = merge_kv_cache_fast
    print("[kv-merge-fast] batched index computation installed (no per-layer host sync)", flush=True)
