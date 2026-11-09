"""Image-attention-guided KV-cache merging (model-agnostic), faithful to the legacy
IKODCache (experiments/utils/kv_cache.py) but operating on a list of per-layer
(key, value) tensors + a persistent per-(layer,token) image-attention score table.

Legacy algorithm (per layer, independently), reproduced here:
  region = [start : T - protect_recent)           # text tokens, excl. image span & last token(s)
  select_num = round(ratio * (T - start - protect_recent))
  anchors  = the `select_num` LOWEST-image-attention tokens in region (these sit near the
             end / query → keep them); throw = the rest.
  each throw token's K/V is averaged into its nearest anchor (by |index distance|).
  kept order: [0:start] (image+system, untouched) ++ anchors ++ [T-protect_recent:T] (last).
Result: a shorter, more image-focused cache used for the second (contrastive) forward.

Score table convention: score[layer, t] = summed attention of token t (as query) onto the
image span, averaged over heads — accumulated across decoding steps by the caller.
"""
from __future__ import annotations
import torch


def _nearest_bucket_assignment(anchor_local: torch.Tensor, region_len: int) -> torch.Tensor:
    """Map every region position to its nearest sorted anchor index."""
    position = torch.arange(region_len, device=anchor_local.device)
    right = torch.searchsorted(anchor_local, position).clamp(max=anchor_local.numel() - 1)
    left = (right - 1).clamp(min=0)
    dist_left = (position - anchor_local[left]).abs()
    dist_right = (anchor_local[right] - position).abs()
    # Match argmin over ascending anchors: an equal-distance tie goes left.
    return torch.where(dist_left <= dist_right, left, right)


def _deterministic_bucket_mean(values: torch.Tensor,
                               anchor_local: torch.Tensor) -> torch.Tensor:
    """Average ``[..., region, dim]`` into nearest-anchor buckets.

    CUDA scatter reductions use racing atomic additions when several entries
    target one anchor. Low-order differences can change an autoregressive
    greedy trajectory. A row-normalized assignment matrix performs the same
    bucket means in FP32 without many-to-one atomic writes.
    """
    assignment = _nearest_bucket_assignment(anchor_local, values.shape[-2])
    weights = torch.nn.functional.one_hot(
        assignment, num_classes=anchor_local.numel()
    ).transpose(0, 1).to(torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    return torch.matmul(weights, values.to(torch.float32)).to(values.dtype)


@torch.no_grad()
def update_score_table(score: torch.Tensor, attentions, image_start: int, image_end: int,
                       num_tokens: int):
    """Update the persistent per-(layer,token) image-attention score table in place.
    `attentions`: tuple/list over layers of [b, h, q, kv] (b assumed 1).
    On prefill q==num_tokens (fill all); on a decode step q==1 (fill last position)."""
    L = len(attentions)
    a0 = attentions[0]
    q = a0.shape[-2]
    for i in range(L):
        a = attentions[i][0]                              # [h, q, kv]
        img = a[:, :, image_start:image_end].sum(-1)      # [h, q] attention to image
        img = img.mean(0)                                 # [q] mean over heads
        if q > 1:                                         # prefill
            score[i, :num_tokens] = img
        else:                                             # decode step
            score[i, num_tokens - 1] = img[-1]
    return score


@torch.no_grad()
def merge_kv_cache(kv_list, score: torch.Tensor, start: int, T: int,
                   ratio: float, protect_recent: int = 1,
                   anchor_strategy: str = "low", merge_mode: str = "mean",
                   append_slots: int = 0):
    """Build a merged (shorter) KV cache. Returns (merged_kv_list, kept_positions_per_layer).

    kv_list: list over layers of (k, v), each [b, n_kv_heads, T, head_dim].
    score:   [num_layers, >=T] image-attention score table (rows used: [:, :T]).
    start:   first text-token index (== image_end; image+system kept untouched).
    T:       current cache length (== num_tokens).
    ``merge_mode='mean'`` averages every non-anchor into its nearest anchor (the
    published operation). ``merge_mode='prune'`` keeps the identical anchors and
    directly drops all non-anchors, providing a same-length control that isolates
    aggregation from generic cache shortening. ``append_slots`` reserves
    uninitialized positions after the compact cache.
    The decoder uses one slot for the replay query, allowing its cache object to
    append in place instead of concatenating (and recopying) a long visual
    prefix in every layer.  The logical compact length remains ``len(kept)``.

    Returns merged_kv_list (list of (k,v) with shorter seq) and kept_positions
    (LongTensor of original indices retained, per layer) for logical-length and
    provenance accounting by the caller. Retained RoPE keys are not re-rotated.
    """
    num_layers = len(kv_list)
    region_len = T - start - protect_recent
    select_num = int(round(ratio * region_len))
    if append_slots < 0:
        raise ValueError(f"append_slots must be non-negative, got {append_slots}")
    if merge_mode not in ("mean", "prune"):
        raise ValueError(f"unknown merge_mode={merge_mode!r}")
    if select_num <= 0 or region_len <= 0:
        # nothing to merge; return original + identity positions
        pos = torch.arange(T, device=kv_list[0][0].device)
        if append_slots == 0:
            return [list(kv) for kv in kv_list], [pos for _ in range(num_layers)]
        copied = []
        for k, v in kv_list:
            shape = (*k.shape[:-2], T + append_slots, k.shape[-1])
            kk = torch.empty(shape, dtype=k.dtype, device=k.device)
            vv = torch.empty(shape, dtype=v.dtype, device=v.device)
            kk[..., :T, :].copy_(k)
            vv[..., :T, :].copy_(v)
            copied.append([kk, vv])
        return copied, [pos for _ in range(num_layers)]

    merged, kept_all = [], []
    front = torch.arange(0, start, device=kv_list[0][0].device)
    tail = torch.arange(T - protect_recent, T, device=kv_list[0][0].device)

    for i, (k, v) in enumerate(kv_list):
        dev = k.device
        s = score[i, start:T - protect_recent]                      # [region_len]
        if anchor_strategy == "low":
            order = torch.argsort(s, stable=True)                   # ascending; position breaks ties
            rank = torch.empty_like(order); rank[order] = torch.arange(region_len, device=dev)
            is_anchor = rank < select_num
        elif anchor_strategy == "high":
            order = torch.argsort(s, descending=True, stable=True)  # descending; position breaks ties
            rank = torch.empty_like(order); rank[order] = torch.arange(region_len, device=dev)
            is_anchor = rank < select_num
        elif anchor_strategy == "recent":
            # Position-only control: keep the latest mergeable entries, without
            # consulting image attention. The separately protected latest KV is
            # outside this region.
            is_anchor = torch.zeros(region_len, dtype=torch.bool, device=dev)
            is_anchor[-select_num:] = True
        elif anchor_strategy == "uniform":
            # Deterministic position-only control. Choose one midpoint from each
            # of ``select_num`` equal-width regions, without consulting attention.
            # Integer midpoint arithmetic is unique whenever select_num <= region_len.
            j = torch.arange(select_num, device=dev)
            uniform = ((2 * j + 1) * region_len) // (2 * select_num)
            is_anchor = torch.zeros(region_len, dtype=torch.bool, device=dev)
            is_anchor[uniform] = True
        else:
            raise ValueError(f"unknown anchor_strategy={anchor_strategy!r}")
        region_idx = torch.arange(start, T - protect_recent, device=dev)
        anchor_idx = region_idx[is_anchor]                          # original positions of anchors
        throw_idx = region_idx[~is_anchor]

        if anchor_idx.numel() == 0 or throw_idx.numel() == 0:
            kept = torch.cat([front, region_idx, tail])
            kk = k.index_select(-2, kept)
            vv = v.index_select(-2, kept)
            if append_slots:
                k_buf = torch.empty((*kk.shape[:-2], kk.shape[-2] + append_slots, kk.shape[-1]),
                                    dtype=kk.dtype, device=kk.device)
                v_buf = torch.empty((*vv.shape[:-2], vv.shape[-2] + append_slots, vv.shape[-1]),
                                    dtype=vv.dtype, device=vv.device)
                k_buf[..., :kk.shape[-2], :].copy_(kk)
                v_buf[..., :vv.shape[-2], :].copy_(vv)
                kk, vv = k_buf, v_buf
            merged.append([kk, vv])
            kept_all.append(kept)
            continue

        # Same-length compression control: preserve the selected anchors but do
        # not inject the discarded K/V states into them. This separates the
        # proposed bucketwise aggregation from benefits caused by merely exposing
        # a shorter auxiliary context.
        if merge_mode == "prune":
            kept = torch.cat([front, anchor_idx, tail])
            kk = k.index_select(-2, kept)
            vv = v.index_select(-2, kept)
            if append_slots:
                logical_len = kept.numel()
                out_shape = (*kk.shape[:-2], logical_len + append_slots, kk.shape[-1])
                k_buf = torch.empty(out_shape, dtype=kk.dtype, device=kk.device)
                v_buf = torch.empty(out_shape, dtype=vv.dtype, device=vv.device)
                k_buf[..., :logical_len, :].copy_(kk)
                v_buf[..., :logical_len, :].copy_(vv)
                kk, vv = k_buf, v_buf
            merged.append([kk, vv])
            kept_all.append(kept)
            continue

        # Represent the partition by a deterministic assignment matrix over the
        # mergeable text region. This removes racing atomic scatter additions
        # while preserving the mathematical bucket means.
        b, h, _, d = k.shape
        a = anchor_idx.numel()
        anchor_local = anchor_idx - start
        k_anchor = _deterministic_bucket_mean(
            k[..., start:T - protect_recent, :], anchor_local
        )
        v_anchor = _deterministic_bucket_mean(
            v[..., start:T - protect_recent, :], anchor_local
        )

        kept = torch.cat([front, anchor_idx, tail])                    # original positions kept
        logical_len = kept.numel()
        # Reserving an append slot is most useful for large multi-head visual
        # prefixes. Smaller grouped-query caches use a regular DynamicCache
        # append; both construction paths share the deterministic reduction.
        use_local_buffer = append_slots and start >= 1024 and start * h >= 16384
        if not use_local_buffer:
            kk = k.index_select(-2, kept)
            vv = v.index_select(-2, kept)
            kk[..., start:start + a, :].copy_(k_anchor)
            vv[..., start:start + a, :].copy_(v_anchor)
            merged.append([kk, vv])
            kept_all.append(kept)
            continue

        out_shape = (*k.shape[:-2], logical_len + append_slots, d)
        kk = torch.empty(out_shape, dtype=k.dtype, device=dev)
        vv = torch.empty(out_shape, dtype=v.dtype, device=dev)
        # The three destination slices are disjoint.  Direct slice copies avoid
        # a full-size index_select temporary and reserve the optional append slot
        # without another allocation.
        kk[..., :start, :].copy_(k[..., :start, :])
        vv[..., :start, :].copy_(v[..., :start, :])
        kk[..., start:start + a, :].copy_(k_anchor)
        vv[..., start:start + a, :].copy_(v_anchor)
        if protect_recent:
            kk[..., start + a:logical_len, :].copy_(k[..., T - protect_recent:T, :])
            vv[..., start + a:logical_len, :].copy_(v[..., T - protect_recent:T, :])
        merged.append([kk, vv])
        kept_all.append(kept)

    return merged, kept_all


if __name__ == "__main__":
    # self-test: shapes + that low-attention tokens are kept as anchors
    torch.manual_seed(0)
    L, H, T, D = 4, 8, 20, 16
    start, ratio, protect = 6, 0.4, 1                       # image+sys = [0:6]
    kv = [(torch.randn(1, H, T, D), torch.randn(1, H, T, D)) for _ in range(L)]
    score = torch.rand(L, T)
    # Use a deliberately nonmonotonic region so the attention-based and
    # position-only controls cannot pass by accidentally selecting the same set.
    region_scores = torch.tensor([.90, .10, .80, .20, .70, .30, .60,
                                  .40, .50, .05, .95, .15, .85])
    score[:, 6:19] = region_scores
    merged, kept = merge_kv_cache(kv, score, start, T, ratio, protect)
    region_len = T - start - protect                        # 13
    sel = int(round(ratio * region_len))                    # 5
    exp_len = start + sel + protect                         # 6 + 5 + 1 = 12
    assert merged[0][0].shape == (1, H, exp_len, D), merged[0][0].shape
    # Lowest-attention anchors are retained in original sequence order.
    anchors = kept[0][start:start + sel].tolist()
    expected_low = [7, 9, 11, 15, 17]
    assert anchors == expected_low, (anchors, "expected", expected_low)
    # no-merge path
    m2, k2 = merge_kv_cache(kv, score, start, T, ratio=0.0)
    assert m2[0][0].shape == (1, H, T, D)
    # The high-attention and recent-position controls select distinct sets.
    _, kept_high = merge_kv_cache(kv, score, start, T, ratio, protect,
                                  anchor_strategy="high")
    expected_high = [6, 8, 10, 16, 18]
    assert kept_high[0][start:start + sel].tolist() == expected_high
    _, kept_recent = merge_kv_cache(kv, score, start, T, ratio, protect,
                                    anchor_strategy="recent")
    expected_recent = list(range(T - protect - sel, T - protect))
    assert kept_recent[0][start:start + sel].tolist() == expected_recent
    _, kept_uniform = merge_kv_cache(kv, score, start, T, ratio, protect,
                                     anchor_strategy="uniform")
    expected_uniform_local = [((2 * j + 1) * region_len) // (2 * sel) for j in range(sel)]
    expected_uniform = [start + j for j in expected_uniform_local]
    assert kept_uniform[0][start:start + sel].tolist() == expected_uniform
    assert len({tuple(expected_low), tuple(expected_high), tuple(expected_recent),
                tuple(expected_uniform)}) == 4
    # The direct-pruning control uses exactly the same retained positions and
    # logical length as bucket-mean merging but leaves anchor values unchanged.
    pruned, kept_pruned = merge_kv_cache(kv, score, start, T, ratio, protect,
                                         merge_mode="prune", append_slots=1)
    assert torch.equal(kept_pruned[0], kept[0])
    assert pruned[0][0].shape[-2] == merged[0][0].shape[-2] + 1
    assert torch.equal(pruned[0][0][..., :exp_len, :],
                       kv[0][0].index_select(-2, kept[0]))
    # Implementation-level check of Theorem 1: for a fixed query, the mean-key
    # merge cannot lower attention on an unchanged image span.  Prefix [0:2)
    # acts as untouched text and [2:start) as image keys.
    q = torch.randn(1, H, 1, D)
    scale = D ** -0.5
    for (k_full, _), (k_short, _) in zip(kv, merged):
        p_full = torch.softmax(torch.matmul(q, k_full.transpose(-1, -2)) * scale, dim=-1)
        p_short = torch.softmax(torch.matmul(q, k_short.transpose(-1, -2)) * scale, dim=-1)
        image_full = p_full[..., 2:start].sum(-1)
        image_short = p_short[..., 2:start].sum(-1)
        assert torch.all(image_short >= image_full - 1e-6), (image_full, image_short)
    print(f"kv_merge self-test OK (T={T} -> merged_len={exp_len}, anchors={anchors})")
