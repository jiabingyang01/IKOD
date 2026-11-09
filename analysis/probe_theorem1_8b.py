"""Exact empirical verification of Theorem 1 (Two-Term Decomposition: Count Compression vs.
Jensen/Post-RoPE Phase Gap) and Value-Aggregation Semantic Fidelity across Anchor Selection
Rules on all 8 LVLM backbones.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import rerun0927_chair_cells_v1 as RC
from ikod.image_span import locate_image_span
from ikod.decode import _cache_to_list
from ikod.kv_merge import _nearest_bucket_assignment

IMG_DIR = Path("/data/yjb/datasets/coco/val2014")
PROMPT = "Please describe this image in detail."


def select_anchors(s_region: torch.Tensor, k_keep: int, strategy: str) -> torch.Tensor:
    """Return sorted 1D tensor of anchor indices in [0, R)."""
    R = int(s_region.numel())
    k_keep = max(1, min(R, k_keep))
    if strategy == "low":
        idx = torch.topk(s_region, k=k_keep, largest=False, sorted=False).indices
    elif strategy == "high":
        idx = torch.topk(s_region, k=k_keep, largest=True, sorted=False).indices
    elif strategy == "recent":
        idx = torch.arange(R - k_keep, R, device=s_region.device)
    elif strategy == "uniform":
        idx = torch.linspace(0, R - 1, steps=k_keep, device=s_region.device).round().long().unique()
    else:
        raise ValueError(strategy)
    return torch.sort(idx).values


@torch.no_grad()
def analyze_backbone(model_name: str, limit: int = 100, ratio: float = 0.4, out_path: str = ""):
    t0 = time.time()
    if model_name == "llava_15_hf":
        from ikod import mrope_decode_fix_v1 as fix
        fix.install()
        import run_llava_pad_v1  # noqa: F401
        from ikod.models import load_vlm, build_inputs
    elif model_name == "qwen3_vl":
        import ikod_qwen3_compat
        ikod_qwen3_compat.install()
        from ikod.models import load_vlm, build_inputs
    elif model_name == "instructblip_vicuna7b":
        import ikod_instructblip_compat as compat
        compat.install()
        load_vlm = lambda name: compat.load_vlm(name, device="cuda", dtype=torch.float32, eager=True)
        build_inputs = lambda name, proc, image, question: compat.build_inputs(name, proc, image, question, device="cuda", dtype=torch.float32)
    else:
        from ikod.models import load_vlm, build_inputs

    try:
        cap_path = RC.path(f"results/chair512_main_v1/{model_name}_greedy.jsonl")
    except BaseException:
        cap_path = Path("./workspace/results/chair512_main_v1") / f"{model_name}_greedy.jsonl"
    rows = [json.loads(line) for line in cap_path.read_text().splitlines()][:limit]

    model, proc, img_tok = load_vlm(model_name)
    model.eval()

    stats = {
        st: {
            "S_orig": [],
            "S_card": [],
            "S_merged": [],
            "dS_total": [],
            "dS_card": [],
            "dS_jensen": [],
            "strict_pos": [],
            "val_rel_l2_all": [],
            "val_cos_all": [],
            "val_rel_l2_grounded": [],
            "val_cos_grounded": [],
            "full_out_cos_grounded": [],
            "grounded_bucket_size": [],
            "lowattn_bucket_size": [],
        }
        for st in ("low", "uniform", "high", "recent", "prune_low", "prune_high")
    }
    rope_norm_ratios = []
    step_curve = {step_bin: {"S_orig": [], "S_card": [], "S_merged": []} for step_bin in (32, 64, 96, 128, 160, 192)}

    n_processed = 0
    for r_idx, row in enumerate(rows):
        img_file = IMG_DIR / row["image"]
        if not img_file.exists():
            continue
        image = Image.open(img_file).convert("RGB")
        prompt_inputs = build_inputs(model_name, proc, image, PROMPT)
        p_ids = prompt_inputs["input_ids"]
        p_len = int(p_ids.shape[1])

        tok = getattr(proc, "tokenizer", proc)
        if hasattr(tok, " tokenizer"):
            tok = tok.tokenizer
        cap_ids = tok.encode(row["text"], add_special_tokens=False)
        if len(cap_ids) < 24:
            continue
        cap_ids = cap_ids[:220]
        cap_tensor = torch.tensor([cap_ids], dtype=p_ids.dtype, device=p_ids.device)
        full_ids = torch.cat([p_ids, cap_tensor], dim=1)

        fwd_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
        # For Qwen2-VL / Qwen2.5-VL / Qwen3-VL, ensure cache_position / position_ids not fixed to prompt length
        fwd_kwargs.pop("position_ids", None)
        fwd_kwargs.pop("cache_position", None)
        full_am = torch.ones_like(full_ids)

        try:
            out = model(input_ids=full_ids, attention_mask=full_am,
                        output_attentions=True, use_cache=True, return_dict=True, **fwd_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        attns = out.attentions
        kv_list = _cache_to_list(out.past_key_values)

        span = locate_image_span(full_ids, img_tok)
        if span is None:
            continue
        im_start, im_end = span
        if im_end <= im_start:
            continue

        T_total = int(full_ids.shape[1])
        L = len(attns)
        layer_indices = list(range(max(0, L // 4), L, max(1, L // 8)))

        s_table = torch.stack([
            attns[l][0, :, :, im_start:im_end].sum(dim=-1).mean(dim=0).float()
            for l in range(L)
        ], dim=0)  # [L, T_total]

        gen_positions = [p_len + offset for offset in (32, 64, 96, 128, 160, 192) if p_len + offset < T_total - 1]
        if not gen_positions and T_total > p_len + 16:
            gen_positions = [T_total - 2]

        for t in gen_positions:
            offset = t - p_len
            reg_start = im_end
            reg_end = t - 1
            R = reg_end - reg_start
            if R < 10:
                continue
            k_keep = max(1, int(round(ratio * R)))

            for l in layer_indices:
                s_reg = s_table[l, reg_start:reg_end]
                q75 = torch.quantile(s_reg, 0.75)
                q50 = torch.quantile(s_reg, 0.50)
                grounded_mask = (s_reg >= q75)
                lowattn_mask = (s_reg <= q50)
                is_grounded_step = bool(s_table[l, t] >= torch.quantile(s_table[l, p_len:T_total], 0.65))

                probs_t = attns[l][0, :, t, :t + 1].float().clamp(min=1e-12)
                H = probs_t.shape[0]

                k_l, v_l = kv_list[l]
                k_reg = k_l[0, :, reg_start:reg_end, :].float()
                v_reg = v_l[0, :, reg_start:reg_end, :].float()
                H_kv = v_reg.shape[0]
                kv_rep = H // H_kv
                v_reg_h = v_reg.repeat_interleave(kv_rep, dim=0) if kv_rep > 1 else v_reg

                S_I = probs_t[:, im_start:im_end].sum(dim=-1)
                S_U = probs_t[:, :im_start].sum(dim=-1) + probs_t[:, reg_end:t + 1].sum(dim=-1)
                probs_reg = probs_t[:, reg_start:reg_end]
                S_P = probs_reg.sum(dim=-1).clamp(min=1e-9)

                w_P_full = probs_reg / S_P.unsqueeze(-1)
                o_P_full = torch.bmm(w_P_full.unsqueeze(1), v_reg_h).squeeze(1)

                v_all_h = v_l[0, :, :t + 1, :].float()
                if kv_rep > 1:
                    v_all_h = v_all_h.repeat_interleave(kv_rep, dim=0)
                o_full_total = torch.bmm(probs_t.unsqueeze(1), v_all_h).squeeze(1)
                o_nonP = o_full_total - S_P.unsqueeze(-1) * o_P_full

                for st_name in ("low", "uniform", "high", "recent", "prune_low", "prune_high"):
                    base_st = "low" if st_name == "prune_low" else ("high" if st_name == "prune_high" else st_name)
                    anchors = select_anchors(s_reg, k_keep, base_st)
                    B_num = int(anchors.numel())

                    if st_name.startswith("prune"):
                        p_kept = probs_reg[:, anchors]
                        A_prime_P = p_kept.sum(dim=-1).clamp(min=1e-9)
                        A_card_P = A_prime_P
                        Z_prime = S_I + S_U + A_prime_P
                        S_prime = S_I / Z_prime
                        S_card = S_prime
                        w_P_merged = p_kept / A_prime_P.unsqueeze(-1)
                        v_kept_h = v_reg_h[:, anchors, :]
                        o_P_merged = torch.bmm(w_P_merged.unsqueeze(1), v_kept_h).squeeze(1)
                        g_bsize = 1.0
                        l_bsize = 1.0
                    else:
                        assign = _nearest_bucket_assignment(anchors, R)
                        onehot = F.one_hot(assign, num_classes=B_num).transpose(0, 1).float()
                        counts = onehot.sum(dim=-1).clamp(min=1.0)
                        W_mean = onehot / counts.unsqueeze(-1)

                        token_bsizes = counts[assign]
                        g_bsize = float(token_bsizes[grounded_mask].mean().item()) if grounded_mask.any() else 1.0
                        l_bsize = float(token_bsizes[lowattn_mask].mean().item()) if lowattn_mask.any() else 1.0

                        bucket_prob_mean = torch.matmul(probs_reg, W_mean.T)
                        A_card_P = bucket_prob_mean.sum(dim=-1)

                        log_probs_reg = torch.log(probs_reg)
                        bucket_log_mean = torch.matmul(log_probs_reg, W_mean.T)
                        p_bucket_geom = torch.exp(bucket_log_mean)
                        A_prime_P = p_bucket_geom.sum(dim=-1).clamp(min=1e-9)

                        Z_card = S_I + S_U + A_card_P
                        Z_prime = S_I + S_U + A_prime_P
                        S_card = S_I / Z_card
                        S_prime = S_I / Z_prime

                        v_bar_h = torch.matmul(W_mean.unsqueeze(0), v_reg_h)
                        w_P_merged = p_bucket_geom / A_prime_P.unsqueeze(-1)
                        o_P_merged = torch.bmm(w_P_merged.unsqueeze(1), v_bar_h).squeeze(1)

                        if st_name == "low":
                            multi_mask = (counts >= 2.0)
                            if multi_mask.any():
                                k_bar = torch.matmul(W_mean.unsqueeze(0), k_reg)
                                k_bar_norm = k_bar.norm(dim=-1)
                                k_orig_norm = k_reg.norm(dim=-1)
                                mean_orig_norm = torch.matmul(k_orig_norm, W_mean.T).clamp(min=1e-9)
                                ratio_multi = (k_bar_norm[:, multi_mask] / mean_orig_norm[:, multi_mask]).mean().item()
                                rope_norm_ratios.append(float(ratio_multi))

                    s0 = float(S_I.mean().item()) * 100.0
                    sc = float(S_card.mean().item()) * 100.0
                    sm = float(S_prime.mean().item()) * 100.0
                    ds_tot = sm - s0
                    ds_crd = sc - s0
                    ds_jen = sm - sc
                    pos_rate = float((S_prime > S_I).float().mean().item()) * 100.0

                    diff_norm = (o_P_merged - o_P_full).norm(dim=-1)
                    full_norm = o_P_full.norm(dim=-1).clamp(min=1e-9)
                    rel_l2 = float((diff_norm / full_norm).mean().item()) * 100.0
                    cos_P = float(F.cosine_similarity(o_P_merged, o_P_full, dim=-1).mean().item()) * 100.0

                    o_merged_total = (o_nonP + A_prime_P.unsqueeze(-1) * o_P_merged) / (S_I + S_U + A_prime_P).unsqueeze(-1)
                    cos_total = float(F.cosine_similarity(o_merged_total, o_full_total, dim=-1).mean().item()) * 100.0

                    d = stats[st_name]
                    d["S_orig"].append(s0)
                    d["S_card"].append(sc)
                    d["S_merged"].append(sm)
                    d["dS_total"].append(ds_tot)
                    d["dS_card"].append(ds_crd)
                    d["dS_jensen"].append(ds_jen)
                    d["strict_pos"].append(pos_rate)
                    d["val_rel_l2_all"].append(rel_l2)
                    d["val_cos_all"].append(cos_P)
                    d["grounded_bucket_size"].append(g_bsize)
                    d["lowattn_bucket_size"].append(l_bsize)
                    if is_grounded_step:
                        d["val_rel_l2_grounded"].append(rel_l2)
                        d["val_cos_grounded"].append(cos_P)
                        d["full_out_cos_grounded"].append(cos_total)

                    if st_name == "low" and offset in step_curve:
                        step_curve[offset]["S_orig"].append(s0)
                        step_curve[offset]["S_card"].append(sc)
                        step_curve[offset]["S_merged"].append(sm)

        del out, attns, kv_list
        torch.cuda.empty_cache()
        n_processed += 1
        if n_processed % 20 == 0:
            print(f"[{model_name}] {n_processed}/{len(rows)} images ({time.time()-t0:.1f}s)", flush=True)

    summary = {
        "model": model_name,
        "n_images": n_processed,
        "ratio": ratio,
        "rope_key_norm_ratio_multibucket": float(np.mean(rope_norm_ratios)) if rope_norm_ratios else None,
        "strategies": {},
        "step_curve": {},
    }
    for st_name, d in stats.items():
        summary["strategies"][st_name] = {k: (float(np.mean(v)) if v else None) for k, v in d.items()}
        if d["dS_total"]:
            tot = float(np.mean(d["dS_total"]))
            crd = float(np.mean(d["dS_card"]))
            jen = float(np.mean(d["dS_jensen"]))
            summary["strategies"][st_name]["card_share_pct"] = (crd / tot * 100.0) if tot > 0 else 0.0
            summary["strategies"][st_name]["jensen_share_pct"] = (jen / tot * 100.0) if tot > 0 else 0.0

    for offset, sc in step_curve.items():
        if sc["S_orig"]:
            summary["step_curve"][str(offset)] = {k: float(np.mean(v)) for k, v in sc.items()}

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"DONE {model_name} in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ratio", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    analyze_backbone(args.model, args.limit, args.ratio, args.out)
