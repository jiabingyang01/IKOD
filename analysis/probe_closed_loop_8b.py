#!/usr/bin/env python3
"""Unified 8-Backbone Closed-Loop Mechanistic Analysis on CHAIR:
1. Experiment 1: Dataset-wide visual attention restoration (S'_t vs S_t across early/late stages
   and Grounded vs. Hallucinated object steps), online decay gate response (alpha_t), and
   step-level Hallucination Reversal vs. Grounded Object Preservation.
2. Experiment 2: Inside KV merging buckets across Anchor/Aggregation rules (Low-attn centroid [IKOD],
   Uniform centroid, High-attn centroid, Hard pruning [low-attn]): lexical composition of selected
   anchors (function/template tokens vs. grounded object nouns), late-span bucket granularity,
   merged-branch visual attention gain dS_t, hallucination reversal rate, and grounded object
   preservation / representation fidelity.
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NLTK_DATA", "/data/yjb/nltk_data")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from chair_score_std_batch_v1 import evaluator
from ikod.image_span import locate_image_span
from ikod.decode import _cache_to_list, _list_to_dynamic_cache
from ikod.kv_merge import update_score_table, merge_kv_cache, _nearest_bucket_assignment
from ikod.adaptive import AdaptiveAlpha, compute_image_attention

SHM_IMG_DIR = Path("/dev/shm/chair500_images")
FALLBACK_IMG_DIR = Path("/data/yjb/datasets/coco/val2014")
PROMPT = "Please describe this image in detail."

MODEL_ALPHA = {
    "llava_15_hf": 2.0,
    "llava_next": 2.0,
    "llava_ov": 2.0,
    "qwen2_vl": 2.0,
    "qwen2_5_vl": 2.0,
    "qwen3_vl": 2.0,
    "internvl3": 0.3,
    "instructblip_vicuna7b": 0.5,
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "up", "about",
    "into", "over", "after", "beneath", "under", "above", "and", "or", "but",
    "as", "if", "while", "that", "this", "these", "those", "it", "its", "they",
    "them", "their", "there", "here", "which", "who", "whom", "can", "also",
    "very", "some", "other", "another", "each", "both", "such", "no", "not",
    "has", "have", "had", "do", "does", "did", "will", "would", "could", "should",
    ",", ".", ";", ":", "!", "?", "-", "(", ")", "'", '"', "'s", "s"
}


def load_rgb_image(fname: str) -> Image.Image | None:
    for d in (SHM_IMG_DIR, FALLBACK_IMG_DIR):
        p = d / fname
        if p.exists():
            for _ in range(3):
                try:
                    return Image.open(p).convert("RGB")
                except OSError:
                    time.sleep(0.5)
    return None


def align_chair_objects_to_tokens(tok, cap_ids: list[int], chair, imid: int):
    gt = chair.imid_to_objects.get(imid, set())
    prefix_strs = [tok.decode(cap_ids[:k + 1], skip_special_tokens=True) for k in range(len(cap_ids))]
    full_cap = prefix_strs[-1] if prefix_strs else ""
    spans = []
    prev = 0
    for s in prefix_strs:
        spans.append((prev, len(s)))
        prev = len(s)

    words, node_words, idxs, double_words = chair.caption_to_words(full_cap)
    lower_cap = full_cap.lower()

    grounded_steps = {}
    hal_steps = {}
    grounded_tok_mask = np.zeros(len(cap_ids), dtype=bool)
    func_tok_mask = np.zeros(len(cap_ids), dtype=bool)

    for k in range(len(cap_ids)):
        piece = tok.decode([cap_ids[k]], skip_special_tokens=True).strip().lower()
        if not piece or piece in STOPWORDS or all(not ch.isalnum() for ch in piece):
            func_tok_mask[k] = True

    denom = max(1, len(double_words) - 1)
    for surface, nw, wi in zip(words, node_words, idxs):
        head_w = surface.split()[0]
        approx_c = (wi / denom) * len(lower_cap)
        pat = r"\b" + re.escape(head_w[:max(3, len(head_w) - 1)])
        matches = [m.start() for m in re.finditer(pat, lower_cap)]
        c_pos = min(matches, key=lambda x: abs(x - approx_c)) if matches else int(approx_c)
        c_end = c_pos + len(surface)
        t_idx = None
        for k, (ts, te) in enumerate(spans):
            if te > c_pos and ts < c_end:
                if t_idx is None:
                    t_idx = k
                if nw in gt:
                    grounded_tok_mask[k] = True
            elif ts >= c_end:
                break
        if t_idx is not None and t_idx >= 4:
            if nw in gt:
                grounded_steps.setdefault(t_idx, (surface, nw))
            else:
                hal_steps.setdefault(t_idx, (surface, nw))

    return grounded_steps, hal_steps, grounded_tok_mask, func_tok_mask


def compute_thm1_visual_share(attentions, im_start: int, im_end: int, Lp: int) -> float:
    """Compute layer-averaged visual attention share of the query token over unique compressed positions."""
    vals = []
    for a in attentions:
        row = a[0, :, -1, :].float()  # [H, Lp + 1]
        img_mass = row[:, im_start:im_end].sum(-1)  # [H]
        if row.shape[-1] == Lp + 1 and Lp > im_end:
            dup_mass = row[:, Lp - 1]
            denom = (1.0 - dup_mass).clamp(min=1e-6)
            vals.append((img_mass / denom).mean())
        else:
            vals.append(img_mass.mean())
    return float(torch.stack(vals).mean().item())


@torch.no_grad()
def run_backbone(model_name: str, limit: int = 120, ratio: float = 0.4, out_path: str = ""):
    t0 = time.time()
    if model_name == "llava_15_hf":
        from ikod import mrope_decode_fix_v1 as fix
        fix.install()
        import run_llava_pad_v1
        run_llava_pad_v1.install()
        from ikod.models import load_vlm, build_inputs
    elif model_name == "qwen3_vl":
        import ikod_qwen3_compat
        ikod_qwen3_compat.install()
        from ikod.models import load_vlm, build_inputs
    elif model_name == "instructblip_vicuna7b":
        import ikod_instructblip_compat as compat
        compat.install()
        load_vlm = lambda name: compat.load_vlm(name, device="cuda", dtype=torch.float32, eager=True)
        build_inputs = lambda name, proc, image, question: compat.build_inputs(
            name, proc, image, question, device="cuda", dtype=torch.float32
        )
    else:
        from ikod import mrope_decode_fix_v1 as fix
        fix.install()
        from ikod.models import load_vlm, build_inputs

    chair = evaluator()
    cap_path = Path(f"./workspace/results/chair512_main_v1/{model_name}_greedy.jsonl")
    all_rows = [json.loads(line) for line in cap_path.read_text().splitlines() if line.strip()]
    hal_rows, clean_rows = [], []
    for r in all_rows:
        imid = int(r["image"].split("_")[-1].split(".")[0])
        gt = chair.imid_to_objects.get(imid, set())
        _, nws, _, _ = chair.caption_to_words(r["text"])
        if any(nw not in gt for nw in nws):
            hal_rows.append(r)
        else:
            clean_rows.append(r)
    rows = (hal_rows + clean_rows)[:limit]

    model, proc, img_tok = load_vlm(model_name)
    model.eval()
    tok = getattr(proc, "tokenizer", proc)
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer

    is_mrope = model_name in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl")
    alpha_fixed = MODEL_ALPHA.get(model_name, 2.0)

    exp1 = {
        "S_early_base": [], "S_early_merged": [], "S_early_thm1": [],
        "S_late_base": [], "S_late_merged": [], "S_late_thm1": [],
        "S_gt_base": [], "S_gt_merged": [], "S_gt_thm1": [],
        "S_hal_base": [], "S_hal_merged": [], "S_hal_thm1": [],
        "alpha_early": [], "alpha_late": [], "alpha_gt": [], "alpha_hal": [],
        "thm1_pos_all": [], "thm1_pos_late": [], "thm1_pos_hal": [],
        "hal_in_late_frac": [],
    }

    strats = ("low", "uniform", "high", "prune_low")
    exp2 = {
        st: {
            "S_merged_late": [], "S_thm1_late": [], "dS_late": [], "dS_thm1_late": [],
            "S_merged_hal": [], "S_thm1_hal": [], "dS_hal": [], "dS_thm1_hal": [],
            "hal_flip_fixed": [], "hal_flip_adap": [], "hal_flip_merged": [],
            "hal_prob_chg_merged": [], "hal_prob_chg_fixed": [], "hal_prob_chg_adap": [],
            "hal_rel_prob_drop": [],
            "gt_keep_fixed": [], "gt_keep_adap": [],
            "gt_prob_chg_fixed": [],
            "func_anchor_frac": [], "gt_anchor_frac": [],
            "late_span_bsize": [], "early_span_bsize": [],
        }
        for st in strats
    }

    n_done = 0
    for r in rows:
        imid = int(r["image"].split("_")[-1].split(".")[0])
        image = load_rgb_image(r["image"])
        if image is None:
            continue
        prompt_inputs = build_inputs(model_name, proc, image, PROMPT)
        p_ids = prompt_inputs["input_ids"]
        P = int(p_ids.shape[1])

        cap_ids = tok.encode(r["text"], add_special_tokens=False)
        if len(cap_ids) < 20:
            continue
        cap_ids = cap_ids[:240]
        M = len(cap_ids)

        grounded_steps, hal_steps, grounded_tok_mask, func_tok_mask = align_chair_objects_to_tokens(
            tok, cap_ids, chair, imid
        )
        for hs in hal_steps:
            exp1["hal_in_late_frac"].append(float(hs >= 0.5 * M) * 100.0)

        cap_t = torch.tensor([cap_ids], dtype=p_ids.dtype, device=p_ids.device)
        full_ids = torch.cat([p_ids, cap_t], dim=1)
        full_am = torch.ones_like(full_ids)
        fwd_kw = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
        fwd_kw.pop("position_ids", None)
        fwd_kw.pop("cache_position", None)

        try:
            out = model(
                input_ids=full_ids, attention_mask=full_am,
                output_attentions=True, use_cache=True, return_dict=True, **fwd_kw
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        kv_list_full = _cache_to_list(out.past_key_values)
        span = locate_image_span(full_ids[0], img_tok)
        if span is None:
            del out, kv_list_full
            continue
        im_start, im_end = span
        L_layers = len(out.attentions)
        T_total = int(full_ids.shape[1])

        score = torch.zeros(L_layers, T_total + 4, device=full_ids.device)
        update_score_table(score, out.attentions, im_start, im_end, T_total)

        pos3d_full = None
        if is_mrope:
            grid = prompt_inputs.get("image_grid_thw")
            pos3d_full, _ = model.model.get_rope_index(full_ids, image_grid_thw=grid, attention_mask=full_am)

        adap = AdaptiveAlpha(mode="p1", alpha_min=0.0, alpha_max=1.0, kappa=10.0)
        step_S = []
        step_alpha = []
        for s_idx in range(M):
            q_pos = P - 1 + s_idx
            S_val = float(score[:, q_pos].mean().item())
            a_val = adap.step(S_t=S_val)
            step_S.append(S_val)
            step_alpha.append(a_val)
            if s_idx <= int(0.25 * M):
                exp1["alpha_early"].append(a_val)
            elif s_idx >= int(0.50 * M):
                exp1["alpha_late"].append(a_val)

        early_probe = min(max(8, int(0.20 * M)), M - 2)
        late_probe = min(max(16, int(0.75 * M)), M - 2)
        gt_probe_list = sorted(grounded_steps.keys())[:4]
        hal_probe_list = sorted(hal_steps.keys())[:5]
        probe_set = sorted(set([early_probe, late_probe] + gt_probe_list + hal_probe_list))

        post_vis_len = T_total - im_end
        full_func_mask = np.ones(post_vis_len, dtype=bool)
        full_gt_mask = np.zeros(post_vis_len, dtype=bool)
        gen_offset = P - im_end
        for k in range(M):
            if 0 <= gen_offset + k < post_vis_len:
                full_func_mask[gen_offset + k] = func_tok_mask[k]
                full_gt_mask[gen_offset + k] = grounded_tok_mask[k]

        for s_idx in probe_set:
            cur_len = P + s_idx
            q_pos = cur_len - 1
            R = cur_len - im_end - 1
            if R < 8:
                continue
            target_tok = cap_ids[s_idx]
            logits_base = out.logits[:, q_pos, :].float()
            probs_base = F.softmax(logits_base, dim=-1)
            pred_base = int(logits_base.argmax(dim=-1).item())
            S_b = step_S[s_idx]
            a_t = step_alpha[s_idx]

            kv_slice = [(k[..., :cur_len, :], v[..., :cur_len, :]) for k, v in kv_list_full]

            for st_key, (ast, mm) in [
                ("low", ("low", "mean")),
                ("uniform", ("uniform", "mean")),
                ("high", ("high", "mean")),
                ("prune_low", ("low", "prune")),
            ]:
                merged_kv, kept = merge_kv_cache(
                    kv_slice, score, im_end, cur_len, ratio, protect_recent=1,
                    anchor_strategy=ast, merge_mode=mm, append_slots=1
                )
                Lp = int(kept[0].numel())
                m_cache = _list_to_dynamic_cache([[k[..., :Lp, :], v[..., :Lp, :]] for k, v in merged_kv])
                last_tok = full_ids[:, q_pos : q_pos + 1]
                c_pos = torch.tensor([Lp], device=full_ids.device)
                if is_mrope:
                    pos = pos3d_full[..., q_pos : q_pos + 1]
                else:
                    pos = torch.tensor([[Lp]], device=full_ids.device)

                out_m = model(
                    input_ids=last_tok, past_key_values=m_cache, position_ids=pos, cache_position=c_pos,
                    use_cache=True, output_attentions=True, return_dict=True
                )
                S_m_raw = float(compute_image_attention(out_m.attentions, im_start, im_end)[0])
                S_m_thm1 = compute_thm1_visual_share(out_m.attentions, im_start, im_end, Lp)
                logits_m = out_m.logits[:, -1, :].float()
                del out_m, m_cache, merged_kv

                cutoff = math.log(0.1) + logits_base.max(dim=-1, keepdim=True).values
                fused_fixed = (logits_base + alpha_fixed * logits_m).masked_fill(logits_base < cutoff, float("-inf"))
                fused_adap = (logits_base + a_t * logits_m).masked_fill(logits_base < cutoff, float("-inf"))
                pred_fixed = int(fused_fixed.argmax(dim=-1).item())
                pred_adap = int(fused_adap.argmax(dim=-1).item())
                pred_merged = int(logits_m.argmax(dim=-1).item())

                # Temperature-matched probability distributions (unmasked convex combination)
                probs_m = F.softmax(logits_m, dim=-1)
                probs_collab_fixed = F.softmax((logits_base + alpha_fixed * logits_m) / (1.0 + alpha_fixed), dim=-1)
                probs_collab_adap = F.softmax((logits_base + a_t * logits_m) / (1.0 + a_t), dim=-1)

                if s_idx == late_probe and R >= 16:
                    mid_l = L_layers // 2
                    kept_l = kept[mid_l]
                    anc_local = (kept_l[im_end:-1] - im_end).clamp(0, R - 1)
                    if anc_local.numel() > 0:
                        anc_np = anc_local.cpu().numpy()
                        reg_func = full_func_mask[:R]
                        reg_gt = full_gt_mask[:R]
                        exp2[st_key]["func_anchor_frac"].append(float(reg_func[anc_np].mean()) * 100.0)
                        exp2[st_key]["gt_anchor_frac"].append(float(reg_gt[anc_np].mean()) * 100.0)
                        if mm == "mean":
                            assign = _nearest_bucket_assignment(anc_local, R)
                            counts = torch.bincount(assign, minlength=anc_local.numel()).float()
                            tok_bs = counts[assign].cpu().numpy()
                            half = R // 2
                            exp2[st_key]["early_span_bsize"].append(float(tok_bs[:half].mean()))
                            exp2[st_key]["late_span_bsize"].append(float(tok_bs[half:].mean()))
                        else:
                            exp2[st_key]["early_span_bsize"].append(1.0)
                            exp2[st_key]["late_span_bsize"].append(1.0)

                if st_key == "low":
                    exp1["thm1_pos_all"].append(float(S_m_raw > S_b) * 100.0)
                    if s_idx == early_probe:
                        exp1["S_early_base"].append(S_b * 100.0)
                        exp1["S_early_merged"].append(S_m_raw * 100.0)
                        exp1["S_early_thm1"].append(S_m_thm1 * 100.0)
                    if s_idx == late_probe:
                        exp1["S_late_base"].append(S_b * 100.0)
                        exp1["S_late_merged"].append(S_m_raw * 100.0)
                        exp1["S_late_thm1"].append(S_m_thm1 * 100.0)
                        exp1["thm1_pos_late"].append(float(S_m_raw > S_b) * 100.0)
                    if s_idx in grounded_steps:
                        exp1["S_gt_base"].append(S_b * 100.0)
                        exp1["S_gt_merged"].append(S_m_raw * 100.0)
                        exp1["S_gt_thm1"].append(S_m_thm1 * 100.0)
                        exp1["alpha_gt"].append(a_t)
                    if s_idx in hal_steps:
                        exp1["S_hal_base"].append(S_b * 100.0)
                        exp1["S_hal_merged"].append(S_m_raw * 100.0)
                        exp1["S_hal_thm1"].append(S_m_thm1 * 100.0)
                        exp1["alpha_hal"].append(a_t)
                        exp1["thm1_pos_hal"].append(float(S_m_raw > S_b) * 100.0)

                if s_idx == late_probe:
                    exp2[st_key]["S_merged_late"].append(S_m_raw * 100.0)
                    exp2[st_key]["S_thm1_late"].append(S_m_thm1 * 100.0)
                    exp2[st_key]["dS_late"].append((S_m_raw - S_b) * 100.0)
                    exp2[st_key]["dS_thm1_late"].append((S_m_thm1 - S_b) * 100.0)

                if s_idx in hal_steps:
                    exp2[st_key]["S_merged_hal"].append(S_m_raw * 100.0)
                    exp2[st_key]["S_thm1_hal"].append(S_m_thm1 * 100.0)
                    exp2[st_key]["dS_hal"].append((S_m_raw - S_b) * 100.0)
                    exp2[st_key]["dS_thm1_hal"].append((S_m_thm1 - S_b) * 100.0)
                    exp2[st_key]["hal_flip_fixed"].append(float(pred_fixed != target_tok) * 100.0)
                    exp2[st_key]["hal_flip_adap"].append(float(pred_adap != target_tok) * 100.0)
                    exp2[st_key]["hal_flip_merged"].append(float(pred_merged != target_tok) * 100.0)
                    p_b = float(probs_base[0, target_tok].item())
                    p_m = float(probs_m[0, target_tok].item())
                    p_f = float(probs_collab_fixed[0, target_tok].item())
                    p_a = float(probs_collab_adap[0, target_tok].item())
                    exp2[st_key]["hal_prob_chg_merged"].append((p_m - p_b) * 100.0)
                    exp2[st_key]["hal_prob_chg_fixed"].append((p_f - p_b) * 100.0)
                    exp2[st_key]["hal_prob_chg_adap"].append((p_a - p_b) * 100.0)
                    if p_b > 0.01:
                        exp2[st_key]["hal_rel_prob_drop"].append((p_m - p_b) / p_b * 100.0)

                if s_idx in grounded_steps and pred_base == target_tok:
                    exp2[st_key]["gt_keep_fixed"].append(float(pred_fixed == target_tok) * 100.0)
                    exp2[st_key]["gt_keep_adap"].append(float(pred_adap == target_tok) * 100.0)
                    p_b = float(probs_base[0, target_tok].item())
                    p_f = float(probs_collab_fixed[0, target_tok].item())
                    exp2[st_key]["gt_prob_chg_fixed"].append((p_f - p_b) * 100.0)

        del out, kv_list_full, score
        torch.cuda.empty_cache()
        n_done += 1
        if n_done % 25 == 0:
            print(f"[{model_name}] {n_done}/{len(rows)} images ({time.time()-t0:.1f}s)", flush=True)

    summary = {
        "model": model_name,
        "n_images": n_done,
        "ratio": ratio,
        "alpha_fixed": alpha_fixed,
        "n_grounded_steps": len(exp1["S_gt_base"]),
        "n_hal_steps": len(exp1["S_hal_base"]),
        "exp1": {k: (float(np.mean(v)) if v else None) for k, v in exp1.items()},
        "exp2": {
            st: {k: (float(np.mean(v)) if v else None) for k, v in d.items()}
            for st, d in exp2.items()
        },
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"DONE {model_name}: {n_done} imgs, gt_steps={len(exp1['S_gt_base'])}, hal_steps={len(exp1['S_hal_base'])} in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--ratio", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_backbone(args.model, args.limit, args.ratio, args.out)
