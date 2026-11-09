"""Extract attention-analysis data to reproduce the AAAI Fig 1/2/3 on modern backbones.
Per model, over N COCO images, greedily generate a caption while capturing, for every generated
token: (relative position, head/layer-averaged image attention S_t), the per-(layer,head) image
attention, and whether the token belongs to a CHAIR-hallucinated object. Saves a JSON consumed by
code/plot_attn_*.py.  Usage: python code/attn_analysis.py --model qwen2_5_vl --n 120 --out results/attn/qwen2_5_vl.json
"""
import os, sys, json, glob, argparse, types, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NLTK_DATA", "/data/yjb/nltk_data")
import torch
from PIL import Image
from ikod.models import load_vlm, build_inputs
from ikod.image_span import locate_image_span

PROMPT = "Please describe this image in detail."
IMGDIR = "/data/yjb/datasets/coco/val2014"


def _setup_chair():
    """Return (chair_obj, ok). chair_obj.imid_to_objects[imid] = set of GT objects;
    chair_obj.caption_to_words(cap) -> (words, node_words, idxs, double_words)."""
    CHAIR_ROOT = "/data/yjb/projects/MLLM/IKOD/instructblip_src"
    sys.path.insert(0, CHAIR_ROOT); sys.path.insert(0, os.path.join(CHAIR_ROOT, "eval"))
    import nltk; nltk.data.path.insert(0, "/data/yjb/nltk_data")
    try:
        from pattern.en import singularize  # noqa
    except Exception:
        import inflect
        _ip = inflect.engine()
        def singularize(w):
            s = _ip.singular_noun(w); return s if s else w
        _m = types.ModuleType("pattern"); _men = types.ModuleType("pattern.en")
        _men.singularize = singularize; _m.en = _men
        sys.modules["pattern"] = _m; sys.modules["pattern.en"] = _men
    os.chdir(CHAIR_ROOT)
    from eval.eval_chair import CHAIR
    ann = "/data/yjb/datasets/coco/annotations"
    return CHAIR, ann


@torch.no_grad()
def per_tok_attn(attentions, s, e):
    """[L,H] image-attention of the current (last) query token across layers and heads."""
    mats = []
    for a in attentions:                       # a: [1, H, q, kv]
        last = a[0, :, -1, :].float()          # [H, kv]
        mats.append(last[:, s:e].sum(-1))      # [H]
    return torch.stack(mats)                    # [L, H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--skip_chair", action="store_true", help="skip slow CHAIR hallucination labeling")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)   # resolve BEFORE _setup_chair() does os.chdir(EXP), else we write to the wrong tree

    model, proc, img_tok = load_vlm(args.model)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    imgs = sorted(glob.glob(f"{IMGDIR}/*.jpg"))[:args.n]
    has_chair = False
    if not args.skip_chair:
        try:
            CHAIR, ann = _setup_chair()
            imids = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in imgs]
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):   # silence the 896k-mask progress flood
                chair = CHAIR(imids, ann); chair.get_annotations()
            has_chair = True
        except Exception as ex:
            print("CHAIR setup failed (skip hallucination labels):", str(ex)[:160], flush=True)
    if not has_chair:
        imids = [None] * len(imgs)

    all_pts, hal_pts, image_records = [], [], []   # (rel_pos, S_t), plus auditable boundaries
    chair_label_errors = 0
    LH_top, LH_bot, n_top, n_bot = None, None, 0, 0   # per-(layer,head) accumulators

    for idx, (path, imid) in enumerate(zip(imgs, imids)):
        image = Image.open(path).convert("RGB")
        inputs = build_inputs(args.model, proc, image, PROMPT)
        ii = inputs["input_ids"]; am = inputs.get("attention_mask")
        gi = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        span = locate_image_span(ii[0], img_tok)
        if span is None:
            continue
        s, e = span
        is_mrope = type(model).__name__ in ("Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration")
        grid = gi.get("image_grid_thw")
        with torch.inference_mode():
            out = model(input_ids=ii, attention_mask=am, use_cache=True, output_attentions=True,
                        return_dict=True, **gi)
            cur = ii.shape[1]
            next3d = None
            if is_mrope:
                pos3d, _ = model.model.get_rope_index(ii, image_grid_thw=grid, attention_mask=am)
                next3d = pos3d[..., -1:] + 1
            gen_ids, per_tok = [], []     # token ids + [L,H] mats
            amx = am
            for _ in range(args.max_new):
                nt = int(out.logits[:, -1, :].argmax(-1))
                eos = getattr(model.generation_config, "eos_token_id", None) or getattr(model.config, "eos_token_id", None)
                eos = eos[0] if isinstance(eos, (list, tuple)) else eos
                if eos is not None and nt == eos:
                    break
                gen_ids.append(nt)
                ntt = torch.tensor([[nt]], device=ii.device)
                if amx is not None:
                    amx = torch.cat([amx, amx.new_ones((1, 1))], 1)
                kw = dict(use_cache=True, output_attentions=True, return_dict=True, attention_mask=amx)
                if is_mrope:
                    kw["position_ids"] = next3d; kw["cache_position"] = torch.tensor([cur], device=ii.device)
                out = model(input_ids=ntt, past_key_values=out.past_key_values, **kw)
                per_tok.append(per_tok_attn(out.attentions, s, e).cpu())   # [L,H] for THIS token
                if is_mrope:
                    next3d = next3d + 1
                cur += 1
        L = len(per_tok)
        if L < 4:
            continue
        # accumulate per-token (pos, S_t) + top/bottom layer-head matrices
        mats = torch.stack(per_tok)               # [L, layers, heads]
        S = mats.mean(dim=(1, 2))                  # [L]
        seq_all, seq_hal = [], []
        for t in range(L):
            # Keep attention at full float precision.  Earlier plotting-only
            # artifacts rounded to 5 decimals, which is unnecessarily coarse
            # for image-attention masses near 1e-4 and can bias bootstrap ratios.
            point = (round(t / (L - 1), 6), float(S[t]))
            all_pts.append(point); seq_all.append(point)
        k = max(1, int(0.2 * L))
        top = mats[:k].mean(0); bot = mats[-k:].mean(0)   # [layers, heads]
        LH_top = top if LH_top is None else LH_top + top; n_top += 1
        LH_bot = bot if LH_bot is None else LH_bot + bot; n_bot += 1
        # Hallucinated-object labels via CHAIR. CHAIR's idxs refer to positions in
        # nltk.word_tokenize(caption), before its compound-word normalization.
        # We map that raw word position proportionally to the generated-token axis;
        # this is auditable and avoids pretending that model tokens equal words.
        cap = tok.decode(gen_ids, skip_special_tokens=True)
        hallucinated_objects = []
        if has_chair and imid is not None:
            try:
                gt = chair.imid_to_objects[imid]
                words, node_words, idxs, _ = chair.caption_to_words(cap)
                # caption_to_words returns idxs[i] = the word index of object node_words[i];
                # use it directly (no substring re-matching) so every hallucinated object is captured.
                # The denominator must use the original NLTK tokenization, not
                # CHAIR's compound-normalized word list.
                raw_word_count = len(__import__("nltk").word_tokenize(cap.lower()))
                word_axis_max = max(1, raw_word_count - 1)
                for i, (surface, nw) in enumerate(zip(words, node_words)):
                    if nw in gt:
                        continue
                    wi = idxs[i] if i < len(idxs) else 0
                    frac = min(1.0, max(0.0, wi / word_axis_max))
                    ti = min(L - 1, max(0, int(frac * (L - 1))))
                    point = (round(float(frac), 6), float(S[ti]))
                    hal_pts.append(point); seq_hal.append(point)
                    hallucinated_objects.append({"surface": surface, "coco_object": nw,
                                                 "raw_word_index": int(wi),
                                                 "mapped_token_index": int(ti)})
            except Exception as ex:
                chair_label_errors += 1
                print(f"CHAIR label failed image={imid}: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
        image_records.append({"image_id": imid, "caption": cap,
                              "generated_token_count": L,
                              "all_tokens": seq_all, "hal_tokens": seq_hal,
                              "hallucinated_objects": hallucinated_objects})
        if (idx + 1) % 20 == 0:
            print(f"[{idx+1}/{len(imgs)}] all={len(all_pts)} hal={len(hal_pts)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model,
               "all_tokens": all_pts, "hal_tokens": hal_pts,
               "layerhead_top": (LH_top / max(1, n_top)).tolist() if LH_top is not None else None,
               "layerhead_bottom": (LH_bot / max(1, n_bot)).tolist() if LH_bot is not None else None,
               "n_images": n_top, "requested_images": args.n, "max_new_tokens": args.max_new,
               "chair_label_errors": chair_label_errors,
               "position_definition": "CHAIR raw NLTK word index/(raw NLTK word count-1), proportionally mapped to generated-token axis",
               "image_records": image_records},
              open(args.out, "w"))
    print(f"ATTN_DONE {args.model}: all={len(all_pts)} hal={len(hal_pts)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
