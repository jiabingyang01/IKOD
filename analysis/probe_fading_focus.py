"""Fading-focus diagnosis (paper Pillar 2): measure how the model's attention to image
tokens evolves with generated-token position, across modern backbones. Greedy-decode K
images recording per-step image-attention S_t; bin by relative position; save curve + figure.

Run: CUDA_VISIBLE_DEVICES=5 python code/fading_focus.py --model llava_ov --k 150 --steps 256
"""
import os, sys, json, glob, argparse, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from ikod.models import load_vlm, build_inputs
from ikod.image_span import locate_image_span
from ikod.adaptive import compute_image_attention

IMG = "/data/yjb/datasets/coco/val2014"


@torch.no_grad()
def run_image(model, proc, img_tok, image, steps, eos_ids):
    inputs = build_inputs_for(model, proc, image)
    ii = inputs["input_ids"]; am = inputs.get("attention_mask")
    gi = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    span = locate_image_span(ii[0], img_tok)
    if span is None:
        return None
    s, e = span
    out = model(input_ids=ii, attention_mask=am, use_cache=True, output_attentions=True, return_dict=True, **gi)
    is_mrope = type(model).__name__ in ("Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration")
    next3d = None
    cur = ii.shape[1]
    amx = am
    if is_mrope:
        pos3d, _ = model.model.get_rope_index(
            ii, image_grid_thw=gi.get("image_grid_thw"), attention_mask=am
        )
        next3d = pos3d[..., -1:] + 1
    S = []
    for _ in range(steps):
        lf = out.logits[:, -1, :]
        nxt = lf.argmax(-1, keepdim=True)
        if eos_ids and int(nxt) in eos_ids:
            break
        if amx is not None:
            amx = torch.cat([amx, amx.new_ones((1, 1))], dim=1)
        step_kw = dict(past_key_values=out.past_key_values, use_cache=True,
                       output_attentions=True, return_dict=True, attention_mask=amx)
        if is_mrope:
            step_kw["position_ids"] = next3d
            step_kw["cache_position"] = torch.tensor([cur], device=ii.device)
        out_next = model(input_ids=nxt, **step_kw)
        St = float(compute_image_attention(out_next.attentions, s, e)[0])
        S.append(St)
        out = out_next
        if is_mrope:
            next3d = next3d + 1
        cur += 1
    return S


def build_inputs_for(model, proc, image):
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Please describe this image in detail."}]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
    inp = proc(images=image, text=prompt, return_tensors="pt").to(model.device)
    if "pixel_values" in inp:
        inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    return inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--start", type=int, default=0,
                    help="offset in the sorted COCO list for disjoint data-parallel shards")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"results/fading/{args.model}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model, proc, img_tok = load_vlm(args.model)
    _e = getattr(model.generation_config, "eos_token_id", None) or getattr(model.config, "eos_token_id", None)
    eos_ids = _e if isinstance(_e, (list, tuple)) else ([_e] if _e is not None else [])
    imgs = sorted(glob.glob(f"{IMG}/*.jpg"))[args.start:args.start + args.k]
    bins = [[] for _ in range(args.bins)]
    raw = []
    for i, p in enumerate(imgs):
        S = run_image(model, proc, img_tok, Image.open(p).convert("RGB"), args.steps, eos_ids)
        if not S or len(S) < 5:
            continue
        raw.append(S)
        n = len(S)
        for t, v in enumerate(S):
            b = min(args.bins - 1, int(args.bins * t / n))
            bins[b].append(v)
        if (i + 1) % 25 == 0:
            cur = [round(sum(b) / len(b), 5) if b else None for b in bins]
            print(f"[{i+1}/{len(imgs)}] curve={cur}", flush=True)
    curve = [sum(b) / len(b) if b else None for b in bins]
    # normalized to first bin
    base = next((c for c in curve if c), 1.0)
    norm = [round(c / base, 4) if c else None for c in curve]
    json.dump({"model": args.model, "k": len(raw), "requested_start": args.start,
               "requested_k": args.k, "bins": args.bins,
               "bin_sum": [sum(b) for b in bins], "bin_count": [len(b) for b in bins],
               "curve": curve,
               "curve_norm": norm, "first_bin": curve[0], "last_bin": curve[-1]},
              open(out, "w"), indent=2)
    print(f"DONE fading {args.model}: first_bin={curve[0]:.4f} last_bin={curve[-1]:.4f} "
          f"drop={(1 - curve[-1]/curve[0])*100:.1f}% -> {out}", flush=True)


if __name__ == "__main__":
    main()
