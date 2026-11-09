"""Where IKOD's decoding time goes: full forward vs merge vs merged forward vs score bookkeeping.

Wraps the three components with CUDA-synchronised timers and runs the published decoder on a few COCO
images. Also reports how many steps the adaptive gate leaves at alpha=0 (those steps could skip the
merged branch entirely without changing a single token).
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ikod.models import load_vlm, build_inputs  # noqa: E402
from ikod import decode as D, kv_merge as KM  # noqa: E402
from ikod.adaptive import AdaptiveAlpha  # noqa: E402

T = {"full_fwd": 0.0, "merged_fwd": 0.0, "merge": 0.0, "score": 0.0, "total": 0.0}
N = {"full_fwd": 0, "merged_fwd": 0, "merge": 0}


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava_15_hf")
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.0, help="0 -> adaptive P1 (the published default)")
    args = ap.parse_args()

    model, proc, img_tok = load_vlm(args.model)
    qs = [json.loads(l) for l in open("/data/yjb/projects/MLLM/IKOD/instructblip_src/data/chair/chair_500.jsonl")][: args.images] \
        if Path("/data/yjb/projects/MLLM/IKOD/instructblip_src/data/chair/chair_500.jsonl").is_file() else None
    if qs is None:
        qs = [json.loads(l) for l in open("results/chair/llava_15_hf_greedy.jsonl")][: args.images]
    prompt = "Please describe this image in detail."

    orig_merge = KM.merge_kv_cache
    def timed_merge(*a, **k):
        sync(); t0 = time.perf_counter(); r = orig_merge(*a, **k); sync()
        T["merge"] += time.perf_counter() - t0; N["merge"] += 1; return r
    KM.merge_kv_cache = timed_merge; D.merge_kv_cache = timed_merge

    orig_score = D.update_score_table
    def timed_score(*a, **k):
        sync(); t0 = time.perf_counter(); r = orig_score(*a, **k); sync()
        T["score"] += time.perf_counter() - t0; return r
    D.update_score_table = timed_score

    orig_call = model.__class__.__call__
    def timed_call(self, *a, **k):
        merged = k.get("cache_position") is not None and k.get("past_key_values") is not None and k.get("attention_mask") is None
        sync(); t0 = time.perf_counter(); r = orig_call(self, *a, **k); sync()
        dt = time.perf_counter() - t0
        key = "merged_fwd" if merged else "full_fwd"
        T[key] += dt; N[key] += 1; return r
    model.__class__.__call__ = timed_call

    alphas = []
    for q in qs:
        image = Image.open("/data/yjb/datasets/coco/val2014/" + q["image"]).convert("RGB")
        inputs = build_inputs(args.model, proc, image, prompt)
        ii = inputs.pop("input_ids"); am = inputs.pop("attention_mask", None)
        sync(); t0 = time.perf_counter()
        gen, meta = D.ikod_generate(model, ii, image_token_id=img_tok, attention_mask=am,
                                    max_new_tokens=args.max_new,
                                    **({"alpha": args.alpha} if args.alpha > 0 else {"adaptive": AdaptiveAlpha(mode="p1", alpha_min=0.0, alpha_max=1.0, kappa=10.0)}),
                                    return_meta=True, **inputs)
        sync(); T["total"] += time.perf_counter() - t0
        alphas += [m["alpha"] for m in meta]
    n = len(alphas)
    print(f"model={args.model} images={args.images} steps={n}")
    print(f"  total            {T['total']:7.2f}s   ({T['total']/max(n,1)*1000:6.1f} ms/step)")
    for k in ("full_fwd", "merged_fwd", "merge", "score"):
        print(f"  {k:15s} {T[k]:7.2f}s  {100*T[k]/max(T['total'],1e-9):5.1f}%   calls={N.get(k,'-')}")
    zero = sum(1 for a in alphas if a == 0.0)
    print(f"  alpha==0 steps: {zero}/{n} ({100*zero/max(n,1):.1f}%)  -> merged branch skippable on those steps")
    print(f"  alpha mean {sum(alphas)/max(n,1):.3f}")


if __name__ == "__main__":
    main()
