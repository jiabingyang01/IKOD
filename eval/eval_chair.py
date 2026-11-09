"""Generate detailed captions for the CHAIR-500 COCO images with modern VLMs,
using greedy / IKOD-fixed / IKOD-adaptive decoding. Writes:
  <out>.jsonl                 {question_id, image, text}
  <out>.json                  single_convert format {overall, imgToEval:{i:{image_id, caption}}}
Then evaluate with code/chair_eval.py.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, argparse, time, datetime, hashlib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from ikod.models import load_vlm, build_inputs
from ikod.decode import ikod_generate
from ikod.adaptive import AdaptiveAlpha

CHAIR_Q = "./workspace/data/chair/chair_500.jsonl"
IMG = "/data/yjb/datasets/coco/val2014"


def _source_fingerprint():
    """Hash the exact decoding sources used by this process for artifact auditing."""
    root = os.path.dirname(os.path.abspath(__file__))
    rels = (
        "gen_chair_modern.py",
        "ikod/decode.py",
        "ikod/kv_merge.py",
        "ikod/adaptive.py",
        "ikod/models.py",
        "ikod/baselines.py",
    )
    hashes = {}
    combined = hashlib.sha256()
    for rel in rels:
        path = os.path.join(root, rel)
        payload = open(path, "rb").read()
        digest = hashlib.sha256(payload).hexdigest()
        hashes[rel] = digest
        combined.update(rel.encode("utf-8") + b"\0" + payload)
    return {"combined_sha256": combined.hexdigest(), "files": hashes}


def main():
    torch.use_deterministic_algorithms(True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["greedy", "ikod", "vcd", "m3id", "dola", "revisit", "vista", "kvsmooth", "pai"], default="ikod")
    # PAI's CHAIR configuration differs from its POPE one: chair_eval.py defaults gamma=2 while
    # pope_eval.py uses 1.1. Alpha and the layer window are shared. Keeping these separate is the
    # same lesson VISTA's per-task lambda taught (ledger A33).
    ap.add_argument("--pai_alpha", type=float, default=0.2)
    ap.add_argument("--pai_gamma", type=float, default=2.0)
    ap.add_argument("--pai_start_layer", type=int, default=2)
    ap.add_argument("--eel", default="all", help="revisit early_exit_layers: all|last")
    ap.add_argument("--rel_top", type=float, default=1e-5, help="revisit APC relative_top threshold")
    ap.add_argument("--vsv_lambda", type=float, default=0.17, help="vista VSV strength (CHAIR 0.17, POPE 0.01)")
    ap.add_argument("--sla_alpha", type=float, default=0.3, help="vista SLA blend")
    # See ledger A37: reading intermediate hidden states without the final norm degenerates on
    # Qwen2-VL. 1 = repaired reading (main results), 0 = literal official behaviour.
    ap.add_argument("--sla_norm", type=int, default=1)
    ap.add_argument("--kvs_lambda_ref", type=float, default=0.9, help="KVSmooth reference EMA coefficient")
    ap.add_argument("--kvs_queue", type=int, default=15, help="KVSmooth row-entropy FIFO length")
    ap.add_argument("--kvs_layer_start", type=int, default=3)
    ap.add_argument("--kvs_layer_end", type=int, default=31)
    ap.add_argument("--kvs_rank_denominator", choices=["fixed", "current"], default="fixed",
                    help="literal fixed-M Eq.13 or warm-up sensitivity control")
    ap.add_argument("--kvs_disable", action="store_true",
                    help="no-op gate for exact agreement of the custom greedy loop")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--full_weight", type=float, default=1.0,
                    help="standard-branch logit weight; 0 gives the merged-branch-only ablation")
    ap.add_argument("--drop_replay_anchor", action="store_true",
                    help="diagnostic only: remove latest protected KV before query replay")
    ap.add_argument("--lam", type=float, default=0.02, help="m3id forgetting factor")
    ap.add_argument("--noise_step", type=int, default=500, help="vcd diffusion noise step")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--ratio", type=float, default=0.4)
    ap.add_argument("--anchor_strategy", choices=["low", "high", "recent", "uniform"], default="low",
                    help="anchor-selection rule; high/recent/uniform are ablation controls")
    ap.add_argument("--merge_mode", choices=["mean", "prune"], default="mean",
                    help="bucket-mean aggregation or same-length direct-pruning control")
    ap.add_argument("--merge_region", choices=["post_visual", "generated", "prompt_suffix"],
                    default="post_visual",
                    help="post-visual text source region transformed in the expert cache")
    ap.add_argument("--plain_query_position", choices=["compact", "original"], default="compact",
                    help="plain-RoPE replay query position; original is a mechanism control")
    ap.add_argument("--adaptive", default="", help="'' | p1 | p1p3 | p2")
    ap.add_argument("--amin", type=float, default=0.0); ap.add_argument("--amax", type=float, default=2.0)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--prompt", default="Please describe this image in detail.")
    ap.add_argument("--qfile", default=CHAIR_Q); ap.add_argument("--img", default=IMG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=1); ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_meta", action="store_true",
                    help="write per-step IKOD alpha/cache-length metadata to a sidecar JSONL")
    args = ap.parse_args()
    source_fingerprint = _source_fingerprint()
    if args.save_meta and args.method != "ikod":
        raise ValueError("--save_meta is currently defined only for --method ikod")

    qs = [json.loads(l) for l in open(args.qfile)]
    if args.limit: qs = qs[:args.limit]
    qs = qs[args.idx::args.chunks]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, proc, img_tok = load_vlm(args.model)
    adaptive = AdaptiveAlpha(mode=args.adaptive, alpha_min=args.amin, alpha_max=args.amax,
                             kappa=args.kappa) if args.adaptive else None

    fjsonl = open(args.out if args.out.endswith(".jsonl") else args.out + ".jsonl", "w")
    base_out = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    fmeta = open(base_out + "_meta.jsonl", "w") if args.save_meta else None
    img_to_eval = {}; t0 = time.time(); lens = []
    for i, q in enumerate(qs):
        image = Image.open(os.path.join(args.img, q["image"])).convert("RGB")
        inputs = build_inputs(args.model, proc, image, args.prompt)
        ii = inputs["input_ids"]; am = inputs.get("attention_mask")
        gen_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            if args.method == "greedy":
                g = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new, use_cache=True)
                out_ids = g[:, ii.shape[1]:]
            elif args.method == "dola":
                g = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new, use_cache=True, dola_layers="high", repetition_penalty=1.2)
                out_ids = g[:, ii.shape[1]:]
            elif args.method == "vcd":
                from ikod.baselines import vcd_generate
                out_ids = vcd_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                       max_new_tokens=args.max_new, alpha=args.alpha, beta=args.beta,
                                       noise_step=args.noise_step, **gen_inputs)
            elif args.method == "m3id":
                from ikod.baselines import m3id_generate
                out_ids = m3id_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                        max_new_tokens=args.max_new, alpha=args.alpha, lam=args.lam,
                                        beta=args.beta, **gen_inputs)
            elif args.method == "revisit":
                from ikod.baselines import revisit_generate
                out_ids = revisit_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                           max_new_tokens=args.max_new, early_exit_layers=args.eel,
                                           relative_top=args.rel_top, **gen_inputs)
            elif args.method == "pai":
                from ikod.baselines import pai_generate
                out_ids = pai_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                       max_new_tokens=args.max_new, alpha=args.pai_alpha,
                                       gamma=args.pai_gamma, start_layer=args.pai_start_layer,
                                       **gen_inputs)
            elif args.method == "vista":
                from ikod.baselines import vista_generate
                out_ids = vista_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                         max_new_tokens=args.max_new, vsv_lambda=args.vsv_lambda,
                                         sla_alpha=args.sla_alpha,
                                         sla_apply_norm=bool(args.sla_norm), **gen_inputs)
            elif args.method == "kvsmooth":
                from ikod.baselines import kvsmooth_generate
                out_ids = kvsmooth_generate(
                    model, ii, attention_mask=am, max_new_tokens=args.max_new,
                    lambda_ref=args.kvs_lambda_ref, queue_len=args.kvs_queue,
                    layer_start=args.kvs_layer_start, layer_end=args.kvs_layer_end,
                    rank_denominator=args.kvs_rank_denominator,
                    enabled=not args.kvs_disable,
                    **gen_inputs,
                )
            else:
                decoded = ikod_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                        max_new_tokens=args.max_new, ratio=args.ratio, beta=args.beta,
                                        alpha=args.alpha, full_weight=args.full_weight,
                                        anchor_strategy=args.anchor_strategy,
                                        merge_mode=args.merge_mode,
                                        merge_region=args.merge_region,
                                        plain_query_position=args.plain_query_position,
                                        adaptive=adaptive,
                                        replay_protected_anchor=not args.drop_replay_anchor,
                                        return_meta=args.save_meta,
                                        **gen_inputs)
                if args.save_meta:
                    out_ids, step_meta = decoded
                else:
                    out_ids = decoded
        text = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        lens.append(len(text.split()))
        image_id = int(q["image"].split("_")[-1].split(".")[0].lstrip("0") or "0")
        fjsonl.write(json.dumps({"question_id": q["question_id"], "image": q["image"], "text": text}) + "\n"); fjsonl.flush()
        if fmeta is not None:
            fmeta.write(json.dumps({"question_id": q["question_id"], "image": q["image"],
                                    "steps": step_meta}) + "\n")
            fmeta.flush()
        img_to_eval[str(i)] = {"image_id": image_id, "caption": text}
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(qs)}] avg_len={sum(lens)/len(lens):.1f} ({(i+1)/(time.time()-t0):.2f} it/s)", flush=True)
    fjsonl.close()
    if fmeta is not None:
        fmeta.close()
    capjson = base_out + ".json"
    json.dump({"overall": {"metric1": 0.0}, "imgToEval": img_to_eval}, open(capjson, "w"), indent=2)
    cfg_path = base_out + "_run_config.json"
    json.dump({
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command_args": vars(args),
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "source_fingerprint": source_fingerprint,
        "model_config_name_or_path": getattr(model.config, "_name_or_path", None),
        "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "evaluated_examples_in_chunk": len(img_to_eval),
    }, open(cfg_path, "w"), indent=2, default=str)
    print(f"DONE gen model={args.model} method={args.method} adaptive={args.adaptive or 'none'} "
          f"alpha={args.alpha} full_weight={args.full_weight} ratio={args.ratio} "
        f"anchor={args.anchor_strategy} "
        f"merge_mode={args.merge_mode} "
          f"beta={args.beta} n={len(img_to_eval)} "
          f"avg_len={sum(lens)/max(1,len(lens)):.1f} time={time.time()-t0:.0f}s -> {capjson}", flush=True)


if __name__ == "__main__":
    main()
