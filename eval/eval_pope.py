"""Standalone POPE eval for modern native-HF VLMs with greedy or IKOD decoding.
Generates yes/no answers, writes jsonl, and prints F1. Supports multi-GPU via --chunks/--idx.

Example:
  CUDA_VISIBLE_DEVICES=0 python code/eval_pope_modern.py --model llava_ov --method ikod \
     --alpha 2 --beta 0.1 --ratio 0.4 --out results/pope_modern/llava_ov_ikod.jsonl
"""
import os, sys, json, argparse, time, datetime, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from ikod.models import load_vlm, build_inputs
from ikod.decode import ikod_generate
from ikod.adaptive import AdaptiveAlpha

POPE = "/data/yjb/projects/MLLM/IKOD/instructblip_src/data/pope/coco/coco_pope_random.json"
IMG = "/data/yjb/datasets/coco/val2014"


def parse_yes_no(text):
    t = text.strip().lower()
    if t.startswith("yes") or " yes" in t[:8]: return "yes"
    if t.startswith("no") or " no" in t[:8]: return "no"
    return "yes" if "yes" in t else ("no" if "no" in t else "no")


def f1_from(gts, preds):
    tp=tn=fp=fn=0; yes=0
    for g,p in zip(gts,preds):
        if g=="yes":
            if p=="yes": tp+=1; yes+=1
            else: fn+=1
        else:
            if p=="no": tn+=1
            else: fp+=1; yes+=1
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
    acc=(tp+tn)/max(1,len(gts))
    return dict(acc=acc*100, precision=prec*100, recall=rec*100, f1=f1*100, yes_ratio=yes/max(1,len(gts))*100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["greedy","ikod","vcd","m3id","dola","revisit","kvsmooth","vista","pai"], default="ikod")
    ap.add_argument("--eel", default="all", help="revisit early_exit_layers: all|last")
    ap.add_argument("--rel_top", type=float, default=1e-5, help="revisit APC relative_top")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--noise_step", type=int, default=500)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--ratio", type=float, default=0.4)
    ap.add_argument("--adaptive", default="", help="'' | p1 | p1p3 | p2")
    ap.add_argument("--amin", type=float, default=0.0); ap.add_argument("--amax", type=float, default=2.0)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--pope", default=POPE); ap.add_argument("--img", default=IMG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=1); ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    # Matched-provenance control. Several checkpoints ship a non-unit repetition_penalty in
    # generation_config.json (Qwen2.5-VL: 1.05), which model.generate silently inherits while
    # the custom ikod_generate loop applies no penalty at all. Passing it explicitly makes the
    # greedy baseline and IKOD use the same decoding rule and records the value in run_config.
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    # DoLa's own penalty, historically hardcoded to 1.2 (its paper's open-ended setting).
    # Exposed so the matched-protocol value 1.0 can be probed; default preserves prior runs.
    ap.add_argument("--dola_rp", type=float, default=1.2)
    # Candidate premature layers. "high"/"low" are HF's half-range shortcuts (text-LLM oriented);
    # the LVLM literature (HALC ICML'24 decoder zoo) sweeps the FULL even-layer range with the
    # final layer as mature. Pass e.g. "0,2,4,...,30" to reproduce that setting.
    ap.add_argument("--dola_layers", default="high")
    # Sampling base (for IKOD sampling-robustness ablation + Mirage sampling-base control).
    # Deterministic via --seed so runs are reproducible; default off preserves greedy behavior.
    ap.add_argument("--do_sample", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # PAI (ECCV'24) official defaults: alpha=0.2, gamma=1.1 for POPE, layers [2, 32).
    ap.add_argument("--pai_alpha", type=float, default=0.2)
    ap.add_argument("--pai_gamma", type=float, default=1.1)
    ap.add_argument("--pai_start_layer", type=int, default=2)
    # VISTA's steering strength is TASK-specific in the official release: run_pope.sh uses 0.01
    # while run_chair.sh uses 0.17. Everything else (greedy, SLA layers 25-30, alpha 0.3) is
    # shared, so lambda is the only knob that has to change between the two benchmarks.
    ap.add_argument("--vista_lambda", type=float, default=0.01)
    ap.add_argument("--vista_sla_alpha", type=float, default=0.3)
    # SLA reads intermediate hidden states, which HF returns BEFORE the stack's final norm.
    # 1 passes them through that norm first (what reading an intermediate layer requires);
    # 0 reproduces the official release literally, which degenerates on Qwen2-VL (ledger A37).
    ap.add_argument("--vista_sla_norm", type=int, default=1)
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(args.pope)]
    if args.limit: qs = qs[:args.limit]
    qs = qs[args.idx::args.chunks]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, proc, img_tok = load_vlm(args.model)
    _gc = getattr(model, "generation_config", None)
    _ckpt_rp = getattr(_gc, "repetition_penalty", None) if _gc is not None else None
    print(f"[provenance] checkpoint generation_config.repetition_penalty={_ckpt_rp} "
          f"| applied to greedy={args.repetition_penalty} | ikod applies no penalty", flush=True)
    adaptive = None
    if args.adaptive:
        adaptive = AdaptiveAlpha(mode=args.adaptive, alpha_min=args.amin, alpha_max=args.amax, kappa=args.kappa)

    # Deterministic sampling RNG (device-matched) so sampled runs reproduce bit-for-bit.
    # The custom IKOD loop takes the explicit generator; HF's generate() reads the global RNG,
    # so seed that too or the greedy-sampling arm would not be reproducible.
    _rng = None
    if args.do_sample:
        _dev = next(model.parameters()).device
        _rng = torch.Generator(device=_dev); _rng.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    fout = open(args.out, "w"); t0=time.time(); gts=[]; preds=[]
    for i, q in enumerate(qs):
        image = Image.open(os.path.join(args.img, q["image"])).convert("RGB")
        question = q["text"] + " Please answer this question with one word."
        inputs = build_inputs(args.model, proc, image, question)
        ii = inputs["input_ids"]; am = inputs.get("attention_mask")
        gen_inputs = {k:v for k,v in inputs.items() if k not in ("input_ids","attention_mask")}
        with torch.inference_mode():
            if args.method == "greedy":
                # Honors --do_sample so this arm can also serve as the plain sampling baseline
                # in the Mirage three-arm control; default 0 keeps the historical greedy behavior.
                _samp = dict(do_sample=True, temperature=args.temperature,
                             top_p=args.top_p, **({"top_k": args.top_k} if args.top_k else {})) \
                        if args.do_sample else dict(do_sample=False)
                g = model.generate(**inputs, max_new_tokens=args.max_new,
                                   use_cache=True, repetition_penalty=args.repetition_penalty, **_samp)
                out_ids = g[:, ii.shape[1]:]
            elif args.method == "dola":
                _dl = args.dola_layers
                if _dl == "auto":
                    # Full even-layer candidate bucket sized to THIS backbone's depth (HALC/OPERA
                    # convention). Fixed [0,2,..,30] mis-fires on 28-layer Qwen2-based VLMs, so read
                    # the language model's actual layer count and generate [0,2,..,L-2].
                    _lm = getattr(model, "language_model", None) or getattr(model, "model", model)
                    _n = getattr(getattr(_lm, "config", None), "num_hidden_layers", None) \
                         or getattr(model.config, "num_hidden_layers", 32)
                    _dl = list(range(0, int(_n) - 1, 2))
                elif _dl not in ("low", "high"):
                    _dl = [int(x) for x in _dl.split(",") if x.strip() != ""]
                g = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new, use_cache=True, dola_layers=_dl, repetition_penalty=args.dola_rp)
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
            elif args.method == "kvsmooth":
                from ikod.baselines import kvsmooth_generate
                out_ids = kvsmooth_generate(model, ii, attention_mask=am,
                                            max_new_tokens=args.max_new, **gen_inputs)
            elif args.method == "pai":
                from ikod.baselines import pai_generate
                out_ids = pai_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                       max_new_tokens=args.max_new, alpha=args.pai_alpha,
                                       gamma=args.pai_gamma, start_layer=args.pai_start_layer,
                                       **gen_inputs)
            elif args.method == "vista":
                from ikod.baselines import vista_generate
                out_ids = vista_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                         max_new_tokens=args.max_new,
                                         vsv_lambda=args.vista_lambda,
                                         sla_alpha=args.vista_sla_alpha,
                                         sla_apply_norm=bool(args.vista_sla_norm),
                                         do_sample=bool(args.do_sample), temperature=args.temperature,
                                         top_p=args.top_p, rng=_rng, **gen_inputs)
            else:
                out_ids = ikod_generate(model, ii, attention_mask=am, image_token_id=img_tok,
                                        max_new_tokens=args.max_new, ratio=args.ratio, beta=args.beta,
                                        alpha=args.alpha, adaptive=adaptive,
                                        do_sample=bool(args.do_sample), temperature=args.temperature,
                                        top_p=args.top_p, top_k=args.top_k, rng=_rng, **gen_inputs)
        text = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        pred = parse_yes_no(text); gts.append(q["label"].lower()); preds.append(pred)
        fout.write(json.dumps({"question_id":q["question_id"],"label":q["label"],"text":text,"pred":pred})+"\n"); fout.flush()
        if (i+1)%200==0:
            m=f1_from(gts,preds); print(f"[{i+1}/{len(qs)}] F1={m['f1']:.2f} acc={m['acc']:.2f} yes={m['yes_ratio']:.1f} ({(i+1)/(time.time()-t0):.2f} it/s)", flush=True)
    fout.close()
    m=f1_from(gts,preds)
    print(f"DONE model={args.model} method={args.method} adaptive={args.adaptive or 'none'} "
          f"alpha={args.alpha} ratio={args.ratio} beta={args.beta} | "
          f"F1={m['f1']:.2f} acc={m['acc']:.2f} prec={m['precision']:.2f} rec={m['recall']:.2f} yes={m['yes_ratio']:.1f} "
          f"n={len(gts)} time={time.time()-t0:.0f}s", flush=True)
    json.dump(m, open(args.out.replace('.jsonl','_metrics.json'),'w'), indent=2)
    json.dump({
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command_args": vars(args),
        "torch": torch.__version__,
        "evaluated_examples_in_chunk": len(gts),
    }, open(args.out.replace('.jsonl', '_run_config.json'), 'w'), indent=2, default=str)


if __name__ == "__main__":
    main()
