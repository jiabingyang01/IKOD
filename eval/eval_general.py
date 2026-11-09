"""Run lmms-eval with IKOD (or a baseline decoder) injected into a base model wrapper.
Constructs the lmms-eval wrapper instance, monkeypatches its inner model.generate to call
ikod_generate (eager attn), and passes the instance to simple_evaluate (which accepts an LM
instance directly — no registry needed).

Usage:
  python code/run_lmms_ikod_v2.py --wrapper qwen2_5_vl --pretrained <path> --tasks mme,mmstar \
     --tag qwen_ikod --alpha 0.3 --ratio 0.4 --beta 0.1 [--adaptive p1 --amax 1.0 --kappa 10] [--limit N] \
     [--out_root results/lmms_ikod]
  --wrapper: qwen2_5_vl | qwen2_vl | llava_hf | llava_hf_textfix | instructblip | ikod_hf

v2 (2026-09-27, capability-table rerun rr0927). Copied from run_lmms_ikod.py (sha256 9f4286b6...) and
changed ONLY in the bookkeeping around the evaluation; the decoding path (patched_generate,
CONTROL_KEYS, the text-only routing to the original HF generate) is verbatim:
  1. --out_root (default results/lmms_ikod, i.e. the v1 behaviour) replaces the five hard-coded
     results/lmms_ikod/... write paths.
  2. Refuses to start if <out_root>/<tag>/{results.json,run_config.json,samples.jsonl.gz} or any
     <out_root>/<tag>_<suffix> per-task directory already exists (checked on every rank before the
     process group and the model are created, so a refusal costs no GPU time). If artifacts appear
     while the run is in flight, the outputs go to <tag>__collision_<utc> and the process exits 3.
  3. The source fingerprint covers this file (not run_lmms_ikod.py), the frozen ikod/ sources, the
     new ikod/lmms_llava_hf_textfix_v1.py wrapper and the three locally patched lmms-eval files
     (api/task.py, models/simple/llava_hf.py, tasks/mmbench/mmbench_en_dev.yaml). It is taken at
     start-up and again before writing; a mismatch is recorded and the process exits 4 without
     printing IKOD_LMMS_DONE.
  4. run_config records the key environment variables (IKOD_FAST_MERGE, PYTORCH_CUDA_ALLOC_CONF,
     HF_HUB_OFFLINE, CUDA_VISIBLE_DEVICES, ...) both as launched and as effective. HF_TOKEN is never
     recorded, only whether it was set.
  5. New --wrapper llava_hf_textfix: identical to the stock lmms-eval LlavaHf (same constructor
     arguments as the llava_hf branch) except that text-only samples get freshly built text-only
     inputs instead of silently reusing the previous sample's inputs (see
     ikod/lmms_llava_hf_textfix_v1.py).
"""
import os, sys, json, gzip, argparse, datetime, hashlib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# v2: snapshot the environment the launcher handed us, before this module's own defaults below.
_ENV_KEYS = (
    "IKOD_FAST_MERGE", "PYTORCH_CUDA_ALLOC_CONF", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES",
    "HF_ENDPOINT", "HF_HUB_DISABLE_XET", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE",
    "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "IKOD_SHARD_MAP", "IKOD_DEBUG",
    "WORLD_SIZE", "LOCAL_WORLD_SIZE", "ACCELERATE_DDP_TIMEOUT", "NCCL_TIMEOUT",
    "TORCH_NCCL_BLOCKING_WAIT", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "CONDA_DEFAULT_ENV",
)
_ENV_AT_LAUNCH = {k: os.environ.get(k) for k in _ENV_KEYS}
_HF_TOKEN_SET_AT_LAUNCH = bool(os.environ.get("HF_TOKEN"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # datasets via mirror (local proxy down -> real HF unreachable)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "0")
# lmms-eval's datasets.load_dataset path forces token auth; the migration lost HF_TOKEN. These are
# public datasets and hf-mirror does not validate tokens, so a dummy token satisfies the check.
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = "hf_dummytokenForPublicMirrorReads000000000"


# v2: files hashed into run_config["source_fingerprint"].
_FP_LOCAL = (
    "run_lmms_ikod_v2.py",
    "ikod/decode.py",
    "ikod/kv_merge.py",
    "ikod/adaptive.py",
    "ikod/models.py",
    "ikod/lmms_wrapper.py",
    "ikod/baselines.py",
    "ikod/lmms_llava_hf_textfix_v1.py",
)
# The three files in which the vendored lmms-eval-src differs from upstream d675c79.
_FP_LMMS_EVAL = (
    "api/task.py",
    "models/simple/llava_hf.py",
    "tasks/mmbench/mmbench_en_dev.yaml",
)


def _lmms_eval_pkg_dir():
    """Directory of the lmms_eval package that `import lmms_eval` will load (resolved without importing it)."""
    import importlib.util
    spec = importlib.util.find_spec("lmms_eval")
    if spec is None or not spec.origin:
        raise RuntimeError("lmms_eval is not importable")
    return os.path.dirname(os.path.abspath(spec.origin))


def _source_fingerprint():
    """Hash the decoding and evaluation bridge used to produce an artifact."""
    root = os.path.dirname(os.path.abspath(__file__))
    lmms_root = _lmms_eval_pkg_dir()
    entries = [(rel, os.path.join(root, rel)) for rel in _FP_LOCAL]
    entries += [("lmms_eval/" + rel, os.path.join(lmms_root, rel)) for rel in _FP_LMMS_EVAL]
    hashes = {}
    combined = hashlib.sha256()
    for rel, path in entries:
        payload = open(path, "rb").read()
        hashes[rel] = hashlib.sha256(payload).hexdigest()
        combined.update(rel.encode("utf-8") + b"\0" + payload)
    return {"combined_sha256": combined.hexdigest(), "files": hashes, "lmms_eval_dir": lmms_root}


# Per-task result directories, {tag}_{suffix} (moved to module level in v2 so the start-up refusal
# check and the writer use one table).
TASK2SUF = {"mme": "mme8", "mmstar": "mmstar8", "mmbench_en_dev": "mmbench_en_dev8",
            "seedbench": "seed8", "scienceqa": "scienceqa8"}


def _artifact_paths(out_root, tag):
    main_dir = os.path.join(out_root, tag)
    paths = [os.path.join(main_dir, n) for n in ("results.json", "run_config.json", "samples.jsonl.gz")]
    paths += [os.path.join(out_root, f"{tag}_{suf}") for suf in TASK2SUF.values()]
    return paths


def _existing_artifacts(out_root, tag):
    return [p for p in _artifact_paths(out_root, tag) if os.path.exists(p)]


def _init_dist_long_timeout():
    """Pre-init the NCCL process group with a long watchdog timeout + bound device_id, so
    lmms-eval's later Accelerator() reuses THIS group (it skips init when already initialized).
    Fixes the 10-min NCCL watchdog SIGABRT that hit AMBER/MMBench: generation length is
    imbalanced across the 8 data-parallel shards, so fast ranks block on the result-gather
    collective long enough to trip the default timeout. device_id also silences the
    'devices currently unknown' barrier warning."""
    import datetime, torch.distributed as dist
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1 and not dist.is_initialized():
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(lr)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=6),
                                device_id=torch.device(f"cuda:{lr}"))


def _destroy_dist():
    """Close the explicitly created process group on every normal exit path."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# 生成结果的返回约定,由 build_wrapper 按 wrapper 设定:
#   True  = 返回 [提示词 + 续写](LLaVA 式)。我们自己的 IKODGenericHf 会
#           `cont[:, input_ids.shape[-1]:]` 切掉前半段,所以必须连提示词一起给。
#   False = 只返回续写。lmms-eval 官方 InstructBLIP wrapper 直接
#           `batch_decode(cont)` **不切**,因为 HF 原生 InstructBlip.generate 只返回新 token;
#           给它 [提示词+续写] 会把整道题原样解码出来(2026-09-17 实测:模型"回答"就是复述题干)。
RETURN_PROMPT_PLUS = True


def build_wrapper(wrapper, pretrained):
    global RETURN_PROMPT_PLUS
    # Under accelerate multi-proc (data-parallel), each rank must load the FULL model on its OWN
    # GPU. device_map="auto" instead triggers transformers tensor-parallel sharding across all
    # visible GPUs, whose input-prep hook crashes our manual forwards (IndexError in
    # tensor_parallel._prepare_input_fn) -> zeroed llava_hf comprehensive runs. Pin to local rank.
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    dmap = f"cuda:{lr}" if ws > 1 else "auto"
    if wrapper == "qwen2_5_vl":
        from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
        return Qwen2_5_VL(pretrained=pretrained, attn_implementation="eager", device_map=dmap, batch_size=1)
    elif wrapper == "qwen2_vl":
        # The lmms-eval qwen2_vl wrapper exposes use_flash_attention_2 (bool), NOT attn_implementation.
        # IKOD needs eager attentions (output_attentions per forward), so load non-flash and pin eager.
        from lmms_eval.models.simple.qwen2_vl import Qwen2_VL
        inst = Qwen2_VL(pretrained=pretrained, device_map=dmap, batch_size=1, use_flash_attention_2=False)
        _force_eager(inst._model)
        return inst
    elif wrapper == "llava_hf":
        from lmms_eval.models.simple.llava_hf import LlavaHf
        return LlavaHf(pretrained=pretrained, attn_implementation="eager", device_map=dmap, dtype="bfloat16", batch_size=1)
    elif wrapper == "llava_hf_textfix":
        # v2 (D1, 2026-09-27): stock LlavaHf with text-only inputs rebuilt per sample. Constructor
        # arguments are kept identical to the llava_hf branch above (eager, bfloat16, per-rank
        # device_map, batch 1) so that the only behavioural change is the text-only fix.
        from ikod.lmms_llava_hf_textfix_v1 import LlavaHfTextFix
        return LlavaHfTextFix(pretrained=pretrained, attn_implementation="eager", device_map=dmap, dtype="bfloat16", batch_size=1)
    elif wrapper == "instructblip":
        # 2026-09-17:InstructBLIP 必须走 lmms-eval 官方 wrapper,不能用通用 ikod_hf。
        # 通用路径下它对不同题目反复输出同一个词(mmstar 1500 题:空 779、"above" 332、"below" 160),
        # 即模型没被问题条件化;官方 wrapper 把 context 原样交给 InstructBlipProcessor
        # (同时产出 LM 与 Q-Former 两路 input_ids,truncation=True,Q-Former 分词上限 512)。
        # 对照:Qwen3-VL 走官方 wrapper 时 mmstar 输出 B/C/A/D 各 439/368/292/235,分布正常。
        # 与 qwen2_vl 分支同理:官方 wrapper 不收 attn_implementation,加载后 _force_eager,
        # 否则 IKOD 的 output_attentions 会静默拿不到注意力。
        # 两条路各缺一半,必须合一:
        #  * 官方 wrapper 的提示词是对的(context 原样交给 InstructBlipProcessor,同时产出 LM 与
        #    Q-Former 两路 input_ids);而通用 IKODGenericHf._build_inputs 自己调 apply_chat_template,
        #    InstructBLIP 没有 chat template,提示词被套进它没见过的格式 -> 模型不被问题条件化,
        #    mmstar 1500 题输出 空779/above332/below160,得分 0.000。
        #    注意 ikod_instructblip_compat.install() 只换了 models.build_inputs,
        #    **管不到 IKODGenericHf 自己的 _build_inputs** —— 那句"绕开 chat template"的注释一直是空头支票。
        #  * 但官方 wrapper 持有的是裸 InstructBlipForConditionalGeneration,它的输出没有
        #    past_key_values,而 ikod/decode.py:156 要读 -> 直接报
        #    'InstructBlipForConditionalGenerationModelOutput' object has no attribute 'past_key_values'。
        # 所以:用官方 wrapper 管输入,把它的 _model 套进 compat 的 LLaVA 式外壳让 IKOD 能驱动。
        from lmms_eval.models.simple.instructblip import InstructBLIP
        import ikod_instructblip_compat as _IB
        inst = InstructBLIP(pretrained=pretrained, device=f"cuda:{lr}" if ws > 1 else "cuda", batch_size=1)
        _force_eager(inst._model)
        inst._model = _IB.InstructBlipLMShim(inst._model)
        RETURN_PROMPT_PLUS = False       # 官方 wrapper 不切提示词
        # 纯文本样本(ScienceQA 有无图题)会把空 visuals 交给 BLIP 图像处理器:
        #   image_processing_blip.py:259  if do_rescale and is_scaled_image(images[0])
        #   IndexError: list index out of range
        # 官方 wrapper 只处理了 len(visuals) > 1(多图拼一张),没有 == 0 的分支。
        # 通用 IKODGenericHf 当初是靠分支走纯文本路径绕开的,那条补丁管不到这里。
        # 这里补一张中性灰占位图,让无图题也能走同一条前向 —— 语言模型仍只被题干条件化,
        # 与通用路径当初的处理一致(见 ikod_instructblip_compat 的说明)。
        from PIL import Image as _PILImage
        _orig_proc = inst._image_processor
        _PLACEHOLDER = _PILImage.new("RGB", (224, 224), (128, 128, 128))

        class _ProcWithPlaceholder:
            def __init__(self, p): self._p = p
            def __getattr__(self, k): return getattr(self._p, k)
            def __call__(self, *a, **kw):
                v = kw.get("images")
                if v is not None and hasattr(v, "__len__") and len(v) == 0:
                    kw["images"] = [_PLACEHOLDER]
                return self._p(*a, **kw)

        inst._image_processor = _ProcWithPlaceholder(_orig_proc)
        return inst
    elif wrapper == "ikod_hf":
        # Generic native-HF wrapper (ikod.models.build_inputs path). Use for llava_15_hf (stock
        # llava_hf mis-expands image tokens -> 0) and internvl3 (stock wrapper uses .chat()).
        # device_map="" lets the wrapper auto-pin per accelerate local rank (multi-proc) or use the
        # single visible GPU (single-proc).
        from ikod.lmms_wrapper import IKODGenericHf
        return IKODGenericHf(pretrained=pretrained, device_map="", batch_size=1)
    raise ValueError(wrapper)


def _force_eager(model):
    """Set eager attention on a model whose wrapper did not accept attn_implementation, so that
    ikod_generate's output_attentions=True forwards actually return attention tensors. Recurses
    over nested module configs (text + vision towers carry their own _attn_implementation)."""
    seen = set()
    def walk(cfg):
        if cfg is None or id(cfg) in seen:
            return
        seen.add(id(cfg))
        try:
            cfg._attn_implementation = "eager"
        except Exception:
            pass
        for attr in ("text_config", "vision_config", "llm_config", "vision_tower_config"):
            walk(getattr(cfg, attr, None))
    walk(getattr(model, "config", None))
    for sub in ("model", "language_model", "visual", "vision_tower", "vision_model"):
        m = getattr(model, sub, None)
        if m is not None and hasattr(m, "config"):
            walk(m.config)


def build_parser():
    """v2: the v1 argument list, unchanged and in the same order, plus --out_root at the end.
    Factored out of main() so that launchers can check a command line without running it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="ikod", choices=["ikod", "vcd", "m3id", "revisit"])
    ap.add_argument("--eel", default="all", help="revisit early_exit_layers: all|last")
    ap.add_argument("--rel_top", type=float, default=1e-5, help="revisit APC relative_top")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--full_weight", type=float, default=1.0,
                    help="standard-branch logit weight; 0 gives the merged-branch-only ablation")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--ratio", type=float, default=0.4)
    ap.add_argument("--adaptive", default="")
    ap.add_argument("--amin", type=float, default=0.0); ap.add_argument("--amax", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--max_new_cap", type=int, default=1024)
    ap.add_argument("--vision_max_patches", type=int, default=0,
                    help="LLaVA-OneVision AnyRes packing cap; 0 keeps the checkpoint default")
    ap.add_argument("--shard_map", default="",
                    help="optional deterministic task/doc-to-rank JSON for load-balanced full runs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_root", default="results/lmms_ikod",
                    help="v2: root directory for <tag>/ and <tag>_<suffix>/ outputs (v1 wrote results/lmms_ikod)")
    return ap


def main():
    # lmms-eval supplies deterministic-generation kwargs such as temperature=None
    # to the text-only fallback.  Recent transformers versions warn once per
    # example about those algebraically inactive flags, producing megabytes of
    # noise and avoidable synchronization overhead across eight ranks.
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    args = build_parser().parse_args()

    # v2 (2): refuse before anything expensive happens.  Every rank runs this check against the same
    # filesystem, so all ranks leave together and no process group or model is ever created.
    existing = _existing_artifacts(args.out_root, args.tag)
    if existing:
        print(f"[run_lmms_ikod_v2] REFUSE: artifacts for tag {args.tag!r} already exist under "
              f"{os.path.abspath(args.out_root)}: {existing}. Move them away explicitly; this runner never overwrites.",
              file=sys.stderr, flush=True)
        sys.exit(2)
    # v2 (3): fingerprint the sources at start-up; compared again before writing.
    fingerprint_start = _source_fingerprint()

    shard_map_provenance = None
    if args.shard_map:
        shard_map = os.path.abspath(args.shard_map)
        if not os.path.isfile(shard_map):
            raise FileNotFoundError(f"shard map not found: {shard_map}")
        shard_bytes = open(shard_map, "rb").read()
        shard_payload = json.loads(shard_bytes)
        if shard_payload.get("world_size") != int(os.environ.get("WORLD_SIZE", "1")):
            raise ValueError("shard-map world_size does not match the launched process count")
        shard_map_provenance = {
            "path": shard_map,
            "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            "format": shard_payload.get("format"),
            "source": shard_payload.get("source"),
            "source_sha256_uncompressed": shard_payload.get("source_sha256_uncompressed"),
            "task_counts": shard_payload.get("task_counts"),
            "proxy_load_max_over_mean": shard_payload.get("proxy_load_max_over_mean"),
        }
        os.environ["IKOD_SHARD_MAP"] = shard_map
        args.shard_map = shard_map

    _init_dist_long_timeout()
    inst = build_wrapper(args.wrapper, args.pretrained)
    if args.vision_max_patches:
        if getattr(inst._model.config, "model_type", "") != "llava_onevision":
            raise ValueError("--vision_max_patches is only supported for LLaVA-OneVision")
        if args.vision_max_patches < 1:
            raise ValueError("--vision_max_patches must be positive")
        vision_ratio = f"anyres_max_{args.vision_max_patches}"
        inst._model.config.vision_aspect_ratio = vision_ratio
        inst._image_processor.vision_aspect_ratio = vision_ratio
    model = inst._model
    img_tok = getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", None))
    orig_generate = model.generate  # real HF generate, captured before we monkeypatch

    from ikod.decode import ikod_generate
    from ikod.adaptive import AdaptiveAlpha
    from ikod.baselines import vcd_generate, m3id_generate, revisit_generate
    adaptive = AdaptiveAlpha(mode=args.adaptive, alpha_min=args.amin, alpha_max=args.amax, kappa=args.kappa) if args.adaptive else None
    # Forward EVERY non-control kwarg (pixel_values, image_grid_thw, image_sizes, image_flags, ...)
    # to the decoder, so this path matches ikod.models.build_inputs across all backbones, including
    # ones whose processor emits image keys we didn't hardcode (e.g. InternVL3).
    # NOTE: image_sizes is NOT here on purpose -- it is a model input for llava-next/ov, so it must
    # be forwarded to the decoder as part of mi, not stripped as a generation-control kwarg.
    CONTROL_KEYS = {"input_ids", "attention_mask", "max_new_tokens", "min_new_tokens", "do_sample",
                    "temperature", "top_p", "top_k", "num_beams", "use_cache", "pad_token_id",
                    "eos_token_id", "bos_token_id", "stopping_criteria", "logits_processor",
                    "return_dict_in_generate", "output_attentions", "output_hidden_states",
                    "output_scores", "synced_gpus"}

    @torch.no_grad()
    def patched_generate(**kw):
      try:
        ii = kw["input_ids"]; am = kw.get("attention_mask")
        # text-only sample (e.g. ScienceQA has image-less questions): IKOD == greedy here,
        # and ikod_generate would raise "no image span" -> use the original HF generate.
        if img_tok is None or int((ii == img_tok).sum()) == 0:
            # 无图样本:IKOD 等价于 greedy。orig_generate 是 shim 的 generate,同样按
            # LLaVA 式返回 [提示词+续写],所以这里也要遵守同一个开关。
            o = orig_generate(**kw)
            return o if RETURN_PROMPT_PLUS else o[:, ii.shape[-1]:]
        mn = min(int(kw.get("max_new_tokens", 512) or 512), args.max_new_cap)
        mi = {k: v for k, v in kw.items() if k not in CONTROL_KEYS and v is not None}
        if args.method == "vcd":
            gen = vcd_generate(model, ii, image_token_id=img_tok, attention_mask=am, max_new_tokens=mn, alpha=args.alpha, beta=args.beta, **mi)
        elif args.method == "m3id":
            gen = m3id_generate(model, ii, image_token_id=img_tok, attention_mask=am, max_new_tokens=mn, alpha=args.alpha, lam=args.lam, beta=args.beta, **mi)
        elif args.method == "revisit":
            gen = revisit_generate(model, ii, image_token_id=img_tok, attention_mask=am, max_new_tokens=mn, early_exit_layers=args.eel, relative_top=args.rel_top, **mi)
        else:
            gen = ikod_generate(model, ii, image_token_id=img_tok, attention_mask=am, max_new_tokens=mn,
                                ratio=args.ratio, beta=args.beta, alpha=args.alpha,
                                full_weight=args.full_weight, adaptive=adaptive, **mi)
        return torch.cat([ii, gen], dim=1) if RETURN_PROMPT_PLUS else gen
      except Exception:
        if os.environ.get("IKOD_DEBUG"):
            import traceback; traceback.print_exc(); sys.stdout.flush()
        raise

    model.generate = patched_generate  # inject

    import lmms_eval
    from lmms_eval import evaluator
    out = evaluator.simple_evaluate(model=inst, tasks=args.tasks.split(","),
                                    limit=(args.limit or None), batch_size=1, verbosity="ERROR")
    # multi-process (accelerate): only main process gets results + writes
    if out is None:
        _destroy_dist()
        return
    # v2 (2)/(3): decide the output tag once, before the first write.  Never write over artifacts that
    # appeared while this run was in flight; park the outputs next to them instead and fail loudly.
    fingerprint_end = _source_fingerprint()
    fingerprint_stable = (fingerprint_end == fingerprint_start)
    out_tag = args.tag
    collision = _existing_artifacts(args.out_root, args.tag)
    if collision:
        out_tag = f"{args.tag}__collision_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        print(f"[run_lmms_ikod_v2] COLLISION: {collision} appeared during the run; writing to {out_tag} instead",
              file=sys.stderr, flush=True)
    out_dir = os.path.join(args.out_root, out_tag)
    os.makedirs(out_dir, exist_ok=True)
    res = out.get("results", {}) if out else {}
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=2, default=str)
    # Keep the exact decoding/evaluation configuration next to every result.  The metric JSON alone
    # is insufficient to distinguish fixed IKOD from IKOD-A, which previously allowed identically
    # configured runs to be mislabeled in a derived table.
    try:
        import transformers
        import lmms_eval
        versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "lmms_eval": getattr(lmms_eval, "__version__", "unknown"),
        }
    except Exception as exc:
        versions = {"metadata_error": str(exc)}
    run_config = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command_args": vars(args),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "versions": versions,
        # v2 (3): the fingerprint of the code that was loaded (start-up), plus the end-of-run check.
        "source_fingerprint": fingerprint_start,
        "source_fingerprint_end": fingerprint_end,
        "source_fingerprint_stable": fingerprint_stable,
        "model_config_name_or_path": getattr(model.config, "_name_or_path", None),
        "cuda_device_names": sorted({torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}),
        "evaluated_tasks": sorted(res),
        "shard_map_provenance": shard_map_provenance,
        # v2 (4)/(5): environment and wrapper provenance.
        "runner": os.path.basename(os.path.abspath(__file__)),
        "cwd": os.getcwd(),
        "env_at_launch": _ENV_AT_LAUNCH,
        "env_effective": {k: os.environ.get(k) for k in _ENV_KEYS},
        "hf_token_set_at_launch": _HF_TOKEN_SET_AT_LAUNCH,
        "wrapper_impl": f"{type(inst).__module__}.{type(inst).__qualname__}",
        "output_tag": out_tag,
        "output_collision": collision,
    }
    json.dump(run_config, open(os.path.join(out_dir, "run_config.json"), "w"),
              indent=2, default=str)
    # Preserve per-example predictions and metrics in a compressed JSONL audit trail.  This makes
    # anomalous aggregate cells diagnosable without spending another full 8-GPU evaluation.
    samples = out.get("samples", {}) if out else {}
    sample_path = os.path.join(out_dir, "samples.jsonl.gz")
    with gzip.open(sample_path, "wt", encoding="utf-8") as sample_file:
        for task, rows in samples.items():
            for row in rows:
                sample_file.write(json.dumps({"task": task, **row}, default=str,
                                             ensure_ascii=False) + "\n")
    # Also emit per-task dirs named {tag}_{suffix} so make_tables.py's noreg_val finds them when a
    # single run covers multiple tasks (e.g. --tag qwen2_vl_ikod --tasks mme,mmstar,... -> qwen2_vl_ikod_mme8).
    # (v2: TASK2SUF lives at module level.)
    for task, metrics in res.items():
        suf = TASK2SUF.get(task)
        if suf:
            d = os.path.join(args.out_root, f"{out_tag}_{suf}")
            os.makedirs(d, exist_ok=True)
            json.dump({task: metrics}, open(f"{d}/results.json", "w"), indent=2, default=str)
            task_config = dict(run_config)
            task_config["evaluated_tasks"] = [task]
            task_config["parent_tag"] = out_tag
            json.dump(task_config, open(f"{d}/run_config.json", "w"), indent=2, default=str)
    print(f"=== {out_tag} ({args.method}) ===")
    for task, m in res.items():
        keep = {k: round(v, 3) for k, v in m.items() if isinstance(v, (int, float)) and "stderr" not in k}
        print(f"  {task}: {keep}")
    if collision or not fingerprint_stable:
        # No IKOD_LMMS_DONE: the launcher must treat this run as failed and a human must look at it.
        if not fingerprint_stable:
            changed = sorted(k for k in fingerprint_start["files"]
                             if fingerprint_start["files"][k] != fingerprint_end["files"].get(k))
            print(f"[run_lmms_ikod_v2] FAIL: source files changed during the run: {changed}",
                  file=sys.stderr, flush=True)
        _destroy_dist()
        sys.exit(3 if collision else 4)
    print("IKOD_LMMS_DONE")
    _destroy_dist()


if __name__ == "__main__":
    main()
