"""Ultra-fast 100% mathematically lossless AMBER Generative evaluator across all 8 LVLMs.

Evaluates all requested `--methods` (`greedy,revisit,m3id,vcd,sid,ikod,ikod_a`) per image in a single loop:
  - Runs `Image.open`, `bounded_image`, `build_inputs`, and the Vision Tower (`visual` / `get_image_features`)
    ONLY ONCE per image across all clean-image methods (`greedy`, `revisit`, `m3id`, `sid`, `ikod`, `ikod_a`).
  - Uses `fast_ikod_generate` (Early-EOS + Plausibility-Singleton short-circuit + PersistentAuxCache + batched merge).
  - Uses `sid_generate(..., fast_eval=True)` and `m3id_generate_v2(..., fast_eval=True, task='chair')`.
  - Uses `fast_vcd_generate` with Early-EOS short-circuit.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("IKOD_FAST_MERGE", "1")
os.environ.setdefault("IKOD_QWEN3_MAX_PIXELS", str(1280 * 28 * 28))
import sys
import math
import json
import time
import random
import argparse
import fcntl
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

AMBER_GEN = "./workspace/data/amber/amber_generative.jsonl"
AMBER_IMG = "./workspace/data/amber/image"

FAMILY_ALPHA = {
    "llava_15_hf": 2.0,
    "llava_next": 2.0,
    "llava_ov": 2.0,
    "qwen2_vl": 2.0,
    "qwen2_5_vl": 2.0,
    "qwen3_vl": 2.0,
    "internvl3": 0.1,
    "instructblip_vicuna7b": 0.5,
}

ALL_METHODS = ["greedy", "revisit", "m3id", "vcd", "sid", "ikod", "ikod_a"]
QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 1280 * 28 * 28


def bounded_image(image: Image.Image, model_name: str) -> Image.Image:
    max_edge = 768 if model_name == "llava_ov" else 1024
    if max(image.size) <= max_edge:
        return image
    scale = max_edge / max(image.size)
    size = tuple(max(1, int(round(v * scale))) for v in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def _setup_model_compat(model_name: str):
    if model_name == "qwen3_vl":
        from ikod.qwen3_vision_cap_v1 import maybe_install as _qwen3_vision_cap
        _qwen3_vision_cap()
        import ikod_qwen3_compat
        ikod_qwen3_compat.install()
    elif model_name == "instructblip_vicuna7b":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import ikod_instructblip_compat as compat
        compat.install()
        _lv = compat.load_vlm
        _bi = compat.build_inputs
        from ikod import models
        models.load_vlm = lambda name, device="cuda", dtype_=None, eager=True, **kw: _lv(
            name, device=device, dtype=torch.float32, eager=eager
        )
        models.build_inputs = lambda name, proc, image, question, device="cuda", dtype_=None: _bi(
            name, proc, image, question, device=device, dtype=torch.float32
        )


def _install_vision_and_proc_cache(model, proc, model_name: str):
    if model_name in ("qwen2_vl", "qwen2_5_vl") and hasattr(proc, "image_processor"):
        ip = proc.image_processor
        ip.min_pixels = int(QWEN_MIN_PIXELS)
        ip.max_pixels = int(QWEN_MAX_PIXELS)
        ip.size = {"shortest_edge": int(QWEN_MIN_PIXELS), "longest_edge": int(QWEN_MAX_PIXELS)}

    if model_name == "instructblip_vicuna7b":
        return

    def _clone_out(x):
        if torch.is_tensor(x):
            return x.clone()
        if isinstance(x, list):
            return [_clone_out(e) for e in x]
        if isinstance(x, tuple):
            return tuple(_clone_out(e) for e in x)
        return x

    vis = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
    if vis is not None and hasattr(vis, "forward"):
        _orig_vis = vis.forward
        _vis_cache = {}

        def _fast_vis(hidden_states, *a, **kw):
            key = (
                hidden_states.data_ptr(),
                hidden_states.shape,
                float(hidden_states.flatten()[0].item()),
                float(hidden_states.flatten()[-1].item()),
            )
            if key in _vis_cache:
                return _clone_out(_vis_cache[key])
            res = _orig_vis(hidden_states, *a, **kw)
            _vis_cache.clear()
            _vis_cache[key] = res
            return _clone_out(res)

        vis.forward = _fast_vis
    elif hasattr(model, "get_image_features"):
        _orig_gif = model.get_image_features
        _gif_cache = {}

        def _fast_gif(pixel_values, *a, **kw):
            key = (
                pixel_values.data_ptr(),
                pixel_values.shape,
                float(pixel_values.flatten()[0].item()),
                float(pixel_values.flatten()[-1].item()),
            )
            if key in _gif_cache:
                return _clone_out(_gif_cache[key])
            res = _orig_gif(pixel_values, *a, **kw)
            _gif_cache.clear()
            _gif_cache[key] = res
            return _clone_out(res)

        model.get_image_features = _fast_gif
        if hasattr(getattr(model, "model", None), "get_image_features"):
            model.model.get_image_features = _fast_gif



@torch.no_grad()
def fast_revisit_generate(
    model, input_ids, *, image_token_id=None, attention_mask=None,
    max_new_tokens=512, early_exit_layers="all", relative_top=1e-5,
    eos_token_id=None, **model_inputs
):
    import torch.nn.functional as F
    from ikod.baselines import _resolve_eos, _is_mrope, _rope_next
    from ikod.image_span import locate_image_span, image_token_id_from_config

    device = input_ids.device
    if image_token_id is None:
        image_token_id = image_token_id_from_config(model.config)
    eos_ids = _resolve_eos(model, eos_token_id)
    is_mrope = _is_mrope(model)
    grid = model_inputs.get("image_grid_thw")
    span = locate_image_span(input_ids[0], image_token_id)
    if span is None:
        raise ValueError("revisit: no image span")
    s, e = span
    img_pos = torch.arange(s, e, device=device)

    final_layer = (
        model.config.num_hidden_layers
        if hasattr(model.config, "num_hidden_layers")
        else model.config.text_config.num_hidden_layers
    )
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    start_layer = 0 if not tied else (2 if final_layer > 2 else (1 if final_layer == 2 else 0))
    layers = list(range(start_layer, final_layer + 1, 2))
    if final_layer % 2 != 0 and final_layer not in layers:
        layers += [final_layer]
    lm_head = model.get_output_embeddings()
    lm_w = lm_head.weight
    lm_b = getattr(lm_head, "bias", None)

    out = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
        output_hidden_states=True, return_dict=True, **model_inputs
    )
    hs = out.hidden_states
    hs_list = [hs[i][:, img_pos, :] for i in layers]
    hs_bank = torch.stack(hs_list, dim=1).view(1, -1, hs_list[0].size(-1)).detach()
    del hs, hs_list
    out.hidden_states = None
    out.logits = out.logits[:, -1:, :]

    cur_len = input_ids.shape[1]
    next3d = _rope_next(model, input_ids, grid, attention_mask) if is_mrope else None
    am = attention_mask
    generated = []
    log_rt = math.log(relative_top)

    for step_idx in range(max_new_tokens):
        mature = out.logits[:, -1, :].float()
        if eos_ids and int(mature.argmax(-1)) in eos_ids:
            break
        mature_lsm = mature.log_softmax(-1)
        thresh = mature_lsm.max(-1, keepdim=True).values + log_rt
        plaus = (mature_lsm >= thresh)[0]
        plaus[mature_lsm[0].argmax()] = True
        mat_l = mature[:, plaus].log_softmax(-1)
        mat_p = mat_l.exp()
        w_sub = lm_w[plaus]
        b_sub = lm_b[plaus] if lm_b is not None else None
        best_jsd = None
        best_prem = None
        for k0 in range(0, hs_bank.size(1), 1024):
            hs_c = hs_bank[:, k0:k0+1024, :]
            pl_c = F.linear(hs_c, w_sub, b_sub).float().log_softmax(-1)
            pp_c = pl_c.exp()
            avg_c = 0.5 * (mat_p[:, None, :] + pp_c)
            kl1_c = F.kl_div(mat_l[:, None, :], avg_c, reduction="none").mean(-1)
            kl2_c = F.kl_div(pl_c, avg_c, reduction="none").mean(-1)
            jsd_c = (0.5 * (kl1_c + kl2_c)).mean(0)
            m_val, m_idx = jsd_c.min(dim=0)
            if best_jsd is None or m_val < best_jsd:
                best_jsd = m_val
                best_prem = pl_c[0, m_idx]
            del pl_c, pp_c, avg_c, kl1_c, kl2_c, jsd_c
        fused = torch.full_like(mature, float("-inf"))
        fused[0, plaus] = mat_l[0] + best_prem
        if eos_ids:
            fused[:, eos_ids] = float("-inf")
        nxt = fused.argmax(-1)
        generated.append(nxt)
        if step_idx + 1 == max_new_tokens:
            break
        nt = nxt.view(1, 1)
        if am is not None:
            am = torch.cat([am, am.new_ones((1, 1))], 1)
        kw = dict(use_cache=True, return_dict=True, attention_mask=am)
        if is_mrope:
            kw["position_ids"] = next3d
            kw["cache_position"] = torch.tensor([cur_len], device=device)
        out = model(input_ids=nt, past_key_values=out.past_key_values, **kw)
        if is_mrope:
            next3d = next3d + 1
        cur_len += 1
    return (
        torch.stack(generated, 1)
        if generated
        else torch.empty((1, 0), dtype=torch.long, device=device)
    )

@torch.no_grad()
def fast_vcd_generate(
    model, input_ids, *, attention_mask=None, max_new_tokens=512,
    alpha=1.0, beta=0.1, noise_step=500, eos_token_id=None, **model_inputs
):
    from ikod.baselines import _resolve_eos, _is_mrope, _rope_next, add_diffusion_noise

    device = input_ids.device
    eos_ids = _resolve_eos(model, eos_token_id)
    is_mrope = _is_mrope(model)
    grid = model_inputs.get("image_grid_thw")

    noisy = dict(model_inputs)
    if "pixel_values" in noisy and noisy["pixel_values"] is not None:
        noisy["pixel_values"] = add_diffusion_noise(noisy["pixel_values"], noise_step)

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                return_dict=True, **model_inputs)
    outn = None
    next3d = _rope_next(model, input_ids, grid, attention_mask) if is_mrope else None
    next3d_n = next3d.clone() if (is_mrope and next3d is not None) else None
    cur_len = input_ids.shape[1]
    cur_len_n = input_ids.shape[1]
    generated = []
    pending_n = []
    am = attention_mask
    log_beta = math.log(beta)

    for step_idx in range(max_new_tokens):
        lc = out.logits[:, -1, :].float()
        if eos_ids and int(lc.argmax(-1)) in eos_ids:
            break
        cutoff = log_beta + lc.max(-1, keepdim=True).values
        pmask = lc >= cutoff
        if eos_ids:
            pmask[:, eos_ids] = False
        need_noisy = int(pmask.sum().item()) > 1

        if not need_noisy:
            nxt = lc.masked_fill(~pmask, float("-inf")).argmax(-1)
        else:
            if outn is None:
                outn = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                             return_dict=True, **noisy)
            for p_nt, p_am in pending_n:
                kwn = dict(use_cache=True, return_dict=True, attention_mask=p_am)
                if is_mrope:
                    kwn["position_ids"] = next3d_n
                    kwn["cache_position"] = torch.tensor([cur_len_n], device=device)
                outn = model(input_ids=p_nt, past_key_values=outn.past_key_values, **kwn)
                if is_mrope:
                    next3d_n = next3d_n + 1
                cur_len_n += 1
            pending_n.clear()
            ln = outn.logits[:, -1, :].float()
            diffs = (1.0 + alpha) * lc - alpha * ln
            if eos_ids:
                diffs[:, eos_ids] = float("-inf")
            scores = diffs.masked_fill(lc < cutoff, float("-inf"))
            nxt = scores.argmax(-1)

        generated.append(nxt)
        if step_idx + 1 == max_new_tokens:
            break
        nt = nxt.view(1, 1)
        if am is not None:
            am = torch.cat([am, am.new_ones((1, 1))], 1)
        kw = dict(use_cache=True, return_dict=True, attention_mask=am)
        if is_mrope:
            kw["position_ids"] = next3d
            kw["cache_position"] = torch.tensor([cur_len], device=device)
        out = model(input_ids=nt, past_key_values=out.past_key_values, **kw)
        if is_mrope:
            next3d = next3d + 1
        cur_len += 1
        if eos_ids and int(out.logits[:, -1, :].argmax(-1)) in eos_ids:
            break
        pending_n.append((nt, am))
    return (
        torch.stack(generated, 1)
        if generated
        else torch.empty((1, 0), dtype=torch.long, device=device)
    )


def _heal_one_disc_chunk_if_needed():
    import subprocess
    disc_dir = "./workspace/results/amber_8b_v1/disc_chunks"
    disc_script = "./workspace/code/eval_amber_disc_fast_8b.py"
    disc_methods = ["greedy", "vcd", "m3id", "sid", "revisit", "ikod", "ikod_a"]
    n_total = 14216
    chunks = 4
    env_sub = os.environ.copy()
    env_sub["IKOD_NO_TAIL_SWEEP"] = "1"
    for m_cand in FAMILY_ALPHA.keys():
        for c_cand in range(chunks):
            t_len = (n_total * (c_cand + 1)) // chunks - (n_total * c_cand) // chunks
            c_files = [
                os.path.join(disc_dir, f"{m_cand}_{m}_chunk{c_cand}of{chunks}.jsonl")
                for m in disc_methods
            ]
            if all(os.path.exists(cf) and sum(1 for _ in open(cf)) >= t_len for cf in c_files):
                continue
            lk_p = os.path.join(disc_dir, f"{m_cand}_chunk{c_cand}of{chunks}.lock")
            try:
                lk_test = open(lk_p, "w")
                fcntl.flock(lk_test, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lk_test, fcntl.LOCK_UN)
                lk_test.close()
            except BlockingIOError:
                continue
            py_bin = "/data/yjb/miniconda3/envs/ikod_qwen3/bin/python" if m_cand == "qwen3_vl" else "/data/yjb/miniconda3/envs/ikodnew/bin/python"
            print(f"[PRE-GEN HEAL DISC] Healing incomplete {m_cand} chunk {c_cand}/{chunks} first", flush=True)
            subprocess.call([
                py_bin, disc_script,
                "--model", m_cand,
                "--chunks", str(chunks),
                "--idx", str(c_cand),
                "--out_dir", disc_dir,
            ], env=env_sub)
            return


def main():
    for _heal_iter in range(8):
        _heal_one_disc_chunk_if_needed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(FAMILY_ALPHA.keys()))
    ap.add_argument("--methods", default="greedy,revisit,m3id,vcd,sid,ikod,ikod_a")
    ap.add_argument("--qfile", default=AMBER_GEN)
    ap.add_argument("--img_dir", default=AMBER_IMG)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--reverse_sidecar", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method: {m}")

    os.makedirs(args.out_dir, exist_ok=True)
    qs_all = [json.loads(l) for l in open(args.qfile)]
    if args.limit:
        qs_all = qs_all[: args.limit]
    qs = qs_all[args.idx :: args.chunks]
    target_n = len(qs)

    todo_methods = []
    out_paths = {}
    lock_files = {}
    done_by_method = {}
    fouts = {}

    _sfx = "_rev" if args.reverse_sidecar else ""
    fwd_paths = {m: os.path.join(args.out_dir, f"{args.model}_{m}_chunk{args.idx}of{args.chunks}.jsonl") for m in methods}
    for m in methods:
        p = os.path.join(args.out_dir, f"{args.model}_{m}_chunk{args.idx}of{args.chunks}{_sfx}.jsonl")
        out_paths[m] = p
        if os.path.exists(p):
            try:
                if sum(1 for _ in open(p)) >= target_n:
                    print(f"[SKIP] {p} already complete ({target_n} rows).", flush=True)
                    continue
            except Exception:
                pass
        lf = open(p + ".lock", "w")
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_files[m] = lf
            todo_methods.append(m)
        except BlockingIOError:
            print(f"[SKIP] {p} locked by another worker.", flush=True)

    if not todo_methods:
        print(f"[ALL DONE] {args.model} chunk {args.idx}/{args.chunks} has nothing left to run.", flush=True)
        return

    for m in todo_methods:
        p = out_paths[m]
        done_qids = set()
        existing_rows = []
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if "question_id" in r and "text" in r:
                            done_qids.add(int(r["question_id"]))
                            existing_rows.append(r)
                    except Exception:
                        pass
        done_by_method[m] = done_qids
        fout = open(p, "w")
        for r in existing_rows:
            fout.write(json.dumps(r) + "\n")
        fout.flush()
        fouts[m] = fout

    _setup_model_compat(args.model)
    from ikod.models import load_vlm, build_inputs
    from ikod.baselines import revisit_generate
    from ikod.fast_decode import fast_ikod_generate
    from ikod.adaptive import AdaptiveAlpha
    from sid_m3id_port import _sid_set, sid_generate, m3id_generate_v2

    model, proc, img_tok = load_vlm(args.model)
    _install_vision_and_proc_cache(model, proc, args.model)
    alpha_fam = FAMILY_ALPHA[args.model]

    t0 = time.time()
    n_imgs_done = 0

    qs_iter = list(reversed(qs)) if args.reverse_sidecar else qs
    for i, q in enumerate(qs_iter):
        qid = int(q["question_id"])
        if args.reverse_sidecar:
            fwd_done = {}
            for m in todo_methods:
                s = set()
                if os.path.exists(fwd_paths[m]):
                    with open(fwd_paths[m]) as _ff:
                        for _ln in _ff:
                            if _ln.strip():
                                try:
                                    s.add(int(json.loads(_ln)["question_id"]))
                                except Exception:
                                    pass
                fwd_done[m] = s
            if all(len(done_by_method[m] | fwd_done[m]) >= len(qs) for m in todo_methods):
                print(f"[REV-MEET] {args.model} chunk {args.idx}/{args.chunks} met forward worker at i={i}! Merging...", flush=True)
                break
            need_m = [m for m in todo_methods if qid not in done_by_method[m] and qid not in fwd_done[m]]
        else:
            need_m = [m for m in todo_methods if qid not in done_by_method[m]]
        if not need_m:
            continue

        img_cand_shm = os.path.join("/dev/shm/amber_images", q["image"])
        img_cand_disk = os.path.join(args.img_dir, q["image"])
        raw_img = None
        for _io_try in range(6):
            try:
                _p = img_cand_shm if os.path.exists(img_cand_shm) else img_cand_disk
                raw_img = Image.open(_p).convert("RGB")
                break
            except OSError:
                time.sleep(0.5 * (_io_try + 1))
                if _io_try >= 2:
                    raw_img = Image.open(img_cand_disk).convert("RGB")
                    break
        image = bounded_image(raw_img, args.model)
        prompt = q.get("text") or args.prompt
        inputs = build_inputs(args.model, proc, image, prompt)
        ii = inputs["input_ids"]
        am = inputs.get("attention_mask")
        gen_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}

        for m in need_m:
            seed = 42 + args.idx
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            with torch.inference_mode():
                _sid_set(model, on=False)
                if m == "greedy":
                    g = model.generate(
                        **inputs, do_sample=False, max_new_tokens=args.max_new,
                        use_cache=True, repetition_penalty=1.0
                    )
                    out_ids = g[:, ii.shape[1]:]
                elif m == "revisit":
                    out_ids = fast_revisit_generate(
                        model, ii, attention_mask=am, image_token_id=img_tok,
                        max_new_tokens=args.max_new, early_exit_layers="all",
                        relative_top=1e-5, **gen_inputs
                    )
                elif m == "m3id":
                    out_ids = m3id_generate_v2(
                        model, ii, attention_mask=am, image_token_id=img_tok,
                        max_new_tokens=args.max_new, alpha=0.3, lam=0.02, beta=0.1,
                        task="chair", use_log_softmax=True, fast_eval=True, **gen_inputs
                    )
                elif m == "vcd":
                    out_ids = fast_vcd_generate(
                        model, ii, attention_mask=am, max_new_tokens=args.max_new,
                        alpha=1.0, beta=0.1, noise_step=500, **gen_inputs
                    )
                elif m == "sid":
                    out_ids = sid_generate(
                        model, ii, attention_mask=am, image_token_id=img_tok,
                        max_new_tokens=args.max_new, alpha=0.5, beta=0.1,
                        retain_ratio=0.10, agg_layer=2, fast_eval=True, **gen_inputs
                    )
                elif m == "ikod":
                    out_ids = fast_ikod_generate(
                        model, ii, attention_mask=am, image_token_id=img_tok,
                        max_new_tokens=args.max_new, ratio=0.4, beta=0.1,
                        alpha=alpha_fam, adaptive=None, **gen_inputs
                    )
                elif m == "ikod_a":
                    ada = AdaptiveAlpha(mode="p1", alpha_min=0.0, alpha_max=1.0, kappa=10.0)
                    out_ids = fast_ikod_generate(
                        model, ii, attention_mask=am, image_token_id=img_tok,
                        max_new_tokens=args.max_new, ratio=0.4, beta=0.1,
                        alpha=alpha_fam, adaptive=ada, **gen_inputs
                    )

            text = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            fouts[m].write(
                json.dumps({"question_id": qid, "image": q["image"], "text": text}) + "\n"
            )
            fouts[m].flush()

        torch.cuda.empty_cache()
        n_imgs_done += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(qs):
            dt = max(1e-6, time.time() - t0)
            print(
                f"[GEN {args.model} chunk {args.idx}/{args.chunks}] "
                f"[{i+1}/{len(qs)}] ({len(need_m)} methods) "
                f"{n_imgs_done/dt:.2f} img/s ({len(need_m)*n_imgs_done/dt:.2f} caps/s) "
                f"elapsed={dt:.0f}s",
                flush=True,
            )

    for f in fouts.values():
        f.close()
    for lf in lock_files.values():
        lf.close()
    if args.reverse_sidecar:
        import subprocess, signal
        sig_fwd = f"--model {args.model} --methods {args.methods} --chunks {args.chunks} --idx {args.idx}"
        ps_lines = subprocess.check_output("ps -eo pid,cmd | grep eval_amber_gen_fast_8b | grep -v grep || true", shell=True).decode().splitlines()
        for pl in ps_lines:
            if sig_fwd in pl and "--reverse_sidecar" not in pl:
                fpid = int(pl.strip().split()[0])
                try:
                    os.kill(fpid, signal.SIGTERM)
                    print(f"[REV-MEET] Stopped forward worker PID {fpid}", flush=True)
                except Exception:
                    pass
        time.sleep(1.5)
        for m in todo_methods:
            merged = {}
            for src_p in (fwd_paths[m], out_paths[m]):
                if os.path.exists(src_p):
                    with open(src_p) as _f:
                        for _ln in _f:
                            if _ln.strip():
                                try:
                                    _r = json.loads(_ln)
                                    if "question_id" in _r and "text" in _r:
                                        merged[int(_r["question_id"])] = _r
                                except Exception:
                                    pass
            if len(merged) >= len(qs):
                tmp_p = fwd_paths[m] + ".tmp"
                with open(tmp_p, "w") as _wf:
                    for _k in sorted(merged.keys()):
                        _wf.write(json.dumps(merged[_k]) + "\n")
                os.replace(tmp_p, fwd_paths[m])
                print(f"[REV-MERGE] Merged {len(merged)}/{len(qs)} rows into {fwd_paths[m]}", flush=True)
        return
    print(
        f"DONE GEN {args.model} chunk {args.idx}/{args.chunks} "
        f"n={len(qs)} methods={todo_methods} in {time.time()-t0:.1f}s",
        flush=True,
    )
    if os.environ.get("IKOD_NO_TAIL_SWEEP") != "1":
        import subprocess
        env_sub = os.environ.copy()
        env_sub["IKOD_NO_TAIL_SWEEP"] = "1"
        log_dir = os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "logs")
        for m_cand in FAMILY_ALPHA.keys():
            for c_cand in range(args.chunks):
                t_len = len(range(c_cand, 1004, args.chunks))
                for gname, mlist in [("gA", "greedy,revisit,m3id,vcd"), ("gB", "sid,ikod,ikod_a")]:
                    log_p = os.path.join(log_dir, f"gen_{m_cand}_{gname}_c{c_cand}of{args.chunks}.log")
                    if not os.path.exists(log_p):
                        continue
                    c_files = [
                        os.path.join(args.out_dir, f"{m_cand}_{m}_chunk{c_cand}of{args.chunks}.jsonl")
                        for m in mlist.split(",")
                    ]
                    if all(os.path.exists(cf) and sum(1 for _ in open(cf)) >= t_len for cf in c_files):
                        continue
                    lk_p = os.path.join(args.out_dir, f"{m_cand}_{gname}_chunk{c_cand}of{args.chunks}.lock")
                    try:
                        lk_test = open(lk_p, "w")
                        fcntl.flock(lk_test, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(lk_test, fcntl.LOCK_UN)
                        lk_test.close()
                    except BlockingIOError:
                        continue
                    py_bin = "/data/yjb/miniconda3/envs/ikod_qwen3/bin/python" if m_cand == "qwen3_vl" else "/data/yjb/miniconda3/envs/ikodnew/bin/python"
                    print(f"[TAIL-HEAL GEN] Healing unlocked incomplete {m_cand} {gname} chunk {c_cand}/{args.chunks}", flush=True)
                    subprocess.call([
                        py_bin, os.path.abspath(__file__),
                        "--model", m_cand,
                        "--methods", mlist,
                        "--chunks", str(args.chunks),
                        "--idx", str(c_cand),
                        "--out_dir", args.out_dir,
                    ], env=env_sub)
                    return


if __name__ == "__main__":
    main()
