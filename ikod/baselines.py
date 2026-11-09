"""Model-agnostic training-free decoding baselines for native-HF VLMs, sharing the
same harness (models.py) and the same fair EOS handling as ikod_generate (stopping is
decided by the original/clean branch; the contrastive branch never forces EOS).

Implemented:
  vcd_generate : Visual Contrastive Decoding (Leng et al., CVPR'24). Contrast clean vs
                 diffusion-noised image:  logits = (1+a)*clean - a*noisy, plausibility on clean.
  kvsmooth_generate : KVSmooth (Jiang et al., CVPR'26). Smooth each generated token's
                 cached K/V with the preceding token using a layerwise row-entropy FIFO rank.

TODO (next): m3id, sid, pai, opera (beam, heavier), only, dcd. Add incrementally.
"""
from __future__ import annotations
import math, sys, torch
import torch.nn.functional as F
from .image_span import locate_image_span, image_token_id_from_config


def _resolve_eos(model, eos_token_id):
    if eos_token_id is None:
        gc = getattr(model, "generation_config", None)
        eos_token_id = getattr(gc, "eos_token_id", None) if gc is not None else None
        if eos_token_id is None:
            eos_token_id = getattr(model.config, "eos_token_id", None)
    return [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id or [])


def add_diffusion_noise(x, noise_step=500, num_steps=1000):
    """Standard VCD diffusion noising of a pixel tensor (q_sample with linear-sigmoid beta)."""
    betas = torch.linspace(-6, 6, num_steps, device=x.device)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    ap = torch.cumprod(alphas, dim=0)
    t = int(max(0, min(num_steps - 1, noise_step)))
    a = ap[t].sqrt().to(x.dtype)
    am = (1 - ap[t]).sqrt().to(x.dtype)
    return a * x + am * torch.randn_like(x)


def _is_mrope(model):
    return type(model).__name__ in ("Qwen2VLForConditionalGeneration",
                                    "Qwen2_5_VLForConditionalGeneration")


def _rope_next(model, ids, grid, am):
    """3D position of the NEXT token for M-RoPE models (get_rope_index then +1). grid=None for
    text-only (no image) -> positions degenerate to a shared arange across the 3 sections."""
    pos3d, _ = model.model.get_rope_index(ids, image_grid_thw=grid, attention_mask=am)
    return pos3d[..., -1:] + 1


def _cache_items(past):
    """Mutable (key, value) tensor references for legacy tuples and HF Cache objects."""
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return [(past.key_cache[i], past.value_cache[i]) for i in range(len(past.key_cache))]
    return list(past)


def _kvsmooth_rank_lambda(history, value, *, queue_len=15, lambda_ref=0.9,
                          rank_denominator="fixed"):
    """Equation (13)--(14) of KVSmooth.

    The paper divides the count by the fixed maximum queue length M (not the
    current warm-up length), then clips it to lambda_ref +/- 0.2.
    """
    history.append(float(value))
    if len(history) > queue_len:
        del history[:-queue_len]
    rank = sum(x < float(value) for x in history)
    if rank_denominator == "fixed":
        denominator = queue_len
    elif rank_denominator == "current":
        denominator = len(history)
    else:
        raise ValueError(f"unknown rank_denominator={rank_denominator!r}")
    raw = rank / float(denominator)
    return max(lambda_ref - 0.2, min(lambda_ref + 0.2, raw))


@torch.no_grad()
def _apply_kvsmooth(past, attentions, histories, *, layer_start=3, layer_end=31,
                    queue_len=15, lambda_ref=0.9, rank_denominator="fixed", eps=1e-8):
    """Post-forward KV update for one generated token.

    Row entropy is computed from the current query's attention distribution,
    averaged over heads as in Eq. (5). The current cached K/V is then replaced
    in place according to Eq. (15). Prompt K/V entries are never changed.
    """
    items = _cache_items(past)
    hi = min(int(layer_end), len(items) - 1, len(attentions) - 1)
    for layer in range(max(0, int(layer_start)), hi + 1):
        a = attentions[layer][0, :, -1, :].float()
        # KVSmooth Eq. (5): -alpha * log(alpha + epsilon). Do not clamp alpha;
        # zero-probability entries contribute exactly zero to the entropy.
        entropy = float(-(a * (a + eps).log()).sum(-1).mean())
        lam = _kvsmooth_rank_lambda(
            histories[layer], entropy, queue_len=queue_len, lambda_ref=lambda_ref,
            rank_denominator=rank_denominator,
        )
        k, v = items[layer]
        if k.shape[-2] < 2:
            continue
        # Clone the preceding entries because the destination aliases the cache.
        k_prev = k[..., -2, :].clone()
        v_prev = v[..., -2, :].clone()
        k[..., -1, :].mul_(1.0 - lam).add_(k_prev, alpha=lam)
        v[..., -1, :].mul_(1.0 - lam).add_(v_prev, alpha=lam)


@torch.no_grad()
def kvsmooth_generate(model, input_ids, *, attention_mask=None, max_new_tokens=512,
                      layer_start=3, layer_end=31, queue_len=15, lambda_ref=0.9,
                      rank_denominator="fixed", enabled=True,
                      eos_token_id=None, **model_inputs):
    """Paper-formula KVSmooth reimplementation for native-HF autoregressive VLMs.

    KVSmooth modifies the single active cache. At each generated-token forward,
    its row entropy sets the EMA coefficient and the just-appended K/V entry is
    smoothed with its immediately preceding entry. This update affects all later
    queries without adding a contrastive branch. The implementation follows the
    CVPR 2026 paper because no official native-HF implementation was available.
    """
    device = input_ids.device
    eos_ids = _resolve_eos(model, eos_token_id)
    is_mrope = _is_mrope(model)
    grid = model_inputs.get("image_grid_thw")
    text_cfg = getattr(model.config, "text_config", None)
    llm_cfg = getattr(model.config, "llm_config", None)
    n_layers = (getattr(model.config, "num_hidden_layers", None)
                or getattr(text_cfg, "num_hidden_layers", None)
                or getattr(llm_cfg, "num_hidden_layers", None))

    # The prompt cache is left untouched; smoothing starts with generated tokens.
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                output_attentions=False, return_dict=True, **model_inputs)
    if n_layers is None:
        n_layers = len(_cache_items(out.past_key_values))
    histories = [[] for _ in range(int(n_layers))]
    next3d = _rope_next(model, input_ids, grid, attention_mask) if is_mrope else None
    cur_len = input_ids.shape[1]
    generated = []
    am = attention_mask

    for step_idx in range(max_new_tokens):
        logits = out.logits[:, -1, :].float()
        if eos_ids and int(logits.argmax(-1)) in eos_ids:
            break
        nxt = logits.argmax(-1)
        generated.append(nxt)
        if step_idx + 1 == max_new_tokens:
            break

        nt = nxt.view(1, 1)
        if am is not None:
            am = torch.cat([am, am.new_ones((1, 1))], 1)
        kw = dict(use_cache=True, output_attentions=enabled, return_dict=True, attention_mask=am)
        if is_mrope:
            kw["position_ids"] = next3d
            kw["cache_position"] = torch.tensor([cur_len], device=device)
        out = model(input_ids=nt, past_key_values=out.past_key_values, **kw)
        if enabled:
            _apply_kvsmooth(
                out.past_key_values, out.attentions, histories,
                layer_start=layer_start, layer_end=layer_end,
                queue_len=queue_len, lambda_ref=lambda_ref,
                rank_denominator=rank_denominator,
            )
        if is_mrope:
            next3d = next3d + 1
        cur_len += 1
    return (torch.stack(generated, 1) if generated else
            torch.empty((1, 0), dtype=torch.long, device=device))


@torch.no_grad()
def vcd_generate(model, input_ids, *, image_token_id=None, attention_mask=None,
                 max_new_tokens=512, alpha=1.0, beta=0.1, noise_step=500,
                 eos_token_id=None, **model_inputs):
    device = input_ids.device
    eos_ids = _resolve_eos(model, eos_token_id)
    is_mrope = _is_mrope(model)
    grid = model_inputs.get("image_grid_thw")

    # noised image inputs (copy, perturb pixel_values)
    noisy = dict(model_inputs)
    if "pixel_values" in noisy and noisy["pixel_values"] is not None:
        noisy["pixel_values"] = add_diffusion_noise(noisy["pixel_values"], noise_step)

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                return_dict=True, **model_inputs)
    outn = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                 return_dict=True, **noisy)
    # clean & noisy branches share input_ids (only pixels differ) -> same positions/cache length
    next3d = _rope_next(model, input_ids, grid, attention_mask) if is_mrope else None
    cur_len = input_ids.shape[1]
    generated = []
    am = attention_mask
    for step_idx in range(max_new_tokens):
        lc = out.logits[:, -1, :].float()
        ln = outn.logits[:, -1, :].float()
        if eos_ids and int(lc.argmax(-1)) in eos_ids:
            break
        diffs = (1 + alpha) * lc - alpha * ln
        if eos_ids:
            diffs[:, eos_ids] = float("-inf")
        cutoff = math.log(beta) + lc.max(-1, keepdim=True).values
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
        outn = model(input_ids=nt, past_key_values=outn.past_key_values, **kw)
        if is_mrope:
            next3d = next3d + 1
        cur_len += 1
    return torch.stack(generated, 1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)


@torch.no_grad()
def m3id_generate(model, input_ids, *, image_token_id=None, attention_mask=None,
                  max_new_tokens=512, alpha=0.3, lam=0.02, beta=0.1,
                  eos_token_id=None, **model_inputs):
    """M3ID (Favero et al., CVPR'24). Contrast conditioned (image) vs text-only (image removed):
       l_hat = l_c + 1[max softmax(l_c) < alpha] * (exp(lam*t)-1) * (l_c - l_u).
    Text-only branch DELETES the image-token span from input_ids (independent KV cache).
    Supports plain-RoPE (LLaVA-*/InternVL3) AND M-RoPE (Qwen2/2.5-VL): the two branches have
    different lengths, so each carries its own 3D position + cache_position.
    """
    device = input_ids.device
    if image_token_id is None:
        image_token_id = image_token_id_from_config(model.config)
    eos_ids = _resolve_eos(model, eos_token_id)
    is_mrope = _is_mrope(model)
    grid = model_inputs.get("image_grid_thw")
    span = locate_image_span(input_ids[0], image_token_id)
    if span is None:
        raise ValueError("m3id: no image span")
    s, e = span
    # conditioned (clean, with image) branch
    out_c = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                  return_dict=True, **model_inputs)
    # text-only branch: remove image-token span + all image model_inputs
    keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device); keep[s:e] = False
    ids_u = input_ids[:, keep]
    am_u = attention_mask[:, keep] if attention_mask is not None else None
    out_u = model(input_ids=ids_u, attention_mask=am_u, use_cache=True, return_dict=True)
    # per-branch next-token positions (M-RoPE): conditioned keeps the image grid, text-only drops it
    next_c = _rope_next(model, input_ids, grid, attention_mask) if is_mrope else None
    next_u = _rope_next(model, ids_u, None, am_u) if is_mrope else None
    cur_c = input_ids.shape[1]; cur_u = ids_u.shape[1]
    generated = []; am_c = attention_mask
    for t in range(max_new_tokens):
        lc = out_c.logits[:, -1, :].float()
        lu = out_u.logits[:, -1, :].float()
        if eos_ids and int(lc.argmax(-1)) in eos_ids:
            break
        w = math.exp(lam * t) - 1.0
        gate = float(lc.softmax(-1).max()) < alpha
        diffs = lc + w * (lc - lu) if gate else lc.clone()
        if eos_ids:
            diffs[:, eos_ids] = float("-inf")
        if beta:
            cutoff = math.log(beta) + lc.max(-1, keepdim=True).values
            diffs = diffs.masked_fill(lc < cutoff, float("-inf"))
        nxt = diffs.argmax(-1)
        generated.append(nxt)
        if t + 1 == max_new_tokens:
            break
        nt = nxt.view(1, 1)
        if am_c is not None:
            am_c = torch.cat([am_c, am_c.new_ones((1, 1))], 1)
        kwc = dict(use_cache=True, return_dict=True, attention_mask=am_c)
        if is_mrope:
            kwc["position_ids"] = next_c; kwc["cache_position"] = torch.tensor([cur_c], device=device)
        out_c = model(input_ids=nt, past_key_values=out_c.past_key_values, **kwc)
        if am_u is not None:
            am_u = torch.cat([am_u, am_u.new_ones((1, 1))], 1)
        kwu = dict(use_cache=True, return_dict=True, attention_mask=am_u)
        if is_mrope:
            kwu["position_ids"] = next_u; kwu["cache_position"] = torch.tensor([cur_u], device=device)
        out_u = model(input_ids=nt, past_key_values=out_u.past_key_values, **kwu)
        if is_mrope:
            next_c = next_c + 1; next_u = next_u + 1
        cur_c += 1; cur_u += 1
    return torch.stack(generated, 1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)


@torch.no_grad()
def revisit_generate(model, input_ids, *, image_token_id=None, attention_mask=None,
                     max_new_tokens=512, early_exit_layers="all", relative_top=1e-5,
                     eos_token_id=None, **model_inputs):
    """ReVisiT (Cho & Kim, ACL'26; arXiv 2506.09522). Training-free, SINGLE-forward decoding that
    additively fuses the LM head's mature next-token distribution with the JSD-closest projection of the model's OWN
    vision-token hidden states through lm_head. A fixed vision-candidate bank is built once at prefill
    (vision hidden states do not change), then each step: APC plausibility filter, re-softmax over the
    surviving subset, pick the bank row (vision token) with minimum JSD to the mature subset distribution,
    and fuse additively in log space (no alpha/beta). Faithful to /tmp/ReVisiT generation/utils.py.
    Model-agnostic incl. M-RoPE (only the prefill reads image positions; decode is a plain single branch).
    """
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

    final_layer = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") \
        else model.config.text_config.num_hidden_layers
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    start_layer = 0 if not tied else (2 if final_layer > 2 else (1 if final_layer == 2 else 0))
    if early_exit_layers == "all":
        layers = list(range(start_layer, final_layer + 1, 2))
        if final_layer % 2 != 0 and final_layer not in layers:
            layers += [final_layer]
    elif early_exit_layers == "last":
        layers = [final_layer]
    elif isinstance(early_exit_layers, (list, tuple)):
        layers = [i for i in early_exit_layers if i <= final_layer]
    else:
        raise ValueError("early_exit_layers in {'all','last',list}")
    lm_head = model.get_output_embeddings()

    # PREFILL: build the FIXED vision candidate bank from image-position hidden states
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                output_hidden_states=True, return_dict=True, **model_inputs)
    hs = out.hidden_states
    bank = [lm_head(hs[i][:, img_pos, :]) for i in layers]      # each [1, n_img, V]
    bank = torch.stack(bank, dim=1)                             # [1, L_sel, n_img, V]
    bank = bank.view(bank.size(0), -1, bank.size(-1)).float()   # [1, K, V]

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
        plaus[mature_lsm[0].argmax()] = True                   # guard: always keep top-1
        mat_l = mature[:, plaus].log_softmax(-1)               # [1,|P|] re-softmax over subset
        mat_p = mat_l.exp()
        prem_l = bank[:, :, plaus].log_softmax(-1)             # [1,K,|P|]
        prem_p = prem_l.exp()
        avg = 0.5 * (mat_p[:, None, :] + prem_p)               # [1,K,|P|]
        kl1 = F.kl_div(mat_l[:, None, :], avg, reduction="none").mean(-1)   # [1,K]
        kl2 = F.kl_div(prem_l, avg, reduction="none").mean(-1)             # [1,K]
        jsd = (0.5 * (kl1 + kl2)).mean(0)                      # [K]
        idx = int(jsd.argmin())
        fused = torch.full_like(mature, float("-inf"))
        fused[0, plaus] = mat_l[0] + prem_l[0, idx]            # additive log-space fusion, no weights
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
        out = model(input_ids=nt, past_key_values=out.past_key_values, **kw)   # no image inputs
        if is_mrope:
            next3d = next3d + 1
        cur_len += 1
    return torch.stack(generated, 1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)


def _decoder_layers(model):
    """Locate the list of decoder layers across native-HF VLM wrappers (LLaVA/Qwen/InternVL)."""
    for path in ("model.language_model.layers", "language_model.model.layers",
                 "model.model.layers", "model.language_model.model.layers",
                 "language_model.layers", "model.layers"):
        obj = model
        ok = True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False; break
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise RuntimeError("vista: cannot locate decoder layers")


@torch.no_grad()
def vista_generate(model, input_ids, *, image_token_id=None, attention_mask=None, max_new_tokens=512,
                   use_vsv=True, vsv_lambda=0.17, vsv_layers=None,
                   use_sla=True, sla_layers=None, sla_alpha=0.3, sla_apply_norm=True,
                   do_sample=False, temperature=1.0, top_p=1.0, rng=None,
                   eos_token_id=None, **model_inputs):
    """VISTA (ICML'25; arXiv 2502.03628). Single-branch decode with two orthogonal training-free parts:
    VSV (visual steering vector injected into each layer's MLP output via forward hooks; the vector is the
    last-token hidden-state difference between the image and image-removed prompt) and SLA (self-logits
    augmentation: convex-blend the final logits with lm_head applied to a window of EARLY hidden states).
    Model-agnostic incl. M-RoPE (steering is in hidden space; SLA reads hidden_states). sla_layers default
    is set relative to the layer count (last ~6 layers before the final one), matching the repo's (25,30)
    for 32-layer LLaVA and (21,26) for 28-layer Qwen."""
    device = input_ids.device
    if image_token_id is None:
        image_token_id = image_token_id_from_config(model.config)
    eos_ids = _resolve_eos(model, eos_token_id)
    is_mrope = _is_mrope(model); grid = model_inputs.get("image_grid_thw")
    layers = _decoder_layers(model)
    nL = len(layers)
    if sla_layers is None:
        sla_layers = (max(0, nL - 7), nL - 2)          # 32->(25,30), 28->(21,26)
    sla_lo, sla_hi = sla_layers
    head = model.get_output_embeddings()
    # The stack's final norm, needed by _sla. Located by walking the usual language-model paths;
    # None means the faithful (un-normed) reading is used regardless of sla_apply_norm.
    final_norm = None
    for _p in ("language_model.model.norm", "model.language_model.norm", "model.norm", "norm"):
        _o = model
        for _a in _p.split("."):
            _o = getattr(_o, _a, None)
            if _o is None:
                break
        if _o is not None and hasattr(_o, "forward"):
            final_norm = _o
            break

    hooks = []
    if use_vsv:
        span = locate_image_span(input_ids[0], image_token_id)
        if span is None:
            raise ValueError("vista: no image span")
        s, e = span
        out_p = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                      output_hidden_states=True, return_dict=True, **model_inputs)
        keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device); keep[s:e] = False
        ids_u = input_ids[:, keep]
        am_u = attention_mask[:, keep] if attention_mask is not None else None
        out_n = model(input_ids=ids_u, attention_mask=am_u, use_cache=False,
                      output_hidden_states=True, return_dict=True)   # image removed
        hp = torch.stack([h[0, -1, :].float() for h in out_p.hidden_states[1:]])
        hn = torch.stack([h[0, -1, :].float() for h in out_n.hidden_states[1:]])
        vsv = hp - hn                                                # [L, D]
        lo = 0 if vsv_layers is None else vsv_layers[0]
        hi = nL if vsv_layers is None else vsv_layers[1]
        for i in range(lo, min(hi, nL)):
            d = F.normalize(vsv[i], dim=-1)
            def make_hook(d=d):
                def hook(mod, inp, out):
                    x = out[0] if isinstance(out, tuple) else out
                    xf = x.float()
                    orig_norm = xf.norm(p=2, dim=-1, keepdim=True)
                    cos = F.cosine_similarity(xf, -d.view(1, 1, -1), dim=-1).clamp(min=0.0).unsqueeze(-1)
                    y = vsv_lambda * (1.0 + cos) * d.view(1, 1, -1)
                    xs = (F.normalize(F.normalize(xf, p=2, dim=-1) + y, p=2, dim=-1) * orig_norm).to(x.dtype)
                    return (xs,) + out[1:] if isinstance(out, tuple) else xs
                return hook
            hooks.append(layers[i].mlp.register_forward_hook(make_hook()))

    def _sla(last_logits, hidden_states):
        hs = hidden_states[1:]
        # HF returns the intermediate entries BEFORE the stack's final norm and only the last one
        # after it, so lm_head applied to a raw intermediate sees a hidden state on a different
        # scale than the one it was trained to read. The official release ignores this, which is
        # harmless on LLaMA-based towers but fatal on Qwen2-VL, where hidden-state norms are ~2.6x
        # larger: the augmented logits then dominate the 0.3/0.7 blend and the model emits a
        # constant out-of-context token. Passing the final norm first is what reading an
        # intermediate layer requires (and what DoLa does); sla_apply_norm=False restores the
        # literal official behaviour for the faithfulness comparison.
        def _read(h):
            return head(final_norm(h) if (sla_apply_norm and final_norm is not None) else h)
        aug = torch.stack([_read(hs[L][:, -1:, :]).float()
                           for L in range(sla_lo, sla_hi + 1)]).mean(0)[:, -1, :]
        return sla_alpha * aug + (1.0 - sla_alpha) * last_logits

    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                    output_hidden_states=use_sla, return_dict=True, **model_inputs)
        next3d = _rope_next(model, input_ids, grid, attention_mask) if is_mrope else None
        cur_len = input_ids.shape[1]; generated = []; am = attention_mask
        for step_idx in range(max_new_tokens):
            last_logits = out.logits[:, -1, :].float()
            logits = _sla(last_logits, out.hidden_states) if use_sla else last_logits
            if do_sample:
                # Sampling analogue for the cross-regime comparison: draw from the blended
                # distribution instead of taking its argmax. VISTA's release only ever decodes
                # greedily, so this is OUR extension and is labelled as such wherever reported.
                sl = logits / max(float(temperature), 1e-6)
                probs = torch.softmax(sl, dim=-1)
                if top_p and top_p < 1.0:
                    sp, si = probs.sort(dim=-1, descending=True)
                    drop = sp.cumsum(-1) - sp > top_p
                    sp = sp.masked_fill(drop, 0.0)
                    probs = torch.zeros_like(probs).scatter_(-1, si, sp)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
                if eos_ids and int(nxt.item()) in eos_ids:
                    break
            else:
                if eos_ids and int(logits.argmax(-1)) in eos_ids:
                    break
                nxt = logits.argmax(-1)
            generated.append(nxt)
            if step_idx + 1 == max_new_tokens:
                break
            nt = nxt.view(1, 1)
            if am is not None:
                am = torch.cat([am, am.new_ones((1, 1))], 1)
            kw = dict(use_cache=True, output_hidden_states=use_sla, return_dict=True, attention_mask=am)
            if is_mrope:
                kw["position_ids"] = next3d
                kw["cache_position"] = torch.tensor([cur_len], device=device)
            out = model(input_ids=nt, past_key_values=out.past_key_values, **kw)
            if is_mrope:
                next3d = next3d + 1
            cur_len += 1
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(generated, 1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)


# ---------------------------------------------------------------- PAI ----
# PAI (Liu et al., ECCV'24; arXiv 2407.21771), ported from the official repo's attention.py
# and CFG.py. Two components, both applied only to the LAST query row:
#   (a) attention amplification, pre-softmax, on the image-token columns of layers
#       [start_layer, end_layer):   w <- w + |w| * alpha
#   (b) CFG-style logit refinement against a branch whose image attention is switched off:
#       out = gamma * (cond - uncond) + uncond, restricted to cond >= log(0.1) + max(cond),
#       with both sides in log-softmax space.
# Official defaults: alpha=0.2, gamma=1.1 (POPE) / 2.0 (CHAIR), start_layer=2, end_layer=32.
#
# The port targets backbones whose attention module dispatches through transformers'
# `attention_interface` (LLaMA- and Qwen2-based language towers, i.e. LLaVA-1.5/NeXT/OneVision
# and InternVL3). Qwen2-VL/2.5-VL still carry the pre-refactor inline-eager class and are
# handled separately; see pai_supported().
_PAI_PATCHED = set()


def _attn_module_of(layer):
    for name in ("self_attn", "attn", "attention"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise RuntimeError("pai: cannot locate attention submodule on decoder layer")


def _install_pai_eager(mod):
    """Wrap the modeling module's `eager_attention_forward` so PAI flags carried on the
    attention module take effect. Idempotent per module."""
    key = mod.__name__
    if key in _PAI_PATCHED:
        return True
    base = getattr(mod, "eager_attention_forward", None)
    if base is None:
        return False

    def pai_eager(module, query, key_, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if not getattr(module, "pai_use_attn", False):
            return base(module, query, key_, value, attention_mask,
                        scaling=scaling, dropout=dropout, **kw)
        n_rep = query.shape[1] // key_.shape[1]
        k = key_.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key_
        v = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value
        w = torch.matmul(query, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, :, : k.shape[-2]]
        s, e = module.pai_img_span
        # official: last query row only, additive |w|*alpha on the image columns
        w[:, :, -1, s:e] = w[:, :, -1, s:e] + w[:, :, -1, s:e].abs() * module.pai_alpha
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        w = torch.nn.functional.dropout(w, p=dropout, training=module.training)
        out = torch.matmul(w, v).transpose(1, 2).contiguous()
        return out, w

    mod.eager_attention_forward = pai_eager
    _PAI_PATCHED.add(key)
    return True


_PAI_LEGACY = ("Qwen2VLAttention", "Qwen2_5_VLAttention")


def pai_supported(model):
    """True when every decoder layer's attention can carry the amplification.

    Two shapes qualify: the refactored classes that dispatch through `eager_attention_forward`
    (LLaMA- and Qwen2-based towers), and the pre-refactor inline-eager Qwen2-VL classes, which get
    an explicit replacement forward instead.
    """
    import sys
    try:
        layers = _decoder_layers(model)
    except RuntimeError:
        return False
    for layer in layers:
        cls = type(_attn_module_of(layer))
        if cls.__name__ in _PAI_LEGACY:
            continue
        mod = sys.modules.get(cls.__module__)
        if mod is None or not hasattr(mod, "eager_attention_forward"):
            return False
    return True


def _pai_set(model, *, on, alpha=0.2, span=None, start_layer=2, end_layer=32):
    import sys, types
    layers = _decoder_layers(model)
    for i, layer in enumerate(layers):
        att = _attn_module_of(layer)
        if on and start_layer <= i < end_layer:
            if type(att).__name__ in _PAI_LEGACY:
                if not getattr(att, "_pai_legacy_installed", False):
                    att.forward = types.MethodType(_pai_legacy_forward, att)
                    att._pai_legacy_installed = True
            else:
                mod = sys.modules[type(att).__module__]
                if not _install_pai_eager(mod):
                    raise RuntimeError(f"pai: {mod.__name__} has no eager_attention_forward to wrap")
            att.pai_use_attn = True
            att.pai_alpha = float(alpha)
            att.pai_img_span = span
        else:
            att.pai_use_attn = False


@torch.no_grad()
def pai_generate(model, input_ids, *, image_token_id=None, attention_mask=None, max_new_tokens=512,
                 alpha=0.2, gamma=1.1, start_layer=2, end_layer=32, cfg_cutoff=0.1,
                 eos_token_id=None, **model_inputs):
    """PAI: image-attention amplification plus CFG refinement against an attention-off branch.

    The uncond branch is the prompt with the image placeholder span DELETED, matching the official
    init_cfg_processor, which rebuilds the negative prompt as bos + text_before + text_after rather
    than masking the image tokens in place. Amplification is off on that branch (official use_cfg
    gates the attention edit off). Stopping follows the conditional branch, matching the EOS
    convention used by the other baselines here.
    """
    if image_token_id is None:
        image_token_id = image_token_id_from_config(model)
    span = locate_image_span(input_ids, image_token_id)
    if span is None:
        raise RuntimeError("pai: no image tokens found in input_ids")
    if not pai_supported(model):
        raise RuntimeError("pai: backbone uses the pre-refactor inline-eager attention class")

    eos = _resolve_eos(model, eos_token_id)
    device = input_ids.device
    n_layers = len(_decoder_layers(model))
    end_layer = min(int(end_layer), n_layers)

    generated = []
    cur_ids, cur_mask = input_ids, attention_mask
    past_c = past_u = None
    extra = dict(model_inputs)

    for step in range(max_new_tokens):
        # conditional branch: amplification ON
        _pai_set(model, on=True, alpha=alpha, span=span, start_layer=start_layer, end_layer=end_layer)
        oc = model(input_ids=cur_ids, attention_mask=cur_mask, past_key_values=past_c,
                   use_cache=True, **(extra if step == 0 else {}))
        past_c = oc.past_key_values
        cond = F.log_softmax(oc.logits[:, -1, :].float(), dim=-1)
        _pai_set(model, on=False)

        if gamma == 1.0:
            scores = cond
        else:
            # uncond branch: prompt with the image placeholder span deleted (no pixel inputs)
            if step == 0:
                u_ids = torch.cat([cur_ids[:, :span[0]], cur_ids[:, span[1]:]], dim=1)
                u_mask = torch.ones_like(u_ids)
                ou = model(input_ids=u_ids, attention_mask=u_mask,
                           past_key_values=None, use_cache=True)
            else:
                u_mask = torch.cat([u_mask, torch.ones((1, 1), dtype=u_mask.dtype, device=device)], 1)
                ou = model(input_ids=cur_ids, attention_mask=u_mask, past_key_values=past_u, use_cache=True)
            past_u = ou.past_key_values
            uncond = F.log_softmax(ou.logits[:, -1, :].float(), dim=-1)
            cutoff = math.log(cfg_cutoff) + cond.max(dim=-1, keepdim=True).values
            scores = gamma * (cond - uncond) + uncond
            scores = scores.masked_fill(cond < cutoff, -float("inf"))

        nxt = scores.argmax(dim=-1)
        generated.append(nxt)
        if eos and int(nxt.item()) in eos:
            break
        cur_ids = nxt.view(1, 1)
        cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=device)], 1) \
                   if cur_mask is not None else None

    _pai_set(model, on=False)
    return torch.stack(generated, 1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)


def _pai_legacy_forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        cache_position=None, position_embeddings=None):
    """Qwen2-VL / Qwen2.5-VL attention with PAI's amplification inserted before the softmax.

    Those two backbones still carry the pre-refactor inline-eager attention class, so they never
    reach `eager_attention_forward` and the registry wrapper cannot see them. This mirrors the
    installed Qwen2VLAttention.forward line for line -- multimodal rotary embedding, cache update,
    kv repetition, the float16 inf guard, fp32 softmax -- and adds only the four lines PAI adds to
    its own llama_new_forward.
    """
    import math
    from torch import nn
    mod = sys.modules[type(self).__module__]
    apply_mrope = mod.apply_multimodal_rotary_pos_emb
    repeat_kv = mod.repeat_kv

    bsz, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_mrope(q, k, cos, sin, self.rope_scaling["mrope_section"])

    if past_key_value is not None:
        k, v = past_key_value.update(k, v, self.layer_idx,
                                     {"sin": sin, "cos": cos, "cache_position": cache_position})
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

    w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : k.shape[-2]]
    if q.dtype == torch.float16:
        w = torch.where(torch.isinf(w), torch.zeros_like(w), w)

    if getattr(self, "pai_use_attn", False):
        s, e = self.pai_img_span
        w[:, :, -1, s:e] = w[:, :, -1, s:e] + w[:, :, -1, s:e].abs() * self.pai_alpha

    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    w = nn.functional.dropout(w, p=self.attention_dropout, training=self.training)
    out = torch.matmul(w, v).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    out = self.o_proj(out)
    return out, (w if output_attentions else None), past_key_value
