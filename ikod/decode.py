"""ikod_generate: model-agnostic IKOD collaborative decoding for native-HF VLMs.

Greedy collaborative decoding (batch=1), faithful to the legacy
cache_generate.greedy_search but written against the modern transformers Cache API:

  per step:
    logits_full   = full incremental forward (output_attentions=True)         # normal cache
    merged_cache  = image-attention-guided KV merge of the full cache         # shorter, image-focused
    logits_merged = replay(last_token, past=merged_cache)                     # positive image branch
    alpha_t       = fixed alpha OR AdaptiveAlpha.step(...)  (core novelty)
    logits        = logits_full + alpha_t * logits_merged                     # ADDITIVE (not subtractive)
    cutoff        = log(beta) + max(logits_full)                              # VCD-style plausibility
    next_token    = argmax( logits.masked_fill(logits_full < cutoff, -inf) )

Plain-RoPE backbones (LLaVA-1.5/NeXT/OneVision, InternVL3) expose the merged cache as a
contiguous sequence, with the query at its compact length Lp. M-RoPE backbones
(Qwen2-VL/2.5-VL) receive the model-produced three-dimensional query position explicitly.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .image_span import locate_image_span, image_token_id_from_config
from .kv_merge import update_score_table, merge_kv_cache
from .adaptive import AdaptiveAlpha, compute_image_attention, jsd


def _cache_to_list(past):
    """Return list over layers of (k, v). Supports legacy tuple cache and HF Cache objects."""
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return [(past.key_cache[i], past.value_cache[i]) for i in range(len(past.key_cache))]
    return [(k, v) for (k, v) in past]


def _list_to_dynamic_cache(kv_list):
    """Build a DynamicCache from a list of (k, v) per layer (modern transformers)."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv_list):
        cache.update(k, v, i)
    return cache


class _AppendOnceCache(DynamicCache):
    """DynamicCache-compatible read-once cache with one preallocated append slot.

    IKOD discards the auxiliary cache immediately after replaying one query.  A
    regular ``DynamicCache`` nevertheless concatenates that query onto every
    compact layer, recopying the unchanged (and sometimes very long) visual
    prefix.  This adapter writes the query into the slot reserved by
    :func:`merge_kv_cache` and returns a logical view.  It deliberately exposes
    the same minimal interface used by the Transformers 4.52 decoder stack.
    """

    is_compileable = False

    def __init__(self, items, base_len):
        super().__init__()
        self.key_cache = [kv[0] for kv in items]
        self.value_cache = [kv[1] for kv in items]
        self._base_len = int(base_len)
        self._seen_tokens = self._base_len

    def get_seq_length(self, layer_idx=0):
        # The preallocated final slot is capacity, not past context.
        return self._base_len

    def get_max_cache_shape(self):
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        q = int(key_states.shape[-2])
        end = self._base_len + q
        k_buf = self.key_cache[layer_idx]
        v_buf = self.value_cache[layer_idx]
        if end > k_buf.shape[-2]:
            raise RuntimeError(
                f"append-once cache capacity {k_buf.shape[-2]} < requested {end}"
            )
        k_buf[..., self._base_len:end, :].copy_(key_states)
        v_buf[..., self._base_len:end, :].copy_(value_states)
        if layer_idx == 0:
            self._seen_tokens = end
        return k_buf[..., :end, :], v_buf[..., :end, :]


def _list_to_append_once_cache(kv_list, logical_len):
    return _AppendOnceCache(kv_list, logical_len)


@torch.no_grad()
def ikod_generate(model, input_ids, *, image_token_id=None, attention_mask=None,
                  max_new_tokens=512, ratio=0.4, beta=0.1, alpha=2.0,
                  full_weight=1.0,
                  adaptive: AdaptiveAlpha | None = None, protect_recent=1,
                  anchor_strategy="low",
                  merge_mode="mean",
                  merge_region="post_visual",
                  plain_query_position="compact",
                  replay_protected_anchor=True,
                  do_sample=False, temperature=1.0, top_p=1.0, top_k=0, rng=None,
                  eos_token_id=None, score_layers=None, return_meta=False,
                  **model_inputs):
    """Greedy IKOD decode. `model_inputs` carries pixel_values / image_grid_thw / etc.
    Returns generated token ids [1, T_new] (and optional per-step meta if return_meta).
    """
    device = input_ids.device
    if plain_query_position not in ("compact", "original"):
        raise ValueError(f"unknown plain_query_position={plain_query_position!r}")
    if merge_region not in ("post_visual", "generated", "prompt_suffix"):
        raise ValueError(f"unknown merge_region={merge_region!r}")
    if image_token_id is None:
        image_token_id = image_token_id_from_config(model.config)
    span = locate_image_span(input_ids[0], image_token_id)
    if span is None:
        raise ValueError("no image tokens found in input_ids; IKOD needs an image span")
    image_start, image_end = span
    if eos_token_id is None:
        gc = getattr(model, "generation_config", None)
        eos_token_id = getattr(gc, "eos_token_id", None) if gc is not None else None
        if eos_token_id is None:
            eos_token_id = getattr(model.config, "eos_token_id", None)
    eos_ids = ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id or []))
    if adaptive is not None:
        adaptive.reset()
    # α=0 short-circuits to the plain base forward — but only under GREEDY selection.
    # Under sampling, α=0 is the "base+plausibility, sampled" arm of the Mirage control and must
    # still run the full selection path, so exclude do_sample from the short-circuit.
    greedy_control = (adaptive is None and float(alpha) == 0.0
                      and float(full_weight) == 1.0 and not do_sample)

    # M-RoPE (Qwen2-VL/2.5-VL) needs explicit 3D position_ids for both forwards.
    is_mrope = type(model).__name__ in ("Qwen2VLForConditionalGeneration",
                                        "Qwen2_5_VLForConditionalGeneration")
    grid = model_inputs.get("image_grid_thw")
    next3d = None

    text_cfg = getattr(model.config, "text_config", None)
    llm_cfg = getattr(model.config, "llm_config", None)
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
        num_layers = text_cfg.num_hidden_layers
    elif llm_cfg is not None and hasattr(llm_cfg, "num_hidden_layers"):
        num_layers = llm_cfg.num_hidden_layers
    else:
        num_layers = len(_cache_to_list(_first_cache(model, input_ids, attention_mask, model_inputs)))
    max_len = input_ids.shape[1] + max_new_tokens + 4
    score = None if greedy_control else torch.zeros(num_layers, max_len, device=device)

    # ---- prefill ----
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                output_attentions=not greedy_control, return_dict=True, **model_inputs)
    cur_len = out.past_key_values.get_seq_length() if hasattr(out.past_key_values, "get_seq_length") \
        else _cache_to_list(out.past_key_values)[0][0].shape[-2]
    prompt_len = int(cur_len)
    if not greedy_control:
        update_score_table(score, out.attentions, image_start, image_end, cur_len)
    if is_mrope:
        pos3d, _ = model.model.get_rope_index(input_ids, image_grid_thw=grid,
                                              attention_mask=attention_mask)
        next3d = pos3d[..., -1:] + 1                       # [3,1,1] position of next token
    generated, meta = [], []

    for step_idx in range(max_new_tokens):
        logits_full = out.logits[:, -1, :].float()                       # [1, V]
        # ---- merged (image-focused) branch ----
        # alpha=0 is exactly standard greedy decoding under the same stopping and
        # plausibility path; avoid an algebraically dead second forward in audit runs.
        if greedy_control:
            Lp = cur_len
            logits_merged = None
        else:
            kv_list = _cache_to_list(out.past_key_values)
            if merge_region == "post_visual":
                merge_start = image_end
                merge_protect = protect_recent
            elif merge_region == "generated":
                # Keep the complete multimodal prompt untouched and transform
                # only generated states preceding the latest processed token.
                merge_start = prompt_len
                merge_protect = protect_recent
            else:
                # Transform only the instruction suffix following the visual
                # block. Keep the last prompt state and every generated state.
                merge_start = image_end
                merge_protect = cur_len - (prompt_len - 1)
            merged_kv, kept = merge_kv_cache(
                kv_list, score, merge_start, cur_len, ratio, merge_protect,
                anchor_strategy=anchor_strategy,
                merge_mode=merge_mode,
                append_slots=1,
            )
            Lp = int(kept[0].numel())
            has_append_slot = merged_kv[0][0].shape[-2] > Lp
            if not replay_protected_anchor:
                # Diagnostic conventional-incremental variant: remove the latest
                # protected KV before replaying its token as the current query.
                # The published/default decoder intentionally keeps that KV as a
                # full-context anchor; this switch is only for mechanism audit.
                Lp -= 1
                if not has_append_slot:
                    merged_kv = [[k[..., :Lp, :], v[..., :Lp, :]] for k, v in merged_kv]
            merged_cache = (_list_to_append_once_cache(merged_kv, Lp)
                            if has_append_slot else _list_to_dynamic_cache(merged_kv))
            # In the default method, query replay is intentional: the latest
            # full-context KV remains as a protected anchor and its token is
            # replayed at the compact position.
            last_tok = input_ids[:, -1:] if not generated else generated[-1].view(1, 1)
            cache_pos = torch.tensor([Lp], device=device)
            if is_mrope:
                pos = next3d - 1
            else:
                query_pos = Lp if plain_query_position == "compact" else cur_len - 1
                pos = torch.tensor([[query_pos]], device=device)
            out_m = model(input_ids=last_tok, past_key_values=merged_cache,
                          position_ids=pos, cache_position=cache_pos,
                          use_cache=True, output_attentions=False, return_dict=True)
            logits_merged = out_m.logits[:, -1, :].float()               # [1, V]
            # Only the merged logits survive this step.  Keeping ``out_m`` and
            # the temporary cache alive through the next full-branch forward
            # can otherwise hold the old full cache, merged cache, and updated
            # full cache at once on high-resolution OneVision examples.
            del out_m, merged_cache, merged_kv, kv_list, kept

        # ---- adaptive alpha (core novelty) ----
        if greedy_control:
            a_t = 0.0
        elif adaptive is None:
            a_t = alpha
        else:
            S_t = float(compute_image_attention(out.attentions, image_start, image_end, score_layers)[0])
            # P1, the published shared schedule, depends only on S_t.  Avoid two
            # full-vocabulary softmax operations that were previously computed
            # and then ignored at every step.  The other ablation modes receive
            # exactly the distributions/signals they use.
            p_full = p_merged = None
            c_t = None
            if adaptive.mode in ("p2", "p1p2"):
                p_full = F.softmax(logits_full, -1)
                p_merged = F.softmax(logits_merged, -1)
            elif adaptive.mode == "p1p3":
                p_full = F.softmax(logits_full, -1)
                c_t = float(p_full.max())
            a_t = adaptive.step(S_t=S_t, p_full=p_full, p_merged=p_merged, c_t=c_t)

        # ---- stopping decided by the ORIGINAL branch (prevents over-generation/degeneration:
        #      the image-focused merge refines WHICH content token, not WHETHER to stop) ----
        if eos_ids and int(logits_full.argmax(dim=-1)) in eos_ids:
            break
        # ---- combine (additive) + plausibility constraint; IKOD never emits EOS itself ----
        diffs = logits_full if greedy_control else float(full_weight) * logits_full + a_t * logits_merged
        if eos_ids:
            diffs[:, eos_ids] = float("-inf")
        cutoff = math.log(beta) + logits_full.max(dim=-1, keepdim=True).values
        scores = diffs.masked_fill(logits_full < cutoff, float("-inf"))
        if do_sample:
            # Sampling base: draw from the combined+plausibility distribution.
            # Same additive/plausibility path as greedy — only the final selection differs, so the
            # Mirage sampling-base three-arm control is exactly this (IKOD) vs full_weight-only.
            s = scores / max(float(temperature), 1e-6)
            if top_k and top_k > 0:
                kth = torch.topk(s, min(int(top_k), s.shape[-1]), dim=-1).values[:, -1, None]
                s = s.masked_fill(s < kth, float("-inf"))
            probs = torch.softmax(s, dim=-1)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sp, dim=-1)
                drop = (cum - sp) > top_p
                sp = sp.masked_fill(drop, 0.0)
                probs = torch.zeros_like(probs).scatter_(-1, si, sp)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_tok = torch.multinomial(probs, 1, generator=rng).squeeze(-1)   # [1]
        else:
            next_tok = scores.argmax(dim=-1)                                 # [1]
        generated.append(next_tok)
        if return_meta:
            meta.append({"alpha": a_t, "Lp": Lp, "cur_len": cur_len,
                         "merge_start": merge_start if not greedy_control else None,
                         "merge_protect": merge_protect if not greedy_control else None})

        # The selected token is already the final requested output.  Advancing
        # the full cache here would compute logits for a token that the caller
        # will never request; this formerly inflated the one-token protocol by
        # one complete incremental forward and added one dead step per caption.
        if step_idx + 1 == max_new_tokens:
            break

        # ---- advance full branch ----
        input_ids = torch.cat([input_ids, next_tok.view(1, 1)], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
        past_full = out.past_key_values
        del out
        if is_mrope:
            out = model(input_ids=next_tok.view(1, 1), past_key_values=past_full,
                        attention_mask=attention_mask, position_ids=next3d,
                        cache_position=torch.tensor([cur_len], device=device),
                        use_cache=True, output_attentions=not greedy_control, return_dict=True)
            next3d = next3d + 1
        else:
            out = model(input_ids=next_tok.view(1, 1), past_key_values=past_full,
                        attention_mask=attention_mask, use_cache=True,
                        output_attentions=not greedy_control, return_dict=True)
        del past_full
        cur_len += 1
        if not greedy_control:
            update_score_table(score, out.attentions, image_start, image_end, cur_len)

    gen = torch.stack(generated, dim=1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)
    return (gen, meta) if return_meta else gen


def _first_cache(model, input_ids, attention_mask, model_inputs):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                output_attentions=False, return_dict=True, **model_inputs)
    return out.past_key_values
