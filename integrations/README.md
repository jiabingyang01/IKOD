# Wiring IKOD into an LVLM Backbone

**English** · [简体中文](README_zh.md)

IKOD and IKOD-A operate directly on the native HuggingFace `DynamicCache` during autoregressive generation. They do not modify model weights, require no external detectors or auxiliary VLMs, and preserve intact visual tokens throughout decoding.

To apply IKOD to any vision-language backbone, only three interface elements are needed:
1. **Visual token span `[u, u+U)`**: Located automatically per forward pass via `locate_image_span(input_ids, image_token_id)` in `ikod/image_span.py`.
2. **Post-visual mergeable region `[u+U, u+U+P)`**: All text KV entries after the visual span except the protected most-recent query token (`P = t - u - U - 1`).
3. **Collaborative fusion weight**: Either a fixed family-level scalar `alpha` (**IKOD**) or the tuning-free online attention-decay schedule `alpha_t` (**IKOD-A**, `--adaptive p1`).

```python
from ikod import ikod_generate, load_vlm

model, processor = load_vlm("llava_15_hf")
output_ids = ikod_generate(
    model,
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    ratio=0.4,          # lambda: retained post-visual KV ratio
    alpha=4.0,          # fixed collaborative weight (or set adaptive="p1" for IKOD-A)
    beta=0.1,           # plausibility threshold
    max_new_tokens=512,
)
```

---

## Environment Switches

| Switch | Default | Description |
|---|---|---|
| `IKOD_FAST_MERGE` | `1` | Vectorized batched bucket-averaging + visual-prefix KV reuse + plausibility-singleton short-circuit |
| `IKOD_CKPT_DIR` | `./checkpoints` | Local checkpoint directory before falling back to HuggingFace Hub |
| `IKOD_DATA_DIR` | `./data` | Local benchmark root directory (COCO, POPE, AMBER) |
| `IKOD_RESULTS_DIR` | `./results` | Output directory for JSONL generations and evaluation reports |

---

## Hyperparameters Used in the Paper

Across all benchmarks and all 12 backbones (8 primary 7B/8B models + 4 scaled 13B–32B models), plausibility threshold is fixed at $\beta = 0.1$, and **IKOD-A** uses a single universal schedule ($\alpha_{\min} = 0, \alpha_{\max} = 1, \kappa = 10, w = 3$) with zero per-backbone alpha tuning:

| Backbone Family | Models Covered | Retained Ratio λ | IKOD Fixed α (Captioning / Probing) | IKOD-A Schedule |
|---|---|:---:|:---:|:---:|
| **LLaVA Family** | LLaVA-1.5-7B/13B, LLaVA-NeXT-7B, LLaVA-OneVision-7B | `0.4` | `4.0` / `1.0` | Universal ($\alpha_{\min}{=}0, \alpha_{\max}{=}1, \kappa{=}10$) |
| **Qwen-VL Family** | Qwen2-VL-7B, Qwen2.5-VL-7B/32B, Qwen3-VL-8B | `0.6` | `4.0` / `0.5` | Universal ($\alpha_{\min}{=}0, \alpha_{\max}{=}1, \kappa{=}10$) |
| **InternVL Family** | InternVL3-8B, InternVL3-14B | `0.4` | `0.1` / `1.0` | Universal ($\alpha_{\min}{=}0, \alpha_{\max}{=}1, \kappa{=}10$) |
| **InstructBLIP Family** | InstructBLIP-7B, InstructBLIP-13B | `0.4` | `0.5` / `0.5` | Universal ($\alpha_{\min}{=}0, \alpha_{\max}{=}1, \kappa{=}10$) |
