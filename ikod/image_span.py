"""Locate the contiguous image-token span in input_ids, per model family.

For native-HF VLMs the image placeholder is expanded inline into N identical
image-token ids before the LLM. We never hardcode N: we find the span from the
placeholder id mask each forward (dynamic for Qwen/InternVL/anyres).
"""
from __future__ import annotations
import torch

# Placeholder token id used to mark image tokens in input_ids, per backbone family.
# Verify against model.config at runtime (config attr names differ); these are defaults.
IMAGE_TOKEN_IDS = {
    "llava_1_5_hf":      32000,    # llava-hf/llava-1.5-7b-hf  (<image>)
    "llava_next_hf":     32000,    # llava-hf/llava-v1.6-vicuna-7b-hf
    "llava_ov_hf":       151646,   # llava-hf/llava-onevision-qwen2-7b-ov-hf (<image>)
    "qwen2_vl":          151655,   # Qwen/Qwen2-VL-7B-Instruct  (<|image_pad|>)
    "qwen2_5_vl":        151655,   # Qwen/Qwen2.5-VL-7B-Instruct
    "internvl3_hf":      151667,   # OpenGVLab/InternVL3-8B-hf  (<IMG_CONTEXT>)
    # legacy LLaVA repo uses IMAGE_TOKEN_INDEX = -200 *before* expansion to 576 tokens
    "llava_legacy":      -200,
}


def image_token_id_from_config(config) -> int | None:
    """Best-effort: read the image placeholder id from a HF config."""
    for attr in ("image_token_id", "image_token_index", "image_context_token_id"):
        v = getattr(config, attr, None)
        if isinstance(v, int):
            return v
    return None


def locate_image_span(input_ids: torch.Tensor, image_token_id: int):
    """Return (start, end) indices [start, end) of the (largest) contiguous run of
    image_token_id in a 1-D input_ids tensor. Returns None if no image tokens.

    Handles the common single-contiguous-block case (LLaVA/Qwen/InternVL). If the
    placeholder appears in multiple blocks (rare), returns the span covering the
    longest contiguous run and logs nothing (caller may union if needed).
    """
    if input_ids.dim() > 1:
        input_ids = input_ids.view(-1)
    mask = (input_ids == image_token_id)
    if not bool(mask.any()):
        return None
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    # find contiguous runs
    runs = []
    s = prev = int(idx[0])
    for i in idx[1:].tolist():
        if i == prev + 1:
            prev = i
        else:
            runs.append((s, prev + 1))
            s = prev = i
    runs.append((s, prev + 1))
    # return the longest run
    start, end = max(runs, key=lambda r: r[1] - r[0])
    return start, end


def image_span_union(input_ids: torch.Tensor, image_token_id: int):
    """Return (min_index, max_index+1) covering ALL image tokens (use if multi-block)."""
    if input_ids.dim() > 1:
        input_ids = input_ids.view(-1)
    idx = torch.nonzero(input_ids == image_token_id, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return None
    return int(idx.min()), int(idx.max()) + 1


if __name__ == "__main__":
    # quick self-test
    ii = torch.tensor([1, 2, 32000, 32000, 32000, 5, 6, 7])
    assert locate_image_span(ii, 32000) == (2, 5), locate_image_span(ii, 32000)
    ii2 = torch.tensor([32000, 32000, 9, 9, 32000, 32000, 32000, 9])  # multi-block
    assert locate_image_span(ii2, 32000) == (4, 7)
    assert image_span_union(ii2, 32000) == (0, 7)
    assert locate_image_span(torch.tensor([1, 2, 3]), 32000) is None
    print("image_span self-test OK")
