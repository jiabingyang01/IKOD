"""IKOD: Image-Attention-Guided Key-Value Merging against Object Hallucination in Decoding.

Core modules:
  decode     : ikod_generate() collaborative decoding with lossless acceleration.
  kv_merge   : Image-attention-guided post-visual KV-cache merging (standard & fast vectorized).
  adaptive   : Online visual-attention-decay gate (IKOD-A) and JSD utilities.
  image_span : Contiguous visual-token span locator across LVLM families.
  models     : Unified native-HF model & processor loader across 8 LVLM backbones (7B-32B).
  baselines  : Training-free decoding baselines (VCD, M3ID, OPERA, ICD, SID, AGLA, ReVisiT).
"""
from .adaptive import AdaptiveAlpha, compute_image_attention, jsd
from .image_span import locate_image_span, IMAGE_TOKEN_IDS
from .kv_merge import update_score_table, merge_kv_cache
from .decode import ikod_generate

__all__ = [
    "ikod_generate",
    "AdaptiveAlpha",
    "compute_image_attention",
    "jsd",
    "update_score_table",
    "merge_kv_cache",
    "locate_image_span",
    "IMAGE_TOKEN_IDS",
]
