"""Unified native-HF VLM registry and loader for 7B/8B and 13B-32B backbones."""
from __future__ import annotations
import os
import torch
from .paths import CKPT_DIR

REGISTRY = {
    "llava_15_hf":   dict(repo="llava-hf/llava-1.5-7b-hf",                  cls="LlavaForConditionalGeneration"),
    "llava_next":    dict(repo="llava-hf/llava-v1.6-vicuna-7b-hf",          cls="LlavaNextForConditionalGeneration"),
    "llava_ov":      dict(repo="llava-hf/llava-onevision-qwen2-7b-ov-hf",   cls="LlavaOnevisionForConditionalGeneration"),
    "qwen2_vl":      dict(repo="Qwen/Qwen2-VL-7B-Instruct",                 cls="Qwen2VLForConditionalGeneration"),
    "qwen2_5_vl":    dict(repo="Qwen/Qwen2.5-VL-7B-Instruct",               cls="Qwen2_5_VLForConditionalGeneration"),
    "qwen3_vl":      dict(repo="Qwen/Qwen3-VL-8B-Instruct",                 cls="AutoModelForVision2Seq"),
    "internvl3":     dict(repo="OpenGVLab/InternVL3-8B-hf",                 cls="AutoModelForCausalLM"),
    "instructblip":  dict(repo="Salesforce/instructblip-vicuna-7b",         cls="InstructBlipForConditionalGeneration"),
    # Scale transfer backbones (13B-32B)
    "llava_15_13b":  dict(repo="llava-hf/llava-1.5-13b-hf",                 cls="LlavaForConditionalGeneration"),
    "instructblip_13b": dict(repo="Salesforce/instructblip-vicuna-13b",     cls="InstructBlipForConditionalGeneration"),
    "internvl3_14b": dict(repo="OpenGVLab/InternVL3-14B-hf",                cls="AutoModelForCausalLM"),
    "qwen2_5_vl_32b": dict(repo="Qwen/Qwen2.5-VL-32B-Instruct",             cls="Qwen2_5_VLForConditionalGeneration"),
}

def resolve_model_path(name: str) -> str:
    cfg = REGISTRY[name]
    local_dir = os.path.join(CKPT_DIR, os.path.basename(cfg["repo"]))
    return local_dir if os.path.exists(local_dir) else cfg["repo"]

def load_vlm(name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, eager: bool = True):
    import transformers
    from transformers import AutoProcessor
    cfg = REGISTRY[name]
    path = resolve_model_path(name)
    ModelCls = getattr(transformers, cfg["cls"])
    proc = AutoProcessor.from_pretrained(path, use_fast=False)
    kw = dict(torch_dtype=dtype)
    if eager:
        kw["attn_implementation"] = "eager"
    model = ModelCls.from_pretrained(path, **kw).to(device).eval()
    return model, proc
