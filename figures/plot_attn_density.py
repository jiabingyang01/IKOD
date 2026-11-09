"""Density figures (fig:density, fig:sdensall, fig:shaldens), v2 (09-30): the main-table protocol.

plot_attn_density.py (imported, not modified) draws from results/attn/*.json with the 300-image / 64-token audit dumps
overriding the older ones.  v2 hands it only the CHAIR-500 / 512-token dumps (results/attn/{model}_chair500.json:
the 500 CHAIR images, 512 new tokens), by filtering the glob inside that module; the KDE, colour maps, panels and
file names are unchanged.

    python code/plot_attn_density_v2.py
"""
import glob as _glob
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "./workspace/code")   # a sandbox copy runs from another directory
import plot_attn_density as base  # noqa: E402

# Output directory; check_figures_fresh_v2 rewrites this line to redirect a sandbox run.
DST = "./paper/figures"
base.DST = DST


def _chair500(pattern: str) -> list[str]:
    out = sorted(f for f in _glob.glob(pattern) if f.endswith("_chair500.json"))
    for f in out:
        d = json.load(open(f))
        if d.get("n_images") != 500 or d.get("max_new_tokens") != 512:
            raise SystemExit(f"{f}: not the CHAIR-500 / 512-token protocol")
    return out


base.glob = types.SimpleNamespace(glob=_chair500)

if __name__ == "__main__":
    ALL = ["llava_15_hf", "llava_next", "llava_ov", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "internvl3", "instructblip_vicuna7b"]
    REP = ["llava_ov", "qwen2_5_vl", "internvl3"]
    have = {json.load(open(f))["model"] for f in _chair500(f"{base.ROOT}/results/attn/*.json")}
    if set(ALL) - have:
        raise SystemExit(f"missing CHAIR-500 dumps: {sorted(set(ALL) - have)}")
    base.make(REP, "all_tokens", "viridis", "attn_density_main", ncols=3)
    base.make(ALL, "all_tokens", "viridis", "attn_density_all", ncols=3)
    base.make(REP, "hal_tokens", "Blues", "hal_density_main", ncols=3)
    base.make(ALL, "hal_tokens", "Blues", "hal_density_all", ncols=3)
    base.make_combined(REP, "density_combined_main")
    base.make_combined(ALL, "density_combined_all")
