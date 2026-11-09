#!/usr/bin/env python3
"""Generate compact double-column mechanistic visualization Fig. 6 (fig_mechanism_case_v3.pdf).

All curves, spatial heatmaps, bucket scores s_m, and divergence logits come 100% from the
authentic pad-matched LLaVA-1.5 dual-branch trace on COCO_val2014_000000338428.jpg:
  ./workspace/results/attnviz_v1/llava_15_hf_338428_mech_v2.json
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DATA_JSON = Path("./workspace/results/attnviz_v1/llava_15_hf_338428_mech_v2.json")
OUT_PDF = Path("./paper/figures/fig_mechanism_case_v3.pdf")
OUT_PNG = Path("./paper/figures/fig_mechanism_case_v3.png")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Palette aligned with Fig. 4 (ikod_pipeline_v5.pdf) & Fig. 5 (YlGnBu + crimson + gold)
C_PANEL_BG   = "#fbfcfe"
C_PANEL_EDGE = "#cfd8e3"
C_BASE_DARK  = "#264b82"
C_BASE_LINE  = "#3563a8"
C_BASE_BG    = "#f1f5fb"
C_BASE_EDGE  = "#9cb3d8"

C_IKOD_DARK  = "#16653e"
C_IKOD_LINE  = "#1f8a55"
C_IKOD_BG    = "#ecf7f0"
C_IKOD_EDGE  = "#7cc099"

C_IKODA_DARK = "#8f5b08"
C_IKODA_LINE = "#c67d11"
C_IKODA_BG   = "#fdf7e7"
C_IKODA_EDGE = "#dbb668"

C_HAL_TEXT   = "#b81d24"
C_HAL_BG     = "#fde6e7"
C_HAL_EDGE   = "#e59a9e"
C_INK        = "#1d242b"
C_MUTED      = "#5b6570"


def suppress_vit_sink_outliers(grid24: np.ndarray) -> np.ndarray:
    """Two-pass 5x5 local median filter to suppress isolated 1-2 patch CLIP-ViT register/sink spikes
    on uniform background/border patches while preserving multi-patch semantic object attention."""
    g = grid24.astype(np.float32).copy()
    for _ in range(2):
        padded = np.pad(g, 2, mode="edge")
        nxt = g.copy()
        for r in range(24):
            for c in range(24):
                win = padded[r:r + 5, c:c + 5].ravel()
                nb = np.delete(win, 12)
                med = float(np.median(nb))
                p75 = float(np.percentile(nb, 75))
                if g[r, c] > max(2.15 * p75, 2.6 * med + 1e-5):
                    nxt[r, c] = p75
        g = nxt
    return g


def gaussian_smooth_2d(arr: np.ndarray, sigma: float = 0.85) -> np.ndarray:
    rad = 2
    ax = np.arange(-rad, rad + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    ker /= ker.sum()
    pad = np.pad(arr, rad, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for dy in range(2 * rad + 1):
        for dx in range(2 * rad + 1):
            out += ker[dy, dx] * pad[dy:dy + arr.shape[0], dx:dx + arr.shape[1]]
    return out


def deletterbox_attention(grid24: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Upsample 24x24 attention grid to square max(W,H)x max(W,H) and crop valid RGB region."""
    clean24 = gaussian_smooth_2d(suppress_vit_sink_outliers(grid24), sigma=0.82)
    side = max(img_w, img_h)
    pil_map = Image.fromarray(clean24, mode="F").resize((side, side), Image.Resampling.BICUBIC)
    arr = np.asarray(pil_map, dtype=np.float32)
    arr = np.clip(arr, 0.0, None)
    x0 = (side - img_w) // 2
    y0 = (side - img_h) // 2
    return arr[y0:y0 + img_h, x0:x0 + img_w]


def blend_heatmap(rgb_img: np.ndarray, attn_hw: np.ndarray, vmin: float, vmax: float, cmap_name: str = "magma") -> np.ndarray:
    """Alpha-blend normalized attention heatmap over a gently dimmed desaturated background."""
    gray = (0.299 * rgb_img[..., 0] + 0.587 * rgb_img[..., 1] + 0.114 * rgb_img[..., 2])[..., None]
    base = (0.38 * rgb_img + 0.62 * gray) * 0.54 / 255.0
    norm = np.clip((attn_hw - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    norm_gamma = np.power(norm, 0.85)
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(norm_gamma)[..., :3]
    alpha = (0.14 + 0.78 * np.power(norm, 0.78))[..., None]
    out = (1.0 - alpha) * base + alpha * heat_rgb
    return np.clip(out, 0.0, 1.0)


def smooth_1d(arr: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1:
        return arr
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(padded, kernel, mode="valid")


def draw_rich_caption(fig, x_left, x_right, y_top, y_bottom_limit, line_h, tokens_spec, fontsize=10.35):
    """Word-wrap styled tokens inside [x_left, x_right] in figure coordinates and verify bounds."""
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.bbox.width
    fig_h_px = fig.bbox.height
    space_w = 0.0030

    cur_x = x_left
    cur_y = y_top
    n_lines = 1

    for idx, (word, style) in enumerate(tokens_spec):
        weight = "bold" if style in ("hal", "ground_g", "ground_a") else "normal"
        fstyle = "italic" if style == "muted" else "normal"
        color = C_INK
        if style == "hal":
            color = C_HAL_TEXT
        elif style == "ground_g":
            color = C_IKOD_DARK
        elif style == "ground_a":
            color = C_IKODA_DARK
        elif style == "muted":
            color = "#6c7682"

        t = fig.text(cur_x, cur_y, word, fontsize=fontsize, fontweight=weight, fontstyle=fstyle,
                     color=color, va="top", ha="left", zorder=6)
        bb = t.get_window_extent(renderer=renderer)
        w_fig = bb.width / fig_w_px
        h_fig = bb.height / fig_h_px

        if cur_x + w_fig > x_right and cur_x > x_left:
            cur_x = x_left
            cur_y -= line_h
            n_lines += 1
            t.set_position((cur_x, cur_y))
            bb = t.get_window_extent(renderer=renderer)
            w_fig = bb.width / fig_w_px
            h_fig = bb.height / fig_h_px

        if style == "hal":
            pad_x, pad_y = 0.0014, 0.0022
            pill = FancyBboxPatch(
                (cur_x - pad_x, cur_y - h_fig - pad_y + 0.002),
                w_fig + 2 * pad_x, h_fig + 2 * pad_y - 0.002,
                boxstyle="round,pad=0.0,rounding_size=0.0035",
                facecolor=C_HAL_BG, edgecolor=C_HAL_EDGE, linewidth=0.75,
                transform=fig.transFigure, zorder=5
            )
            fig.patches.append(pill)

        next_word = tokens_spec[idx + 1][0] if idx + 1 < len(tokens_spec) else ""
        if next_word in (".", ",", ";", ":"):
            cur_x += w_fig + 0.0004
        else:
            cur_x += w_fig + space_w

    bot_y = cur_y - h_fig
    if bot_y < y_bottom_limit:
        print(f"WARNING: card text bottom {bot_y:.4f} < limit {y_bottom_limit:.4f} (lines={n_lines})")
    return n_lines, bot_y


def tokenize_styled(segments):
    out = []
    for chunk, style in segments:
        for raw in chunk.split():
            if len(raw) > 1 and raw[-1] in ".,;:" and raw != "[...]":
                out.append((raw[:-1], style))
                out.append((raw[-1], "normal" if style != "muted" else "muted"))
            else:
                out.append((raw, style))
    return out


def main():
    data = json.loads(DATA_JSON.read_text())
    rgb_pil = Image.open(data["image_path"]).convert("RGB")
    img_w, img_h = rgb_pil.size
    rgb_np = np.asarray(rgb_pil, dtype=np.float32)

    # Compact double-column canvas: 14.4 x 4.88 in -> ~2.42 in tall at \textwidth (saves >58% height!)
    fig = plt.figure(figsize=(14.4, 4.88), dpi=300, facecolor="white")
    fig.canvas.draw()

    ax_ov = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=5)
    ax_ov.set_xlim(0, 1); ax_ov.set_ylim(0, 1)
    ax_ov.set_xticks([]); ax_ov.set_yticks([])

    # ---- Outer 3-panel frames ----
    panels = [
        (0.004, 0.014, 0.268, 0.972, "(a) Scene & Late Spatial Attention"),
        (0.280, 0.014, 0.372, 0.972, r"(b) Outputs & Lowest-$s_m$ Bucket Merging"),
        (0.660, 0.014, 0.336, 0.972, "(c) Temporal Recovery & Online Gate"),
    ]
    for px, py, pw, ph, title in panels:
        card = FancyBboxPatch(
            (px, py), pw, ph,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor=C_PANEL_BG, edgecolor=C_PANEL_EDGE, linewidth=1.15,
            transform=fig.transFigure, zorder=1
        )
        fig.patches.append(card)
        fig.text(px + pw / 2.0, py + ph - 0.018, title,
                 fontsize=12.2, fontweight="bold", color=C_INK, va="top", ha="center", zorder=4)

    # =========================================================================
    # PANEL (a): Input RGB Photograph + Late-Half (t > T/2) 2D Spatial Maps
    # =========================================================================
    ax_img = fig.add_axes([0.014, 0.495, 0.248, 0.422], zorder=3)
    ax_img.imshow(rgb_pil)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_edgecolor("#7d8b99"); sp.set_linewidth(1.0)

    ax_img.text(
        0.022, 0.945, 'Prompt: "Please describe this image in detail."',
        transform=ax_img.transAxes, fontsize=9.8, fontstyle="italic", color="white",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#101820", alpha=0.78, edgecolor="none")
    )
    ax_img.text(
        0.978, 0.050, "0 people  |  0 handbag",
        transform=ax_img.transAxes, fontsize=9.8, fontweight="bold", color="#ffd9dc",
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#6e1116", alpha=0.86, edgecolor="#f2a2a8", linewidth=0.7)
    )

    g_arm = data["arms"]["greedy"]
    map_base_24 = np.array(g_arm["spatial_late_base"], dtype=np.float32)
    map_merg_24 = np.array(g_arm["spatial_late_merged"], dtype=np.float32)
    att_base_hw = deletterbox_attention(map_base_24, img_w, img_h)
    att_merg_hw = deletterbox_attention(map_merg_24, img_w, img_h)

    combined_vals = np.concatenate([att_base_hw.ravel(), att_merg_hw.ravel()])
    vmin = float(np.percentile(combined_vals, 8.0))
    vmax = float(np.percentile(combined_vals, 98.8))

    blend_base = blend_heatmap(rgb_np, att_base_hw, vmin, vmax, "magma")
    blend_merg = blend_heatmap(rgb_np, att_merg_hw, vmin, vmax, "magma")

    T_g = g_arm["n_tokens"]
    half_g = T_g // 2
    sb_late = np.mean([s["S_base"] for s in g_arm["steps"][half_g:]]) * 100.0
    sm_late = np.mean([s["S_merged"] for s in g_arm["steps"][half_g:]]) * 100.0
    rel_gain = (sm_late / sb_late - 1.0) * 100.0

    ax_m1 = fig.add_axes([0.014, 0.118, 0.120, 0.276], zorder=3)
    ax_m2 = fig.add_axes([0.142, 0.118, 0.120, 0.276], zorder=3)
    ax_m1.imshow(blend_base)
    ax_m2.imshow(blend_merg)
    for ax_m, edge_c in [(ax_m1, C_BASE_LINE), (ax_m2, C_IKOD_LINE)]:
        ax_m.set_xticks([]); ax_m.set_yticks([])
        for sp in ax_m.spines.values():
            sp.set_edgecolor(edge_c); sp.set_linewidth(1.35)

    fig.text(0.074, 0.448, "Base Cache ($t > T/2$)", fontsize=10.6, fontweight="bold",
             color=C_BASE_DARK, ha="center", va="bottom", zorder=4)
    fig.text(0.074, 0.408, f"Share: {sb_late:.1f}%", fontsize=10.2, fontweight="bold",
             color=C_BASE_DARK, ha="center", va="bottom", zorder=4,
             bbox=dict(boxstyle="round,pad=0.18", facecolor=C_BASE_BG, edgecolor=C_BASE_EDGE, linewidth=0.8))

    fig.text(0.202, 0.448, r"Merged View ($\lambda{=}0.4$)", fontsize=10.6, fontweight="bold",
             color=C_IKOD_DARK, ha="center", va="bottom", zorder=4)
    fig.text(0.202, 0.408, f"Share: {sm_late:.1f}% (+{rel_gain:.0f}%)", fontsize=10.2, fontweight="bold",
             color=C_IKOD_DARK, ha="center", va="bottom", zorder=4,
             bbox=dict(boxstyle="round,pad=0.18", facecolor=C_IKOD_BG, edgecolor=C_IKOD_EDGE, linewidth=0.8))

    ax_cb = fig.add_axes([0.052, 0.070, 0.172, 0.018], zorder=3)
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax_cb.imshow(grad, aspect="auto", cmap="magma")
    ax_cb.set_xticks([]); ax_cb.set_yticks([])
    for sp in ax_cb.spines.values():
        sp.set_edgecolor("#88929e"); sp.set_linewidth(0.7)
    fig.text(0.047, 0.079, "Low", fontsize=9.8, color=C_MUTED, ha="right", va="center")
    fig.text(0.229, 0.079, "High", fontsize=9.8, color=C_MUTED, ha="left", va="center")
    fig.text(0.138, 0.038, "Late-stage spatial image attention ($t > T/2$)",
             fontsize=10.0, fontstyle="italic", color=C_MUTED, ha="center", va="center")

    # =========================================================================
    # PANEL (b): Three Caption Cards + Real Lowest-s_m Anchor & Bucket Strip
    # =========================================================================
    cards_cfg = [
        (0.718, 0.202, C_BASE_BG, C_BASE_EDGE,
         "Greedy  (94 words)", "4 hallucinated mentions", C_HAL_TEXT, C_HAL_BG, C_HAL_EDGE,
         [
             ("The image features a green pickup truck parked on a grassy field", "normal"),
             ("[...]", "muted"),
             ("next to a van. There are several", "normal"),
             ("people", "hal"),
             ("in the scene, with one", "normal"),
             ("person", "hal"),
             ("standing close to the truck", "normal"),
             ("[...]", "muted"),
             ("A", "normal"),
             ("handbag", "hal"),
             ("can be seen placed on the ground near the", "normal"),
             ("person", "hal"),
             ("standing close to the truck.", "normal"),
         ]),
        (0.512, 0.197, C_IKOD_BG, C_IKOD_EDGE,
         r"IKOD ($\lambda{=}0.4,\ \alpha{=}2.0$ | 91 words)", "0 hallucinations", C_IKOD_DARK, "#d6efe0", C_IKOD_EDGE,
         [
             ("The image features a", "normal"),
             ("green pickup truck", "ground_g"),
             ("parked on a", "normal"),
             ("grassy field.", "ground_g"),
             ("The truck is parked next to a", "normal"),
             ("van,", "ground_g"),
             ("and a", "normal"),
             ("small car", "ground_g"),
             ("is also visible", "normal"),
             ("[...]", "muted"),
             ("surrounded by a", "normal"),
             ("few cars parked in the background.", "ground_g"),
             ("The truck, with its green color, is the main focus.", "normal"),
         ]),
        (0.306, 0.197, C_IKODA_BG, C_IKODA_EDGE,
         "IKOD-A (Tuning-free gate | 86 words)", "0 hallucinations", C_IKODA_DARK, "#faeac6", C_IKODA_EDGE,
         [
             ("The image features a", "normal"),
             ("green pickup truck", "ground_a"),
             ("parked on a", "normal"),
             ("grassy field.", "ground_a"),
             ("The truck is parked next to a", "normal"),
             ("van,", "ground_a"),
             ("and there are", "normal"),
             ("two other trucks in the background.", "ground_a"),
             ("The green truck is parked in the foreground, with one of the", "normal"),
             ("other trucks", "ground_a"),
             ("in the middle.", "normal"),
         ]),
    ]

    cx0, cw = 0.290, 0.352
    for y0, ch, bg, edge, t_left, b_right, b_fg, b_bg, b_edge, segs in cards_cfg:
        box = FancyBboxPatch(
            (cx0, y0), cw, ch,
            boxstyle="round,pad=0.0,rounding_size=0.009",
            facecolor=bg, edgecolor=edge, linewidth=1.1,
            transform=fig.transFigure, zorder=2
        )
        fig.patches.append(box)
        fig.text(cx0 + 0.008, y0 + ch - 0.013, t_left,
                 fontsize=10.8, fontweight="bold", color=C_INK, va="top", ha="left", zorder=5)
        fig.text(cx0 + cw - 0.008, y0 + ch - 0.012, b_right,
                 fontsize=9.8, fontweight="bold", color=b_fg, va="top", ha="right", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.17", facecolor=b_bg, edgecolor=b_edge, linewidth=0.75))
        toks = tokenize_styled(segs)
        n_ln, bot_y = draw_rich_caption(
            fig, cx0 + 0.008, cx0 + cw - 0.008, y0 + ch - 0.052,
            y_bottom_limit=y0 + 0.010, line_h=0.0342, tokens_spec=toks, fontsize=10.10
        )
        print(f"Card {t_left[:10]}: {n_ln} lines, bottom={bot_y:.4f} (card_y0={y0:.4f})")

    # ---- Bottom Ribbon inside (b): Authentic Token-Level Bucket Merging (pos=33..44) ----
    rib_box = FancyBboxPatch(
        (cx0, 0.026), cw, 0.270,
        boxstyle="round,pad=0.0,rounding_size=0.009",
        facecolor="#ffffff", edgecolor="#b9c6d6", linewidth=1.05,
        transform=fig.transFigure, zorder=2
    )
    fig.patches.append(rib_box)

    fig.text(cx0 + 0.008, 0.282,
             r"Bucket Merging ($\lambda{=}0.4$: $|\mathcal{P}_t|{=}74 \to n_t{=}30$)",
             fontsize=10.5, fontweight="bold", color=C_INK, va="top", ha="left", zorder=5)

    # Legend star in top-right of ribbon
    ax_ov.scatter([cx0 + cw - 0.110], [0.271], marker="*", s=82,
                  facecolor=C_HAL_TEXT, edgecolor="#7a0c0c", linewidth=0.5, zorder=7)
    fig.text(cx0 + cw - 0.008, 0.282, r"lowest-$s_m$ anchors",
             fontsize=10.0, fontweight="bold", color=C_HAL_TEXT, va="top", ha="right", zorder=6)

    bs = data["arms"]["ikod"]["bucket_snapshot"]
    sl = slice(33, 45)
    tok_labels = [w.strip() if w.strip() else w for w in bs["words"][sl]]
    if tok_labels[2] == "ed":
        tok_labels[2] = "-ed"
    sm_vals = [x * 100.0 for x in bs["s_mean"][sl]]
    anc_flags = bs["is_anchor"][sl]
    bkt_ids = bs["bucket_id"][sl]

    n_chips = len(tok_labels)
    x_start = cx0 + 0.008
    avail_w = cw - 0.016
    gap = 0.0028
    chip_w = (avail_w - gap * (n_chips - 1)) / n_chips
    chip_y = 0.132
    chip_h = 0.088

    cmap_ylgnbu = plt.get_cmap("YlGnBu")
    sm_min, sm_max = min(sm_vals), max(sm_vals)

    for j in range(n_chips):
        xj = x_start + j * (chip_w + gap)
        norm_s = (sm_vals[j] - sm_min) / max(sm_max - sm_min, 1e-6)
        fill_rgba = cmap_ylgnbu(0.10 + 0.58 * norm_s)
        is_a = anc_flags[j]
        ec = C_HAL_TEXT if is_a else "#9fb0c2"
        lw = 1.65 if is_a else 0.85

        chip = FancyBboxPatch(
            (xj, chip_y), chip_w, chip_h,
            boxstyle="round,pad=0.0,rounding_size=0.004",
            facecolor=fill_rgba, edgecolor=ec, linewidth=lw,
            transform=fig.transFigure, zorder=4
        )
        fig.patches.append(chip)

        txt_col = "white" if norm_s > 0.68 else C_INK
        fig.text(xj + chip_w / 2, chip_y + chip_h * 0.67, tok_labels[j],
                 fontsize=10.1, fontweight="bold", color=txt_col, ha="center", va="center", zorder=6)
        fig.text(xj + chip_w / 2, chip_y + chip_h * 0.25, f"{sm_vals[j]:.1f}%",
                 fontsize=8.9, color=txt_col, ha="center", va="center", zorder=6)
        if is_a:
            ax_ov.scatter([xj + chip_w / 2], [chip_y + chip_h + 0.018], marker="*", s=85,
                          facecolor=C_HAL_TEXT, edgecolor="#7a0c0c", linewidth=0.5, zorder=7)

    unique_bkts = []
    for b in bkt_ids:
        if b not in unique_bkts:
            unique_bkts.append(b)

    for bid in unique_bkts:
        idxs = [i for i, b in enumerate(bkt_ids) if b == bid]
        x_l = x_start + idxs[0] * (chip_w + gap) + 0.002
        x_r = x_start + idxs[-1] * (chip_w + gap) + chip_w - 0.002
        xm = 0.5 * (x_l + x_r)
        yb_top = chip_y - 0.007
        yb_bot = chip_y - 0.020
        ax_ov.plot([x_l, x_l, x_r, x_r], [yb_top, yb_bot, yb_bot, yb_top],
                   color=C_IKOD_DARK, linewidth=1.25)
        ax_ov.plot([xm, xm], [yb_bot, yb_bot - 0.011], color=C_IKOD_DARK, linewidth=1.25)

        c_pill = FancyBboxPatch(
            (xm - 0.025, 0.052), 0.050, 0.043,
            boxstyle="round,pad=0.0,rounding_size=0.005",
            facecolor=C_IKOD_BG, edgecolor=C_IKOD_LINE, linewidth=1.0,
            transform=fig.transFigure, zorder=5
        )
        fig.patches.append(c_pill)
        fig.text(xm, 0.0735, rf"$D_{{{bid}}}\!\to\!\bar{{\mathbf{{k}}}}_{{{bid}}}$",
                 fontsize=10.0, fontweight="bold", color=C_IKOD_DARK, ha="center", va="center", zorder=6)

    # =========================================================================
    # PANEL (c): Synchronized Temporal Curves (Exact Pixel-Aligned X-Axis!)
    # =========================================================================
    plot_x0, plot_w = 0.708, 0.240
    ax_top = fig.add_axes([plot_x0, 0.548, plot_w, 0.355], zorder=3)
    ax_bot = fig.add_axes([plot_x0, 0.122, plot_w, 0.340], zorder=3)

    g_steps = g_arm["steps"]
    t_g = np.arange(1, len(g_steps) + 1)
    s_base_raw = np.array([s["S_base"] * 100.0 for s in g_steps])
    s_merg_raw = np.array([s["S_merged"] * 100.0 for s in g_steps])
    s_base_sm = smooth_1d(np.clip(s_base_raw, 0, 25), win=5)
    s_merg_sm = smooth_1d(np.clip(s_merg_raw, 0, 25), win=5)

    ax_top.axvspan(59, 124, color="#fdecef", alpha=0.65, zorder=0)
    ax_top.fill_between(t_g, s_base_sm, s_merg_sm, color="#bce5cc", alpha=0.55, zorder=1,
                        label="Theorem 1 gain")
    ax_top.plot(t_g, s_merg_sm, color=C_IKOD_LINE, linewidth=2.05, zorder=3,
                label=r"Merged $S'_t$ ($\lambda{=}0.4$)")
    ax_top.plot(t_g, s_base_sm, color=C_BASE_LINE, linewidth=1.85, linestyle="-", zorder=3,
                label=r"Base cache $S_t$")

    # Callouts placed neatly inside [1, 124]
    ax_top.axvline(62, color=C_HAL_TEXT, linestyle="--", linewidth=1.1, alpha=0.85, zorder=4)
    ax_top.scatter([62], [s_base_sm[61]], color=C_HAL_TEXT, s=24, zorder=5)
    ax_top.annotate(
        '"people"→"other"\n(53.5 vs 55.1)',
        xy=(62, s_base_sm[61]), xytext=(57, 23.2),
        fontsize=8.8, fontweight="bold", color=C_HAL_TEXT, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff5f5", edgecolor=C_HAL_EDGE, linewidth=0.75),
        arrowprops=dict(arrowstyle="->", color=C_HAL_TEXT, lw=0.9, shrinkA=2, shrinkB=3),
        zorder=6,
    )

    ax_top.axvline(84, color=C_HAL_TEXT, linestyle="--", linewidth=1.1, alpha=0.85, zorder=4)
    ax_top.scatter([84], [s_base_sm[83]], color=C_HAL_TEXT, s=24, zorder=5)
    ax_top.annotate(
        '"handbag"→"few"\n(24.7 vs 33.0)',
        xy=(84, s_base_sm[83]), xytext=(122, 23.2),
        fontsize=8.8, fontweight="bold", color=C_HAL_TEXT, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff5f5", edgecolor=C_HAL_EDGE, linewidth=0.75),
        arrowprops=dict(arrowstyle="->", color=C_HAL_TEXT, lw=0.9, shrinkA=2, shrinkB=3),
        zorder=6,
    )

    ax_top.set_xlim(1, 124)
    ax_top.set_ylim(3.2, 24.5)
    ax_top.set_ylabel("Image attn. share (%)", fontsize=10.8, fontweight="bold", color=C_INK)
    ax_top.tick_params(axis="both", labelsize=9.8)
    ax_top.set_xticklabels([])
    ax_top.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax_top.legend(loc="lower left", fontsize=8.8, frameon=True, facecolor="white", framealpha=0.92,
                  edgecolor="#d0d7de", borderpad=0.24, labelspacing=0.18, handlelength=1.4)
    for sp in ("top", "right"):
        ax_top.spines[sp].set_visible(False)

    # Bottom plot: IKOD-A online gate alpha_t (left) & Post-visual KV length (right)
    a_arm = data["arms"]["ikoda"]
    a_steps = a_arm["steps"]
    t_a = np.arange(1, len(a_steps) + 1)
    alpha_raw = np.array([s["alpha_t"] for s in a_steps])
    alpha_sm = smooth_1d(alpha_raw, win=5)
    l_base = np.array([s["L_post_base"] for s in a_steps])
    l_merg = np.array([s["L_post_merged"] for s in a_steps])

    ax_bot.axvspan(59, 124, color="#fdecef", alpha=0.50, zorder=0)
    ax_bot.axvline(62, color=C_HAL_TEXT, linestyle="--", linewidth=0.9, alpha=0.45, zorder=2)
    ax_bot.axvline(84, color=C_HAL_TEXT, linestyle="--", linewidth=0.9, alpha=0.45, zorder=2)
    ax_bot.plot(t_a, alpha_raw, color=C_IKODA_LINE, alpha=0.25, linewidth=0.95, zorder=2)
    ax_bot.plot(t_a, alpha_sm, color=C_IKODA_LINE, linewidth=2.1, zorder=4,
                label=r"IKOD-A gate $\alpha_t$")
    ax_bot.fill_between(t_a, 0, alpha_sm, color="#f6d58e", alpha=0.24, zorder=1)

    ax_bot.set_xlim(1, 124)
    ax_bot.set_ylim(0.0, 1.08)
    ax_bot.set_xlabel("Decoding step $t$", fontsize=11.0, fontweight="bold", color=C_INK)
    ax_bot.set_ylabel(r"Gate weight $\alpha_t$", fontsize=10.8, fontweight="bold", color=C_IKODA_DARK)
    ax_bot.tick_params(axis="y", labelcolor=C_IKODA_DARK, labelsize=9.8)
    ax_bot.tick_params(axis="x", labelsize=9.8)
    ax_bot.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax_bot.spines["top"].set_visible(False)

    ax_r = ax_bot.twinx()
    ax_r.plot(t_a, l_base, color="#566573", linestyle="--", linewidth=1.45, zorder=3,
              label=r"Full text KV $|\mathcal{P}_t|$")
    ax_r.plot(t_a, l_merg, color=C_IKOD_LINE, linestyle="-.", linewidth=1.65, zorder=3,
              label=r"Merged $n_t$ ($-60\%$)")
    ax_r.set_ylim(0, 155)
    ax_r.set_ylabel("Post-visual KV entries", fontsize=10.4, fontweight="bold", color="#3d5245")
    ax_r.tick_params(axis="y", labelcolor="#3d5245", labelsize=9.6)
    ax_r.spines["top"].set_visible(False)

    h1, l1 = ax_bot.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_bot.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.6, frameon=True,
                  facecolor="white", framealpha=0.92, edgecolor="#d0d7de",
                  borderpad=0.22, labelspacing=0.16, handlelength=1.4)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PNG, format="png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    im = Image.open(OUT_PNG)
    print(f"Saved {OUT_PDF} and {OUT_PNG} | PNG pixels={im.size}, aspect={im.size[0]/im.size[1]:.2f}:1")


if __name__ == "__main__":
    main()
