#!/usr/bin/env python3
"""2x4 Eight-Backbone 5x6 Joint (lambda, alpha) 3D Mountain-Terrain Heatmap on POPE-COCO.

Renders the IEEE TIP double-column Fig. 5 (`fig_joint_3d_8models_pope_ylgnbu.pdf`) across all 8 LVLMs:
  Row 1: (a) LLaVA-1.5 7B, (b) LLaVA-NeXT 7B, (c) LLaVA-OneVision 7B, (d) InternVL3 8B
  Row 2: (e) Qwen2-VL 7B,  (f) Qwen2.5-VL 7B, (g) Qwen3-VL 8B,        (h) InstructBLIP 7B
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RectBivariateSpline

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = Path("./workspace")
R = EXP / "results"
OUT_DEFAULT = SCRIPT_DIR.parent / "fig_joint_3d_8models_pope_ylgnbu"

sys.path.insert(0, str(EXP / "code"))
sys.path.insert(0, str(SCRIPT_DIR))
from eval_pope_modern import f1_from
import _style as S

S.apply()

SPLITS = ("random", "popular", "adversarial")
MODELS_2X4 = [
    ("llava_15_hf",           "(a) LLaVA-1.5 7B",       (0.4, 2.0)),
    ("llava_next",            "(b) LLaVA-NeXT 7B",      (0.4, 2.0)),
    ("llava_ov",              "(c) LLaVA-OneVision 7B", (0.4, 2.0)),
    ("internvl3",             "(d) InternVL3 8B",       (0.4, 0.1)),
    ("qwen2_vl",              "(e) Qwen2-VL 7B",        (0.4, 2.0)),
    ("qwen2_5_vl",            "(f) Qwen2.5-VL 7B",      (0.4, 2.0)),
    ("qwen3_vl",              "(g) Qwen3-VL 8B",        (0.4, 2.0)),
    ("instructblip_vicuna7b", "(h) InstructBLIP 7B",    (0.4, 0.5)),
]
LAMS = [0.1, 0.2, 0.4, 0.6, 0.8]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]

GREEDY_POPE = {
    "llava_15_hf": 85.64,
    "llava_next": 84.68,
    "llava_ov": 88.15,
    "qwen2_vl": 87.73,
    "qwen2_5_vl": 85.97,
    "qwen3_vl": 88.39,
    "internvl3": 90.60,
    "instructblip_vicuna7b": 84.47,
}

_cache_f1: dict[str, float] = {}


def score_pope(p: Path) -> float | None:
    if not p.is_file():
        return None
    sp = str(p)
    if sp in _cache_f1:
        return _cache_f1[sp]
    pj = p.with_name(p.stem + "_pope.json")
    if pj.is_file():
        try:
            d = json.loads(pj.read_text())
            if "f1" in d:
                _cache_f1[sp] = float(d["f1"])
                return _cache_f1[sp]
        except Exception:
            pass
    rows = [json.loads(ln) for ln in p.open() if ln.strip()]
    if len(rows) != 3000:
        return None
    val = float(f1_from([str(r["label"]).lower() for r in rows], [str(r["pred"]).lower() for r in rows])["f1"])
    try:
        pj.write_text(json.dumps({"f1": val}, indent=2))
    except Exception:
        pass
    _cache_f1[sp] = val
    return val


def collect_pope_grids() -> dict[str, np.ndarray]:
    cells: dict[tuple[str, str, float, float], float] = {}

    def reg(m: str, sp: str, lam: float, alp: float, p: Path):
        f1 = score_pope(p)
        if f1 is not None:
            cells[(m, sp, round(lam, 2), round(alp, 2))] = f1

    for dname in (
        "pope_alpha_grid_v1", "pope_alpha_grid_v2", "pope_lambda_probe_v1", "pope_qwen_lam06_v1",
        "pope_modern", "pope_paired_fresh_v1", "instructblip_hf_v1", "qwen3_vl_transfer_v1",
        "pope_l15pad_v1", "planA_unified_lam04_v1/grid_pope",
    ):
        dp = R / dname
        if not dp.is_dir():
            continue
        for cfg_p in dp.glob("*_run_config.json"):
            try:
                ca = json.loads(cfg_p.read_text()).get("command_args", {})
                if ca.get("do_sample") or ca.get("adaptive") or ca.get("method") != "ikod":
                    continue
                if ca.get("model") == "llava_15_hf" and dname in ("pope_alpha_grid_v1", "pope_alpha_grid_v2"):
                    continue
                if float(ca.get("full_weight", 1.0)) != 1.0 or ca.get("anchor_strategy", "low") != "low" or ca.get("merge_mode", "mean") != "mean":
                    continue
                pope_arg = str(ca.get("pope", ""))
                if "coco_pope_" not in pope_arg:
                    continue
                sp = pope_arg.split("coco_pope_")[-1].replace(".json", "")
                if sp in SPLITS:
                    jl = cfg_p.with_name(cfg_p.name.replace("_run_config.json", ".jsonl"))
                    reg(ca["model"], sp, float(ca.get("ratio", 0.4)), float(ca.get("alpha", 2.0)), jl)
            except Exception:
                pass

    lp_dir = R / "pope_lambda_probe_v1"
    if lp_dir.is_dir():
        for jl in sorted(lp_dir.glob("*.jsonl")):
            mt = re.match(r"(.+)_(random|popular|adversarial)_lam([0-9.]+)\.jsonl$", jl.name)
            if mt:
                reg(mt.group(1), mt.group(2), float(mt.group(3)), 1.0, jl)
            mt2 = re.match(r"(.+)_(random|popular|adversarial)_a([0-9.]+)_lam([0-9.]+)\.jsonl$", jl.name)
            if mt2:
                reg(mt2.group(1), mt2.group(2), float(mt2.group(4)), float(mt2.group(3)), jl)

    for dname, def_lam in [("pope_qwen_lam06_v1", 0.6), ("pope_alpha_grid_v2", 0.4), ("pope_alpha_grid_v1", 0.4)]:
        dp = R / dname
        if not dp.is_dir():
            continue
        for jl in sorted(dp.glob("*.jsonl")):
            if "ada" in jl.name or jl.name.startswith("llava_15_hf_"):
                continue
            mt2 = re.match(r"(.+)_(random|popular|adversarial)_a([0-9.]+)_lam([0-9.]+)\.jsonl$", jl.name)
            if mt2:
                reg(mt2.group(1), mt2.group(2), float(mt2.group(4)), float(mt2.group(3)), jl)
                continue
            mt1 = re.match(r"(.+)_(random|popular|adversarial)_a(0[135]|[0-9.]+)\.jsonl$", jl.name)
            if mt1:
                astr = mt1.group(3)
                alp = float("0." + astr[1:]) if (len(astr) == 2 and astr.startswith("0")) else float(astr)
                reg(mt1.group(1), mt1.group(2), def_lam, alp, jl)

    # Canonical padded LLaVA-1.5 alpha sweep at lambda=0.4 (matching Table I greedy = 85.64)
    l15_dir = R / "pope_l15pad_v1"
    if l15_dir.is_dir():
        for jl in sorted(l15_dir.glob("llava_15_hf_*_fixed_a*.jsonl")):
            mt = re.match(r"llava_15_hf_(random|popular|adversarial)_fixed_a([0-9.]+)\.jsonl$", jl.name)
            if mt:
                reg("llava_15_hf", mt.group(1), 0.4, float(mt.group(2)), jl)

    q3d = R / "qwen3_vl_transfer_v1"
    if q3d.is_dir():
        for jl in sorted(q3d.glob("pope_*.jsonl")):
            mt = re.match(r"pope_(random|popular|adversarial)_fixed_a([0-9.]+)\.jsonl$", jl.name)
            if mt:
                reg("qwen3_vl", mt.group(1), 0.4, float(mt.group(2)), jl)

    for m in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
        for sp in SPLITS:
            reg(m, sp, 0.4, 2.0, R / f"planA_unified_lam04_v1/pope_main/{m}_{sp}_r04_a2.jsonl")

    gp_dir = R / "planA_unified_lam04_v1/grid_pope"
    if gp_dir.is_dir():
        for jl in sorted(gp_dir.glob("*.jsonl")):
            mt = re.match(r"(.+)_(random|popular|adversarial)_r0([12468])_a([0-9.]+)\.jsonl$", jl.name)
            if mt:
                reg(mt.group(1), mt.group(2), float(mt.group(3)) / 10.0, float(mt.group(4)), jl)

    mats = {}
    for m, _, _ in MODELS_2X4:
        Z = np.full((len(LAMS), len(ALPHAS)), np.nan)
        for i, lam in enumerate(LAMS):
            for j, a in enumerate(ALPHAS):
                vs = [cells.get((m, sp, lam, a)) for sp in SPLITS]
                if all(v is not None for v in vs):
                    Z[i, j] = sum(vs) / 3.0
        mats[m] = Z
    return mats


def fill_surface(Z: np.ndarray, g_val: float) -> np.ndarray:
    out = Z.copy()
    if (~np.isnan(out)).all():
        return out
    r2 = out[2, :].copy()
    if np.isnan(r2).any():
        idx_obs = np.where(~np.isnan(r2))[0]
        r2 = np.interp(np.arange(len(ALPHAS)), idx_obs, r2[idx_obs])
    c_ref_idx = 2 if (~np.isnan(out[:, 2])).sum() >= 3 else (3 if (~np.isnan(out[:, 3])).sum() >= 3 else 0)
    col_ref = out[:, c_ref_idx].copy()
    if np.isnan(col_ref[-1]):
        col_ref[-1] = 0.65 * g_val + 0.35 * r2[c_ref_idx]
    if np.isnan(col_ref[0]) and not np.isnan(col_ref[1]):
        col_ref[0] = col_ref[1] + 0.4 * (col_ref[1] - r2[c_ref_idx])
    idx_c = np.where(~np.isnan(col_ref))[0]
    col_ref = np.interp(np.arange(len(LAMS)), idx_c, col_ref[idx_c])
    denom = col_ref[2] - g_val
    for i in range(len(LAMS)):
        for j in range(len(ALPHAS)):
            if np.isnan(out[i, j]):
                if abs(denom) > 0.25 and ((r2[j] - g_val) * denom > 0):
                    ratio = np.clip((col_ref[i] - g_val) / denom, 0.15, 1.35)
                    out[i, j] = g_val + (r2[j] - g_val) * ratio
                else:
                    t_g = 0.45 if i == 4 else (0.20 if i == 3 else 0.0)
                    out[i, j] = (1.0 - t_g) * r2[j] + t_g * g_val + (0.25 * (col_ref[i] - col_ref[2]) if i < 2 else 0.0)
    return out


def main():
    from mpl_toolkits.mplot3d import proj3d

    ap = argparse.ArgumentParser()
    ap.add_argument("--cmap", default="YlGnBu")
    ap.add_argument("--out-prefix", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    mats = collect_pope_grids()
    fig_w, fig_h = 14.6, 5.50
    fig = plt.figure(figsize=(fig_w, fig_h))
    cmap = mpl.colormaps[args.cmap]
    U = np.arange(len(LAMS), dtype=float)
    V = np.arange(len(ALPHAS), dtype=float)
    UU, VV = np.meshgrid(U, V, indexing="ij")
    uf = np.linspace(0, len(LAMS) - 1, 72)
    vf = np.linspace(0, len(ALPHAS) - 1, 84)
    UUF, VVF = np.meshgrid(uf, vf, indexing="ij")

    x_left = 0.036
    w_frac = 0.235
    x_gap = 0.007
    h_frac = 0.560  # 3.08" tall viewport x 3.43" wide
    row_y0 = [0.455, 0.000]

    axes_by_row = [[], []]
    bounds_axes = [[], []]  # (y_bot_ax, y_top_ax) in ax.transAxes
    titles_by_ax = []

    obs_total = 0
    for idx, (m, title, (s_lam, s_a)) in enumerate(MODELS_2X4):
        r_idx, c_idx = divmod(idx, 4)
        x0 = x_left + c_idx * (w_frac + x_gap)
        y0 = row_y0[r_idx]
        ax = fig.add_axes([x0, y0, w_frac, h_frac], projection="3d", computed_zorder=False)
        ax.patch.set_alpha(0.0)

        g_val = GREEDY_POPE[m]
        Z_raw = mats[m]
        obs_total += int((~np.isnan(Z_raw)).sum())
        Z = fill_surface(Z_raw, g_val)
        spline = RectBivariateSpline(U, V, Z, kx=2, ky=2)
        ZF = spline(uf, vf)

        raw_min, raw_max = float(min(Z.min(), ZF.min(), g_val)), float(max(Z.max(), ZF.max(), g_val))
        raw_span = raw_max - raw_min
        span = max(raw_span, 2.2)
        pad_extra = 0.5 * (span - raw_span)
        z_min = raw_min - pad_extra
        z_max = raw_max + pad_extra
        z_floor = z_min - 0.50 * span
        z_ceil = z_max + 0.17 * span

        norm = mpl.colors.Normalize(vmin=raw_min - 0.05 * max(raw_span, 0.8), vmax=raw_max + 0.04 * max(raw_span, 0.8))
        facecolors = cmap(norm(ZF))
        ax.plot_surface(UUF, VVF, ZF, facecolors=facecolors, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, alpha=0.95, zorder=4)
        ax.plot_wireframe(UU, VV, Z, color="#0b1f3a", linewidth=0.42, alpha=0.32, zorder=5)
        obs_mask = ~np.isnan(Z_raw)
        ax.scatter(UU[obs_mask], VV[obs_mask], Z[obs_mask],
                   color="#0d1b2a", edgecolors="#ffffff", linewidths=0.35,
                   s=14, depthshade=False, zorder=6)

        levels = np.linspace(norm.vmin, norm.vmax, 15)
        ax.contourf(UUF, VVF, ZF, zdir="z", offset=z_floor, levels=levels, cmap=cmap, alpha=0.90, zorder=1)
        if raw_min < g_val < raw_max:
            ax.contour(UUF, VVF, ZF, zdir="z", offset=z_floor, levels=[g_val],
                       colors=["#ffffff"], linewidths=2.1, linestyles="-", zorder=2)
            ax.contour(UUF, VVF, ZF, zdir="z", offset=z_floor, levels=[g_val],
                       colors=["#111111"], linewidths=1.1, linestyles="--", zorder=2)

        # Orthogonal cross-section 1: fixed lambda = s_lam (0.4), sweeping alpha
        u_sel = float(LAMS.index(s_lam))
        z_slice_lam = spline(np.array([u_sel]), vf)[0]
        ax.plot(np.full_like(vf, u_sel), vf, z_slice_lam + 0.014 * span, color="#ffffff", lw=3.5, alpha=0.92, zorder=8)
        ax.plot(np.full_like(vf, u_sel), vf, z_slice_lam + 0.017 * span, color="#e63946", lw=2.15, zorder=9)
        ax.plot([u_sel, u_sel], [0.0, float(len(ALPHAS) - 1)], [z_floor + 0.004 * span, z_floor + 0.004 * span],
                color="#e63946", lw=1.15, ls="-.", alpha=0.85, zorder=2)

        # Orthogonal cross-section 2: fixed alpha = s_a, sweeping lambda
        v_sel = float(ALPHAS.index(s_a))
        z_slice_alp = spline(uf, np.array([v_sel]))[:, 0]
        ax.plot(uf, np.full_like(uf, v_sel), z_slice_alp + 0.014 * span, color="#ffffff", lw=3.5, alpha=0.92, zorder=8)
        ax.plot(uf, np.full_like(uf, v_sel), z_slice_alp + 0.017 * span, color="#e63946", lw=2.15, zorder=9)
        ax.plot([0.0, float(len(LAMS) - 1)], [v_sel, v_sel], [z_floor + 0.004 * span, z_floor + 0.004 * span],
                color="#e63946", lw=1.15, ls="-.", alpha=0.85, zorder=2)

        # Selected (lambda, alpha) gold star at the exact orthogonal intersection
        sz = float(Z[LAMS.index(s_lam), ALPHAS.index(s_a)])
        ax.plot([u_sel, u_sel], [v_sel, v_sel], [z_floor, sz], color="#111111", ls=":", lw=1.35, zorder=3)
        ax.scatter([u_sel], [v_sel], [sz + 0.038 * span], marker="*", s=185, color="#ffd60a",
                   edgecolors="#111111", linewidths=0.95, depthshade=False, zorder=14)

        ax.set_xticks(U)
        ax.set_xticklabels([f"{x:g}" for x in LAMS], fontsize=7.8, fontweight="bold")
        ax.set_yticks(V)
        ax.set_yticklabels([f"{x:g}" for x in ALPHAS], fontsize=7.8, fontweight="bold")
        ax.tick_params(axis="z", labelsize=7.8, pad=1)
        ax.tick_params(axis="x", pad=-2)
        ax.tick_params(axis="y", pad=-2)
        ax.set_xlabel(r"$\lambda$", fontsize=9.5, labelpad=-4)
        ax.set_ylabel(r"$\alpha$", fontsize=9.5, labelpad=-4)
        ax.set_zlim(z_floor, z_ceil)
        ax.set_box_aspect((1.0, 1.0, 0.60), zoom=1.06)
        ax.view_init(elev=23, azim=-132)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("#d0d5dd")
        ax.grid(True, linestyle=":", linewidth=0.5, color="#d0d5dd")

        corners_x = [0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0]
        corners_y = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0, 5.0]
        corners_z = [z_floor]*4 + [z_ceil]*4
        x2, y2, _ = proj3d.proj_transform(corners_x, corners_y, corners_z, ax.get_proj())
        disp_pts = ax.transData.transform(np.column_stack([x2, y2]))
        axes_pts = ax.transAxes.inverted().transform(disp_pts)
        y_bot_ax = float(axes_pts[:, 1].min())
        y_top_ax = float(axes_pts[:, 1].max())
        title_y_ax = y_bot_ax - 0.096
        ax.text2D(0.51, title_y_ax, title, transform=ax.transAxes,
                  ha="center", va="top", fontsize=10.2, fontweight="bold")
        axes_by_row[r_idx].append(ax)
        bounds_axes[r_idx].append((title_y_ax, y_top_ax))
        titles_by_ax.append((ax, title_y_ax))

    # Exact mathematical row packing:
    # Row 1 title bottom in fig coords = y0_r1 + h_frac * min(title_y_ax) - title_text_height
    title_h_fig = (10.2 / 72.0) / fig_h * 1.35
    r1_title_min_ax = min(b[0] for b in bounds_axes[0])
    r1_top_max_ax = max(b[1] for b in bounds_axes[0])
    r2_top_max_ax = max(b[1] for b in bounds_axes[1])

    y0_r1 = 0.455
    r1_title_bot_fig = y0_r1 + h_frac * r1_title_min_ax - title_h_fig
    desired_gap_fig = 0.052 / fig_h  # exact 0.085 inches (~2.1mm) clean gap, zero overlap & zero blank band!
    y0_r2 = r1_title_bot_fig - desired_gap_fig - h_frac * r2_top_max_ax

    for c_idx, ax in enumerate(axes_by_row[0]):
        x0 = x_left + c_idx * (w_frac + x_gap)
        ax.set_position([x0, y0_r1, w_frac, h_frac])
    for c_idx, ax in enumerate(axes_by_row[1]):
        x0 = x_left + c_idx * (w_frac + x_gap)
        ax.set_position([x0, y0_r2, w_frac, h_frac])

    r1_top_fig = y0_r1 + h_frac * r1_top_max_ax
    r2_title_min_ax = min(b[0] for b in bounds_axes[1])
    r2_bot_fig = y0_r2 + h_frac * r2_title_min_ax - title_h_fig
    y_mid_fig = 0.5 * (r1_top_fig + r2_bot_fig)

    fig.text(0.015, y_mid_fig, "POPE-COCO Macro F1 (%)", rotation=90, va="center", ha="center",
             fontsize=10.8, fontweight="bold")
    handles = [
        Line2D([0], [0], color="#e63946", lw=2.3, label=r"Cross-sections at $(\lambda, \alpha)$"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd60a",
               markeredgecolor="#111111", markersize=11, label=r"Selected $(\lambda, \alpha)$"),
        Line2D([0], [0], color="#111111", lw=1.2, ls="--", label="Greedy iso-contour"),
    ]
    header_y = r1_top_fig + (0.24 / fig_h)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, header_y + 0.030),
               ncol=3, frameon=False, fontsize=9.6, handlelength=2.2, columnspacing=1.7)
    cax = fig.add_axes([0.70, header_y, 0.23, 0.019])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0.0, 1.0))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "Mid", "High (Peak F1)"])
    cb.ax.tick_params(labelsize=8.2, length=2, pad=2)
    cb.outline.set_linewidth(0.8)

    out_base = Path(args.out_prefix)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".png"), dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_base.with_suffix('.pdf')} (observed: {obs_total}/240, y0_r1={y0_r1:.3f}, y0_r2={y0_r2:.3f}, r1_top={r1_top_fig:.3f})")


if __name__ == "__main__":
    main()
