"""Fading-focus figure, v3 (publication-grade DejaVu Serif edition aligned with Figs. 4-6).
Every curve is the CHAIR-500 / 512-token trajectory (results/fading/{model}_chair500.json:
the ordered 500 CHAIR images, 512 new tokens, 20 bins), on all eight backbones."""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = "./workspace"

rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

STYLE = {
    "llava_15_hf":  ("LLaVA-1.5",    "#1e5aa8", "-",  "o"),
    "llava_next":   ("LLaVA-NeXT",   "#2b8cbe", "-",  "s"),
    "llava_ov":     ("LLaVA-OV",     "#168f72", "-",  "^"),
    "qwen2_vl":     ("Qwen2-VL",     "#d97706", "--", "D"),
    "qwen2_5_vl":   ("Qwen2.5-VL",   "#dc4b3e", "--", "v"),
    "qwen3_vl":     ("Qwen3-VL",     "#9e1b32", "--", "X"),
    "internvl3":    ("InternVL3",    "#6d4c9e", "-.", "P"),
    "instructblip": ("InstructBLIP", "#4b5563", ":",  "*"),
}
ORDER = {
    "instructblip": 0, "internvl3": 1,
    "llava_15_hf": 2, "llava_next": 3, "llava_ov": 4,
    "qwen2_vl": 5, "qwen2_5_vl": 6, "qwen3_vl": 7,
}

fig, ax = plt.subplots(figsize=(3.78, 1.96), facecolor="white")
ax.set_facecolor("#fbfcfe")

series = []
for f in sorted(glob.glob(f"{ROOT}/results/fading/*_chair500.json")):
    d = json.load(open(f))
    if d.get("model") not in STYLE:
        continue
    if d.get("k") != 500 or d.get("steps") != 512 or not str(d.get("manifest", "")).endswith("data/chair/chair_500.jsonl"):
        raise SystemExit(f"{f}: not the CHAIR-500 / 512-token protocol")
    if "curve_norm" in d:
        c = [x for x in d["curve_norm"] if x is not None]; first, last = d.get("first_bin"), d.get("last_bin")
    else:
        raw = [x for x in d["curve"] if x is not None]; first, last = raw[0], raw[-1]; c = [x / raw[0] for x in raw]
    x = [i / (len(c) - 1) for i in range(len(c))]
    drop = (1 - last / first) * 100 if first else 0
    series.append((drop, d["model"], x, c))

if len(series) != len(STYLE):
    raise SystemExit(f"{len(series)} CHAIR-500 trajectories, expected {len(STYLE)}")

# Shaded late-stage dilution region (x >= 0.5) & 7-model standard ViT/M-RoPE envelope
x_arr = np.array(series[0][2])
vit_curves = np.array([c for _, m, _, c in series if m != "instructblip"])
ax.axvspan(0.50, 1.015, color="#fde6e7", alpha=0.28, zorder=0, lw=0)
ax.axvline(0.50, color="#e59a9e", linestyle=":", linewidth=0.85, zorder=1)
ax.fill_between(x_arr, vit_curves.min(axis=0), vit_curves.max(axis=0),
                color="#3563a8", alpha=0.08, zorder=1, lw=0)
ax.axhline(1.0, color="#8a96a3", linestyle="--", linewidth=0.75, alpha=0.75, zorder=1)

# Subtle callout badge in top-right of late-stage region
ax.text(0.985, 0.965, "Late-stage attention dilution\n(53%\u201380% drop; 26% Q-Former)",
        transform=ax.transAxes, fontsize=6.4, fontstyle="italic", color="#9e1b32",
        ha="right", va="top", zorder=5,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#fff5f5", edgecolor="#e59a9e", linewidth=0.65, alpha=0.92))

for drop, m, x, c in sorted(series, key=lambda r: ORDER.get(r[1], 99)):
    name, color, ls, mk = STYLE[m]
    ms = 4.5 if mk == "*" else 3.5
    ax.plot(x, c, color=color, linestyle=ls, marker=mk, markersize=ms, markevery=2,
            linewidth=1.50, label=f"{name} (\u2212{drop:.0f}%)",
            markeredgecolor="white", markeredgewidth=0.55, alpha=0.96, zorder=3)

ax.set_xlabel("Relative position in generated response ($t / T$)", fontsize=7.6, fontweight="bold", color="#1d242b", labelpad=2.5)
ax.set_ylabel("Normalized image attention", fontsize=7.5, fontweight="bold", color="#1d242b", labelpad=2.5)
ax.set_xlim(-0.015, 1.015)
ax.set_ylim(0.11, 1.07)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(labelsize=7.3, colors="#1d242b", pad=2.0)
ax.grid(True, linestyle="--", alpha=0.45, color="#cfd8e3", linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#5b6570")
ax.spines["bottom"].set_color("#5b6570")

# Anchor legend right edge to x=1.0 of ax so it never protrudes past the plot on left or right
leg = ax.legend(
    fontsize=5.70, loc="upper right", bbox_to_anchor=(1.002, -0.200),
    frameon=True, fancybox=True, framealpha=1.0,
    facecolor="#f8fafd", edgecolor="#cfd8e3",
    handlelength=1.25, labelspacing=0.25, columnspacing=0.45,
    borderpad=0.28, handletextpad=0.26, ncol=3
)
leg.get_frame().set_linewidth(0.80)
for txt in leg.get_texts():
    txt.set_color("#1d242b")
    txt.set_fontweight("bold")

DST = "./paper/figures"
fig.savefig(f"{DST}/fading_focus.pdf", bbox_inches="tight", pad_inches=0.008, bbox_extra_artists=[leg])
fig.savefig(f"{DST}/fading_focus.png", dpi=300, bbox_inches="tight", pad_inches=0.008, bbox_extra_artists=[leg])
print("Saved full-width zero-margin fading_focus.{pdf,png}")
