import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

models = [
    "LLaVA-1.5", "LLaVA-NeXT", "LLaVA-OV",
    "Qwen2-VL", "Qwen2.5-VL", "Qwen3-VL",
    "InternVL3", "InstructBLIP", "Average"
]
n_m = len(models)
y = np.arange(n_m)[::-1].astype(float)
y[-1] = -0.15

late_rel = np.array([74.8, 133.0, 102.0, 57.6, 50.7, 37.2, 39.5, 45.3, 67.5])
hal_rel  = np.array([15.9,  40.7,  57.8, 37.1, 36.6, 27.6, 18.3, 37.7, 34.0])
win_late = np.array([99.2, 100.0,  96.7, 100.0, 98.3, 97.5, 89.2, 100.0, 97.6])
win_hal  = np.array([72.3,  90.5,  92.3, 100.0, 99.5, 91.8, 76.8, 100.0, 90.4])

prob_drop  = np.array([69.2, 66.1, 66.9, 66.8, 65.1, 56.3, 67.8, 72.1, 66.3])
rev_merged = np.array([71.9, 68.4, 69.9, 71.8, 68.4, 64.2, 77.3, 80.5, 71.6])
rev_collab = np.array([54.6, 40.3, 36.7, 27.7, 29.3, 14.5, 41.2, 46.0, 36.3])
gt_ikod    = np.array([82.1, 85.8, 84.4, 87.7, 90.4, 92.5, 90.4, 90.2, 87.9])
gt_ikoda   = np.array([91.4, 93.5, 92.7, 94.1, 94.5, 94.7, 88.7, 89.5, 92.4])

fig = plt.figure(figsize=(7.30, 2.50), facecolor="white")

ax_a  = fig.add_axes([0.095, 0.195, 0.282, 0.695])
ax_b  = fig.add_axes([0.398, 0.195, 0.276, 0.695])
ax_c1 = fig.add_axes([0.762, 0.642, 0.228, 0.248])
ax_c2 = fig.add_axes([0.762, 0.195, 0.228, 0.355])

for ax in (ax_a, ax_b, ax_c1, ax_c2):
    ax.set_facecolor("#fbfcfe")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5b6570")
    ax.spines["bottom"].set_color("#5b6570")

# ==============================================================================
# Panel (a): Closed-Loop Visual Attention Recovery + Theorem 1 Pill Column
# ==============================================================================
ax_a.axhspan(-0.72, 0.44, color="#eef4fb", alpha=0.95, zorder=0)
ax_a.axhline(0.44, color="#8a96a3", linestyle="--", linewidth=0.75, zorder=1)

ax_a.axvspan(184, 260, color="#fff6f5", alpha=0.92, zorder=1)
ax_a.axvline(184, color="#e5b8b8", linestyle=":", linewidth=0.85, zorder=2)
ax_a.text(222, n_m - 0.02, r"$\mathbf{Pr(S'_t > S_t)}$" + "\nLate / Hal.",
          fontsize=5.6, color="#9e1b32", ha="center", va="center", linespacing=1.02,
          bbox=dict(boxstyle="round,pad=0.12", facecolor="#fff0f0", edgecolor="#e59a9e", lw=0.55))

for i in range(n_m):
    yi = y[i]
    lw = 2.2 if i == n_m - 1 else 1.55
    ax_a.plot([hal_rel[i], late_rel[i]], [yi, yi], color="#6baed6", linewidth=lw, zorder=2, solid_capstyle="round")

ax_a.scatter(hal_rel[:-1], y[:-1], s=24, color="#2ca25f", edgecolor="white", linewidth=0.55, zorder=4, label=r"Hal.-Onset $\Delta S_t / S_t$")
ax_a.scatter(late_rel[:-1], y[:-1], s=27, color="#1e5aa8", marker="D", edgecolor="white", linewidth=0.55, zorder=4, label=r"Late-Stage $\Delta S_t / S_t$")
ax_a.scatter([hal_rel[-1]], [y[-1]], s=38, color="#2ca25f", edgecolor="#0f172a", linewidth=0.8, zorder=5)
ax_a.scatter([late_rel[-1]], [y[-1]], s=40, color="#1e5aa8", marker="D", edgecolor="#0f172a", linewidth=0.8, zorder=5)

for i in range(n_m):
    yi = y[i]
    fw = "bold" if i == n_m - 1 else "normal"
    dx_left  = 8.8 if i == n_m - 1 else 6.8
    dx_right = 9.5 if i == n_m - 1 else 7.5
    ax_a.text(hal_rel[i] - dx_left,   yi, f"+{hal_rel[i]:.0f}%",  va="center", ha="right", fontsize=5.5, fontweight=fw, color="#15803d")
    ax_a.text(late_rel[i] + dx_right, yi, f"+{late_rel[i]:.0f}%", va="center", ha="left",  fontsize=5.7, fontweight=fw, color="#1e5aa8")
    w_str = f"{win_late[i]:.0f}/{win_hal[i]:.0f}%" if win_late[i] == 100.0 else f"{win_late[i]:.1f}/{win_hal[i]:.1f}%"
    ax_a.text(222, yi, w_str, va="center", ha="center", fontsize=5.8, fontweight=fw, color="#9e1b32" if i == n_m - 1 else "#7f1d1d")

ax_a.set_yticks(y)
ax_a.set_yticklabels(models, fontsize=7.0)
ax_a.get_yticklabels()[-1].set_fontweight("bold")
ax_a.set_xlim(-26, 260)
ax_a.set_xticks([0, 40, 80, 120])
ax_a.set_xticklabels(["0%", "+40%", "+80%", "+120%"], fontsize=6.5)
ax_a.set_ylim(-0.74, n_m + 0.56)
ax_a.set_xlabel("Relative Visual Attention Gain (%)", fontsize=7.0, fontweight="bold", color="#1d242b", labelpad=1.8)
ax_a.grid(axis="x", linestyle="--", alpha=0.45, color="#cfd8e3", linewidth=0.55)
ax_a.legend(loc="upper center", bbox_to_anchor=(0.50, 1.125), ncol=2, frameon=True, fancybox=True,
            facecolor="#f8fafd", edgecolor="#cfd8e3", fontsize=5.7, handletextpad=0.2, columnspacing=0.55, borderpad=0.18)
ax_a.text(0.5, -0.175, "(a) Closed-loop visual recovery", transform=ax_a.transAxes,
          fontsize=7.5, fontweight="bold", color="#0f172a", ha="center", va="top")

# ==============================================================================
# Panel (b): Diverging Butterfly Plot
# ==============================================================================
ax_b.axhspan(-0.72, 0.44, color="#eef4fb", alpha=0.95, zorder=0)
ax_b.axhline(0.44, color="#8a96a3", linestyle="--", linewidth=0.75, zorder=1)
ax_b.axvline(0, color="#334155", linewidth=0.95, zorder=3)

bh = 0.35
ax_b.barh(y + bh/2, -prob_drop, height=bh, color="#3b82f6", alpha=0.85, edgecolor="white", linewidth=0.35, zorder=2, label="Hal. Prob. Drop")
ax_b.barh(y - bh/2, -rev_merged, height=bh, color="#0d9488", alpha=0.88, edgecolor="white", linewidth=0.35, zorder=2, label="Merged Rev.")
ax_b.scatter(-rev_collab, y - bh/2, s=16, color="#fef08a", edgecolor="#0f172a", linewidth=0.55, zorder=4, label="Collab. Rev.")

ax_b.barh(y + bh/2, gt_ikod, height=bh, color="#f59e0b", alpha=0.85, edgecolor="white", linewidth=0.35, zorder=2, label="Keep (IKOD)")
ax_b.barh(y - bh/2, gt_ikoda, height=bh, color="#ea580c", alpha=0.90, edgecolor="white", linewidth=0.35, zorder=2, label="Keep (IKOD-A)")

ax_b.text(-prob_drop[-1]  - 1.8, y[-1] + bh/2 + 0.03, f"{prob_drop[-1]:.1f}%",  va="center", ha="right", fontsize=5.3, fontweight="bold", color="#1d4ed8")
ax_b.text(-rev_merged[-1] - 1.8, y[-1] - bh/2 - 0.03, f"{rev_merged[-1]:.1f}%", va="center", ha="right", fontsize=5.3, fontweight="bold", color="#0f766e")
ax_b.text(gt_ikod[-1]     + 1.8, y[-1] + bh/2 + 0.03, f"{gt_ikod[-1]:.1f}%",    va="center", ha="left",  fontsize=5.3, fontweight="bold", color="#b45309")
ax_b.text(gt_ikoda[-1]    + 1.8, y[-1] - bh/2 - 0.03, f"{gt_ikoda[-1]:.1f}%",   va="center", ha="left",  fontsize=5.3, fontweight="bold", color="#c2410c")

ax_b.text(-54, n_m + 0.02, "← Hal. Suppression", fontsize=5.9, fontweight="bold", color="#0f766e", ha="center", va="center")
ax_b.text(54,  n_m + 0.02, "GT Preservation →", fontsize=5.9, fontweight="bold", color="#c2410c", ha="center", va="center")

ax_b.set_yticks(y)
ax_b.set_yticklabels([""] * n_m)
ax_b.set_xlim(-110, 118)
ax_b.set_xticks([-80, -40, 0, 40, 80])
ax_b.set_xticklabels(["80%", "40%", "0%", "40%", "80%"], fontsize=6.5)
ax_b.set_ylim(-0.74, n_m + 0.56)
ax_b.set_xlabel("Hal. Suppression  |  Grounded Keep (%)", fontsize=7.0, fontweight="bold", color="#1d242b", labelpad=1.8)
ax_b.grid(axis="x", linestyle="--", alpha=0.45, color="#cfd8e3", linewidth=0.55)
ax_b.legend(loc="upper center", bbox_to_anchor=(0.50, 1.135), ncol=3, frameon=True, fancybox=True,
            facecolor="#f8fafd", edgecolor="#cfd8e3", fontsize=5.25, handletextpad=0.16, columnspacing=0.32, handlelength=0.80, borderpad=0.18)
ax_b.text(0.5, -0.175, "(b) Step-level reversal vs. GT keep", transform=ax_b.transAxes,
          fontsize=7.5, fontweight="bold", color="#0f172a", ha="center", va="top")

# ==============================================================================
# Panel (c1): Top - Lexical Composition of Selected Anchors & Late Bucket Size
# ==============================================================================
anchor_types = ["High-Attn\n(10.2 tok)", "Uniform\n(2.6 tok)", "IKOD\n(3.7 tok)"]
ay = np.arange(3)[::-1]
func_pct  = np.array([53.2, 55.5, 55.6])
other_pct = np.array([40.75, 39.30, 40.17])
gt_pct    = np.array([6.05, 5.20, 4.23])

ax_c1.barh(ay, func_pct, color="#38bdf8", edgecolor="white", height=0.58, label="Func./Punct.")
ax_c1.barh(ay, other_pct, left=func_pct, color="#cbd5e1", edgecolor="white", height=0.58, label="Other")
ax_c1.barh(ay, gt_pct, left=func_pct + other_pct, color="#ef4444", edgecolor="white", height=0.58, label="GT Noun")

for i in range(3):
    fw = "bold" if i == 2 else "normal"
    ax_c1.text(func_pct[i]/2, ay[i], f"{func_pct[i]:.1f}%", va="center", ha="center", fontsize=5.8, fontweight=fw, color="#0f172a")
    ax_c1.text(101.5, ay[i], f"{gt_pct[i]:.1f}%", va="center", ha="left", fontsize=5.8, fontweight=fw, color="#b91c1c")

ax_c1.set_yticks(ay)
ax_c1.set_yticklabels(anchor_types, fontsize=6.2)
ax_c1.get_yticklabels()[-1].set_fontweight("bold")
ax_c1.set_xlim(0, 122)
ax_c1.set_xticks([0, 50, 100])
ax_c1.set_xticklabels(["0%", "50%", "100%"], fontsize=6.2)
ax_c1.set_ylim(-0.48, 2.52)
ax_c1.legend(loc="upper center", bbox_to_anchor=(0.50, 1.35), ncol=3, frameon=True, fancybox=True,
             facecolor="#f8fafd", edgecolor="#cfd8e3", fontsize=5.35, handletextpad=0.16, columnspacing=0.35, handlelength=0.80, borderpad=0.18)

# ==============================================================================
# Panel (c2): Bottom - Restored Grounded Value Cosine (%) vs. Hal. Gain
# ==============================================================================
val_cos = np.array([68.1, 79.3, 85.0, 76.7])
dS_hal  = np.array([5.38, 2.44, 5.68, 6.17])

ax_c2.axhspan(5.45, 6.85, xmin=0.35, xmax=1.0, color="#ecfdf5", alpha=0.85, zorder=0)

colors_c2 = ["#dc2626", "#64748b", "#d97706", "#059669"]
markers_c2 = ["X", "s", "^", "*"]
sizes_c2 = [42, 36, 40, 95]

for i in range(4):
    ax_c2.scatter(val_cos[i], dS_hal[i], s=sizes_c2[i], color=colors_c2[i], marker=markers_c2[i],
                  edgecolor="#0f172a" if i == 3 else "white", linewidth=0.7, zorder=4)

ax_c2.annotate("Prune-low\n" + r"($C_I$=9.2%)",
               xy=(68.1, 5.38), xytext=(66.2, 3.55),
               fontsize=5.6, color="#991b1b", ha="left",
               arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.6, shrinkA=1, shrinkB=2.5),
               bbox=dict(boxstyle="round,pad=0.12", facecolor="#fef2f2", edgecolor="#fca5a5", lw=0.5))

ax_c2.annotate("High-attn\n" + r"($C_I$=8.4%)",
               xy=(79.3, 2.44), xytext=(71.8, 2.35),
               fontsize=5.6, color="#334155", ha="left", va="center",
               arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.6, shrinkA=1, shrinkB=2.5))

ax_c2.annotate("Uniform\n" + r"($C_I$=8.4%)",
               xy=(85.0, 5.68), xytext=(82.2, 4.10),
               fontsize=5.6, color="#92400e", ha="center",
               arrowprops=dict(arrowstyle="->", color="#d97706", lw=0.6, shrinkA=1, shrinkB=2.5))

ax_c2.annotate("IKOD (Ours)\n" + r"$\mathbf{C_I=8.0\%}$",
               xy=(76.7, 6.17), xytext=(66.2, 6.08),
               fontsize=5.8, fontweight="bold", color="#065f46", ha="left", va="center",
               arrowprops=dict(arrowstyle="->", color="#059669", lw=0.75, shrinkA=1, shrinkB=3.5),
               bbox=dict(boxstyle="round,pad=0.14", facecolor="#d1fae5", edgecolor="#34d399", lw=0.6))

ax_c2.set_xlim(65.2, 87.8)
ax_c2.set_ylim(1.8, 6.85)
ax_c2.set_xticks([68, 74, 80, 86])
ax_c2.set_xticklabels(["68%", "74%", "80%", "86%"], fontsize=6.3)
ax_c2.set_yticks([2.5, 4.0, 5.5, 6.5])
ax_c2.set_yticklabels(["+2.5%", "+4.0%", "+5.5%", "+6.5%"], fontsize=6.3)
ax_c2.set_xlabel("Grounded Value Cosine (%)", fontsize=7.0, fontweight="bold", color="#1d242b", labelpad=1.8)
ax_c2.set_ylabel(r"Hal. Gain $\Delta S_t^{\mathrm{hal}}$", fontsize=6.8, fontweight="bold", color="#1d242b", labelpad=1.5)
ax_c2.grid(True, linestyle="--", alpha=0.45, color="#cfd8e3", linewidth=0.55)
ax_c2.text(0.45, -0.343, "(c) Inside KV merging buckets", transform=ax_c2.transAxes,
           fontsize=7.5, fontweight="bold", color="#0f172a", ha="center", va="top")

out_pdf = "./paper/figures/fig_mechanism_8b_main.pdf"
plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.015)
print("Saved", out_pdf)
