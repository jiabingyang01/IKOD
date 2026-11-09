"""Adaptive collaborative-weight alpha for IKOD (core novelty P1/P2/P3).

Given quantities IKOD already computes each step:
  S_t  = head/layer-averaged image attention of the last (query) token (full cache),
  p_full, p_merged = next-token distributions from the full and merged-cache forwards,
  c_t  = top-1 probability of p_full (model confidence),
we derive a per-step collaborative weight alpha_t with NO per-task tuning.

Modes
-----
fixed : alpha_t = alpha_max (the paper's hand-tuned scalar; baseline/parent).
p1    : degradation-gated  alpha_t = a_min + (a_max-a_min)*sigmoid(kappa*(D_t - D_med)),
        D_t = max(0,(S_0 - S_t)/S_0),  D_med = running median of D over the sequence.
p2    : divergence-weighted alpha_t = a_max * Δ_t/(Δ_t + tau),  Δ_t = JSD(p_full||p_merged).
p1p3  : p1 multiplied by confidence gate (1 - c_t)  [calibration-safeguard ablation].
p1p2  : geometric-ish combine of p1 schedule with the p2 divergence gate.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence (base 2, in [0,1]) between two prob vectors (last dim)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(-1)
    kl_qm = (q * (q / m).log()).sum(-1)
    return (0.5 * (kl_pm + kl_qm) / math.log(2)).clamp(0.0, 1.0)


def compute_image_attention(attentions, image_start: int, image_end: int,
                            layers=None) -> torch.Tensor:
    """S_t: mean over (selected layers, heads) of the last query row's attention mass
    on the image span. `attentions` = tuple/list over layers of [b, h, q, kv].
    Returns a scalar tensor per batch item averaged to a float (batch=1 typical)."""
    vals = []
    L = len(attentions)
    sel = range(L) if layers is None else layers
    for i in sel:
        a = attentions[i]                      # [b, h, q, kv]
        last = a[:, :, -1, :]                  # [b, h, kv]
        img = last[:, :, image_start:image_end].sum(-1)   # [b, h]
        vals.append(img.mean(-1))              # [b]  mean over heads
    s = torch.stack(vals, 0).mean(0)           # [b]  mean over layers
    return s


class AdaptiveAlpha:
    def __init__(self, mode: str = "fixed", alpha_min: float = 0.0,
                 alpha_max: float = 2.0, kappa: float = 10.0, tau: float = 0.1,
                 layers=None, conf_power: float = 1.0, tau_len: float = 16.0):
        self.mode = mode
        self.a_min = float(alpha_min)
        self.a_max = float(alpha_max)
        self.kappa = float(kappa)
        self.tau = float(tau)
        self.tau_len = float(tau_len)
        self.layers = layers
        self.conf_power = float(conf_power)
        self.reset()

    def reset(self):
        self.S0 = None          # image attention at the query predicting the first generated token
        self._D_hist = []       # history of degradation D_t for running median
        self._t = 0             # decode step index (for the length-only schedule baseline)

    def _running_median(self, x: float) -> float:
        self._D_hist.append(x)
        h = sorted(self._D_hist)
        n = len(h)
        return h[n // 2] if n % 2 else 0.5 * (h[n // 2 - 1] + h[n // 2])

    def step(self, S_t: float | None = None, p_full: torch.Tensor | None = None,
             p_merged: torch.Tensor | None = None, c_t: float | None = None) -> float:
        """Return alpha_t for the current decoding step."""
        self._t += 1
        if self.mode == "fixed":
            return self.a_max
        # length-only schedule (ablation baseline): alpha grows with POSITION, ignoring the measured
        # degradation S_t. This is the M3ID-style fixed length ramp, used to isolate whether the
        # degradation gate (p1) beats a trivial position schedule.
        if self.mode == "length":
            ramp = self._t / (self._t + self.tau_len)
            return self.a_min + (self.a_max - self.a_min) * ramp

        # --- degradation D_t (needs S_t) ---
        alpha_p1 = self.a_max
        if S_t is not None:
            if self.S0 is None:
                self.S0 = max(float(S_t), 1e-6)
            D_t = max(0.0, (self.S0 - float(S_t)) / self.S0)   # in [0,1]
            D_med = self._running_median(D_t)
            sig = 1.0 / (1.0 + math.exp(-self.kappa * (D_t - D_med)))
            alpha_p1 = self.a_min + (self.a_max - self.a_min) * sig

        # --- divergence gate Δ_t (needs both distributions) ---
        alpha_p2 = self.a_max
        if p_full is not None and p_merged is not None:
            delta = float(jsd(p_full, p_merged).mean())
            alpha_p2 = self.a_max * delta / (delta + self.tau)

        if self.mode == "p1":
            alpha = alpha_p1
        elif self.mode == "p2":
            alpha = alpha_p2
        elif self.mode == "p1p3":
            gate = (1.0 - float(c_t)) ** self.conf_power if c_t is not None else 1.0
            alpha = alpha_p1 * gate
        elif self.mode == "p1p2":
            # p1 sets the schedule; p2 gates by genuine disagreement
            gate = (alpha_p2 / self.a_max) if self.a_max > 0 else 1.0
            alpha = alpha_p1 * gate
        else:
            raise ValueError(f"unknown adaptive mode {self.mode}")
        return float(max(0.0, alpha))


if __name__ == "__main__":
    # self-test (no model needed)
    V = 100
    p = F.softmax(torch.randn(1, V), -1)
    q = F.softmax(torch.randn(1, V), -1)
    assert 0.0 <= float(jsd(p, p)) < 1e-5
    assert 0.0 <= float(jsd(p, q)) <= 1.0
    # synthetic attentions: 4 layers, 8 heads, q=3, kv=20, image span [2:10)
    att = [torch.softmax(torch.randn(1, 8, 3, 20), -1) for _ in range(4)]
    S = compute_image_attention(att, 2, 10)
    assert S.shape == (1,)
    aa = AdaptiveAlpha(mode="p1", alpha_min=0.0, alpha_max=2.0, kappa=10.0)
    # simulate degrading image attention -> alpha should rise then settle around schedule
    seq = [0.5, 0.45, 0.4, 0.3, 0.2, 0.15]
    out = [round(aa.step(S_t=s), 3) for s in seq]
    assert all(0.0 <= a <= 2.0 for a in out)
    aa2 = AdaptiveAlpha(mode="fixed", alpha_max=2.0)
    assert aa2.step(S_t=0.1) == 2.0
    aa3 = AdaptiveAlpha(mode="p1p3", alpha_min=0.0, alpha_max=2.0)
    a_lowconf = aa3.step(S_t=0.2, c_t=0.1)
    aa3.reset()
    a_hiconf = aa3.step(S_t=0.2, c_t=0.95)
    assert a_hiconf <= a_lowconf + 1e-6   # high confidence -> smaller alpha
    print("adaptive self-test OK; p1 alpha trajectory:", out)
