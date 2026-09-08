#!/usr/bin/env python3
"""Train an axis-aligned tree and a soft-routed oblique tree on a diagonal-stripe
dataset, and plot their decision surfaces to justify the FADO base learner."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "/Users/tiagosilva/Documents/Aranyani2.0/DOCS/presentation/assets/oblique_demo.pdf"
PNG = "/private/tmp/claude-501/-Users-tiagosilva-Documents-Aranyani2-0/55d6d5df-c507-417b-a93c-07d0a53cb8a5/scratchpad/oblique_demo.png"

RED  = "#DA0725"
GREY = "#5B6470"
LRED = "#F6C9CE"
LGRY = "#DCE1E7"

rng = np.random.default_rng(1)

# ---------- dataset: a diagonal stripe (class 1 inside |x1-x2|<c) ----------
N = 420
X = rng.uniform(-3, 3, size=(N, 2))
u = (X[:, 0] - X[:, 1]) / np.sqrt(2.0)          # oblique coordinate
y = (np.abs(u) < 1.15).astype(float)
flip = rng.random(N) < 0.05                      # 5% label noise
y[flip] = 1 - y[flip]

# ================= axis-aligned greedy tree (Gini) =================
def gini(yv):
    if len(yv) == 0:
        return 0.0
    p = yv.mean()
    return 1.0 - p * p - (1 - p) ** 2

class Node:
    __slots__ = ("f", "t", "l", "r", "v")

def build(Xa, ya, depth, min_leaf=8):
    nd = Node(); nd.f = nd.t = nd.l = nd.r = None
    if depth == 0 or len(ya) < 2 * min_leaf or ya.mean() in (0.0, 1.0):
        nd.v = ya.mean() if len(ya) else 0.5
        return nd
    best = (gini(ya), None, None)
    base_n = len(ya)
    for f in (0, 1):
        thresholds = np.quantile(Xa[:, f], np.linspace(0.1, 0.9, 17))
        for t in thresholds:
            mask = Xa[:, f] <= t
            if mask.sum() < min_leaf or (~mask).sum() < min_leaf:
                continue
            g = (mask.sum() * gini(ya[mask]) + (~mask).sum() * gini(ya[~mask])) / base_n
            if g < best[0] - 1e-9:
                best = (g, f, t)
    if best[1] is None:
        nd.v = ya.mean(); return nd
    _, f, t = best
    nd.f, nd.t = f, t
    m = Xa[:, f] <= t
    nd.l = build(Xa[m], ya[m], depth - 1, min_leaf)
    nd.r = build(Xa[~m], ya[~m], depth - 1, min_leaf)
    nd.v = ya.mean()
    return nd

def predict_tree(nd, P):
    out = np.empty(len(P))
    for i, p in enumerate(P):
        n = nd
        while n.f is not None:
            n = n.l if p[n.f] <= n.t else n.r
        out[i] = n.v
    return out

tree = build(X, y, depth=3)
acc_axis = ((predict_tree(tree, X) > 0.5) == y).mean()

# ================= soft-routed oblique tree (depth 2, analytic backprop) =====
mu, sd = X.mean(0), X.std(0)
Xs = (X - mu) / sd

def forward(params, Z):
    W, b, th = params["W"], params["b"], params["th"]
    a = Z @ W.T + b                       # (N,3) pre-activations for nodes 0,1,2
    g = 1.0 / (1.0 + np.exp(-a))          # gates (prob go right)
    g0, g1, g2 = g[:, 0], g[:, 1], g[:, 2]
    P = np.stack([(1 - g0) * (1 - g1),    # leaf LL
                  (1 - g0) * g1,          # leaf LR
                  g0 * (1 - g2),          # leaf RL
                  g0 * g2], axis=1)       # leaf RR
    s = 1.0 / (1.0 + np.exp(-th))         # leaf class-1 probs
    p = P @ s
    return p, P, g, s

def loss_grad(params, Z, yv):
    p, P, g, s = forward(params, Z)
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    L = -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
    dLdp = (p - yv) / (p * (1 - p)) / len(yv)
    g0, g1, g2 = g[:, 0], g[:, 1], g[:, 2]
    Lleft  = s[0] * (1 - g1) + s[1] * g1
    Lright = s[2] * (1 - g2) + s[3] * g2
    dpda0 = (Lright - Lleft) * g0 * (1 - g0)
    dpda1 = (1 - g0) * (s[1] - s[0]) * g1 * (1 - g1)
    dpda2 = g0 * (s[3] - s[2]) * g2 * (1 - g2)
    dA = np.stack([dpda0, dpda1, dpda2], axis=1) * dLdp[:, None]   # (N,3)
    gW = dA.T @ Z
    gb = dA.sum(0)
    gth = (P * (s * (1 - s))[None, :] * dLdp[:, None]).sum(0)
    return L, {"W": gW, "b": gb, "th": gth}

# init
params = {"W": rng.normal(0, 0.6, (3, 2)), "b": rng.normal(0, 0.3, 3),
          "th": rng.normal(0, 0.5, 4)}

# gradient check vs finite differences
def numgrad(params, Z, yv, key, eps=1e-5):
    base = params[key].copy(); g = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        params[key][idx] = base[idx] + eps; lp = loss_grad(params, Z, yv)[0]
        params[key][idx] = base[idx] - eps; lm = loss_grad(params, Z, yv)[0]
        g[idx] = (lp - lm) / (2 * eps); params[key][idx] = base[idx]
    return g
_, ga = loss_grad(params, Xs, y)
maxerr = max(np.abs(ga[k] - numgrad(params, Xs, y, k)).max() for k in params)
print(f"grad-check max abs error = {maxerr:.2e}")

# train (momentum GD)
vel = {k: np.zeros_like(v) for k, v in params.items()}
lr, mom = 0.5, 0.9
for it in range(4000):
    L, gr = loss_grad(params, Xs, y)
    for k in params:
        vel[k] = mom * vel[k] - lr * gr[k]
        params[k] += vel[k]
    if it % 1000 == 0:
        print(f"  step {it:4d}  loss {L:.4f}")
p_final, _, _, _ = forward(params, Xs)
acc_soft = ((p_final > 0.5) == y).mean()
print(f"axis-aligned acc={acc_axis:.3f}  soft-oblique acc={acc_soft:.3f}")

# ================= plot =================
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "svg.fonttype": "none"})
gx, gy = np.meshgrid(np.linspace(-3.2, 3.2, 300), np.linspace(-3.2, 3.2, 300))
grid = np.c_[gx.ravel(), gy.ravel()]
cmap = LinearSegmentedColormap.from_list("cg", [LGRY, "#FFFFFF", LRED])

fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.35))

def scatter(a):
    a.scatter(X[y == 0, 0], X[y == 0, 1], s=11, c=GREY, edgecolors="none", alpha=.9)
    a.scatter(X[y == 1, 0], X[y == 1, 1], s=11, c=RED, edgecolors="none", alpha=.9)
    a.set_xlim(-3.2, 3.2); a.set_ylim(-3.2, 3.2)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor("#B9C0C8")

# panel A: axis-aligned
za = predict_tree(tree, grid).reshape(gx.shape)
ax[0].contourf(gx, gy, za, levels=[-.1, .5, 1.1], colors=[LGRY, LRED], alpha=.55)
ax[0].contour(gx, gy, za, levels=[.5], colors=[GREY], linewidths=1.1)
scatter(ax[0])
ax[0].set_title("Axis-aligned tree (depth 3)", fontsize=13, color="#141414", pad=8)
ax[0].text(-3.05, 2.72, f"acc {acc_axis:.2f}", fontsize=11, color=GREY)

# panel B: soft oblique
zb = forward(params, (grid - mu) / sd)[0].reshape(gx.shape)
cf = ax[1].contourf(gx, gy, zb, levels=np.linspace(0, 1, 21), cmap=cmap, alpha=.9)
ax[1].contour(gx, gy, zb, levels=[.5], colors=["#141414"], linewidths=1.2)
# draw the 3 learned oblique split lines  w.x_std + b = 0  ->  in original coords
W, b = params["W"], params["b"]
xx = np.linspace(-3.2, 3.2, 10)
for j in range(3):
    w0, w1 = W[j] / sd                       # de-standardise slope
    c = b[j] - (W[j] * mu / sd).sum()
    if abs(w1) > 1e-6:
        ax[1].plot(xx, -(w0 * xx + c) / w1, ls="--", lw=1.0, color="#141414", alpha=.45)
scatter(ax[1])
ax[1].set_title("Soft-routed oblique tree (depth 2)", fontsize=13, color="#141414", pad=8)
ax[1].text(-3.05, 2.72, f"acc {acc_soft:.2f}", fontsize=11, color="#141414")

plt.tight_layout(pad=0.6)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(PNG, dpi=150, bbox_inches="tight")
print("saved", OUT)
