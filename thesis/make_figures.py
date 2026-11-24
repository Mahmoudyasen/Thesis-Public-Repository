"""Thesis figures that are not produced by an existing script, drawn directly from the result files (ml env).

  figures/det_froc.png         pooled ten-fold detector FROC with 95 % bootstrap band (own_candidates_all.csv)
  figures/det_folds_size.png   (a) test CPM per fold with 95 % CI, (b) detector recall by nodule diameter

Published systems are not plotted: their full FROC curves could not be traced to a verified source; their reported
CPMs are compared in Table "Pooled detector CPM against published LUNA16 detectors" instead.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from lungcad.eval.froc import evaluate                     # noqa: E402

D = Path(os.environ.get("LUNGCAD_DATA", "data"), "work", "wp0"); S1 = D / "stage1"
OUT = REPO / "thesis" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
FP = [0.125, 0.25, 0.5, 1, 2, 4, 8]; LAB = ["1/8", "1/4", "1/2", "1", "2", "4", "8"]
BLUE, GREY = "#2a5ea8", "#8a8a80"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---- pooled FROC
res = evaluate(pd.read_csv(S1 / "own_candidates" / "own_candidates_all.csv"), n_bootstrap=1000, seed=0)
s, lo, hi = np.array(res["sens_at_fp"]), np.array(res["sens_at_fp_ci95_low"]), np.array(res["sens_at_fp_ci95_high"])
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.log2(FP)
ax.fill_between(x, lo, hi, color=BLUE, alpha=.18, lw=0, label="95 % bootstrap interval")
ax.plot(x, s, "-o", color=BLUE, lw=2, ms=4, label=f"detector, 10 folds pooled (CPM {res['cpm']:.3f})")
for xi, si in zip(x, s): ax.annotate(f"{si:.3f}", (xi, si), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.set_xticks(x, LAB); ax.set_xlabel("false positives per scan"); ax.set_ylabel("sensitivity")
ax.set_ylim(0.65, 1.0); ax.grid(alpha=.25); ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "det_froc.png", dpi=200); plt.close(fig)

# ---- per fold + recall by size
folds = [json.load(open(S1 / f"fold{k}" / "seed0" / f"froc_fold{k}_test.json")) for k in range(10)]
h = pd.read_csv(D / "analysis" / "e5_misses" / "all_nodules_hit_status.csv")
bins = [3, 4, 5, 6, 8, 10, 15, np.inf]; names = ["3–4", "4–5", "5–6", "6–8", "8–10", "10–15", "≥15"]
h["bin"] = pd.cut(h.diameter_mm, bins, labels=names, right=False)
g = h.groupby("bin", observed=False).hit.agg(["sum", "count"])
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
c = np.array([f["cpm"] for f in folds]); e = np.array([f["cpm_ci95"] for f in folds]).T
ax[0].bar(range(10), c, color=BLUE, width=.65)
ax[0].errorbar(range(10), c, yerr=[c - e[0], e[1] - c], fmt="none", ecolor="black", capsize=3, lw=1)
ax[0].axhline(res["cpm"], color=GREY, ls="--", lw=1, label=f"pooled {res['cpm']:.3f}")
ax[0].set_xticks(range(10)); ax[0].set_xlabel("fold (test subset)"); ax[0].set_ylabel("test CPM"); ax[0].set_ylim(0.75, 1.0)
ax[0].legend(frameon=False, fontsize=8, loc="lower right"); ax[0].set_title("(a) test CPM per fold, 95 % CI", loc="left", fontsize=10)
r = g["sum"] / g["count"]
ax[1].bar(range(len(g)), r, color=BLUE, width=.65)
for i, (k, n) in enumerate(zip(g["sum"], g["count"])): ax[1].text(i, r.iloc[i] + .004, f"{k}/{n}", ha="center", fontsize=8)
ax[1].set_xticks(range(len(g)), names); ax[1].set_xlabel("nodule diameter (mm)"); ax[1].set_ylabel("recall (any score)")
ax[1].set_ylim(0.85, 1.01)
ax[1].set_title(f"(b) recall by size, {int(h.hit.sum())}/{len(h)} = {100 * h.hit.mean():.1f} %", loc="left", fontsize=10)
for a in ax: a.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(OUT / "det_folds_size.png", dpi=200); plt.close(fig)
print("wrote det_froc.png, det_folds_size.png;", f"pooled CPM {res['cpm']:.4f}, recall {h.hit.mean():.4f}")
