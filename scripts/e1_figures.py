"""E1 figures from e1_results.csv / e1_drop_vs_clean.csv -> report/e1_cpm_curves.png, report/e1_delta_forest.png."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C

E1 = C.WORK / "wp0" / "analysis" / "e1_dose_fold0"; REP = Path(__file__).resolve().parents[1] / "report"
COL = {"S3": "#2a78d6", "S4": "#eb6834"}; LAB = {"S3": "S3 (no dose aug.)", "S4": "S4 (dose + kernel aug.)"}
INK, MUTE = "#0b0b0b", "#52514e"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.25); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=MUTE)


def main():
    r = pd.read_csv(E1 / "e1_results.csv"); m = r[r.model.isin(["S3", "S4"])].set_index(["model", "condition"])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1, 0.8, 1.15]})
    # (a) noise sweep
    ax = axs[0]; sig = [0, 20, 40, 60, 80, 100]; names = ["clean"] + [f"noise{s}" for s in sig[1:]]
    ax.axvspan(10, 60, color="#f0efec", zorder=0); ax.text(35, 0.975, "S4 training range", ha="center", fontsize=7.5, color=MUTE)
    for k, off in (("S3", -1.2), ("S4", 1.2)):
        y = [m.loc[(k, n), "cpm"] for n in names]; lo = [m.loc[(k, n), "ci_lo"] for n in names]; hi = [m.loc[(k, n), "ci_hi"] for n in names]
        x = np.array(sig) + off
        ax.fill_between(x, lo, hi, color=COL[k], alpha=.12, lw=0); ax.plot(x, y, "-o", color=COL[k], lw=2, ms=6, mec="white", mew=1.2, label=LAB[k])
    ax.set_xticks(sig); ax.set_xlabel("added noise σ (HU)", fontsize=9); ax.set_ylabel("fold-0 test CPM", fontsize=9)
    ax.set_title("a  White (voxel-independent) noise", fontsize=10, loc="left"); ax.set_ylim(0.4, 1.0); style(ax); ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    # (b) correlated noise
    ax = axs[1]; cs = [0, 40, 80]
    for k, off in (("S3", -1.2), ("S4", 1.2)):
        for corr, ls, mk in (("c1", "--", "s"), ("c2", "-", "o")):
            names = ["clean", f"cnoise40_{corr}", f"cnoise80_{corr}"]
            if not all((k, n) in m.index for n in names):
                continue
            y = [m.loc[(k, n), "cpm"] for n in names]; lo = [m.loc[(k, n), "ci_lo"] for n in names]; hi = [m.loc[(k, n), "ci_hi"] for n in names]
            x = np.array(cs) + off
            if corr == "c2":
                ax.fill_between(x, lo, hi, color=COL[k], alpha=.12, lw=0)
            ax.plot(x, y, ls, marker=mk, color=COL[k], lw=2 if corr == "c2" else 1.3, ms=6, mec="white", mew=1.2, label=f"{k}, grain {'1.4' if corr == 'c2' else '0.7'} mm")
    ax.set_xticks(cs); ax.set_xlabel("added correlated noise σ (HU)", fontsize=9); ax.set_ylim(0.4, 1.0)
    ax.set_title("b  Correlated noise", fontsize=10, loc="left"); style(ax); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    # (c) kernel sweep
    ax = axs[2]; kn = ["soft1.5", "soft1.0", "soft0.5", "clean", "sharp0.5", "sharp1.0", "sharp1.5"]
    kl = ["soft\n1.5", "soft\n1.0", "soft\n0.5", "clean", "sharp\n0.5", "sharp\n1.0", "sharp\n1.5"]; xx = np.arange(len(kn))
    ax.axvspan(0.5, 2.5, color="#f0efec", zorder=0); ax.axvspan(3.5, 5.5, color="#f0efec", zorder=0)
    ax.text(1.5, 0.975, "S4 training range", ha="center", fontsize=7.5, color=MUTE); ax.text(4.5, 0.975, "S4 training range", ha="center", fontsize=7.5, color=MUTE)
    for k, off in (("S3", -0.06), ("S4", 0.06)):
        y = [m.loc[(k, n), "cpm"] for n in kn]; lo = [m.loc[(k, n), "ci_lo"] for n in kn]; hi = [m.loc[(k, n), "ci_hi"] for n in kn]
        ax.fill_between(xx + off, lo, hi, color=COL[k], alpha=.12, lw=0); ax.plot(xx + off, y, "-o", color=COL[k], lw=2, ms=6, mec="white", mew=1.2, label=LAB[k])
    ax.set_xticks(xx); ax.set_xticklabels(kl, fontsize=8); ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("reconstruction kernel: Gaussian blur σ (voxels) / unsharp-mask amount", fontsize=9)
    ax.set_title("c  Kernel change (softer ← → sharper)", fontsize=10, loc="left"); style(ax)
    fig.tight_layout(); fig.savefig(REP / "e1_cpm_curves.png", dpi=150); plt.close(fig)

    # forest: ΔCPM S4 - S3 (paired) per condition
    d = r[r.model == "S4-S3"].reset_index(drop=True); order = ["clean", "noise20", "noise40", "noise60", "noise80", "noise100", "soft0.5", "soft1.0", "soft1.5",
                                                                "sharp0.5", "sharp1.0", "sharp1.5", "sharp1.0+noise60", "soft1.0+noise60",
                                                                "cnoise40_c1", "cnoise80_c1", "cnoise40_c2", "cnoise80_c2"]
    order = [o for o in order if o in set(d.condition)]
    d = d.set_index("condition").loc[order]; y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(order) + 0.9)); ax.axvline(0, color=MUTE, lw=1)
    sigc = (d.ci_lo > 0) | (d.ci_hi < 0)
    for yi, (c, row) in zip(y, d.iterrows()):
        col = "#eb6834" if row.ci_lo > 0 else ("#2a78d6" if row.ci_hi < 0 else "#8a8984")
        ax.plot([row.ci_lo, row.ci_hi], [yi, yi], color=col, lw=2, solid_capstyle="round"); ax.plot(row.cpm, yi, "o", color=col, ms=7, mec="white", mew=1.2)
        ax.text(1.01, yi, f"{row.cpm:+.3f}  [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]", va="center", fontsize=7.8, color=INK, transform=ax.get_yaxis_transform(), family="DejaVu Sans")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8.5); ax.set_xlabel("ΔCPM = S4 − S3 (paired bootstrap 95 % CI, 2,000 resamples)", fontsize=9)
    ax.set_xlim(min(d.ci_lo) - 0.02, max(d.ci_hi) + 0.02); style(ax); ax.grid(axis="y", alpha=0)
    ax.set_title("S4 better →  (orange = CI excludes 0 in S4's favour)", fontsize=9, loc="right", color=MUTE)
    fig.tight_layout(rect=(0, 0, 0.76, 1)); fig.savefig(REP / "e1_delta_forest.png", dpi=150); plt.close(fig)
    print("significant:", list(d.index[sigc]))


if __name__ == "__main__":
    main()
