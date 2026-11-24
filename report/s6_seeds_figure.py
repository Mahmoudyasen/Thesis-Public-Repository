"""S6 across seeds: pooled CPM, paired model contrasts and FPs/scan per seed, from 08_results_tables.py output.

    python report/s6_seeds_figure.py --tables <dir with paired_bootstrap.json + fp_reduction.json> [--out report/figs_s6/fig9_seeds.png]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s6_figures import COL, INK, MUTE, style, REPO

MARK = {0: "o", 1: "s", 2: "D"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default=str(REPO / "report" / "figs_s6" / "fig9_seeds.png"))
    a = ap.parse_args(); t = Path(a.tables)
    paired = json.load(open(t / "paired_bootstrap.json")); fpr = json.load(open(t / "fp_reduction.json"))
    seeds = sorted({r["seed"] for r in paired}); dx = {s: (i - (len(seeds) - 1) / 2) * 0.22 for i, s in enumerate(seeds)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.25, 1.2, 1]})

    # (a) pooled CPM per system and seed
    ax = axes[0]; style(ax)
    systems = [("official", "stage2", m) for m in "ABC"] + [("own", "fused", m) for m in "ABC"]
    for x, (reg, sc, m) in enumerate(systems):
        for s in seeds:
            r = next((r for r in paired if r["seed"] == s and r["regime"] == reg and r["score"] == sc and (r["a"] == m or r["b"] == m)), None)
            if r is None: continue
            cpm, ci = (r["cpm_a"], r["cpm_a_ci95"]) if r["a"] == m else (r["cpm_b"], r["cpm_b_ci95"])
            ax.errorbar(x + dx[s], cpm, yerr=[[cpm - ci[0]], [ci[1] - cpm]], fmt=MARK[s], color=COL[m], ms=5, capsize=0, lw=1.2,
                        mfc=COL[m] if s == 0 else "white")
    det = next(r for r in fpr if r["seed"] == seeds[0])["cpm_base"]
    ax.axhline(det, color=COL["det"], ls="--", lw=1); ax.text(-0.4, det + 0.0015, f"detector alone {det:.3f}", ha="left", fontsize=7.5, color=MUTE)
    ax.set_xticks(range(len(systems))); ax.set_xticklabels([f"{m}\n{'official' if reg == 'official' else 'own fused'}" for reg, sc, m in systems], fontsize=8)
    ax.set_ylabel("pooled test CPM (95 % CI)", fontsize=8.5); ax.set_title("(a) pooled CPM per seed (888 scans)", loc="left", fontsize=9.5)
    ax.legend(handles=[plt.Line2D([], [], marker=MARK[s], ls="", color=MUTE, mfc=MUTE if s == 0 else "white", label=f"seed {s}") for s in seeds],
              fontsize=7.5, frameon=False, loc="lower right")

    # (b) paired contrasts per seed
    ax = axes[1]; style(ax, grid="x")
    rows = [("own", "fused", "C", "B"), ("own", "fused", "B", "A"), ("own", "stage2", "C", "B"), ("own", "stage2", "B", "A"),
            ("official", "stage2", "C", "B"), ("official", "stage2", "B", "A")]
    for y, (reg, sc, ma, mb) in enumerate(rows):
        for s in seeds:
            r = next((r for r in paired if r["seed"] == s and r["regime"] == reg and r["score"] == sc and r["a"] == ma and r["b"] == mb), None)
            if r is None: continue
            d, (lo, hi) = r["delta"], r["delta_ci95"]
            ax.errorbar(d, y - dx[s], xerr=[[d - lo], [hi - d]], fmt=MARK[s], color=COL[ma], ms=5, lw=1.2, mfc=COL[ma] if s == 0 else "white")
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(range(len(rows))); ax.invert_yaxis()
    ax.set_yticklabels([f"{ma} − {mb}  ({'own, fused' if sc == 'fused' else 'own, stage 2 alone' if reg == 'own' else 'official'})" for reg, sc, ma, mb in rows], fontsize=8)
    ax.set_xlabel("paired ΔCPM (95 % CI)", fontsize=8.5); ax.set_title("(b) model contrasts per seed", loc="left", fontsize=9.5)

    # (c) FPs/scan at 90 / 95 % sensitivity per seed, fused models
    ax = axes[2]; style(ax)
    for j, tgt in enumerate((90, 95)):
        base = next(r for r in fpr if r["seed"] == seeds[0])[f"fp_base@{tgt}"]
        ax.plot([j * 4 - 0.4, j * 4 + 0.4], [base, base], color=COL["det"], lw=2.5)
        for k, m in enumerate("ABC"):
            for s in seeds:
                r = next((r for r in fpr if r["seed"] == s and r["model"] == m and r["score"] == "fused"), None)
                if r is None: continue
                v, ci = r[f"fp_model@{tgt}"], r[f"fp_model@{tgt}_ci95"]
                ax.errorbar(j * 4 + 1 + k + dx[s], v, yerr=[[v - ci[0]], [ci[1] - v]], fmt=MARK[s], color=COL[m], ms=5, lw=1.2, mfc=COL[m] if s == 0 else "white")
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7]); ax.set_xticklabels(["det", "A", "B", "C", "det", "A", "B", "C"], fontsize=8)
    ax.text(1.5, -0.2, "at 90 % sensitivity", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=MUTE)
    ax.text(5.5, -0.2, "at 95 % sensitivity", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=MUTE)
    ax.set_ylabel("false positives per scan (95 % CI)", fontsize=8.5); ax.set_title("(c) FPs/scan, own regime, fused", loc="left", fontsize=9.5)

    fig.suptitle(f"S6 across seeds ({', '.join(map(str, seeds))}); filled = seed {seeds[0]}, open = later seeds", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=200); plt.close(fig); print("wrote", a.out)


if __name__ == "__main__":
    main()
