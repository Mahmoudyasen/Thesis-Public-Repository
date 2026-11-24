"""S6 thesis figures and summary, built from whatever S6 runs are scored so far.

    python report/s6_figures.py [--seed 0] [--n-boot 500] [--out report/figs_s6] [--model B] [--only fig3_score_dist fig4_confusion]

Partial results are allowed and labelled "n/10 folds": every comparison pools only the folds that ALL its series have
finished, so the series in one panel are always scored on the same scans. Final thesis numbers: rerun after S6 with
--n-boot 2000 (or more). Read-only: it never writes into the S6 run folders.

  fig1_froc.png            FROC curves: official A/B/C; own regime detector alone vs. stage 2 alone vs. fused A/B/C
  fig2_fp_at_sens.png      FPs/scan at 90 % and 95 % sensitivity, detector vs. fused models, % reduction (paired CI)
  fig3_score_dist.png      score distributions, true nodules vs. false positives: detector, stage 2, fused (+ AUC)
  fig4_confusion.png       nodule / false-positive confusion matrices at the 90 % and 95 % sensitivity operating points
  fig5_forest.png          paired ΔCPM with 95 % CIs: model vs. model, and every stage-2 score vs. the detector
  fig6_cpm_matrix.png      fold x model test-CPM matrix (both regimes, all score types)
  fig7_funnel.png          candidate funnel: candidates/scan and FPs/scan from the official list to the fused output
  fig8_fusion_folds.png    fusion weight chosen per fold, and validation vs. test ΔCPM per fold
  s6_summary.json          every number behind the figures
Counting follows the official LUNA16 rules (lungcad.eval.froc): a nodule is detected by its best-scoring hit, marks on
excluded findings are ignored, at most 100 marks per scan.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from lungcad import config as C
from lungcad.data.luna16_index import load_index, get_split
from lungcad.eval.froc import per_scan_results, froc_from_results
from lungcad.eval.bootstrap import paired_delta_cpm
from lungcad.eval.operating_points import paired_fp_reduction

MODELS = ("A", "B", "C")
FILES = {"stage2": "test_candidates.csv", "fused": "test_candidates_fused.csv", "lr": "test_candidates_lr.csv"}
DONE = {"stage2": "froc_test.json", "fused": "fusion.json", "lr": "fusion.json"}
COL = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eb6834", "det": "#52514e"}
NOD, FPC = "#2a78d6", "#eb6834"             # true nodule / false positive
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FP_TICKS = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def style(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTE)
    ax.tick_params(labelsize=8, colors=MUTE)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)


class S6:
    def __init__(self, root: Path, seed: int):
        self.root, self.seed, self.model = root, seed, None
        self.index = load_index(); self.split = {k: get_split(self.index, k) for k in range(10)}
        self.det_all = pd.read_csv(C.STAGE1 / "own_candidates" / "own_candidates_all.csv", dtype={"seriesuid": str})
        self._cache = {}

    def run(self, regime, model, k):
        return self.root / regime / model / f"fold{k}" / f"seed{self.seed}"

    def folds(self, regime, model, score) -> list[int]:
        return [k for k in range(10) if (self.run(regime, model, k) / DONE[score]).exists() and (self.run(regime, model, k) / FILES[score]).exists()]

    def uids(self, folds):
        return [u for k in folds for u in self.split[k]["test"]]

    def cands(self, regime, model, score, folds) -> pd.DataFrame:
        key = (regime, model, score, tuple(folds))
        if key not in self._cache:
            self._cache[key] = pd.concat([pd.read_csv(self.run(regime, model, k) / FILES[score], dtype={"seriesuid": str}) for k in folds],
                                         ignore_index=True)
        return self._cache[key]

    def detector(self, folds) -> pd.DataFrame:
        u = set(self.uids(folds)); d = self.det_all[self.det_all.seriesuid.isin(u)].copy()
        d["cand_id"] = d.index.values
        return d

    def series(self, name, folds):
        """'det' or (regime, model, score) -> candidate table on these folds."""
        return self.detector(folds) if name == "det" else self.cands(*name, folds)

    def common(self, names):
        f = set(range(10))
        for n in names:
            if n != "det": f &= set(self.folds(*n))
        return sorted(f)


def label(n):
    if n == "det": return "detector alone"
    regime, m, s = n
    return f"{m} " + {"stage2": "stage 2 alone", "fused": "fused", "lr": "fused (LR)"}[s] + ("" if regime == "own" else " (official)")


def threshold_at(results, target):
    """Score threshold where pooled sensitivity first reaches `target` (official counting) -> (thr, sens, fp/scan) or None."""
    n_nod = sum(r.n_nodules for r in results); tp = np.sort(np.concatenate([r.tp_probs for r in results]))[::-1]
    k = int(np.ceil(target * n_nod - 1e-9))
    if k == 0 or k > len(tp): return None
    thr = tp[k - 1]; fp = sum(int((r.fp_probs >= thr).sum()) for r in results)
    return float(thr), float((tp >= thr).sum() / n_nod), fp / len(results)


def fig_froc(s6, out, summ):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True); made = False
    for ax, (title, names) in zip(axes, [("Official candidates (classifier benchmark)", [("official", m, "stage2") for m in MODELS]),
                                         ("Own candidates (end-to-end system)", ["det", ("own", "C", "stage2")] + [("own", m, "fused") for m in MODELS])]):
        style(ax, "both"); names = [n for n in names if n == "det" or s6.folds(*n)]
        folds = s6.common(names)
        if not folds or not [n for n in names if n != "det"]:
            ax.text(0.5, 0.5, "no scored runs yet", ha="center", transform=ax.transAxes, color=MUTE); ax.set_title(title, loc="left", fontsize=9.5); continue
        uids = s6.uids(folds)
        for n in names:
            fps, sens, s_at, cpm = froc_from_results(per_scan_results(s6.series(n, folds), uids))
            c = COL["det"] if n == "det" else COL[n[1]]; ls = "--" if n == "det" else (":" if n[2] == "stage2" and n[0] == "own" else "-")
            ax.step(np.r_[1e-3, fps], np.r_[0, sens], where="post", color=c, ls=ls, lw=2 if ls == "-" else 1.5, label=f"{label(n)}  CPM {cpm:.3f}")
            summ.setdefault("froc", {})[f"{title[:3]}|{label(n)}"] = {"folds": folds, "cpm": cpm, "sens_at_fp": s_at.tolist()}
        ax.set_xscale("log", base=2); ax.set_xlim(1 / 16, 16); ax.set_xticks(FP_TICKS); ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
        ax.set_ylim(0.5, 1.0); ax.set_xlabel("false positives per scan", fontsize=8.5); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        ax.set_title(f"{title} · {len(folds)}/10 folds", loc="left", fontsize=9.5, color=INK); made = True
    axes[0].set_ylabel("sensitivity", fontsize=8.5)
    if made: fig.tight_layout(); fig.savefig(out / "fig1_froc.png", dpi=200)
    plt.close(fig); return made


def fig_fp(s6, out, summ, n_boot):
    names = [("own", m, "fused") for m in MODELS if s6.folds("own", m, "fused")]
    if not names: return False
    folds = s6.common(names); uids = s6.uids(folds); base = s6.detector(folds)
    res = {n: paired_fp_reduction(base, s6.series(n, folds), uids, n_boot=n_boot) for n in names}
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); style(ax); w = 0.8 / (len(names) + 1)
    for gi, t in enumerate(("90", "95")):
        x0 = gi; first = res[names[0]]
        ax.bar(x0, first[f"fp_base@{t}"], w * 0.92, color=COL["det"], label="detector alone" if gi == 0 else None)
        lo, hi = first[f"fp_base@{t}_ci95"]; ax.plot([x0, x0], [lo, hi], color=INK, lw=1)
        ax.text(x0, hi + 0.03, f"{first[f'fp_base@{t}']:.2f}", ha="center", fontsize=7.5, color=INK)
        for i, n in enumerate(names, 1):
            r = res[n]; x = x0 + i * w
            ax.bar(x, r[f"fp_model@{t}"], w * 0.92, color=COL[n[1]], label=label(n) if gi == 0 else None)
            lo, hi = r[f"fp_model@{t}_ci95"]; ax.plot([x, x], [lo, hi], color=INK, lw=1)
            red = r[f"fp_reduction@{t}"]
            ax.text(x, hi + 0.03, f"{r[f'fp_model@{t}']:.2f}\n{-red:+.0%}" if np.isfinite(red) else "n/a", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks([0 + w * len(names) / 2, 1 + w * len(names) / 2]); ax.set_xticklabels(["at 90 % sensitivity", "at 95 % sensitivity"], fontsize=9)
    ax.set_ylabel("false positives per scan", fontsize=8.5); ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title(f"False positives at fixed sensitivity, own regime · {len(folds)}/10 folds · whiskers: 95 % CI", loc="left", fontsize=9.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout(); fig.savefig(out / "fig2_fp_at_sens.png", dpi=200); plt.close(fig)
    summ["fp_at_sensitivity"] = {label(n): {"folds": folds} | {k: v for k, v in r.items()} for n, r in res.items()}
    return True


def merged_own(s6, model, folds):
    """Own-regime candidates of one model with p1 (detector), p2 (stage 2), pf (fused) and the candidate label."""
    s2 = s6.cands("own", model, "stage2", folds); fu = s6.cands("own", model, "fused", folds)
    assert (s2.cand_id.values == fu.cand_id.values).all()
    m = s2[["seriesuid", "coordX", "coordY", "coordZ", "cand_id", "label"]].copy()
    m["p1"] = s6.det_all.probability.values[s2.cand_id.values]; m["p2"] = s2.probability.values; m["pf"] = fu.probability.values
    return m


def pick_model(s6):
    if s6.model: return s6.model if s6.folds("own", s6.model, "fused") else None
    for m in ("C", "B", "A"):
        if s6.folds("own", m, "fused"): return m
    return None


def fig_dist(s6, out, summ):
    from sklearn.metrics import roc_auc_score
    m = pick_model(s6)
    if not m: return False
    folds = s6.folds("own", m, "fused"); d = merged_own(s6, m, folds)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4)); bins = np.linspace(0, 1, 26); summ["score_auc"] = {}
    for ax, (col, t) in zip(axes, [("p1", "detector p₁"), ("p2", f"stage 2 ({m}) p₂"), ("pf", f"fused ({m})")]):
        style(ax); y = d.label.values; p = d[col].values
        ax.hist(p[y == 0], bins, density=True, color=FPC, alpha=0.75, label=f"non-nodule candidates (n = {(y == 0).sum():,})")
        ax.hist(p[y == 1], bins, density=True, histtype="step", lw=2, color=NOD, label=f"true nodule candidates (n = {(y == 1).sum():,})")
        auc = roc_auc_score(y, p); summ["score_auc"][col] = auc
        ax.set_yscale("log"); ax.set_xlabel("score", fontsize=8.5); ax.set_title(f"{t} · AUC {auc:.3f}", loc="left", fontsize=9.5)
    axes[0].set_ylabel("density (log)", fontsize=8.5); axes[0].legend(fontsize=7, frameon=False, loc="upper center")
    fig.suptitle(f"Own regime, test candidates, {len(folds)}/10 folds: separation of true nodules and other candidates (incl. marks on excluded findings, which the FROC ignores)", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / "fig3_score_dist.png", dpi=200); plt.close(fig); return True


def fig_confusion(s6, out, summ):
    m = pick_model(s6)
    if not m: return False
    folds = s6.folds("own", m, "fused"); uids = s6.uids(folds)
    series = [("det", "detector alone"), (("own", m, "fused"), f"detector + stage 2 {m} (fused)")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6)); summ["confusion"] = {}
    for j, (n, name) in enumerate(series):
        res = per_scan_results(s6.series(n, folds), uids); n_nod = sum(r.n_nodules for r in res)
        fp_all = np.concatenate([r.fp_probs for r in res]); tp_all = np.concatenate([r.tp_probs for r in res])
        for i, target in enumerate((0.90, 0.95)):
            ax = axes[i, j]; op = threshold_at(res, target)
            if op is None:
                ax.axis("off"); ax.text(0.5, 0.5, f"{target:.0%} sensitivity not reachable", ha="center", transform=ax.transAxes); continue
            thr, sens, fps = op
            M = np.array([[int((tp_all >= thr).sum()), n_nod - int((tp_all >= thr).sum())],
                          [int((fp_all >= thr).sum()), int((fp_all < thr).sum())]])
            R = M / M.sum(1, keepdims=True)
            ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{M[a, b]:,}\n{R[a, b]:.1%}", ha="center", va="center", fontsize=9, color="white" if R[a, b] > 0.6 else INK)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["flagged", "not flagged"], fontsize=8); ax.set_yticks([0, 1])
            ax.set_yticklabels(["true nodule", "false-positive\ncandidate"], fontsize=8)
            ax.set_title(f"{name}\n{target:.0%} sens. point: threshold {thr:.3f}, {fps:.2f} FP/scan", fontsize=8.5, loc="left")
            summ["confusion"][f"{name}|{int(target * 100)}"] = {"threshold": thr, "sensitivity": sens, "fp_per_scan": fps, "matrix": M.tolist()}
    fig.suptitle(f"Own regime, {len(folds)}/10 folds, {len(uids)} test scans. Rows: truth; columns: score ≥ threshold. "
                 "Missed nodules include those the detector never proposed.", fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / "fig4_confusion.png", dpi=200); plt.close(fig); return True


def fig_forest(s6, out, summ, n_boot):
    pairs = []
    for regime, score in [("own", "fused"), ("official", "stage2"), ("own", "stage2")]:
        for a, b in [("C", "B"), ("B", "A"), ("C", "A")]:
            pairs.append(((regime, a, score), (regime, b, score)))
    for m in MODELS:
        for score in ("fused", "lr", "stage2"):
            pairs.append((("own", m, score), "det"))
    rows = []
    for a, b in pairs:
        if a != "det" and not s6.folds(*a) or b != "det" and not s6.folds(*b): continue
        folds = s6.common([a, b])
        if not folds: continue
        r = paired_delta_cpm(s6.series(a, folds), s6.series(b, folds), s6.uids(folds), n_boot=n_boot)
        rows.append((f"{label(a)} − {label(b)}" + ("" if a[0] == "official" else " (own)" if b != "det" else ""), r, len(folds), a))
    if not rows: return False
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.2)); style(ax, "x")
    for i, (name, r, nf, a) in enumerate(rows[::-1]):
        lo, hi = r["delta_ci95"]; c = COL[a[1]]
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round"); ax.plot([r["delta"]], [i], "o", color=c, ms=6, mec="white", mew=1)
        ax.text(1.02, i, f"{r['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]  {nf}/10", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=MUTE)
    ax.axvline(0, color=MUTE, lw=1); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7.5)
    ax.set_xlabel("paired ΔCPM on the pooled test scans (95 % bootstrap CI)", fontsize=8.5)
    ax.set_title("Paired comparisons (seed %d); a claim needs the CI to exclude 0 for every seed" % s6.seed, loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "fig5_forest.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    summ["paired"] = {name: {"folds": nf} | r for name, r, nf, _ in rows}
    return True


def fig_matrix(s6, out, summ):
    cols = [("official", m, "stage2") for m in MODELS] + ["det"] + [("own", m, s) for s in ("stage2", "fused") for m in MODELS]
    M = np.full((10, len(cols)), np.nan)
    for j, n in enumerate(cols):
        for k in range(10):
            if n == "det":
                M[k, j] = froc_from_results(per_scan_results(s6.detector([k]), s6.uids([k])))[3]
            else:
                f = s6.run(n[0], n[1], k) / ("froc_test.json" if n[2] == "stage2" else f"froc_test_{n[2]}.json")
                if f.exists(): M[k, j] = json.load(open(f))["cpm"]
    if np.isnan(np.delete(M, cols.index("det"), 1)).all(): return False
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(M, cmap="Blues", vmin=np.nanmin(M) - 0.02, vmax=np.nanmax(M), aspect="auto")
    for k in range(10):
        for j in range(len(cols)):
            v = M[k, j]; ax.text(j, k, "—" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                                 color=MUTE if np.isnan(v) else ("white" if v > np.nanmean(M) + 0.01 else INK))
    names = [("official\n" if n != "det" and n[0] == "official" else "own\n") + (label(n).replace(" (official)", "")) for n in cols]
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(names, fontsize=7.2); ax.set_yticks(range(10)); ax.set_yticklabels([f"fold {k}" for k in range(10)], fontsize=8)
    for x in (2.5, 3.5, 6.5): ax.axvline(x, color="white", lw=3)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).ax.tick_params(labelsize=7)
    ax.set_title(f"Test CPM per fold (seed {s6.seed}); each fold's test scans = one LUNA16 subset", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "fig6_cpm_matrix.png", dpi=200); plt.close(fig)
    summ["cpm_matrix"] = {"columns": [label(n) + (" (own)" if n != "det" and n[0] == "own" else "") for n in cols], "rows": M.tolist()}
    return True


def fig_funnel(s6, out, summ):
    m = pick_model(s6)
    if not m: return False
    folds = s6.folds("own", m, "fused"); uids = s6.uids(folds); ns = len(uids); steps = []
    off = [k for k in folds if (s6.run("official", m, k) / FILES["stage2"]).exists()]
    if off:
        o = s6.cands("official", m, "stage2", off); steps.append((f"official candidate list\n(candidates_V2, {len(off)}/10 folds)", len(o) / len(s6.uids(off)), None))
    det = s6.detector(folds); r_det = per_scan_results(det, uids)
    steps.append(("stage-1 detector\n(all proposals)", len(det) / ns, sum(len(r.fp_probs) for r in r_det) / ns))
    fu = per_scan_results(s6.cands("own", m, "fused", folds), uids)
    for t in (0.95, 0.90):
        op = threshold_at(fu, t)
        if op: steps.append((f"+ stage 2 {m}, fused\nat {t:.0%} sensitivity", sum(int((np.r_[r.tp_probs, r.fp_probs] >= op[0]).sum()) for r in fu) / ns, op[2]))
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(steps) + 1)); style(ax, "x")
    for i, (name, c, fp) in enumerate(steps[::-1]):
        ax.barh(i, c, 0.6, color=COL["det"] if "detector" in name else (COL[m] if "fused" in name else "#9ec5f4"))
        ax.text(c * 1.15, i, f"{c:,.1f} marks/scan" + (f" · {fp:.2f} FP/scan" if fp is not None else ""), va="center", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_yticks(range(len(steps))); ax.set_yticklabels([s[0] for s in steps[::-1]], fontsize=8)
    ax.set_xlim(0.1, max(s[1] for s in steps) * 20); ax.set_xlabel("marks per scan (log)", fontsize=8.5)
    ax.set_title(f"Candidate funnel, {len(folds)}/10 folds", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "fig7_funnel.png", dpi=200); plt.close(fig)
    summ["funnel"] = [{"step": s[0].replace("\n", " "), "marks_per_scan": s[1], "fp_per_scan": s[2]} for s in steps]
    return True


def fig_fusion(s6, out, summ):
    rows = []
    for m in MODELS:
        for k in s6.folds("own", m, "fused"):
            f = json.load(open(s6.run("own", m, k) / "fusion.json"))
            rows.append({"model": m, "fold": k, "w": f["w"], "val_dcpm": f["val"]["fused"]["cpm"] - f["val"]["detector"]["cpm"],
                         "test_dcpm": f["test"]["fused"]["delta_cpm"]})
    if not rows: return False
    d = pd.DataFrame(rows); fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    style(a); off = {"A": -0.012, "B": 0, "C": 0.012}
    for m, g in d.groupby("model"):
        a.plot(g.w + off[m], g.fold, "o", color=COL[m], ms=7, mec="white", mew=1, label=f"model {m}")
    a.set_xlim(-0.08, 1.08); a.set_yticks(range(10)); a.set_ylabel("fold", fontsize=8.5); a.invert_yaxis()
    a.set_xlabel("w chosen on that fold's validation (0 = detector, 1 = stage 2)", fontsize=8.5); a.legend(fontsize=7.5, frameon=False)
    a.set_title("(a) fusion weight per fold", loc="left", fontsize=9.5)
    style(b, "both")
    for m, g in d.groupby("model"):
        b.plot(g.val_dcpm, g.test_dcpm, "o", color=COL[m], ms=7, mec="white", mew=1, label=f"model {m}")
    lim = [min(d.val_dcpm.min(), d.test_dcpm.min(), 0) - 0.01, max(d.val_dcpm.max(), d.test_dcpm.max()) + 0.01]
    b.plot(lim, lim, color=GRID, lw=1); b.axhline(0, color=MUTE, lw=0.8); b.axvline(0, color=MUTE, lw=0.8)
    b.set_xlabel("validation ΔCPM (fused − detector)", fontsize=8.5); b.set_ylabel("test ΔCPM (fused − detector)", fontsize=8.5)
    b.set_title("(b) does the validation gain carry over to test?", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "fig8_fusion_folds.png", dpi=200); plt.close(fig)
    summ["fusion_per_fold"] = rows
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default=str(REPO / "report" / "figs_s6")); ap.add_argument("--root", default=str(C.STAGE2))
    ap.add_argument("--model", choices=list("ABC"), default=None, help="model shown in fig3/fig4 (default: first of C, B, A with fused runs)")
    ap.add_argument("--only", nargs="+", default=None, help="build only these figures; s6_summary.json is updated, not replaced")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s6 = S6(Path(a.root), a.seed); s6.model = a.model; summ = {"seed": a.seed, "n_boot": a.n_boot}
    if a.only and (out / "s6_summary.json").exists():
        summ = {**json.load(open(out / "s6_summary.json")), **summ}
    for name, f in [("fig1_froc", lambda: fig_froc(s6, out, summ)), ("fig2_fp_at_sens", lambda: fig_fp(s6, out, summ, a.n_boot)),
                    ("fig3_score_dist", lambda: fig_dist(s6, out, summ)), ("fig4_confusion", lambda: fig_confusion(s6, out, summ)),
                    ("fig5_forest", lambda: fig_forest(s6, out, summ, a.n_boot)), ("fig6_cpm_matrix", lambda: fig_matrix(s6, out, summ)),
                    ("fig7_funnel", lambda: fig_funnel(s6, out, summ)), ("fig8_fusion_folds", lambda: fig_fusion(s6, out, summ))]:
        if a.only and name not in a.only: continue
        print(f"{name}: {'made' if f() else 'skipped (no data yet)'}")
    json.dump(summ, open(out / "s6_summary.json", "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
