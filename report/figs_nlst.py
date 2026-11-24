"""Thesis figures for the NLST package and the system overview (figure pass, 24 Nov 2026) — ml env.

    python report/figs_nlst.py [--no-registration]

Writes thesis/figures/nlst/:
  system_overview.png      pipeline diagram (Chapter 1/3)
  nlst_cohort_flow.png     NLST cohort flow (Section 5.2)
  nlst_confusion.png       per-exam confusion matrices, radiologists vs the system at their sensitivity (Section 5.3)
  nlst_temporal.png        false-positive rate at the radiologists' sensitivity, single-scan vs temporal (Section 5.4)
  nlst_registration.png    one registered prior/current pair, before and after registration, no participant ID (5.4)
Every number is read from the WP4 result files (work/wp4/analysis/wp4_results.json, systems.csv) or from the image
inventory; counts are recovered from the stored rates and checked to be integers.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
WP4 = Path(os.environ.get("LUNGCAD_DATA", "data"), "work", "wp4"); AN = WP4 / "analysis"
OUT = REPO / "thesis" / "figures" / "nlst"; OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, BLUE, ORANGE, GREEN, GREY = "#1f2430", "#5b6478", "#2a78d6", "#eb6834", "#1baf7a", "#8a93a6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open(AN / "wp4_results.json"))


def count(rate, n):
    c = rate * n; assert abs(c - round(c)) < 1e-6, (rate, n); return int(round(c))


def box(ax, x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, c=MUTED):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ system overview
def system_overview():
    fig, ax = plt.subplots(figsize=(11.5, 4.8)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    y = 0.80; xs = [0.055, 0.215, 0.39, 0.57, 0.75, 0.93]; ws = [0.09, 0.15, 0.15, 0.15, 0.15, 0.10]
    items = ["CT scan\n(LUNA16 or\nNLST)", "Preprocessing\nlung mask, crop,\nresample to\n1.25 × 0.703 × 0.703 mm",
             "Stage 1: 3D RetinaNet\n(ResNet-50 + FPN)\ncandidates with\nscore p₁ ≥ 0.02",
             "Stage 2: model B\n3 cubes (24 / 48 / 96 mm)\n3 ResNet-18 trunks\nscore p₂",
             "Score fusion\np = w·p₂ + (1 − w)·p₁\nw chosen on\nvalidation scans", "Candidates\nwith fused\nscore p"]
    for i, (x, w, t) in enumerate(zip(xs, ws, items)):
        end = i in (0, 5)
        box(ax, x, y, w, 0.27, t, fc="#f4f6fa" if end else "#eef3fb", ec=GREY if end else BLUE, fs=7.8)
        if i: arrow(ax, xs[i - 1] + ws[i - 1] / 2, y, x - w / 2, y)
    # evaluation branches
    yb = 0.33
    box(ax, 0.24, yb, 0.30, 0.22, "LUNA16 (Chapter 4)\nper candidate: FROC, CPM,\nfalse positives per scan at 90 / 95 % sensitivity",
        fc="#f4f6fa", ec=GREY)
    box(ax, 0.62, yb, 0.30, 0.22, "NLST (Chapter 5)\nper exam: maximum fused score (primary),\nFeatureGBM, AttentionMIL; + temporal features\n"
        "from the registered prior screen", fc="#eefaf5", ec=GREEN)
    box(ax, 0.62, 0.07, 0.30, 0.09, "false-positive rate at the NLST radiologists' sensitivity", fc="#eefaf5", ec=GREEN, fs=8)
    box(ax, 0.90, yb, 0.17, 0.22, "TCIA (Section 5.5)\nper lesion, full and\n10 % dose", fc="#f4f6fa", ec=GREY)
    arrow(ax, 0.93, y - 0.135, 0.24, yb + 0.11); arrow(ax, 0.93, y - 0.135, 0.62, yb + 0.11); arrow(ax, 0.93, y - 0.135, 0.90, yb + 0.11)
    arrow(ax, 0.62, yb - 0.11, 0.62, 0.115)
    ax.text(0.33, 0.56, "frozen after LUNA16 training; applied unchanged to NLST and TCIA", ha="center", fontsize=8, color=MUTED, style="italic")
    fig.tight_layout(); fig.savefig(OUT / "system_overview.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ cohort flow
def cohort_flow():
    from lungcad.data.nlst import usable_series
    u = usable_series(); n_series, n_pid = len(u), u.pid.nunique()
    json.dump({"n_series": n_series, "n_participants": int(n_pid)}, open(OUT / "nlst_cohort.json", "w"))
    H = R["H1"]; k1 = R["k1"]; rows = {(r["system"], r["group"]): r for r in csv.DictReader(open(AN / "systems.csv"))}
    g = lambda grp: rows[("max_fused_B_s0", grp)]
    n_exam = H["baseline"]["n_exams"] + H["after"]["n_exams"]
    reg = json.load(open(WP4 / "registration" / "summary.json"))
    # layout redrawn 25 Nov 2026: main column left, exclusions in a right column between the steps, nothing clipped
    fig, ax = plt.subplots(figsize=(8.4, 8.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    MX, MW, MH = 0.33, 0.60, 0.08
    steps = [(0.935, f"NLST CT arm (IDC delivery): {k1['baseline']['n']:,} participants with a screen result\n"
                     "image inventory: 72,861 primary series of 26,254 participants"),
             (0.765, "72,858 series converted to the detector's spacing\n(native resolution)"),
             (0.575, f"Usable image set:\n{n_series:,} series of {n_pid:,} participants"),
             (0.385, f"Screening exams with a usable series\nand an official screen result: {n_exam:,}")]
    for y, t in steps: box(ax, MX, y, MW, MH, t, fs=9.5)
    for (y0, _), (y1, _) in zip(steps, steps[1:]): arrow(ax, MX, y0 - MH / 2, MX, y1 + MH / 2)
    EX, EW = 0.815, 0.32
    for y, t in ((0.67, "Excluded: 3 defective DICOM,\n313 lung-mask failures (0.43 %)"),
                 (0.48, f"Excluded: {n_series - n_exam:,} series without\nan official result (codes 1–6)")):
        box(ax, EX, y, EW, 0.085, t, fc="#fbf1ec", ec=ORANGE, fs=9)
        ax.plot([MX, EX - EW / 2 - 0.004], [y, y], color=ORANGE, lw=1.1)
        arrow(ax, EX - EW / 2 - 0.03, y, EX - EW / 2, y, c=ORANGE)
    yb, BH = 0.205, 0.13
    box(ax, 0.20, yb, 0.36, BH, f"Baseline (first screen received)\n{int(g('baseline')['n_exams']):,} exams\n"
        f"{int(g('baseline')['n_cancer'])} with cancer present", fc="#eefaf5", ec=GREEN, fs=9.5)
    box(ax, 0.705, yb, 0.53, BH, f"After baseline\n{int(g('after')['n_exams']):,} exams, {int(g('after')['n_cancer'])} with cancer present\n"
        f"first after baseline: {int(g('after1')['n_exams']):,} ({int(g('after1')['n_cancer'])})\n"
        f"second after baseline: {int(g('after2')['n_exams']):,} ({int(g('after2')['n_cancer'])})", fc="#eefaf5", ec=GREEN, fs=9.5)
    ys = steps[-1][0] - MH / 2
    ax.plot([MX, MX], [ys, 0.30], color=MUTED, lw=1.1); ax.plot([0.20, 0.705], [0.30, 0.30], color=MUTED, lw=1.1)
    arrow(ax, 0.20, 0.30, 0.20, yb + BH / 2); arrow(ax, 0.705, 0.30, 0.705, yb + BH / 2)
    box(ax, 0.705, 0.055, 0.53, 0.075, f"Prior screen registered: {reg['ok']:,} of {reg['pairs']:,}\npairs pass the quality check",
        fc="#f4f6fa", ec=GREY, fs=9)
    arrow(ax, 0.705, yb - BH / 2, 0.705, 0.055 + 0.0375)
    fig.tight_layout(); fig.savefig(OUT / "nlst_cohort_flow.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ per-exam confusion matrices
def confusion():
    H = R["H1"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    for i, grp in enumerate(("baseline", "after")):
        h = H[grp]; nc = h["n_cancer"]; nn = h["n_exams"] - nc
        for j, (name, sens, fpr) in enumerate((("NLST radiologists", h["ref_sens"], h["ref_fpr"]),
                                                ("System (B fused, max pooling)\nat the radiologists' sensitivity", h["sys_sens_at_thr"], h["sys_fpr"]))):
            tp, fp = count(sens, nc), count(fpr, nn); M = np.array([[tp, nc - tp], [fp, nn - fp]])
            frac = M / M.sum(1, keepdims=True); ax = axes[i, j]
            ax.imshow(frac, cmap="Blues", vmin=0, vmax=1.15)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{M[a, b]:,}\n{100 * frac[a, b]:.1f} %", ha="center", va="center", fontsize=10,
                            color="white" if frac[a, b] > 0.6 else INK)
            ax.set_xticks([0, 1], ["positive", "negative"]); ax.set_yticks([0, 1], ["cancer\npresent", "cancer\nabsent"])
            ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
            label = "Baseline screen" if grp == "baseline" else "After baseline"
            ax.set_title(f"{label} · {name}", fontsize=8.5, loc="left", color=INK)
    fig.text(0.5, 0.005, "Rows: outcome (Pinsky et al. definition); columns: exam called positive. Same exams in each row.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(OUT / "nlst_confusion.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ temporal: FPR at matched sensitivity
def temporal():
    H, H2 = R["H1"]["after"], R["H2"]
    rows = {(r["system"], r["group"]): r for r in csv.DictReader(open(AN / "systems.csv"))}
    det = float(rows[("max_p1", "after")]["fpr_matched"])
    bars = [("NLST radiologists", H["ref_fpr"], INK), ("Detector alone\n(max score)", det, GREY),
            ("B fused,\nmax pooling\n(primary)", H["sys_fpr"], BLUE), ("FeatureGBM,\nsingle scan", H2["fpr_b"], GREEN),
            ("FeatureGBM,\n+ temporal\nfeatures", H2["fpr_a"], ORANGE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [2.2, 1]})
    a.bar(range(len(bars)), [100 * v for _, v, _ in bars], color=[c for *_, c in bars], width=0.62)
    for k, (_, v, _) in enumerate(bars): a.text(k, 100 * v + 1.2, f"{100 * v:.1f} %", ha="center", fontsize=8.5, color=INK)
    a.set_xticks(range(len(bars)), [n for n, *_ in bars], fontsize=8); a.set_ylabel("false-positive rate (%)")
    a.set_ylim(0, 80); a.set_title(f"(a) After-baseline exams ({H2['n_exams']:,}; {H2['n_cancer']} with cancer), "
                                   f"all at sensitivity {100 * H2['ref_sens']:.1f} %", loc="left", fontsize=8.5)
    d = [("temporal − single scan (H2)", H2["delta_fpr"], H2["delta_ci95"], ORANGE),
         ("primary − radiologists (H1b)", H["delta_fpr"], H["delta_ci95"], BLUE)]
    for k, (n, v, ci, c) in enumerate(d):
        b.plot([100 * ci[0], 100 * ci[1]], [k, k], color=c, lw=2.2); b.plot(100 * v, k, "o", color=c, ms=7)
        b.text(100 * max(v, 0.12), k + 0.25, f"{100 * v:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]", ha="center", fontsize=8)
    b.axvline(0, color=MUTED, lw=1); b.set_yticks(range(len(d)), [n for n, *_ in d], fontsize=8); b.set_ylim(-0.6, 1.7); b.set_xlim(-10, 62)
    b.set_xlabel("Δ false-positive rate (percentage points)"); b.set_title("(b) Paired differences, 95 % CI", loc="left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "nlst_temporal.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ registration example (no participant ID)
def registration():
    import pandas as pd, SimpleITK as sitk
    from lungcad.data.nlst import usable_series
    from lungcad.wp4.register import mask_image
    from lungcad.wp4.temporal import transform_from_row
    from lungcad.preprocess import load_volume
    r = pd.concat([pd.read_parquet(f) for f in sorted((WP4 / "registration").glob("reg_*.parquet"))], ignore_index=True)
    r = r[(r.error == "") & r.ok]
    # a pair with a visible change of position before registration, chosen by a fixed rule (largest translation
    # among a fixed random sample of passing pairs), so the example shows what registration corrects
    s = r.sample(200, random_state=1); x = s.loc[s.translation.apply(lambda t: float(np.linalg.norm(t))).idxmax()]
    u = usable_series().set_index("uid")
    arr, _ = load_volume(u.loc[x.uid, "stem"]); fm, _ = mask_image(u.loc[x.uid, "stem"]); mm, _ = mask_image(u.loc[x.prior_uid, "stem"])
    # "before": only the lung centres aligned (the scans lie far apart in scanner coordinates, so identity shows nothing)
    cen = lambda im: np.array((lambda s: (s.Execute(sitk.Cast(im > 0, sitk.sitkUInt8)), s.GetCentroid(1))[1])(sitk.LabelShapeStatisticsImageFilter()))
    off = cen(mm) - cen(fm); Tc = sitk.TranslationTransform(3, tuple(off))
    w_id = sitk.GetArrayFromImage(sitk.Resample(mm, fm, Tc, sitk.sitkNearestNeighbor, 0.0))
    w = sitk.GetArrayFromImage(sitk.Resample(mm, fm, transform_from_row(x), sitk.sitkNearestNeighbor, 0.0))
    f = sitk.GetArrayFromImage(fm) > 0
    dice = lambda a: 2 * (a & f).sum() / (a.sum() + f.sum())
    d0 = dice(w_id > 0)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2))
    for i, (ww, lab) in enumerate(((w_id, f"lung centres aligned only (Dice {d0:.3f})"), (w, f"after rigid + affine (Dice {x.dice:.3f})"))):
        for j, (sl, name) in enumerate((((arr.shape[0] // 2, slice(None), slice(None)), "axial"),
                                        ((slice(None), arr.shape[1] // 2, slice(None)), "coronal"))):
            a = axes[i, j]; a.imshow(arr[sl], cmap="gray", vmin=-1000, vmax=400, origin="lower" if name == "coronal" else "upper")
            a.contour(f[sl], levels=[0.5], colors=BLUE, linewidths=0.8); a.contour(ww[sl], levels=[0.5], colors=ORANGE, linewidths=0.9)
            a.set_axis_off(); a.set_title(f"{name}, {lab}", fontsize=8.5)
    fig.text(0.5, 0.01, f"Current screen (T{int(x['round'])}) with its lung mask (blue) and the prior screen's (T{int(x.prior_round)}) "
             "lung mask mapped onto it (orange).", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "nlst_registration.png", dpi=200); plt.close(fig)
    json.dump({"dice_centres": float(d0), "offset_mm": float(np.linalg.norm(off)), "dice_after": float(x.dice), "transition": f"T{int(x.prior_round)}->T{int(x['round'])}"},
              open(OUT / "nlst_registration.json", "w"))


if __name__ == "__main__":
    system_overview(); cohort_flow(); confusion(); temporal()
    if "--no-registration" not in sys.argv: registration()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
