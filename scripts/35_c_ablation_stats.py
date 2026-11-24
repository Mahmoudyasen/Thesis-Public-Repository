"""Model C ablations: analysis exactly as fixed in report/c_ablation_prespec.md (commit ee66d7a).

    python scripts/35_c_ablation_stats.py --check-only        # known-answer checks against the S6 files, nothing else
    python scripts/35_c_ablation_stats.py --dry-run           # pipeline test: S6 C seed 1 posing as a "variant" vs seed 0
    python scripts/35_c_ablation_stats.py                     # the real analysis (variants with all 10 folds done)

Steps (the script stops at the first failed check):
  1. Known-answer checks on S6 files (no ablation file is read before they pass):
       - per-fold CPM recomputed from test_candidates.csv == froc_test.json (official C seed 0, all 10 folds)
       - fold-mean of those CPMs == 0.8916 (the number quoted in the prespec)
       - pooled 888-scan CPM == S6 paired_bootstrap.json (official C seed 0: 0.89143…)
       - paired bootstrap of C vs B (official, seed 0, 10,000 resamples) reproduces the stored ΔCPM 95 % CI exactly
       - FPs/scan at 90 / 95 % sensitivity of own C seed 0 fused == S6 fp_reduction.json
  2. Seed range of the C baseline over S6 seeds 0/1/2, computed from the files on BOTH bases (pooled and fold-mean).
  3. Arm A (official, stage 2): ΔCPM = variant - C seed 0, paired bootstrap over the 888 scans (10,000), 95 % percentile
     CI, two-sided P = 2·min(#Δ≤0+1, #Δ≥0+1)/(B+1) (capped at 1), Holm across the variants. Verdict per the prespec rule:
     "changes C" only if P_Holm < 0.05 AND |ΔCPM| > seed range. Secondary (descriptive): FPs/scan at 90/95 % (S6
     operating_points method, 2,000 resamples), sens at 1/8 FP/scan, per-fold CPM, parameters, training time.
  4. Arm B (own, fused = headline and stage 2 alone): ΔCPM + 95 % CI, descriptive only (no P, no Holm).
  5. Outputs in work/wp0/analysis/c_ablations[_DRYRUN]/: armA.json, armB.json, per_fold.csv, runs.csv,
     known_answers.json, seed_range.json, forest.png/.pdf.

Seed-range basis (--seed-range-basis): the prespec quotes 0.8916/0.8923/0.8876 (range 0.0047) as "pooled", but those are
fold-means of the per-fold CPMs; the pooled 888-scan values are 0.8914/0.8773/0.8892 (range 0.0141). The primary
endpoint is pooled ΔCPM, so the like-for-like threshold is the pooled range. The script computes both and reports the
verdict under each; the basis used for the headline verdict must be recorded in report/c_ablation_prespec_NOTES.md.
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.luna16_index import load_index
from lungcad.eval.froc import per_scan_results, froc_from_results
from lungcad.eval.operating_points import paired_fp_reduction, fp_at_sensitivity

ROOT = C.STAGE2
S6_DIR = C.STAGE2.parent / "analysis" / "s6_final_10k"
VARIANTS = {"C_fusionconcat": "V1 concat (no cross-scale attention)",
            "C_use_scale_embeddingfalse": "V2 no scale embedding",
            "C_depth6": "V3 depth 6"}
PRESPEC_FOLDMEAN_SEED0 = 0.8916
N_BOOT, N_BOOT_FP, ALPHA = 10000, 2000, 0.05
TOL = 1e-12


# ---------------------------------------------------------------- loading
def run_dir(regime, model, fold, seed):
    return ROOT / regime / model / f"fold{fold}" / f"seed{seed}"


def run_complete(d: Path, regime: str) -> bool:
    need = ["done.json", "froc_test.json", "test_candidates.csv"] + (["fusion.json", "test_candidates_fused.csv"] if regime == "own" else [])
    return all((d / f).is_file() and (d / f).stat().st_size > 0 for f in need)


def pooled(regime, model, seed, score="stage2"):
    f = "test_candidates.csv" if score == "stage2" else "test_candidates_fused.csv"
    dirs = [run_dir(regime, model, k, seed) for k in range(10)]
    if not all(run_complete(d, regime) for d in dirs):
        return None
    return pd.concat([pd.read_csv(d / f) for d in dirs], ignore_index=True)


def pooled_cpm(df, uids):
    return froc_from_results(per_scan_results(df, uids))[3]


def fold_cpms(regime, model, seed, score="stage2"):
    f = "froc_test.json" if score == "stage2" else "froc_test_fused.json"
    return [json.load(open(run_dir(regime, model, k, seed) / f))["cpm"] for k in range(10)]


def run_info(d: Path) -> dict:
    out = {"run_tag": json.load(open(d / "done.json")).get("run_tag")}
    m = re.search(r"params ([\d.]+)M", (d / "stdout.log").read_text(errors="ignore")) if (d / "stdout.log").exists() else None
    out["params_M"] = float(m.group(1)) if m else np.nan
    ep, val = {}, {}
    for line in (d / "train.log").read_text(errors="ignore").splitlines():
        m = re.match(r"epoch (\d+)/\d+ .*? (\d+)s mem", line)
        if m:
            e = int(m.group(1)); ep[e] = int(m.group(2))          # last line per epoch wins (resume-safe)
            v = re.search(r"\| val CPM .*\((\d+)s\)", line); val[e] = int(v.group(1)) if v else 0
    out["epochs_logged"] = len(ep); out["train_h"] = (sum(ep.values()) + sum(val.values())) / 3600
    return out


# ---------------------------------------------------------------- statistics
def paired_boot(ca: pd.DataFrame, cb: pd.DataFrame, uids, n_boot=N_BOOT, seed=0) -> dict:
    """Same resampling as lungcad.eval.bootstrap.paired_delta_cpm (same RNG stream), but also returns the two-sided P."""
    ra, rb = per_scan_results(ca, uids), per_scan_results(cb, uids)
    _, _, sa, xa = froc_from_results(ra); _, _, sb, xb = froc_from_results(rb)
    rng = np.random.RandomState(seed); d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(uids), len(uids))
        d[i] = froc_from_results([ra[j] for j in idx])[3] - froc_from_results([rb[j] for j in idx])[3]
    p = min(1.0, 2 * min(((d <= 0).sum() + 1) / (n_boot + 1), ((d >= 0).sum() + 1) / (n_boot + 1)))
    return {"cpm_variant": xa, "cpm_base": xb, "delta": xa - xb, "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(p), "sens_1_8_variant": float(sa[0]), "sens_1_8_base": float(sb[0]), "n_boot": n_boot, "n_scans": len(uids)}


def holm(ps: list[float]) -> list[float]:
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def verdict(p_holm, delta, thr):
    sig, big = p_holm < ALPHA, abs(delta) > thr
    if sig and big:
        return "changes C"
    if sig:
        return "statistically detectable but within seed-to-seed variation"
    return "no evidence that this component matters"


# ---------------------------------------------------------------- step 1: known answers
def known_answers(uids, idx) -> dict:
    ka, ok = {}, True
    def chk(name, got, want, tol=TOL):
        nonlocal ok
        g, w = np.asarray(got, float), np.asarray(want, float); passed = bool(g.shape == w.shape and np.all(np.abs(g - w) <= tol))
        ok &= passed; ka[name] = {"got": g.tolist(), "want": w.tolist(), "tol": tol, "pass": passed}
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: got {np.round(g, 6).tolist()} want {np.round(w, 6).tolist()}")

    re_cpm = []
    for k in range(10):
        d = run_dir("official", "C", k, 0)
        fu = idx.loc[idx.fold == k, "seriesuid"].astype(str).tolist()
        re_cpm.append(pooled_cpm(pd.read_csv(d / "test_candidates.csv"), fu))
    chk("official C s0 per-fold CPM recomputed == froc_test.json", re_cpm, fold_cpms("official", "C", 0))
    chk("official C s0 fold-mean CPM == prespec 0.8916 (4 dp)", round(float(np.mean(re_cpm)), 4), PRESPEC_FOLDMEAN_SEED0)

    pb = json.load(open(S6_DIR / "paired_bootstrap.json"))
    s6 = next(r for r in pb if (r["regime"], r["score"], r["seed"], r["a"], r["b"]) == ("official", "stage2", 0, "C", "B"))
    c0, b0 = pooled("official", "C", 0), pooled("official", "B", 0)
    chk("official C s0 pooled CPM == S6 paired_bootstrap.json", pooled_cpm(c0, uids), s6["cpm_a"])
    r = paired_boot(c0, b0, uids, n_boot=s6["n_boot"])
    chk("official C vs B s0 paired-bootstrap ΔCPM + 95% CI == S6", [r["delta"]] + r["delta_ci95"], [s6["delta"]] + s6["delta_ci95"])

    fr = json.load(open(S6_DIR / "fp_reduction.json"))
    s6f = next(x for x in fr if (x["seed"], x["model"], x["score"]) == (0, "C", "fused"))
    fp = fp_at_sensitivity(per_scan_results(pooled("own", "C", 0, "fused"), uids))
    chk("own C s0 fused FPs/scan @90/95 == S6 fp_reduction.json", fp, [s6f["fp_model@90"], s6f["fp_model@95"]])
    return {"all_pass": bool(ok), "checks": ka}


# ---------------------------------------------------------------- step 2: seed range
def seed_range(uids) -> dict:
    out = {}
    for regime, score in [("official", "stage2"), ("own", "fused"), ("own", "stage2")]:
        pc = [pooled_cpm(pooled(regime, "C", s, score), uids) for s in (0, 1, 2)]
        fm = [float(np.mean(fold_cpms(regime, "C", s, score))) for s in (0, 1, 2)]
        out[f"{regime}/{score}"] = {"pooled": pc, "pooled_range": max(pc) - min(pc), "foldmean": fm, "foldmean_range": max(fm) - min(fm)}
        print(f"  C {regime}/{score}: pooled {np.round(pc, 4).tolist()} range {max(pc) - min(pc):.4f} | "
              f"fold-mean {np.round(fm, 4).tolist()} range {max(fm) - min(fm):.4f}")
    return out


# ---------------------------------------------------------------- figure
def forest(armA, armB, sr, basis, out: Path, dry: bool):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    INK, MUTE, GRID, BAND, PT = "#0b0b0b", "#52514e", "#e6e5e1", "#ecebe7", "#2a78d6"
    panels = [("(a) Arm A: official candidates, stage 2", armA, sr["official/stage2"][f"{basis}_range"])]
    for sc, lab in (("fused", "(b) Arm B: own candidates, fused score"), ("stage2", "(c) Arm B: own candidates, stage 2 alone")):
        rows = [r for r in armB if r["score"] == sc]
        if rows:
            panels.append((lab, rows, sr[f"own/{sc}"][f"{basis}_range"]))
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 1.25 + 0.55 * sum(len(p[1]) for p in panels) + 0.5 * len(panels)), squeeze=False)
    lo = min(min(r["delta_ci95"][0] for r in p[1]) for p in panels); hi = max(max(r["delta_ci95"][1] for r in p[1]) for p in panels)
    pad = 0.15 * (hi - lo + 1e-9); xl = (min(lo, -max(p[2] for p in panels)) - pad, max(hi, max(p[2] for p in panels)) + pad)
    for ax, (title, rows, thr) in zip(axes[:, 0], panels):
        y = np.arange(len(rows))[::-1]
        ax.axvspan(-thr, thr, color=BAND, lw=0, zorder=0); ax.axvline(0, color=MUTE, lw=1, zorder=1)
        for yi, r in zip(y, rows):
            ax.plot(r["delta_ci95"], [yi, yi], color=PT, lw=2, solid_capstyle="round", zorder=2)
            filled = r.get("verdict") == "changes C" if "score" not in r else r["exceeds_seed_range"]
            ax.plot(r["delta"], yi, "o", ms=8, color=PT if filled else "white", mec=PT, mew=2, zorder=3)
            ax.annotate(f"{r['delta']:+.4f} [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}]", (1.01, yi), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=8, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5, color=INK)
        ax.set_xlim(*xl); ax.set_ylim(-0.7, len(rows) - 0.3); ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTE); ax.tick_params(axis="x", labelsize=8, colors=MUTE); ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[-1, 0].set_xlabel(f"ΔCPM vs model C (S6 seed 0), pooled 888 scans, 95 % paired-bootstrap CI\n"
                           f"grey band = ± C seed-to-seed range ({basis}); filled dot = arm A: 'changes C' (Holm P < 0.05 and |Δ| > range); arm B: |Δ| > range", fontsize=8, color=INK)
    if dry:
        fig.suptitle("DRY RUN: S6 seed 1 vs seed 0 — not an ablation result", fontsize=9, color="#c0392b")
    fig.tight_layout(); fig.savefig(out / "forest.png", dpi=200, bbox_inches="tight"); fig.savefig(out / "forest.pdf", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check-only", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed-range-basis", choices=["pooled", "foldmean"], default="pooled"); ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--allow-partial", action="store_true", help="analyse whatever variants are complete (Holm then marked provisional)")
    a = ap.parse_args()
    out = C.STAGE2.parent / "analysis" / ("c_ablations_DRYRUN" if a.dry_run else "c_ablations"); out.mkdir(parents=True, exist_ok=True)
    idx = load_index(); uids = idx.seriesuid.astype(str).tolist(); assert len(uids) == 888

    print("1. known-answer checks (S6 files)")
    ka = known_answers(uids, idx); json.dump(ka, open(out / "known_answers.json", "w"), indent=1)
    if not ka["all_pass"]:
        sys.exit("known-answer check FAILED - nothing else is computed")
    print("2. seed range of the C baseline (S6 seeds 0/1/2)")
    sr = seed_range(uids); json.dump(sr, open(out / "seed_range.json", "w"), indent=1)
    if a.check_only:
        return

    if a.dry_run:   # S6 seed 1 posing as one "variant"; same code path as the real analysis
        variants = {"C": "DRY RUN: C seed 1"}; vseed = 1
    else:
        variants = VARIANTS; vseed = 0

    print("3. arm A: official, stage 2")
    base = pooled("official", "C", 0); armA, runs, folds = [], [], []
    for v, lab in variants.items():
        cand = pooled("official", v, vseed)
        if cand is None:
            print(f"  {v}: not all 10 folds complete - skipped"); continue
        r = paired_boot(cand, base, uids, n_boot=a.n_boot)
        f = paired_fp_reduction(base, cand, uids, n_boot=N_BOOT_FP)
        r |= {"variant": v, "label": lab, "fp_base@90": f["fp_base@90"], "fp_variant@90": f["fp_model@90"], "fp_variant@90_ci95": f["fp_model@90_ci95"],
              "fp_base@95": f["fp_base@95"], "fp_variant@95": f["fp_model@95"], "fp_variant@95_ci95": f["fp_model@95_ci95"]}
        fv, fb = fold_cpms("official", v, vseed), fold_cpms("official", "C", 0)
        r["fold_cpm_mean_variant"], r["fold_cpm_mean_base"] = float(np.mean(fv)), float(np.mean(fb))
        r["folds_variant_better"] = int(sum(x > y for x, y in zip(fv, fb)))
        folds += [{"arm": "A", "variant": v, "fold": k, "cpm_variant": x, "cpm_base": y, "delta": x - y} for k, (x, y) in enumerate(zip(fv, fb))]
        for k in range(10):
            runs.append({"regime": "official", "variant": v, "fold": k, **run_info(run_dir("official", v, k, vseed))})
        armA.append(r); print(f"  {lab}: Δ {r['delta']:+.4f} {np.round(r['delta_ci95'], 4).tolist()} P {r['p_two_sided']:.4f}")

    complete = len(armA) == len(variants)
    if armA and (complete or a.allow_partial):
        ph = holm([r["p_two_sided"] for r in armA])
        for r, p in zip(armA, ph):
            r["p_holm"] = p; r["holm_m"] = len(armA); r["holm_provisional"] = not complete
            for basis in ("pooled", "foldmean"):
                r[f"verdict_{basis}"] = verdict(p, r["delta"], sr["official/stage2"][f"{basis}_range"])
            r["verdict"] = r[f"verdict_{a.seed_range_basis}"]
            print(f"  {r['label']}: P_Holm {p:.4f} -> {r['verdict']}" + (f" (fold-mean basis: {r['verdict_foldmean']})" if r['verdict_foldmean'] != r['verdict'] else ""))
    elif armA:
        print("  arm A incomplete - Holm / verdicts not computed (use --allow-partial for a provisional look)")

    print("4. arm B: own, fused + stage 2 alone (descriptive)")
    armB = []
    for v, lab in variants.items():
        for score in ("fused", "stage2"):
            cand, b = pooled("own", v, vseed, score), pooled("own", "C", 0, score)
            if cand is None:
                continue
            r = paired_boot(cand, b, uids, n_boot=a.n_boot); r.pop("p_two_sided")     # descriptive only: no P in arm B
            r |= {"variant": v, "label": lab, "score": score, "exceeds_seed_range": abs(r["delta"]) > sr[f"own/{score}"][f"{a.seed_range_basis}_range"]}
            fv, fb = fold_cpms("own", v, vseed, score), fold_cpms("own", "C", 0, score)
            folds += [{"arm": f"B-{score}", "variant": v, "fold": k, "cpm_variant": x, "cpm_base": y, "delta": x - y} for k, (x, y) in enumerate(zip(fv, fb))]
            armB.append(r); print(f"  {lab} [{score}]: Δ {r['delta']:+.4f} {np.round(r['delta_ci95'], 4).tolist()}")
        if pooled("own", v, vseed) is not None:
            runs += [{"regime": "own", "variant": v, "fold": k, **run_info(run_dir("own", v, k, vseed))} for k in range(10)]

    meta = {"prespec_commit": "ee66d7a", "seed_range_basis": a.seed_range_basis, "n_boot": a.n_boot, "n_boot_fp": N_BOOT_FP, "dry_run": a.dry_run}
    json.dump({"meta": meta, "results": armA}, open(out / "armA.json", "w"), indent=1, default=float)
    json.dump({"meta": meta, "results": armB}, open(out / "armB.json", "w"), indent=1, default=float)
    pd.DataFrame(folds).to_csv(out / "per_fold.csv", index=False); pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
    if runs:
        tags = {r["run_tag"] for r in runs}; print(f"  run tags: {sorted(tags)}")
    if armA or armB:
        forest(armA, armB, sr, a.seed_range_basis, out, a.dry_run)
    print(f"outputs -> {out}")


if __name__ == "__main__":
    main()
