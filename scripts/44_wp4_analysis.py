"""WP4 analysis exactly as fixed in report/wp4_nlst_prespec.md, Section 7.

    python scripts/44_wp4_analysis.py [--n-boot 10000] [--n-boot-secondary 2000]   # -> work/wp4/analysis/

Order: guard (plan committed, splits hash) -> K1 + K4 must pass -> primary (H1a, H1b with Holm; H2) -> secondaries
(per group: AUC, matched-sensitivity FPR vs the radiologists, PPV, transferred threshold; every system and aggregator vs
the primary; subgroups; cancer characteristics at the primary operating point; interval cancers; outcome sensitivity
analyses; agreement of the temporal matching with the radiologists' pre-existing / growth records) -> figures -> numbers
for Chapter 5 (ch5_numbers.json). Nothing here tunes anything.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.wp4 import metrics as M
from lungcad.wp4.labels import image_cohort, known_answer_k1, PINSKY

WP4 = C.WORK / "wp4"; OUT = WP4 / "analysis"
PRIMARY = "max_fused_B_s0"
SYSTEMS = {"max_fused_B_s0": "B fused, max pooling (primary)", "max_p1": "detector alone", "max_p2_B_s0": "B stage 2 alone",
           "max_fused_B_s1": "B fused seed 1", "max_fused_B_s2": "B fused seed 2", "max_fused_Bens": "B fused, seed ensemble",
           "max_fused_A_s0": "A fused", "max_fused_C_s0": "C fused", "gbm_single": "FeatureGBM (single scan)", "mil": "AttentionMIL"}
LUNGRADS = {"baseline": (0.849, 0.128), "after": (0.786, 0.053)}           # Pinsky 2015 Table 4 (sens, FPR)
GROUPS = ["baseline", "after", "after1", "after2"]


def group_mask(ex, g):
    return {"baseline": ex.baseline.values, "after": ~ex.baseline.values, "after1": (ex.group == "after1").values, "after2": (ex.group == "after2").values}[g]


def auc_ci(pid, y, s, n_boot, seed=0):
    rng = np.random.RandomState(seed); a = roc_auc_score(y, s); b = []
    for _ in range(n_boot):
        w = M._participant_weights(pid, rng)
        if (w * y).sum() == 0:
            continue
        b.append(roc_auc_score(y, s, sample_weight=w))
    return float(a), M._ci(np.array(b))


def delta_auc(pid, y, s_a, s_b, n_boot, seed=0):
    rng = np.random.RandomState(seed); d0 = roc_auc_score(y, s_a) - roc_auc_score(y, s_b); b = []
    for _ in range(n_boot):
        w = M._participant_weights(pid, rng); b.append(roc_auc_score(y, s_a, sample_weight=w) - roc_auc_score(y, s_b, sample_weight=w))
    return float(d0), M._ci(np.array(b))


def transferred(ex, s, m):
    """Deployable operating point: for each held-out fold k the threshold is set on the other folds' exams of the same
    group (system sensitivity >= radiologists' sensitivity there), then applied to fold k. -> sens, FPR, PPV (pooled)."""
    pred = np.zeros(len(ex), bool)
    for k in range(5):
        tr = m & (ex.fold.values != k); te = m & (ex.fold.values == k)
        S = M._Sorted(s[tr], ex.cancer.values[tr]); w = np.ones(tr.sum())
        sr, _ = M._ref(w, ex.cancer.values[tr], ex.rad_pos.values[tr]); _, _, thr = S.fpr_at(w, sr)
        pred[te] = s[te] >= thr
    c = ex.cancer.values[m]; p = pred[m]
    return {"sens": float((c & p).sum() / c.sum()), "fpr": float((~c & p).sum() / (~c).sum()), "ppv": float((c & p).sum() / max(1, p.sum())),
            "n_pos": int(p.sum())}


def ppv_at_matched(ex, s, m, thr):
    c = ex.cancer.values[m]; p = s[m] >= thr
    return float((c & p).sum() / max(1, p.sum()))


def run(ex: pd.DataFrame, n_boot: int, n_boot2: int, test: bool = False) -> dict:
    res = {"meta": {"n_boot": n_boot, "n_boot_secondary": n_boot2, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    pid = ex.pid.values; y = ex.cancer.values.astype(bool); rad = ex.rad_pos.values.astype(bool)
    # ---------------- primary
    prim = {}
    for g in ("baseline", "after"):
        m = group_mask(ex, g)
        prim[g] = M.matched_fpr_vs_reader(pid[m], y[m], ex[PRIMARY].values[m], rad[m], n_boot=n_boot)
    ph = M.holm([prim["baseline"]["p_two_sided"], prim["after"]["p_two_sided"]])
    for g, p in zip(("baseline", "after"), ph):
        prim[g]["p_holm"] = p; prim[g]["verdict"] = M.verdict(p, prim[g]["delta_ci95"])
    res["H1"] = prim
    # ceiling of the matched-sensitivity rule (NOTES 4): a cancer exam without any candidate scores 0, tied with every other
    # candidate-free exam, so if the radiologists' sensitivity exceeds this ceiling the matched FPR jumps across that block
    res["sensitivity_ceiling"] = {}
    for g in ("baseline", "after"):
        m = group_mask(ex, g); s0 = ex[PRIMARY].values[m]
        res["sensitivity_ceiling"][g] = {"cancer_exams_with_candidate": float((s0[y[m]] > 0).mean()), "exams_without_candidate": float((s0 == 0).mean()),
                                         "radiologist_sensitivity": prim[g]["ref_sens"], "rule_on_plateau": bool((s0[y[m]] > 0).mean() < prim[g]["ref_sens"])}
    for g, v in res["sensitivity_ceiling"].items():
        print(f"  ceiling {g}: {v['cancer_exams_with_candidate']:.3f} of cancer exams have a candidate (radiologists {v['radiologist_sensitivity']:.3f})" + ("  ** RULE ON PLATEAU **" if v["rule_on_plateau"] else ""))
    m = group_mask(ex, "after")
    h2 = M.matched_fpr_two_systems(pid[m], y[m], ex.gbm_temporal_after.values[m], ex.gbm_single_after.values[m], rad[m], n_boot=n_boot)
    h2["verdict"] = M.verdict(h2["p_two_sided"], h2["delta_ci95"]); res["H2"] = h2
    print(f"H1a baseline: Δ {prim['baseline']['delta_fpr']:+.4f} {np.round(prim['baseline']['delta_ci95'], 4).tolist()} P_Holm {prim['baseline']['p_holm']:.4f} -> {prim['baseline']['verdict']}")
    print(f"H1b after:    Δ {prim['after']['delta_fpr']:+.4f} {np.round(prim['after']['delta_ci95'], 4).tolist()} P_Holm {prim['after']['p_holm']:.4f} -> {prim['after']['verdict']}")
    print(f"H2 temporal:  Δ {h2['delta_fpr']:+.4f} {np.round(h2['delta_ci95'], 4).tolist()} P {h2['p_two_sided']:.4f} -> {h2['verdict']}")
    # ---------------- secondary: per system x group
    sec = []
    for col, lab in SYSTEMS.items():
        if col not in ex or ex[col].isna().all():
            continue
        for g in GROUPS:
            m = group_mask(ex, g); s = ex[col].values
            r = M.matched_fpr_vs_reader(pid[m], y[m], s[m], rad[m], n_boot=n_boot2)
            a, aci = auc_ci(pid[m], y[m], s[m], n_boot2)
            row = {"system": col, "label": lab, "group": g, "n_exams": r["n_exams"], "n_cancer": r["n_cancer"], "auc": a, "auc_ci95": aci,
                   "ref_sens": r["ref_sens"], "ref_fpr": r["ref_fpr"], "fpr_matched": r["sys_fpr"], "delta_fpr_vs_rad": r["delta_fpr"],
                   "delta_ci95": r["delta_ci95"], "ppv_matched": ppv_at_matched(ex, s, m, r["thr"]), "thr": r["thr"],
                   "transferred": transferred(ex, s, m)}
            if col != PRIMARY and g in ("baseline", "after"):
                t = M.matched_fpr_two_systems(pid[m], y[m], s[m], ex[PRIMARY].values[m], rad[m], n_boot=n_boot2)
                row["delta_fpr_vs_primary"] = t["delta_fpr"]; row["delta_fpr_vs_primary_ci95"] = t["delta_ci95"]
                row["delta_auc_vs_primary"], row["delta_auc_vs_primary_ci95"] = delta_auc(pid[m], y[m], s[m], ex[PRIMARY].values[m], n_boot2)
            sec.append(row)
    res["systems"] = sec
    # ---------------- subgroups (primary, matched within subgroup) and cancer characteristics at the group threshold
    sub = []
    thk = pd.cut(ex.thk_mm, [0, 1.5, 2.5, 99], labels=["<=1.5 mm", "1.5-2.5 mm", ">2.5 mm"]).astype(str)
    for g in ("baseline", "after"):
        mg = group_mask(ex, g)
        for name, col in (("vendor", ex.vendor.astype(str)), ("kernel", ex.kernel_class.astype(str)), ("thickness", thk)):
            for v in sorted(col[mg].unique()):
                m = mg & (col.values == v)
                if y[m].sum() < 10:
                    sub.append({"group": g, "by": name, "value": v, "n_exams": int(m.sum()), "n_cancer": int(y[m].sum()), "note": "< 10 cancers, not estimated"}); continue
                r = M.matched_fpr_vs_reader(pid[m], y[m], ex[PRIMARY].values[m], rad[m], n_boot=n_boot2)
                sub.append({"group": g, "by": name, "value": v, "n_exams": r["n_exams"], "n_cancer": r["n_cancer"], "ref_sens": r["ref_sens"], "ref_fpr": r["ref_fpr"],
                            "fpr_matched": r["sys_fpr"], "delta_fpr_vs_rad": r["delta_fpr"], "delta_ci95": r["delta_ci95"], "auc": auc_ci(pid[m], y[m], ex[PRIMARY].values[m], 0)[0]})
    res["subgroups"] = sub
    cc = []
    for g in ("baseline", "after"):
        m = group_mask(ex, g); thr = prim[g]["thr"]; e = ex[m & y]
        det = e[PRIMARY].values >= thr; rd = e.rad_pos.values
        for name, col in (("stage", e.de_stag_7thed.astype(str)), ("histology", e.de_type.astype(str)),
                          ("size", pd.cut(e.lesionsize, [0, 10, 20, 30, 1e9], labels=["<10 mm", "10-20", "20-30", ">30"]).astype(str)),
                          ("can_scr", e.can_scr.astype(str))):
            for v in sorted(col.unique()):
                k = (col == v).values
                cc.append({"group": g, "by": name, "value": v, "n": int(k.sum()), "system_detected": int(det[k].sum()), "radiologist_positive": int(rd[k].sum())})
    res["cancer_characteristics"] = cc
    # ---------------- outcome sensitivity analyses (primary)
    sa = {}
    for name, keep in (("fu_ge_365", ex.cancer.values | ex.fu_ge_365.values), ("round_based", None)):
        for g in ("baseline", "after"):
            if keep is None:
                m = (ex["round"].values == 0) if g == "baseline" else (ex["round"].values > 0)
            else:
                m = group_mask(ex, g) & keep
            r = M.matched_fpr_vs_reader(pid[m], y[m], ex[PRIMARY].values[m], rad[m], n_boot=n_boot2)
            sa[f"{name}/{g}"] = {k: r[k] for k in ("n_exams", "n_cancer", "ref_sens", "ref_fpr", "sys_fpr", "delta_fpr", "delta_ci95")}
    res["sensitivity_analyses"] = sa
    # ---------------- temporal matching vs radiologist records (descriptive, exam level)
    if "present_prior_of_max" in ex and not test:
        res["matching_vs_ctabc"] = matching_agreement(ex)
    res["k1"] = {g: {k: v for k, v in known_answer_k1()[g].items() if k != "pinsky"} for g in ("baseline", "after")} if not test else "TEST"
    return res


def matching_agreement(ex):
    """Exam level: among after-baseline exams where the radiologist recorded a pre-existing non-calcified nodule >= 4 mm
    (ctabc sct_ab_preexist = 2, code 51), share whose top candidate is matched on the prior / that have any matched
    confident candidate; among exams with recorded growth (sct_ab_gwth = 2), share with a positive diameter change of the
    top candidate. Slice-level matching (plan: ±5 slices) needs the slice orientation, verified first on the Farina nodules."""
    t = pd.read_csv(C.NLST_CLINICAL / "nlst_780_ctabc_idc_20210527.csv", low_memory=False); t = t[t.sct_ab_code == 51]
    pre = t[t.sct_ab_preexist == 2].groupby(["pid", "study_yr"]).size().rename("n_pre").reset_index()
    gro = t[t.sct_ab_gwth == 2].groupby(["pid", "study_yr"]).size().rename("n_growth").reset_index()
    e = ex[~ex.baseline & (ex.no_prior == 0)].merge(pre, left_on=["pid", "round"], right_on=["pid", "study_yr"], how="left").drop(columns="study_yr")
    e = e.merge(gro, left_on=["pid", "round"], right_on=["pid", "study_yr"], how="left").drop(columns="study_yr")
    hp = e.n_pre.fillna(0) > 0; hg = e.n_growth.fillna(0) > 0
    return {"exams_with_prior": int(len(e)), "exams_with_preexisting": int(hp.sum()),
            "top_candidate_matched_if_preexisting": float(e.present_prior_of_max[hp].mean()),
            "top_candidate_matched_if_no_record": float(e.present_prior_of_max[~hp].mean()),
            "any_confident_matched_if_preexisting": float((e.n_conf_matched[hp] > 0).mean()),
            "exams_with_growth": int(hg.sum()), "positive_diam_change_if_growth": float((e.diam_change_of_max[hg] > 0).mean()),
            "positive_diam_change_if_no_growth_record": float((e.diam_change_of_max[~hg & hp] > 0).mean())}


def figures(ex, res, out: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    INK, MUTE, GRID = "#1d2433", "#5b6475", "#e6e5e1"; COL = {"max_fused_B_s0": "#2a78d6", "max_p1": "#eb6834", "gbm_single": "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, g, title in zip(axes, ("baseline", "after"), ("(a) Baseline screen", "(b) After baseline")):
        m = group_mask(ex, g); y = ex.cancer.values[m]
        for col, c in COL.items():
            if col in ex and not ex[col].isna().all():
                f, t, _ = roc_curve(y, ex[col].values[m]); ax.plot(f, t, color=c, lw=1.6, label=SYSTEMS[col])
        p = res["H1"][g]; ax.plot(p["ref_fpr"], p["ref_sens"], "o", ms=8, color=INK, mec="white", mew=1.5, label="NLST radiologists (this cohort)")
        ax.plot(LUNGRADS[g][1], LUNGRADS[g][0], "s", ms=7, color=MUTE, mec="white", mew=1.5, label="Lung-RADS (Pinsky 2015)")
        ax.plot(p["sys_fpr"], p["ref_sens"], "o", ms=8, color="white", mec=COL[PRIMARY], mew=2, label="primary at matched sensitivity")
        ax.set_xlim(0, 0.6); ax.set_ylim(0.5, 1.0); ax.set_xlabel("false-positive rate (exams without cancer)", fontsize=8, color=INK)
        ax.set_ylabel("sensitivity (exams with cancer)", fontsize=8, color=INK); ax.set_title(title, fontsize=9, color=INK, loc="left")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=MUTE); ax.grid(color=GRID, lw=0.6)
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(out / "fig_roc.png", dpi=200); fig.savefig(out / "fig_roc.pdf"); plt.close(fig)
    rows = [r for r in res["systems"] if r["group"] in ("baseline", "after")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, g in zip(axes, ("baseline", "after")):
        rr = [r for r in rows if r["group"] == g]; yy = np.arange(len(rr))[::-1]
        ax.axvline(0, color=MUTE, lw=1)
        for yi, r in zip(yy, rr):
            ax.plot(r["delta_ci95"], [yi, yi], color="#2a78d6", lw=2, solid_capstyle="round")
            ax.plot(r["delta_fpr_vs_rad"], yi, "o", ms=6, color="#2a78d6" if r["system"] == PRIMARY else "white", mec="#2a78d6", mew=1.6)
        ax.set_yticks(yy); ax.set_yticklabels([r["label"] for r in rr], fontsize=7); ax.set_title(g, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("ΔFPR vs radiologists at their sensitivity", fontsize=7.5, color=INK)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, colors=MUTE); ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(out / "fig_forest_systems.png", dpi=200); fig.savefig(out / "fig_forest_systems.pdf"); plt.close(fig)


def load_exams():
    from lungcad.wp4.guard import SPLITS
    ex = image_cohort().merge(pd.read_csv(SPLITS), on="pid")
    sc = pd.read_parquet(WP4 / "exam_scores.parquet"); tf = pd.read_parquet(WP4 / "temporal_features.parquet")
    ex = ex.merge(sc.drop(columns=["uid", "fold"]), on=["pid", "round"], how="left").merge(tf, on=["pid", "round"], how="left")
    assert ex[PRIMARY].notna().all(), "exams without a primary score"
    return ex


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--n-boot-secondary", type=int, default=2000); a = ap.parse_args()
    from lungcad.wp4.guard import require_unblinding_allowed
    g = require_unblinding_allowed()
    k1, k4 = known_answer_k1(), M.known_answer_k4()
    if not (k1["pass"] and k4["pass"]):
        sys.exit("K1 or K4 failed - nothing computed")
    OUT.mkdir(parents=True, exist_ok=True)
    ex = load_exams(); res = run(ex, a.n_boot, a.n_boot_secondary); res["meta"] |= g
    json.dump(res, open(OUT / "wp4_results.json", "w"), indent=1, default=float)
    pd.DataFrame(res["systems"]).to_csv(OUT / "systems.csv", index=False); pd.DataFrame(res["subgroups"]).to_csv(OUT / "subgroups.csv", index=False)
    figures(ex, res, OUT); print("outputs ->", OUT)


if __name__ == "__main__":
    main()
