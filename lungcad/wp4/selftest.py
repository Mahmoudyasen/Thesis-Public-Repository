"""End-to-end test of scripts 43 (aggregators) and 44 (analysis) on a fully SYNTHETIC cohort: no NLST label, image or
model output is used. Planted answers:
  * a 'system' equal to the radiologists' decision (max_p1 := rad_pos) must give ΔFPR vs radiologists = 0 exactly
  * an informative score (cancer exams get one high-scoring candidate) must beat the radiologists (Δ < 0)
  * a temporal feature that marks benign stable nodules must make the temporal GBM better than the single-scan GBM (Δ < 0)
    python -m lungcad.wp4.selftest
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from lungcad import config as C

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def synthetic(n_part=2500, seed=0):
    rng = np.random.RandomState(seed); rows = []
    for p in range(n_part):
        first = 0 if rng.rand() > 0.01 else 1; ca_round = rng.choice([0, 1, 2]) if rng.rand() < 0.04 else -1
        for i, r in enumerate(range(first, 3)):
            c = r == ca_round
            rows.append({"pid": 100000 + p, "round": r, "uid": f"u{p}_{r}", "cancer": c, "baseline": i == 0, "group": ["baseline", "after1", "after2"][i],
                         "rad_pos": rng.rand() < (0.93 if c else (0.26 if i == 0 else 0.20)), "fold": p % 5,
                         "vendor": rng.choice(["GE", "SIEMENS", "PHILIPS", "TOSHIBA"]), "kernel_class": rng.choice(["soft", "sharp"]), "thk_mm": rng.choice([1.0, 2.0, 2.5, 3.2]),
                         "de_stag_7thed": rng.choice([110, 120, 310]) if c else np.nan, "de_type": rng.choice([8140, 8070]) if c else np.nan,
                         "lesionsize": rng.uniform(5, 40) if c else np.nan, "can_scr": rng.choice([1, 2]) if c else 0, "fu_ge_365": rng.rand() > 0.02})
    ex = pd.DataFrame(rows); cand = []; d = 32
    stable = {}
    for e in ex.itertuples():
        n = rng.poisson(5); stable[(e.pid, e.round)] = int(rng.rand() < 0.5 and not e.cancer)
        for j in range(n):
            s = rng.beta(1, 6) + (0.35 * stable[(e.pid, e.round)] * (j == 0))
            cand.append((e.uid, e.pid, e.round, min(s, 1)))
        if e.cancer:
            cand.append((e.uid, e.pid, e.round, rng.uniform(0.5, 1.0)))
    c = pd.DataFrame(cand, columns=["series", "pid", "round", "fused_B_s0"])
    for col in ["p1", "fused_B_s1", "fused_B_s2", "fused_A_s0", "fused_C_s0", "p2_B_s0"]:
        c[col] = np.clip(c.fused_B_s0 + rng.normal(0, 0.05, len(c)), 0, 1)
    c["dz_mm"], c["dy_mm"], c["dx_mm"] = rng.uniform(3, 20, (3, len(c))); c["z_frac"] = rng.rand(len(c))
    emb = rng.normal(0, 1, (len(c), d)).astype(np.float16); emb[:, 0] += (c.fused_B_s0.values * 3).astype(np.float16)
    after = ex[~ex.baseline]
    tf = pd.DataFrame({"pid": after.pid.values, "round": after["round"].values, "no_prior": 0, "reg_failed": 0, "gap_days": 365.0,
                       "prior_max_score": rng.rand(len(after)), "present_prior_of_max": [stable[(p, r)] for p, r in zip(after.pid, after["round"])],
                       "diam_change_of_max": rng.normal(0, 1, len(after)), "score_change_of_max": rng.normal(0, .1, len(after)), "n_conf_new": 0, "n_conf_matched": 0})
    return ex, c, emb, tf


def main():
    agg, ana = _load("43_wp4_aggregate"), _load("44_wp4_analysis")
    ex, c, emb, tf = synthetic(); c["fused_Bens"] = c[["fused_B_s0", "fused_B_s1", "fused_B_s2"]].mean(1)
    sc = agg.aggregate(c, emb, ex, tf, device="cpu")
    sc["max_p1"] = ex.rad_pos.values.astype(float)                      # planted: system == radiologists
    full = ex.merge(sc.drop(columns=["uid", "fold"]), on=["pid", "round"]).merge(tf, on=["pid", "round"], how="left")
    res = ana.run(full, n_boot=300, n_boot2=60, test=True)
    out = C.WORK / "wp4" / "selftest"; out.mkdir(parents=True, exist_ok=True); ana.figures(full, res, out)
    ident = [r for r in res["systems"] if r["system"] == "max_p1" and r["group"] in ("baseline", "after")]
    checks = {"identity_delta_zero": all(r["delta_fpr_vs_rad"] == 0.0 and r["delta_ci95"] == [0.0, 0.0] for r in ident),
              "informative_beats_radiologists_at_baseline": res["H1"]["baseline"]["delta_fpr"] < 0 and res["H1"]["baseline"]["delta_ci95"][1] < 0,
              "temporal_helps": res["H2"]["delta_fpr"] < 0,
              "no_nan_scores": bool(sc[["max_fused_B_s0", "gbm_single", "mil"]].notna().all().all()),
              "after_only_scores_only_after": bool(sc.gbm_temporal_after[ex.baseline.values].isna().all() and sc.gbm_temporal_after[~ex.baseline.values].notna().all()),
              "holm_two": len([g for g in res["H1"] if "p_holm" in res["H1"][g]]) == 2}
    print(checks); print("SELFTEST", "PASS" if all(checks.values()) else "FAIL"); print("figures ->", out)
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
