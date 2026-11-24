"""Score fusion on fold 0 (own regime): stage-1 detector probability p1 combined with stage-2 probability p2.

    python scripts/s6_fusion_fold0.py              # phase 1: choose the fusion weight on VALIDATION only
    python scripts/s6_fusion_fold0.py --test       # phase 2: apply the frozen choice once to the TEST scans

Fixed before any fused score was computed (written to prereg.json by phase 1):
  * primary method : weighted geometric mean  p = p1^(1-w) * p2^w,  w in {0, 0.1, ..., 1.0}
                     (w = 0 is the detector alone, w = 1 is stage 2 alone); w = the highest validation CPM,
                     ties -> the smaller w (closer to the frozen detector)
  * secondary      : logistic regression on [logit p1, logit p2], fitted on the validation candidates (reported only)
  * primary model  : own-regime C, 40 epochs (the decided S6 setting); C 60 epochs reported as secondary
  * success        : on validation, fused CPM > detector CPM AND fewer FPs/scan at 90 % sensitivity than the detector
  * test           : confirmation only (paired ΔCPM and FP reduction vs. the detector, 95 % bootstrap CIs); it never
                     changes the choice
p1 for validation scans is out-of-fold: own_candidates_all.csv holds, for every scan, the candidates of the frozen
stage-1 model whose test split contained it. p2 comes from <run>/val_candidates.csv (lungcad.stage2.infer --split val).
Analysis only: frozen stage-1 and stage-2 files are read, never written. The S6 queue applies the same method
per fold with lungcad.stage2.fuse (shared code in lungcad.eval.fusion).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.luna16_index import load_index, get_split
from lungcad.eval.fusion import W_GRID, attach_detector, choose_w, fit_lr, geo, summary, with_prob
from lungcad.eval.operating_points import paired_fp_reduction

FOLD = 0
RUNS = {"C_e40": C.WORK / "wp0" / "stage2_dev" / "epoch_study" / "own" / "C_e40" / "fold0" / "seed0",
        "C_e60": C.WORK / "wp0" / "stage2_dev" / "epoch_study" / "own" / "C_e60" / "fold0" / "seed0"}
PRIMARY = "C_e40"
OUT = C.WORK / "wp0" / "stage2_dev" / "fusion_fold0"

PREREG = {"method": "weighted geometric mean p1^(1-w)*p2^w", "w_grid": W_GRID, "select": "max validation CPM, ties -> smaller w",
          "secondary": "logistic regression on [logit p1, logit p2] fitted on validation candidates (reported only)",
          "primary_model": PRIMARY, "success": "validation: fused CPM > detector CPM AND fused FPs/scan@90% < detector FPs/scan@90%",
          "test": "applied once after the choice; confirmation only"}


def merged(run: Path, split: str) -> pd.DataFrame:
    return attach_detector(pd.read_csv(run / f"{split}_candidates.csv"), C.STAGE1 / "own_candidates")


def phase_val(uids: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prereg.json").write_text(json.dumps(PREREG, indent=1))
    res = {"prereg": PREREG, "n_val_scans": len(uids), "runs": {}}
    for name, run in RUNS.items():
        df = merged(run, "val")
        best, grid = choose_w(df, uids)
        m, f = fit_lr(df)
        lr = summary(df, f(df), uids) | {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
        det, fus = grid["0.0"], grid[str(best)]
        ok = fus["cpm"] > det["cpm"] and np.isfinite(fus["fp@90"]) and fus["fp@90"] < det["fp@90"]
        res["runs"][name] = {"grid": grid, "w_best": best, "lr_in_sample": lr, "success": bool(ok)}
        print(f"{name} val: detector CPM {det['cpm']:.4f} FP@90 {det['fp@90']:.3f} | stage2 {grid['1.0']['cpm']:.4f} {grid['1.0']['fp@90']:.3f}"
              f" | fused w={best} {fus['cpm']:.4f} {fus['fp@90']:.3f} | LR(in-sample) {lr['cpm']:.4f} {lr['fp@90']:.3f} | success={ok}")
        print("   CPM by w: " + " ".join(f"{w}:{grid[str(w)]['cpm']:.3f}" for w in W_GRID))
    (OUT / "fusion_val.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT / 'fusion_val.json'}")


def phase_test(val_uids: list[str], test_uids: list[str], n_boot: int):
    val = json.loads((OUT / "fusion_val.json").read_text())
    res = {"note": "w and LR fixed on validation (fusion_val.json); test applied once", "n_test_scans": len(test_uids), "runs": {}}
    for name, run in RUNS.items():
        w = val["runs"][name]["w_best"]
        df = merged(run, "test"); base = with_prob(df, df.p1.values)
        _, f = fit_lr(merged(run, "val"))
        r = {"w": w}
        for tag, p in [("stage2_alone", df.p2.values), (f"fused_w{w}", geo(df, w)), ("lr", f(df))]:
            r[tag] = paired_fp_reduction(base, with_prob(df, p), test_uids, n_boot=n_boot) | {"sens_at_fp": summary(df, p, test_uids)["sens_at_fp"]}
            x = r[tag]
            print(f"{name} test {tag:>14}: CPM {x['cpm_base']:.3f} -> {x['cpm_model']:.3f}  ΔCPM {x['delta_cpm']:+.3f} "
                  f"[{x['delta_cpm_ci95'][0]:+.3f}, {x['delta_cpm_ci95'][1]:+.3f}]  FP@90 {x['fp_base@90']:.2f} -> {x['fp_model@90']:.2f}"
                  f"  FP@95 {x['fp_base@95']:.2f} -> {x['fp_model@95']:.2f}")
        res["runs"][name] = r
    (OUT / "fusion_test.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT / 'fusion_test.json'}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    split = get_split(load_index(), FOLD)
    if a.test:
        phase_test(split["val"], split["test"], a.n_boot)
    else:
        phase_val(split["val"])


if __name__ == "__main__":
    main()
