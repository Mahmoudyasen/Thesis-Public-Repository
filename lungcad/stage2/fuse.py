"""Fuse one own-regime stage-2 run with the frozen detector: choose w on the run's VALIDATION candidates, apply it once
to its TEST candidates (method in lungcad.eval.fusion).

    python -m lungcad.stage2.fuse --run-dir work/wp0/stage2/own/C/fold3/seed0
Needs val_candidates.csv (stage2.infer --split val) and test_candidates.csv in the run dir. Writes
test_candidates_fused.csv, test_candidates_lr.csv, froc_test_fused.json, froc_test_lr.json, and fusion.json last
(the queue's done marker for this step).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
from .. import config as C
from ..data.luna16_index import load_index, get_split
from ..eval.froc import evaluate
from ..eval.operating_points import paired_fp_reduction
from ..eval.fusion import METHOD, W_GRID, attach_detector, choose_w, fit_lr, geo, summary, with_prob


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--own-dir", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000); ap.add_argument("--n-boot-paired", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.run_dir); man = json.loads((out / "run_manifest.json").read_text())
    if man["regime"] != "own":
        raise SystemExit(f"fusion needs the own regime (detector probabilities); {out} is {man['regime']}")
    own_dir = Path(a.own_dir) if a.own_dir else C.STAGE1 / "own_candidates"
    split = get_split(load_index(), int(man["fold"]))
    val = attach_detector(pd.read_csv(out / "val_candidates.csv"), own_dir)
    test = attach_detector(pd.read_csv(out / "test_candidates.csv"), own_dir)
    if set(val.seriesuid) - set(split["val"]) or set(test.seriesuid) - set(split["test"]):
        raise SystemExit("candidate files do not match this fold's val/test split")

    w, grid = choose_w(val, split["val"])
    lr, predict = fit_lr(val)
    res = {"method": METHOD, "secondary": "logistic regression on [logit p1, logit p2], fitted on validation (reported only)",
           "fold": man["fold"], "seed": man["seed"], "model": man["model"], "w": w, "w_grid": W_GRID,
           "val": {"detector": grid["0.0"], "stage2": grid["1.0"], "fused": grid[str(w)], "lr_in_sample": summary(val, predict(val), split["val"]),
                   "grid": grid},
           "lr": {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}, "test": {}}
    base = with_prob(test, test.p1.values)
    for tag, p in [("stage2", test.p2.values), ("fused", geo(test, w)), ("lr", predict(test))]:
        if tag != "stage2":
            c = with_prob(test, p); c["cand_id"] = test.cand_id.values; c["label"] = test.label.values
            c.to_csv(out / f"test_candidates_{tag}.csv", index=False)
            evaluate(c, split["test"], out_dir=out, n_bootstrap=a.bootstrap, tag=f"test_{tag}")
        res["test"][tag] = paired_fp_reduction(base, with_prob(test, p), split["test"], n_boot=a.n_boot_paired) \
            | {"sens_at_fp": summary(test, p, split["test"])["sens_at_fp"]}
    tmp = out / "fusion.json.tmp"; tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out / "fusion.json")
    v, t = res["val"], res["test"]["fused"]
    print(f"w={w}  val CPM detector {v['detector']['cpm']:.4f} -> fused {v['fused']['cpm']:.4f} (FP@90 {v['detector']['fp@90']:.3f} -> {v['fused']['fp@90']:.3f})"
          f"  |  test CPM {t['cpm_base']:.4f} -> {t['cpm_model']:.4f}  dCPM {t['delta_cpm']:+.4f} {t['delta_cpm_ci95']}  FP@90 {t['fp_base@90']:.3f} -> {t['fp_model@90']:.3f}")


if __name__ == "__main__":
    main()
