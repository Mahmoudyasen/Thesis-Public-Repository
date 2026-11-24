"""Collect every stage-2 run under work/wp0/stage2 into result tables.

    python scripts/08_results_tables.py [--root DIR] [--n-boot 10000]

  stage2_all_runs.csv    one row per run and score type: froc_test.json (stage 2 alone) and, in the own regime,
                         froc_test_fused.json (detector x stage 2, w chosen on that fold's validation) and froc_test_lr.json
  stage2_cpm_table.csv   mean ± SD over runs per regime / model / score type
  fp_reduction.json      own regime, pooled 10-fold test per seed: FPs/scan at 90 / 95 % sensitivity and ΔCPM against
                         the frozen detector alone, paired bootstrap (the title's numbers)
  paired_bootstrap.json  pooled 10-fold test per seed: C vs B, B vs A, C vs A (official: stage 2; own: fused = primary,
                         stage 2 alone = secondary). "C beats B" needs a CI above 0 with the same sign for every seed.
A pooled comparison uses a (regime, model, seed) only when all 10 folds are present.
"""
import argparse, glob, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.eval.bootstrap import paired_delta_cpm
from lungcad.eval.fusion import attach_detector, with_prob
from lungcad.eval.operating_points import paired_fp_reduction
from lungcad.data.luna16_index import load_index

SCORES = {"stage2": ("froc_test.json", "test_candidates.csv"), "fused": ("froc_test_fused.json", "test_candidates_fused.csv"),
          "lr": ("froc_test_lr.json", "test_candidates_lr.csv")}
MODELS = ("A", "B", "C")


def pooled(root: Path, regime: str, model: str, seed: int, score: str):
    files = sorted(glob.glob(str(root / regime / model / "fold*" / f"seed{seed}" / SCORES[score][1])))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if len(files) == 10 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(C.STAGE2)); ap.add_argument("--out", default=str(C.RESULTS))
    ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--own-dir", default=str(C.STAGE1 / "own_candidates"))
    a = ap.parse_args(); root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for score, (fj, _) in SCORES.items():
        for f in glob.glob(str(root / "*" / "*" / "fold*" / "seed*" / fj)):
            p = Path(f); r = json.load(open(f))
            rows.append({"regime": p.parts[-5], "model": p.parts[-4], "score": score, "fold": int(p.parts[-3][4:]), "seed": int(p.parts[-2][4:]),
                         "cpm": r["cpm"], **{f"s@{fp}": s for fp, s in zip(r["fp_points"], r["sens_at_fp"])}})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no results yet")
    df.sort_values(["regime", "model", "score", "seed", "fold"]).to_csv(out / "stage2_all_runs.csv", index=False)
    tab = df.groupby(["regime", "model", "score"]).agg(cpm_mean=("cpm", "mean"), cpm_std=("cpm", "std"), n_runs=("cpm", "size")).reset_index()
    print(tab.to_string()); tab.to_csv(out / "stage2_cpm_table.csv", index=False)

    uids = load_index().seriesuid.astype(str).tolist(); fp_red, boots = [], []
    for seed in sorted(df.seed.unique()):
        for model in MODELS:
            s2 = pooled(root, "own", model, seed, "stage2")
            if s2 is None:
                continue
            det = attach_detector(s2, Path(a.own_dir)); base = with_prob(det, det.p1.values)
            for score in ("stage2", "fused", "lr"):
                c = s2 if score == "stage2" else pooled(root, "own", model, seed, score)
                if c is None:
                    continue
                r = paired_fp_reduction(base, c, uids, n_boot=min(a.n_boot, 2000)) | {"seed": int(seed), "model": model, "score": score}
                fp_red.append(r)
                print(f"own seed{seed} {model} {score:>6}: CPM {r['cpm_base']:.3f} -> {r['cpm_model']:.3f}  ΔCPM {r['delta_cpm']:+.3f} {r['delta_cpm_ci95']}"
                      f"  FP@90 {r['fp_base@90']:.3f} -> {r['fp_model@90']:.3f} (reduction {r['fp_reduction@90']:+.0%})  FP@95 {r['fp_base@95']:.3f} -> {r['fp_model@95']:.3f}")
        for regime, score in [("official", "stage2"), ("own", "fused"), ("own", "stage2")]:
            pool = {m: pooled(root, regime, m, seed, score) for m in MODELS}
            for x, y in [("C", "B"), ("B", "A"), ("C", "A")]:
                if pool[x] is not None and pool[y] is not None:
                    r = paired_delta_cpm(pool[x], pool[y], uids, n_boot=a.n_boot) | {"regime": regime, "score": score, "seed": int(seed), "a": x, "b": y}
                    boots.append(r); print(f"{regime} {score} seed{seed} {x} vs {y}: Δ {r['delta']:+.4f} {r['delta_ci95']}")
    json.dump(fp_red, open(out / "fp_reduction.json", "w"), indent=1)
    json.dump(boots, open(out / "paired_bootstrap.json", "w"), indent=1)


if __name__ == "__main__":
    main()
