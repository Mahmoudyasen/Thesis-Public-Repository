"""Rule 3 check (ch3 §3.6): does a more complex model need fewer false positives at 90 % sensitivity than a simpler one?

    python scripts/13_fp_between_models.py [--seeds 0 1 2] [--out DIR] [--n-boot 2000]

Own regime, fused scores, pooled 10-fold test (888 scans) per seed: paired bootstrap of FPs/scan at 90 % and 95 %
sensitivity, B vs A, C vs B and C vs A (reduction = 1 - FP(model) / FP(base), same code as the detector comparison).
  fp_between_models.json
"""
import argparse, glob, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.luna16_index import load_index
from lungcad.eval.operating_points import paired_fp_reduction

PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]


def pooled(root: Path, model: str, seed: int) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "own" / model / "fold*" / f"seed{seed}" / "test_candidates_fused.csv")))
    assert len(files) == 10, f"own/{model}/seed{seed}: {len(files)} of 10 folds"
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(C.STAGE2)); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=str(C.RESULTS)); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(); root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    uids = load_index().seriesuid.astype(str).tolist(); res = []
    for s in a.seeds:
        P = {m: pooled(root, m, s) for m in "ABC"}
        for base, model in PAIRS:
            r = paired_fp_reduction(P[base], P[model], uids, n_boot=a.n_boot) | {"seed": s, "base": base, "model": model}
            res.append(r)
            print(f"seed{s} {model} vs {base}: FP@90 {r['fp_base@90']:.3f} -> {r['fp_model@90']:.3f} ({r['fp_reduction@90']:+.0%}, CI {r['fp_reduction@90_ci95'][0]:+.0%} to {r['fp_reduction@90_ci95'][1]:+.0%})"
                  f"  FP@95 {r['fp_base@95']:.3f} -> {r['fp_model@95']:.3f} ({r['fp_reduction@95']:+.0%}, CI {r['fp_reduction@95_ci95'][0]:+.0%} to {r['fp_reduction@95_ci95'][1]:+.0%})", flush=True)
    json.dump(res, open(out / "fp_between_models.json", "w"), indent=1)


if __name__ == "__main__":
    main()
