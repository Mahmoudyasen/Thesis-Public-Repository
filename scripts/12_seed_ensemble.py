"""Seed ensemble (pre-declared 12 Oct 2026, before the S6 results): a SECONDARY, labelled system.

    python scripts/12_seed_ensemble.py [--seeds 0 1 2] [--out DIR] [--n-boot 2000]

Per regime / model / fold, the stage-2 probability of every candidate is averaged over the seeds' runs
(test_candidates.csv; own regime also val_candidates.csv). Official regime: the mean is the score. Own regime: the mean is
fused with the frozen detector exactly as in lungcad.stage2.fuse (w re-chosen on that fold's validation, applied once to
test). The 10 test folds are then pooled (888 scans) and scored like 08_results_tables.py. The 3-seed mean stays the main
result; the ensemble never enters the C-vs-B rule.

  ensemble_folds.csv      per regime / model / fold: test CPM of the ensemble (stage 2, fused) and the chosen w
  ensemble_fp.json        own regime, pooled: ensemble (stage 2 alone, fused) vs the detector alone (paired bootstrap)
  ensemble_paired.json    pooled: C vs B, B vs A, C vs A on the ensemble; and per model ensemble vs each single seed
Check: with --seeds 0 the fused w and test CPM per fold must equal that seed's fusion.json / froc_test_fused.json.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.luna16_index import load_index, get_split
from lungcad.eval.bootstrap import paired_delta_cpm
from lungcad.eval.fusion import attach_detector, choose_w, geo, summary, with_prob
from lungcad.eval.operating_points import paired_fp_reduction

MODELS = ("A", "B", "C")


def mean_over_seeds(root: Path, regime: str, model: str, fold: int, seeds: list[int], name: str) -> pd.DataFrame:
    tabs = [pd.read_csv(root / regime / model / f"fold{fold}" / f"seed{s}" / name).sort_values("cand_id").reset_index(drop=True) for s in seeds]
    for t in tabs[1:]:
        assert (t.cand_id.values == tabs[0].cand_id.values).all(), f"{regime}/{model}/fold{fold} {name}: candidate sets differ between seeds"
    out = tabs[0].drop(columns="probability").copy()
    out["probability"] = sum(t.probability.values for t in tabs) / len(tabs)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(C.STAGE2)); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=str(C.RESULTS / "seed_ensemble")); ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--own-dir", default=str(C.STAGE1 / "own_candidates"))
    a = ap.parse_args(); root, out, own_dir = Path(a.root), Path(a.out), Path(a.own_dir); out.mkdir(parents=True, exist_ok=True)
    idx = load_index(); uids = idx.seriesuid.astype(str).tolist()
    rows, pooled = [], {}
    for regime in ("official", "own"):
        for model in MODELS:
            parts = {"stage2": [], "fused": [], "det": []}
            for fold in range(10):
                split = get_split(idx, fold)
                test = mean_over_seeds(root, regime, model, fold, a.seeds, "test_candidates.csv")
                row = {"regime": regime, "model": model, "fold": fold, "cpm_stage2": summary(test, test.probability.values, split["test"])["cpm"]}
                parts["stage2"].append(test)
                if regime == "own":
                    val = attach_detector(mean_over_seeds(root, regime, model, fold, a.seeds, "val_candidates.csv"), own_dir)
                    t = attach_detector(test, own_dir)
                    w, _ = choose_w(val, split["val"])
                    fused = with_prob(t, geo(t, w))
                    row |= {"w": w, "cpm_fused": summary(t, geo(t, w), split["test"])["cpm"]}
                    parts["fused"].append(fused); parts["det"].append(with_prob(t, t.p1.values))
                rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            for score, lst in parts.items():
                if lst:
                    pooled[(regime, model, score)] = pd.concat(lst, ignore_index=True)
    pd.DataFrame(rows).to_csv(out / "ensemble_folds.csv", index=False)

    fp = []
    for model in MODELS:
        base = pooled[("own", model, "det")]
        for score in ("stage2", "fused"):
            c = pooled[("own", model, score)]
            r = paired_fp_reduction(base, with_prob(c, c.probability.values), uids, n_boot=a.n_boot) | {"model": model, "score": score, "seeds": a.seeds}
            fp.append(r)
            print(f"own ensemble {model} {score:>6}: CPM {r['cpm_base']:.3f} -> {r['cpm_model']:.3f}  ΔCPM {r['delta_cpm']:+.3f} {r['delta_cpm_ci95']}"
                  f"  FP@90 {r['fp_base@90']:.3f} -> {r['fp_model@90']:.3f} ({r['fp_reduction@90']:+.0%})  FP@95 {r['fp_base@95']:.3f} -> {r['fp_model@95']:.3f} ({r['fp_reduction@95']:+.0%})", flush=True)
    json.dump(fp, open(out / "ensemble_fp.json", "w"), indent=1)

    boots = []
    for regime, score in [("official", "stage2"), ("own", "fused"), ("own", "stage2")]:
        for x, y in [("C", "B"), ("B", "A"), ("C", "A")]:
            r = paired_delta_cpm(pooled[(regime, x, score)], pooled[(regime, y, score)], uids, n_boot=a.n_boot) \
                | {"kind": "model", "regime": regime, "score": score, "a": x, "b": y}
            boots.append(r); print(f"ensemble {regime} {score} {x} vs {y}: Δ {r['delta']:+.4f} {r['delta_ci95']}", flush=True)
        if len(a.seeds) > 1:
            fname = {"stage2": "test_candidates.csv", "fused": "test_candidates_fused.csv"}[score]
            for model in MODELS:
                for s in a.seeds:
                    single = pd.concat([pd.read_csv(root / regime / model / f"fold{k}" / f"seed{s}" / fname) for k in range(10)], ignore_index=True)
                    r = paired_delta_cpm(pooled[(regime, model, score)], single, uids, n_boot=a.n_boot) \
                        | {"kind": "ensemble_vs_seed", "regime": regime, "score": score, "model": model, "seed": s}
                    boots.append(r); print(f"ensemble vs seed{s} {regime} {score} {model}: Δ {r['delta']:+.4f} {r['delta_ci95']}", flush=True)
    json.dump(boots, open(out / "ensemble_paired.json", "w"), indent=1)


if __name__ == "__main__":
    main()
