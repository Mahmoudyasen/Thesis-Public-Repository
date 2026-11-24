"""False-positive reduction at fixed sensitivity: the numbers the thesis title needs.

  fp_at_sensitivity : FPs/scan needed to reach a target sensitivity (first point on the FROC curve with sens >= target;
                      NaN if the candidate set never reaches it, e.g. above the stage-1 recall ceiling in the own regime)
  paired_fp_reduction : paired bootstrap over scans between a baseline and a model scored on the SAME scans -> FPs/scan of
                      each, % reduction (1 - model/baseline) with 95 % CI, plus paired ΔCPM. In the own regime the
                      baseline is the frozen stage-1 candidate list with its own detector probabilities.
Uses the official counting rules via froc.per_scan_results (≤ 100 marks per scan, excluded findings ignored).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .froc import per_scan_results, froc_from_results

TARGETS = (0.90, 0.95)


def fp_at_sensitivity(results, targets=TARGETS) -> np.ndarray:
    fps, sens, _, _ = froc_from_results(results)
    out = np.full(len(targets), np.nan)
    for i, t in enumerate(targets):
        k = np.nonzero(sens >= t - 1e-12)[0]
        if len(k):
            out[i] = fps[k[0]]
    return out


def _ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]


def paired_fp_reduction(base: pd.DataFrame, model: pd.DataFrame, uids: list[str], targets=TARGETS, n_boot: int = 2000, seed: int = 0) -> dict:
    rb = per_scan_results(base, uids); rm = per_scan_results(model, uids)
    fb, fm = fp_at_sensitivity(rb, targets), fp_at_sensitivity(rm, targets)
    cb, cm = froc_from_results(rb)[3], froc_from_results(rm)[3]
    rng = np.random.RandomState(seed); B = np.full((n_boot, len(targets)), np.nan); M = B.copy(); dc = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(uids), len(uids)); sb = [rb[j] for j in idx]; sm = [rm[j] for j in idx]
        B[i], M[i] = fp_at_sensitivity(sb, targets), fp_at_sensitivity(sm, targets)
        dc[i] = froc_from_results(sm)[3] - froc_from_results(sb)[3]
    with np.errstate(divide="ignore", invalid="ignore"):
        red = 1 - M / B
    out = {"n_scans": len(uids), "n_boot": n_boot, "cpm_base": cb, "cpm_model": cm, "delta_cpm": cm - cb, "delta_cpm_ci95": _ci(dc),
           "p_delta_cpm_le_0": float((dc <= 0).mean())}
    for i, t in enumerate(targets):
        k = f"{int(round(t * 100))}"
        out[f"fp_base@{k}"] = float(fb[i]); out[f"fp_model@{k}"] = float(fm[i])
        out[f"fp_reduction@{k}"] = float(1 - fm[i] / fb[i]) if np.isfinite(fb[i]) and fb[i] > 0 and np.isfinite(fm[i]) else np.nan
        out[f"fp_reduction@{k}_ci95"] = _ci(red[:, i]); out[f"fp_base@{k}_ci95"] = _ci(B[:, i]); out[f"fp_model@{k}_ci95"] = _ci(M[:, i])
        out[f"reachable_frac_boot@{k}"] = float(np.isfinite(red[:, i]).mean())
    return out
