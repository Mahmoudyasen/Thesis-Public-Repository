"""Detector x stage-2 score fusion for the own regime, always chosen on VALIDATION only.

  primary   : weighted geometric mean  p = p1^(1-w) * p2^w,  w in W_GRID (0 = detector alone, 1 = stage 2 alone);
              w = the highest validation CPM, ties -> the smaller w (closer to the frozen detector)
  secondary : logistic regression on [logit p1, logit p2], fitted on the validation candidates (reported only)
p1 is the frozen stage-1 probability from own_candidates_all.csv, attached through cand_id (load_candidates numbers
the rows of that file 0..n-1). For every scan it is out-of-fold: it comes from the stage-1 model whose test split held it.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from .froc import per_scan_results, froc_from_results
from .operating_points import fp_at_sensitivity

W_GRID = [round(0.1 * i, 1) for i in range(11)]
EPS = 1e-6
METHOD = "weighted geometric mean p1^(1-w)*p2^w; w = max validation CPM over W_GRID, ties -> smaller w"


def attach_detector(s2: pd.DataFrame, own_dir: Path) -> pd.DataFrame:
    """Stage-2 candidate table (probability = p2) -> same rows with columns p1 (detector) and p2 (stage 2)."""
    det = pd.read_csv(Path(own_dir) / "own_candidates_all.csv")
    d = det.iloc[s2.cand_id.values]
    assert (d.seriesuid.values == s2.seriesuid.values).all() and np.allclose(d.coordX.values, s2.coordX.values), \
        "cand_id does not match own_candidates_all.csv (was the frozen candidate file changed?)"
    out = s2.rename(columns={"probability": "p2"}).copy(); out["p1"] = d.probability.values
    return out


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def geo(df: pd.DataFrame, w: float) -> np.ndarray:
    return np.exp((1 - w) * np.log(np.clip(df.p1.values, EPS, 1)) + w * np.log(np.clip(df.p2.values, EPS, 1)))


def with_prob(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    out = df[["seriesuid", "coordX", "coordY", "coordZ"]].copy(); out["probability"] = p
    return out


def summary(df: pd.DataFrame, p: np.ndarray, uids: list[str]) -> dict:
    r = per_scan_results(with_prob(df, p), uids)
    _, _, s_at, cpm = froc_from_results(r); fp = fp_at_sensitivity(r)
    return {"cpm": cpm, "sens_at_fp": [float(x) for x in s_at], "fp@90": float(fp[0]), "fp@95": float(fp[1])}


def choose_w(val: pd.DataFrame, uids: list[str]) -> tuple[float, dict]:
    grid = {str(w): summary(val, geo(val, w), uids) for w in W_GRID}
    best = max(W_GRID, key=lambda w: (round(grid[str(w)]["cpm"], 10), -w))
    return best, grid


def fit_lr(val: pd.DataFrame):
    """-> (model, predict(df) -> probability). Fitted on the validation candidates only."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[logit(val.p1.values), logit(val.p2.values)], val.label.values)
    return m, lambda d: m.predict_proba(np.c_[logit(d.p1.values), logit(d.p2.values)])[:, 1]
