"""WP4 temporal features (plan Section 6): prior-exam pairing, candidate matching through the registration, features.

Label-free: pairing uses only which rounds have a usable series (image availability) and the DICOM StudyDate.
  pairs_table()          one row per (current series, prior series): prior = most recent earlier round with a usable series
  match_candidates()     current candidate -> nearest prior candidate in the prior frame (T maps current -> prior world);
                         match iff distance <= max(10 mm, 1.5 x the larger of the two diameters)
  temporal_exam_features()  per current exam: features of the exam's top candidate (present_prior, diameter change, score
                         change) + exam-level (prior exam's max score, gap days, any matched confident candidate,
                         no-usable-prior indicator, registration failure indicator)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from lungcad.preprocess import load_meta

MATCH_MIN_MM, MATCH_DIAM_FACTOR = 10.0, 1.5
SCORE = "fused_B_s0"


def study_date(stem) -> pd.Timestamp:
    d = load_meta(stem).get("dicom", {}).get("StudyDate")
    return pd.to_datetime(d, format="%Y%m%d", errors="coerce")


def pairs_table(series: pd.DataFrame) -> pd.DataFrame:
    """series: usable_series() rows (pid, round, uid, stem). -> current/prior pairs."""
    s = series.sort_values(["pid", "round"])
    prev = s.groupby("pid").shift(1)
    p = s.assign(prior_round=prev["round"], prior_uid=prev["uid"], prior_stem=prev["stem"]).dropna(subset=["prior_uid"])
    p["prior_round"] = p.prior_round.astype(int)
    return p[["pid", "round", "uid", "stem", "prior_round", "prior_uid", "prior_stem"]].reset_index(drop=True)


def transform_from_row(r):
    import SimpleITK as sitk
    T = sitk.AffineTransform(3); T.SetCenter(tuple(r["center"])); T.SetMatrix(tuple(r["matrix"])); T.SetTranslation(tuple(r["translation"]))
    return T


def _diam(df):
    return df[["dz_mm", "dy_mm", "dx_mm"]].max(1).values


def match_candidates(cur: pd.DataFrame, prior: pd.DataFrame, T) -> pd.DataFrame:
    """-> cur with columns present_prior (0/1), prior_dist_mm, prior_score, diam_change_mm, score_change (NaN if unmatched)."""
    out = cur.copy()
    out["present_prior"] = 0; out["prior_dist_mm"] = np.nan; out["prior_score"] = np.nan; out["diam_change_mm"] = np.nan; out["score_change"] = np.nan
    if not len(cur) or not len(prior):
        return out
    xyz = cur[["coordX", "coordY", "coordZ"]].values
    mapped = np.array([T.TransformPoint(tuple(float(v) for v in p)) for p in xyz])
    Q = prior[["coordX", "coordY", "coordZ"]].values
    d = np.linalg.norm(mapped[:, None, :] - Q[None], axis=2); j = d.argmin(1); dmin = d[np.arange(len(cur)), j]
    dc, dp = _diam(cur), _diam(prior)[j]
    ok = dmin <= np.maximum(MATCH_MIN_MM, MATCH_DIAM_FACTOR * np.maximum(dc, dp))
    ps = prior[SCORE].values[j]
    out["present_prior"] = ok.astype(int); out["prior_dist_mm"] = dmin
    out.loc[ok, "prior_score"] = ps[ok]; out.loc[ok, "diam_change_mm"] = (dc - dp)[ok]; out.loc[ok, "score_change"] = (cur[SCORE].values - ps)[ok]
    return out


def temporal_exam_features(cands: pd.DataFrame, pairs: pd.DataFrame, reg: pd.DataFrame, exams: pd.DataFrame) -> pd.DataFrame:
    """cands: all candidates (series, pid, round, coords, sizes, SCORE); pairs: pairs_table(); reg: registration results
    (uid -> matrix/translation/center/ok); exams: the exams to describe (pid, round, uid).
    -> one row per exam: pid, round + temporal features (exams without a usable prior get no_prior = 1, others NaN-filled)."""
    by = {k: g for k, g in cands.groupby("series")}; empty = cands.iloc[:0]
    pr = pairs.set_index("uid"); rg = reg.set_index("uid") if len(reg) else reg
    rows = []
    for r in exams.itertuples():
        f = {"pid": r.pid, "round": r.round, "no_prior": 1, "reg_failed": 0, "gap_days": np.nan, "prior_max_score": np.nan,
             "present_prior_of_max": np.nan, "diam_change_of_max": np.nan, "score_change_of_max": np.nan, "n_conf_new": np.nan, "n_conf_matched": np.nan}
        if r.uid in pr.index:
            p = pr.loc[r.uid]; prior = by.get(p.prior_uid, empty)
            f["no_prior"] = 0; f["prior_max_score"] = float(prior[SCORE].max()) if len(prior) else 0.0
            f["gap_days"] = float((study_date(p.stem) - study_date(p.prior_stem)).days) if "stem" in p else np.nan
            ok = r.uid in rg.index and bool(rg.loc[r.uid, "ok"])
            if not ok:                                    # plan: a registration failure gets the missing-prior indicator
                f["reg_failed"] = 1; f["no_prior"] = 1
            else:
                cur = by.get(r.uid, empty)
                if len(cur):
                    m = match_candidates(cur, prior, transform_from_row(rg.loc[r.uid])).sort_values(SCORE, ascending=False)
                    top = m.iloc[0]
                    f["present_prior_of_max"] = float(top.present_prior); f["diam_change_of_max"] = float(top.diam_change_mm)
                    f["score_change_of_max"] = float(top.score_change)
                    conf = m[m[SCORE] >= 0.5]; f["n_conf_new"] = int((conf.present_prior == 0).sum()); f["n_conf_matched"] = int(conf.present_prior.sum())
        rows.append(f)
    return pd.DataFrame(rows)
