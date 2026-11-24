"""NLST clinical labels from the local CDAS delivery (nlst_780, IDC 2021-05-27): prsn / canc / screen / ctab / ctabc.

Provides the patient-level cohort table and the fixed-time-horizon screen label used by WP4:
    screen_label(round r) = 1  iff  cancer diagnosed within `horizon_days` after screen r
                                    (candx_days - scr_days_r in [0, horizon])
Cohort study (full-cohort labels: 53,452 participants, 26,254 with images locally).
"""
from __future__ import annotations
import pandas as pd, numpy as np
from .. import config as C


def load_tables() -> dict:
    d = C.NLST_CLINICAL
    return {k: pd.read_csv(d / f"nlst_780_{k}_idc_20210527.csv", low_memory=False) for k in ["prsn", "canc", "screen", "ctab", "ctabc"]}


def patient_table(horizon_days: int = 365) -> pd.DataFrame:
    t = load_tables(); p = t["prsn"].copy()
    p["cancer"] = p.candx_days.notna().astype(int)
    for r in range(3):
        dt = p.candx_days - p[f"scr_days{r}"]
        p[f"label_T{r}_h{horizon_days}"] = ((dt >= 0) & (dt <= horizon_days)).astype(int)
        p[f"label_T{r}_any"] = (dt >= 0).astype(int)
        p[f"screen_result_T{r}"] = p[f"scr_res{r}"]
    stage = t["canc"].sort_values("candx_days").drop_duplicates("pid")[["pid", "de_stag_7thed", "de_type", "lesionsize", "lc_topog"]]
    return p.merge(stage, on="pid", how="left")


def abnormality_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ctab: per-abnormality (type code 51 = non-calcified nodule >= 4 mm, diameters in mm, slice number);
    ctabc: interval-change records (growth, attenuation change, pre-existing) — WP1 temporal ground truth."""
    t = load_tables(); return t["ctab"], t["ctabc"]


def farina_nodules() -> pd.DataFrame:
    """Farina et al. 2025 (Zenodo): 1,996 nodule locations for 450 NLST patients as (i,j,k) voxel indices in the
    native series (columns i,j,k = column,row,slice), with a per-nodule cancer label and SeriesInstanceUID."""
    return pd.read_csv(C.NLST_FARINA)


if __name__ == "__main__":
    p = patient_table()
    print("participants", len(p), "cancer", int(p.cancer.sum()))
    for r in range(3):
        print(f"T{r}: screened {int(p[f'scr_days{r}'].notna().sum())}, cancer within 365 d {int(p[f'label_T{r}_h365'].sum())}, screen result codes {p[f'scr_res{r}'].value_counts().to_dict()}")
