"""WP4 exam table: one row per NLST CT screening exam with the Pinsky et al. (2015) outcome definition.

Plan: report/wp4_nlst_prespec.md, Section 2.
  * exam = participant x round with an official screen result scr_res{r} in 1..6
  * radiologist positive = scr_res in {4, 5, 6}
  * baseline = the participant's FIRST screen received (may be T1 or T2); every later screen = after baseline
  * cancer present (Pinsky): diagnosed within 365 d of the screen and before the next screen, OR the screen was positive
    and the NLST attributed the cancer to that screen's study year (cancyr == r: the diagnostic-procedure chain)
  * cancer absent = not present (primary); sensitivity analysis drops not-present exams with < 365 d cancer-free follow-up
known_answer_k1() reproduces Pinsky Table 4 on the full CT arm of the IDC delivery.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from lungcad import config as C

POSITIVE_CODES = (4, 5, 6)
PINSKY = {"baseline": {"present": 292, "tp": 273, "absent": 26090, "fp": 6939, "screens": 26455},
          "after": {"present": 401, "tp": 376, "absent": 48197, "fp": 10512, "screens": 48671}}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(C.NLST_CLINICAL / f"nlst_780_{name}_idc_20210527.csv", low_memory=False)


def exam_table() -> pd.DataFrame:
    """All CT-arm exams (no image filter). Columns: pid, round, scr_res, rad_pos, scr_days, baseline, group,
    cancer (Pinsky present), fu_days (cancer-free follow-up after the screen), fu_ge_365, can_scr, cancyr."""
    p, s = _read("prsn"), _read("screen")
    p = p[p.pid.isin(set(s.pid))]                         # CT arm = participants with CT screen records
    rows = []
    for r in range(3):
        x = p[["pid", "candx_days", "canc_free_days", "cancyr", "can_scr", "de_stag_7thed", "de_type", "lesionsize"]].copy()
        x["round"] = r; x["scr_res"] = p[f"scr_res{r}"]; x["scr_days"] = p[f"scr_days{r}"]
        x["next_days"] = p[f"scr_days{r + 1}"] if r < 2 else np.nan
        rows.append(x)
    e = pd.concat(rows, ignore_index=True)
    e = e[e.scr_res.between(1, 6)].copy()
    e["rad_pos"] = e.scr_res.isin(POSITIVE_CODES)
    e["baseline"] = e.groupby("pid")["round"].transform("min") == e["round"]
    # position among the screens received (screening history, as for baseline); reindex so the order aligns by label
    # (fix 24 Nov 2026: without reindex, np.where read the sorted Series by position and scrambled after1/after2)
    order = e.sort_values(["pid", "round"]).groupby("pid").cumcount().reindex(e.index).values
    e["group"] = np.where(e.baseline, "baseline", np.where(order == 1, "after1", "after2"))
    dt = e.candx_days - e.scr_days
    within = (dt >= 0) & (dt <= 365) & (e.next_days.isna() | (e.candx_days < e.next_days))
    e["cancer"] = (within | (e.rad_pos & (e.cancyr == e["round"]))).astype(bool)
    e["fu_days"] = e.canc_free_days - e.scr_days
    e["fu_ge_365"] = e.fu_days >= 365
    return e.drop(columns=["next_days"]).reset_index(drop=True)


def op_point(e: pd.DataFrame, pos_col: str = "rad_pos") -> dict:
    c = e.cancer.values; q = e[pos_col].values
    return {"n": len(e), "present": int(c.sum()), "tp": int((c & q).sum()), "absent": int((~c).sum()), "fp": int((~c & q).sum()),
            "sens": float((c & q).sum() / max(1, c.sum())), "fpr": float((~c & q).sum() / max(1, (~c).sum()))}


def known_answer_k1(e: pd.DataFrame | None = None) -> dict:
    """K1: present counts and TP numerators must equal Pinsky exactly; FPR within 0.1 percentage point."""
    e = exam_table() if e is None else e
    out, ok = {}, True
    for g, m in (("baseline", e.baseline), ("after", ~e.baseline)):
        r = op_point(e[m]); ref = PINSKY[g]
        exact = r["present"] == ref["present"] and r["tp"] == ref["tp"]
        fpr_ok = abs(r["fpr"] - ref["fp"] / ref["absent"]) < 0.001
        ok &= exact and fpr_ok
        out[g] = r | {"pinsky": ref, "pinsky_fpr": ref["fp"] / ref["absent"], "exact_present_tp": exact, "fpr_within_0.1pp": fpr_ok}
    out["pass"] = bool(ok)
    return out


def image_cohort(e: pd.DataFrame | None = None) -> pd.DataFrame:
    """Exams that also have a usable converted series (lungcad.data.nlst.usable_series); adds stem/vendor/kernel."""
    from lungcad.data.nlst import usable_series
    e = exam_table() if e is None else e
    u = usable_series()[["pid", "round", "uid", "stem", "vendor", "kernel_class", "thk_mm"]]
    return e.merge(u, on=["pid", "round"], how="inner")


if __name__ == "__main__":
    import json
    k = known_answer_k1()
    print(json.dumps({g: {x: k[g][x] for x in ("present", "tp", "absent", "fp", "sens", "fpr", "exact_present_tp", "fpr_within_0.1pp")} for g in ("baseline", "after")}, indent=1))
    print("K1", "PASS" if k["pass"] else "FAIL")
