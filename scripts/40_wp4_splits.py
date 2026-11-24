"""WP4 participant-level 5-fold split for the trained aggregators (plan Section 5), with known-answer check K5.

    python scripts/40_wp4_splits.py            # writes nlst/meta/wp4_splits.csv (refuses to overwrite) + prints SHA-256

Participants of the image cohort (exams with a usable series, plan Section 2.1), stratified by "has any cancer-present
exam" (Pinsky definition), StratifiedKFold(5, shuffle, random_state=0) over participants sorted by pid. Uses labels
only; no model output exists or is read.
K5: every participant in exactly one fold; cancer participants per fold within ±1 of each other; all exams of a
participant share a fold; the file's SHA-256 is printed for the NOTES file.
"""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.wp4.labels import image_cohort

OUT = C.NLST_PRE.parent / "meta" / "wp4_splits.csv"


def main():
    if OUT.exists():
        print(f"{OUT} exists - not overwritten. sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}"); return
    e = image_cohort()
    part = e.groupby("pid").cancer.any().reset_index().sort_values("pid").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0); part["fold"] = -1
    for k, (_, te) in enumerate(skf.split(part.pid, part.cancer)):
        part.loc[te, "fold"] = k
    # K5
    assert (part.fold >= 0).all() and part.pid.is_unique
    per = part.groupby("fold").cancer.sum(); assert per.max() - per.min() <= 1, per.to_dict()
    ex = e.merge(part[["pid", "fold"]], on="pid"); assert ex.groupby("pid").fold.nunique().max() == 1 and len(ex) == len(e)
    part[["pid", "fold"]].to_csv(OUT, index=False)
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"K5 PASS: {len(part)} participants, cancer participants per fold {per.to_dict()}, exams per fold {ex.fold.value_counts().sort_index().to_dict()}")
    print(f"wrote {OUT}\nsha256 {h}")


if __name__ == "__main__":
    main()
