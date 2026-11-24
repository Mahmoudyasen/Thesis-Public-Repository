"""WP4 secondary: detection sensitivity for the Farina et al. 2025 NLST nodule locations (plan Section 7).

    python scripts/46_wp4_farina.py map            # label-free: native (i, j, k) -> world, orientation chosen on image HU
    python scripts/46_wp4_farina.py sensitivity    # after inference; needs the committed plan (guard)

map: Farina gives (i, j, k) = (column, row, slice) voxel indices in the native series. World = native origin + index x
native spacing (direction is identity for these series; asserted). Unknown: whether k counts slices upward or
downward and whether i/j are as documented. All four variants are tried; for each, the share of nodules whose
3 x 3 x 3 neighbourhood in the 1.5 mm archive reaches > -500 HU (soft tissue) is compared with the same share for
random in-lung points. The variant with the highest share is used, if it clearly beats random (>= 0.80 and >= 2x random);
otherwise the script stops. No label is read.
sensitivity: a nodule is detected at an operating point if a candidate with score >= the threshold lies within 10 mm.
Operating points (fixed on LUNA16, not NLST): the thresholds at which the system reaches 90 % and 95 % sensitivity on the
pooled 888 LUNA16 test scans (own regime, seed 0), plus 'any candidate' (0.02). Reported overall and by Farina's
per-nodule cancer label, for the detector alone and for B fused.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.preprocess import load_meta, load_volume, world_to_voxel

WP4 = C.WORK / "wp4"; OUT = WP4 / "farina"; HIT_MM = 10.0


def variants(r, meta):
    o = np.array(meta["original_origin_xyz"], float); sp = np.array(meta["original_spacing_zyx"], float)[::-1]; nz = meta["original_shape_zyx"][0]
    out = {}
    for flip in (False, True):
        for swap in (False, True):
            i, j = (r.j, r.i) if swap else (r.i, r.j); k = (nz - 1 - r.k) if flip else r.k
            out[(flip, swap)] = o + np.array([i, j, k], float) * sp
    return out


def dense(arr, v):
    z, y, x = np.round(v).astype(int)
    if not (0 <= z < arr.shape[0] and 0 <= y < arr.shape[1] and 0 <= x < arr.shape[2]):
        return False
    nb = arr[max(0, z - 1):z + 2, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
    return bool(nb.max() > -500)


def map_nodules():
    from lungcad.data.nlst import usable_series
    f = pd.read_csv(C.NLST_FARINA, dtype={"series": str}); u = usable_series()
    m = f.merge(u[["uid", "stem", "pid", "round"]], left_on="series", right_on="uid"); rng = np.random.RandomState(0)
    hits = {v: [] for v in [(a, b) for a in (False, True) for b in (False, True)]}; rand = []; worlds = {v: [] for v in hits}
    for stem, g in m.groupby("stem"):
        arr, meta = load_volume(stem); assert np.allclose(meta["direction"], np.eye(3).ravel()), stem
        lung = np.argwhere(arr < -500)
        for r in g.itertuples():
            for v, w in variants(r, meta).items():
                worlds[v].append(w); hits[v].append(dense(arr, world_to_voxel(meta, w)[0]))
            p = lung[rng.randint(len(lung))]; rand.append(dense(arr, p))
    share = {f"flip_k={a},swap_ij={b}": float(np.mean(h)) for (a, b), h in hits.items()}; base = float(np.mean(rand))
    best = max(hits, key=lambda v: np.mean(hits[v])); bs = float(np.mean(hits[best]))
    res = {"n_nodules": int(len(m)), "n_farina": int(len(f)), "dense_share": share, "random_in_lung_share": base, "chosen": {"flip_k": best[0], "swap_ij": best[1]},
           "accepted": bool(bs >= 0.80 and bs >= 2 * base)}
    print(json.dumps(res, indent=1))
    if not res["accepted"]:
        sys.exit("no orientation clearly fits the images - mapping NOT written")
    W = np.array(worlds[best]); OUT.mkdir(parents=True, exist_ok=True)
    m = m.sort_values("stem", kind="stable").reset_index(drop=True)   # same order as the loop (groupby sorts by stem)
    out = m[["PatientID", "Time", "ID_nodule", "series", "pid", "round", "i", "j", "k"]].copy()
    out["coordX"], out["coordY"], out["coordZ"] = W[:, 0], W[:, 1], W[:, 2]; out["dense"] = hits[best]
    out.to_csv(OUT / "farina_world.csv", index=False); json.dump(res, open(OUT / "mapping.json", "w"), indent=1)
    print("wrote", OUT / "farina_world.csv")


def luna_thresholds():
    """Thresholds where detector alone and B fused (own regime, seed 0, pooled 888 scans) first reach 90 / 95 % sensitivity."""
    from lungcad.eval.froc import per_scan_results
    from lungcad.data.luna16_index import load_index
    uids = load_index().seriesuid.astype(str).tolist(); out = {}
    fused = pd.concat([pd.read_csv(C.STAGE2 / "own" / "B" / f"fold{k}" / "seed0" / "test_candidates_fused.csv") for k in range(10)])
    det = pd.read_csv(C.STAGE1 / "own_candidates" / "own_candidates_all.csv")
    for name, df in (("p1", det), ("fused_B_s0", fused)):
        rs = per_scan_results(df, uids); n = sum(r.n_nodules for r in rs)
        tp = np.sort(np.concatenate([r.tp_probs for r in rs]))[::-1]
        out[name] = {f"sens{int(t * 100)}": float(tp[int(np.ceil(t * n)) - 1]) for t in (0.90, 0.95)} | {"any": 0.02}
    return out


def sensitivity():
    from lungcad.wp4.guard import require_unblinding_allowed
    import importlib.util
    g = require_unblinding_allowed()
    spec = importlib.util.spec_from_file_location("agg", Path(__file__).with_name("43_wp4_aggregate.py")); agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
    dec = json.load(open(WP4 / "input_decision.json")); cands, _ = agg.load_inference(WP4 / "inference" / f"{dec['input']}_all")
    nod = pd.read_csv(OUT / "farina_world.csv", dtype={"series": str}); lab = pd.read_csv(C.NLST_FARINA, dtype={"series": str})[["ID_nodule", "series", "Cancer"]]
    nod = nod.merge(lab, on=["ID_nodule", "series"], how="left"); thr = luna_thresholds(); by = {k: v for k, v in cands.groupby("series")}
    rows = []
    for r in nod.itertuples():
        c = by.get(r.series)
        d = np.linalg.norm(c[["coordX", "coordY", "coordZ"]].values - np.array([r.coordX, r.coordY, r.coordZ]), axis=1) if c is not None else np.zeros(0)
        row = {"ID_nodule": r.ID_nodule, "series": r.series, "Cancer": r.Cancer}
        for col, ts in thr.items():
            best = c[col].values[d <= HIT_MM].max() if c is not None and (d <= HIT_MM).any() else 0.0
            for tn, t in ts.items():
                row[f"{col}@{tn}"] = bool(best >= t)
        rows.append(row)
    res = pd.DataFrame(rows); res.to_csv(OUT / "farina_detection.csv", index=False)
    summ = {"thresholds_from_luna16": thr, **g}
    for subset, m in (("all", np.ones(len(res), bool)), ("cancer", res.Cancer.values == 1), ("benign", res.Cancer.values == 0)):
        summ[subset] = {"n": int(m.sum()), **{c: float(res[c][m].mean()) for c in res.columns if "@" in c}}
    json.dump(summ, open(OUT / "farina_sensitivity.json", "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["map", "sensitivity"]); a = ap.parse_args()
    map_nodules() if a.mode == "map" else sensitivity()
