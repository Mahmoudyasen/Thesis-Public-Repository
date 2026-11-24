"""WP4 registration of every current exam to its prior (plan Section 6), CPU, parallel, resumable. Label-free.

    python scripts/42_wp4_register.py --procs 6 [--limit N]     # all pairs -> work/wp4/registration/reg_NNNNN.parquet
    python scripts/42_wp4_register.py --qc-montage 30           # the 30-pair visual check (PNG pages for visual review)
    python scripts/42_wp4_register.py --summary                 # success rate (Dice >= 0.85), Dice distribution

Each row: uid (current), pid, round, prior_uid, prior_round, matrix (9), translation (3), center (3), dice_rigid, dice, ok,
mask methods, seconds. T maps current world -> prior world (lungcad.wp4.register).
"""
import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C

OUT = C.WORK / "wp4" / "registration"; CHUNK = 500


def job(r):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(os.environ.get("REG_THREADS", "4")))
    from lungcad.wp4.register import mask_image, register_masks, DICE_MIN
    t = time.time()
    try:
        fm, fmeta = mask_image(r["stem"]); mm, mmeta = mask_image(r["prior_stem"])
        T, q = register_masks(fm, mm)
        return {"uid": r["uid"], "pid": r["pid"], "round": r["round"], "prior_uid": r["prior_uid"], "prior_round": r["prior_round"],
                "matrix": list(T.GetMatrix()), "translation": list(T.GetTranslation()), "center": list(T.GetCenter()),
                "dice_rigid": q["dice_rigid"], "dice": q["dice"], "ok": q["dice"] >= DICE_MIN, "mask_cur": fmeta["reg_mask_method"],
                "mask_prior": mmeta["reg_mask_method"], "seconds": time.time() - t, "error": ""}
    except Exception as e:                                         # recorded as a failure, never dropped
        return {"uid": r["uid"], "pid": r["pid"], "round": r["round"], "prior_uid": r["prior_uid"], "prior_round": r["prior_round"],
                "matrix": [np.nan] * 9, "translation": [np.nan] * 3, "center": [np.nan] * 3, "dice_rigid": np.nan, "dice": np.nan, "ok": False,
                "mask_cur": "", "mask_prior": "", "seconds": time.time() - t, "error": repr(e)[:300]}


def load_results() -> pd.DataFrame:
    fs = sorted(OUT.glob("reg_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def run(procs, limit):
    from lungcad.data.nlst import usable_series
    from lungcad.wp4.temporal import pairs_table
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pairs_table(usable_series()); done = load_results()
    todo = pairs[~pairs.uid.isin(set(done.uid))] if len(done) else pairs
    if limit:
        todo = todo.head(limit)
    print(f"{len(pairs)} pairs, {len(done)} done, {len(todo)} to go", flush=True)
    k = len(list(OUT.glob("reg_*.parquet"))); buf = []; t0 = time.time()
    with Pool(procs) as pool:
        for n, res in enumerate(pool.imap_unordered(job, todo.to_dict("records"), chunksize=4), 1):
            buf.append(res)
            if len(buf) >= CHUNK or n == len(todo):
                tmp = OUT / f"reg_{k:05d}.parquet.tmp"; pd.DataFrame(buf).to_parquet(tmp, index=False); tmp.replace(OUT / f"reg_{k:05d}.parquet")
                k += 1; buf = []; el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / n:.2f} s/pair (wall)  ETA {el / n * (len(todo) - n) / 3600:.1f} h", flush=True)


def summary():
    r = load_results()
    if r.empty:
        print("no results"); return
    print(f"pairs {len(r)}  ok (Dice >= 0.85) {r.ok.mean():.4f}  errors {(r.error != '').sum()}")
    print("Dice quantiles", r.dice.quantile([.01, .05, .25, .5, .75]).round(3).to_dict())
    print("mask fallback (current / prior)", (r.mask_cur == "slice_border_rule").mean().round(4), (r.mask_prior == "slice_border_rule").mean().round(4))


def montage(n):
    """n random registered pairs: mid-axial, mid-coronal slice of the current mask (grey) with the registered prior mask
    contour (orange). For visual QC before scaling up; 6 pairs per page."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, SimpleITK as sitk
    from lungcad.data.nlst import usable_series
    from lungcad.wp4.register import mask_image
    from lungcad.wp4.temporal import transform_from_row
    from lungcad.preprocess import load_volume
    r = load_results(); r = r[r.error == ""].sample(min(n, len(r)), random_state=0); u = usable_series().set_index("uid")
    pages = [r.iloc[i:i + 6] for i in range(0, len(r), 6)]
    for pi, pg in enumerate(pages):
        fig, axes = plt.subplots(len(pg), 2, figsize=(7, 2.3 * len(pg)), squeeze=False)
        for (_, x), ax in zip(pg.iterrows(), axes):
            arr, meta = load_volume(u.loc[x.uid, "stem"]); fm, _ = mask_image(u.loc[x.uid, "stem"]); mm, _ = mask_image(u.loc[x.prior_uid, "stem"])
            w = sitk.GetArrayFromImage(sitk.Resample(mm, fm, transform_from_row(x), sitk.sitkNearestNeighbor, 0.0))
            for a, sl, name in ((ax[0], (arr.shape[0] // 2, slice(None), slice(None)), "axial"), (ax[1], (slice(None), arr.shape[1] // 2, slice(None)), "coronal")):
                a.imshow(arr[sl], cmap="gray", vmin=-1000, vmax=400, origin="lower" if name == "coronal" else "upper")
                a.contour(w[sl], levels=[0.5], colors="#eb6834", linewidths=0.8); a.set_axis_off()
                a.set_title(f"pid {x.pid} T{x.prior_round}→T{x['round']} {name}  Dice {x.dice:.3f}", fontsize=7)
        fig.tight_layout(); f = OUT / f"qc_montage_{pi + 1}.png"; fig.savefig(f, dpi=130); plt.close(fig); print("wrote", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", action="store_true"); ap.add_argument("--qc-montage", type=int, default=0); a = ap.parse_args()
    if a.summary:
        summary()
    elif a.qc_montage:
        montage(a.qc_montage)
    else:
        run(a.procs, a.limit)
