"""WP4 inference (plan Section 4): frozen fold-0 detector + own-regime stage-2 models (A, B, C) + S6 score fusion.

    python scripts/41_wp4_infer.py --k2                                   # known-answer check K2 on LUNA16 fold-0 test
    python scripts/41_wp4_infer.py --cohort sample1955 --input archive    # Section 3 check, archive input
    python scripts/41_wp4_infer.py --cohort sample1955 --input native     # Section 3 check, native input
    python scripts/41_wp4_infer.py --cohort all --input <archive|native>  # full NLST (resumable)
    LUNGCAD_NLST_NATIVE=$LUNGCAD_NLST_HDD/nlst_native_detector python scripts/41_wp4_infer.py --cohort all --input native [--converted-only]
        # WP4 full run (native chosen 17 Nov); --converted-only = passes while scripts/47 is still converting

BLINDING: this script reads the image series table (lungcad.data.nlst.usable_series) and the image files only. It
never reads the NLST clinical tables; outcomes are joined later by the analysis script.

Per series: one int16 HU volume at detector spacing (LUNA16: the stored archive; NLST native: the stored native crop;
NLST archive: the 1.5 mm archive B-spline-resampled to detector spacing, windowed and rounded to int16 like the LUNA16
archive) -> detector (sliding window, AMP, the checkpoint's own infer config: score >= 0.02, NMS 0.22) -> every candidate
scored by each stage-2 model on the same 3-scale patches as S6 (patches.extract_block / make_scales) -> fused score
p1^(1-w) * p2^w with the run's own validation-chosen w (fusion.json).
Outputs (OUT/<input>_<cohort>/): shard_NNNNN.parquet (one row per candidate), emb_NNNNN.npy (B seed-0 embeddings, row-
aligned), done.txt (finished series), setup.json (checkpoints, run tags, w). Resume = rerun the same command.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, SimpleITK as sitk, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.preprocess import normalize, load_volume, voxel_to_world, world_to_voxel
from lungcad.stage1.model import build_detector, set_inferer
from lungcad.stage2.models import build_model
from lungcad.stage2.patches import extract_block, make_scales
from lungcad.eval.fusion import EPS

DET = C.STAGE1 / "fold0" / "seed0" / "best.pt"
S2 = {"B_s0": "own/B/fold0/seed0", "B_s1": "own/B/fold0/seed1", "B_s2": "own/B/fold0/seed2",
      "A_s0": "own/A/fold0/seed0", "C_s0": "own/C/fold0/seed0"}
EMB_MODEL = "B_s0"
OUT_ROOT = C.WORK / "wp4" / "inference"
SHARD = 200


# ---------------------------------------------------------------- models
def load_detector():
    ck = torch.load(DET, map_location="cuda", weights_only=False); cfg = ck["cfg"]
    det = build_detector(cfg["model"]); det.network.load_state_dict(ck["model"])
    set_inferer(det, cfg["infer"]["roi_size"], cfg["infer"]["overlap"], cfg["infer"]["sw_batch_size"]); det.eval()
    return det, cfg, ck.get("run_tag")


def load_stage2():
    out = {}
    for name, rel in S2.items():
        d = C.STAGE2 / rel; ck = torch.load(d / "best.pt", map_location="cuda", weights_only=False)
        m = build_model(ck["model_name"], **{k: v for k, v in ck["model_kwargs"].items() if k not in ("lr", "warmup_iters")}).cuda()
        m.load_state_dict(ck["model"]); m.eval()
        out[name] = {"model": m, "w": float(json.load(open(d / "fusion.json"))["w"]), "run_tag": ck.get("run_tag"), "dir": str(d),
                     "amp": ck["cfg"]["train"]["amp"], "bs": ck["cfg"]["train"]["eval_batch_size"]}
    return out


# ---------------------------------------------------------------- volumes (int16 HU at detector spacing + meta)
def volume_luna(uid, subset):
    from lungcad.data.luna16_preprocess import stem_for
    return load_volume(stem_for(uid, subset))


def volume_nlst(stem, kind):
    from lungcad.data.nlst import native_stem
    if kind == "native":
        return load_volume(native_stem(stem))
    arr, meta = load_volume(stem)                                 # 1.5 mm archive -> detector spacing (as load_for_detector)
    img = sitk.GetImageFromArray(arr.astype(np.float32)); sp = C.LUNA_SPACING
    img.SetSpacing(tuple(float(s) for s in meta["spacing_zyx"][::-1])); img.SetOrigin(tuple(meta["origin_xyz"])); img.SetDirection(tuple(meta["direction"]))
    size = [int(round(n * s / t)) for n, s, t in zip(img.GetSize(), img.GetSpacing(), sp[::-1])]
    res = sitk.GetArrayFromImage(sitk.Resample(img, size, sitk.Transform(), sitk.sitkBSpline, img.GetOrigin(), tuple(float(s) for s in sp[::-1]),
                                               img.GetDirection(), C.HU_MIN, sitk.sitkFloat32))
    res = np.rint(np.clip(res, C.HU_MIN, C.HU_MAX)).astype(np.int16)
    return res, {**meta, "archive_spacing_zyx": meta["spacing_zyx"], "spacing_zyx": [float(s) for s in sp], "shape_zyx": list(res.shape)}


# ---------------------------------------------------------------- one series
@torch.no_grad()
def run_series(series_id, arr, meta, det, dcfg, s2):
    x = torch.from_numpy(normalize(arr.astype(np.float32)))[None, None]
    with torch.autocast("cuda", dtype=torch.float16, enabled=dcfg["train"]["amp"]):
        out = det(x, use_inferer=True)[0]
    boxes = out[det.target_box_key].float().cpu().numpy(); p1 = out[det.pred_score_key].float().cpu().numpy().astype(np.float64)
    if not len(boxes):
        return pd.DataFrame(), None
    sp = np.array(meta["spacing_zyx"], np.float64); ctr = (boxes[:, :3] + boxes[:, 3:]) / 2
    world = voxel_to_world(meta, ctr); size = (boxes[:, 3:] - boxes[:, :3]) * sp
    vox = world_to_voxel(meta, world)                             # same route as S6 (world -> voxel of the stored volume)
    df = pd.DataFrame({"series": series_id, "cand": np.arange(len(p1)), "coordX": world[:, 0], "coordY": world[:, 1], "coordZ": world[:, 2],
                       "p1": p1, "dz_mm": size[:, 0], "dy_mm": size[:, 1], "dx_mm": size[:, 2],
                       "z_frac": vox[:, 0] / arr.shape[0], "y_frac": vox[:, 1] / arr.shape[1], "x_frac": vox[:, 2] / arr.shape[2]})
    blk = torch.from_numpy(np.stack([extract_block(arr, v, sp.astype(np.float32)) for v in vox]))
    sub = torch.tensor((vox - np.round(vox)) * sp, dtype=torch.float32); spt = torch.tensor(np.tile(sp.astype(np.float32), (len(vox), 1)))
    emb = None
    for name, m in s2.items():
        probs, embs = [], []
        for i in range(0, len(vox), m["bs"]):
            xs = make_scales(normalize(blk[i:i + m["bs"]].cuda().float())[:, None], spt[i:i + m["bs"]].cuda(), augment=False,
                             subvoxel_offset_zyx=sub[i:i + m["bs"]].cuda())
            with torch.autocast("cuda", dtype=torch.float16, enabled=m["amp"]):
                if name == EMB_MODEL:
                    e = m["model"].embed(xs); logit = m["model"].head(e).squeeze(1); embs.append(e.float().cpu())
                else:
                    logit = m["model"](xs)
            probs.append(torch.sigmoid(logit.float()).cpu())
        p2 = torch.cat(probs).numpy().astype(np.float64); df[f"p2_{name}"] = p2
        df[f"fused_{name}"] = np.exp((1 - m["w"]) * np.log(np.clip(p1, EPS, 1)) + m["w"] * np.log(np.clip(p2, EPS, 1)))
        if name == EMB_MODEL:
            emb = torch.cat(embs).numpy().astype(np.float16)
    return df, emb


# ---------------------------------------------------------------- cohorts
def cohort(name):
    from lungcad.data.nlst import usable_series
    u = usable_series()
    if name == "all":
        return u
    if name == "sample1955":
        s = pd.read_csv(C.WORK / "wp0" / "analysis" / "p2_nlst" / "sample.csv", dtype={"uid": str})
        return u[u.uid.isin(set(s.uid))]
    raise SystemExit(f"unknown cohort {name}")


def converted_only(s: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Series listed as ok/skip in the converter log (written after both files exist), so a pass can run while scripts/47
    is still converting. Rerun without the flag at the end: then every series must be present."""
    from lungcad.data.nlst import NATIVE
    assert kind == "native", "--converted-only is for --input native"
    ok = set()
    for line in open(NATIVE / "convert.log"):
        f = line.split()
        if len(f) >= 5 and f[4] in ("ok", "skip"):
            ok.add(f[3])
    keep = s[[Path(x).name in ok for x in s.stem]]
    print(f"--converted-only: {len(keep)}/{len(s)} series converted so far", flush=True)
    return keep


def run(series: pd.DataFrame, kind: str, out: Path, setup: dict):
    out.mkdir(parents=True, exist_ok=True); json.dump(setup, open(out / "setup.json", "w"), indent=1)
    done_f = out / "done.txt"; done = set(done_f.read_text().split()) if done_f.exists() else set()
    todo = series[~series.uid.isin(done)]; print(f"{len(done)} done, {len(todo)} to go -> {out}", flush=True)
    det, dcfg, _ = load_detector(); s2 = load_stage2(); buf, ebuf, ids = [], [], []; t0 = time.time()
    shard = len(list(out.glob("shard_*.parquet")))
    def flush():
        nonlocal shard, buf, ebuf, ids
        if not ids:
            return
        df = pd.concat(buf, ignore_index=True) if buf else pd.DataFrame()
        tmp = out / f"shard_{shard:05d}.parquet.tmp"; df.to_parquet(tmp, index=False); tmp.replace(out / f"shard_{shard:05d}.parquet")
        np.save(out / f"emb_{shard:05d}.npy", np.concatenate(ebuf) if ebuf else np.zeros((0, 1), np.float16))
        with open(done_f, "a") as f: f.write("\n".join(ids) + "\n")
        shard += 1; buf, ebuf, ids = [], [], []
    for n, r in enumerate(todo.itertuples(), 1):
        arr, meta = volume_nlst(r.stem, kind)
        df, emb = run_series(r.uid, arr, meta, det, dcfg, s2)
        if len(df):
            df.insert(1, "pid", r.pid); df.insert(2, "round", r.round); buf.append(df); ebuf.append(emb)
        ids.append(r.uid)
        if len(ids) >= SHARD:
            flush(); el = time.time() - t0; print(f"  {n}/{len(todo)}  {el / n:.2f} s/series  ETA {el / n * (len(todo) - n) / 3600:.1f} h", flush=True)
    flush(); print(f"finished {len(todo)} series in {(time.time() - t0) / 3600:.2f} h", flush=True)


# ---------------------------------------------------------------- K2
def k2():
    """WP4 code path on LUNA16 fold-0 test scans must reproduce the frozen fold-0 candidates and S6 own/B/fold0/seed0."""
    from lungcad.data.luna16_index import load_index, get_split
    from lungcad.eval.froc import evaluate
    index = load_index(); uids = get_split(index, 0)["test"]; sub = index.set_index("seriesuid").subset
    det, dcfg, _ = load_detector(); s2 = {k: v for k, v in load_stage2().items() if k == "B_s0"}
    rows = []
    for uid in uids:
        arr, meta = volume_luna(uid, int(sub[uid])); df, _ = run_series(uid, arr, meta, det, dcfg, s2)
        rows.append(df)
    mine = pd.concat(rows, ignore_index=True)
    frozen = pd.read_csv(C.STAGE1 / "own_candidates" / "own_candidates_all.csv"); frozen = frozen[frozen.seriesuid.isin(set(uids))]
    s6 = pd.read_csv(C.STAGE2 / S2["B_s0"] / "test_candidates_fused.csv")
    key = lambda d, x, y, z: np.round(d[[x, y, z]].values, 3)
    res = {"n_mine": len(mine), "n_frozen": len(frozen), "n_s6": len(s6)}
    a = mine.sort_values(["series", "coordX", "coordY", "coordZ"]).reset_index(drop=True)
    b = frozen.sort_values(["seriesuid", "coordX", "coordY", "coordZ"]).reset_index(drop=True)
    res["same_candidate_set"] = bool(len(a) == len(b) and (a.series.values == b.seriesuid.values).all() and np.abs(key(a, "coordX", "coordY", "coordZ") - key(b, "coordX", "coordY", "coordZ")).max() <= 1e-3)
    if res["same_candidate_set"]:
        res["max_abs_dp1"] = float(np.abs(a.p1.values - b.probability.values).max())
        c = s6.sort_values(["seriesuid", "coordX", "coordY", "coordZ"]).reset_index(drop=True)
        res["max_abs_dfused"] = float(np.abs(a.fused_B_s0.values - c.probability.values).max())
    fus = mine.rename(columns={"series": "seriesuid", "fused_B_s0": "probability"})[["seriesuid", "coordX", "coordY", "coordZ", "probability"]]
    res["cpm_mine"] = evaluate(fus, uids, n_bootstrap=0)["cpm"]; res["cpm_s6"] = json.load(open(C.STAGE2 / S2["B_s0"] / "froc_test_fused.json"))["cpm"]
    res["pass"] = bool(res["same_candidate_set"] and res.get("max_abs_dp1", 1) <= 1e-3 and res.get("max_abs_dfused", 1) <= 1e-3
                       and abs(res["cpm_mine"] - res["cpm_s6"]) <= 1e-4)
    print(json.dumps(res, indent=1)); out = OUT_ROOT / "k2"; out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "k2.json", "w"), indent=1); mine.to_parquet(out / "k2_candidates.parquet", index=False)
    print("K2", "PASS" if res["pass"] else "FAIL")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k2", action="store_true"); ap.add_argument("--cohort", choices=["sample1955", "all"])
    ap.add_argument("--input", choices=["archive", "native"])
    ap.add_argument("--converted-only", action="store_true", help="native: only series the converter has finished (scripts/47 log)")
    a = ap.parse_args()
    if a.k2:
        return k2()
    if not (OUT_ROOT / "k2" / "k2.json").exists() or not json.load(open(OUT_ROOT / "k2" / "k2.json"))["pass"]:
        sys.exit("K2 has not passed - run --k2 first")
    s = cohort(a.cohort)
    if a.converted_only:
        s = converted_only(s, a.input)
    det_tag = torch.load(DET, map_location="cpu", weights_only=False).get("run_tag")
    setup = {"detector": str(DET), "detector_run_tag": det_tag, "input": a.input, "cohort": a.cohort, "n_series": len(s),
             "stage2": {k: {"dir": str(C.STAGE2 / v), "w": float(json.load(open(C.STAGE2 / v / "fusion.json"))["w"])} for k, v in S2.items()},
             "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if a.input == "native":
        from lungcad.data.nlst import NATIVE; setup["native_root"] = str(NATIVE)
    run(s, a.input, OUT_ROOT / f"{a.input}_{a.cohort}", setup)


if __name__ == "__main__":
    main()
