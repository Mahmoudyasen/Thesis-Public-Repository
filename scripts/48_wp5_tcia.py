"""WP5: nodule-level check of the full system (detector + stage-2 fusion, the WP4 models) on TCIA LDCT chest at full and
10 % dose. Plan: report/wp5_tcia_prespec.md (descriptive; patient bootstrap; no P-values).

    python scripts/48_wp5_tcia.py --selftest     # W1: reference lesions as candidates -> sensitivity 1, 0 FP, CPM 1 (CPU)
    python scripts/48_wp5_tcia.py                # inference (GPU, resumable) -> W2 -> scoring
    python scripts/48_wp5_tcia.py --score        # scoring only

Volumes, reference and the unresolved-lesion rule are P2's (scripts/16, 19, 22: imported, unchanged). The inference is
scripts/41_wp4_infer.run_series (K2-validated). Output: work/wp5/.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from lungcad import config as C
from lungcad.eval.froc import froc_from_results
import p2_stats as S


def _mod(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


T22 = _mod("22_tcia_e1")                       # PRE, EVAL, check_lesions_in_crop, ignore_unresolved
PRE, EVAL = T22.PRE, T22.EVAL
OUT = C.WORK / "wp5"; CAND = OUT / "candidates"
DOSES = ("full", "low")
SYSTEMS = {"detector": "p1", "B_fused": "fused_B_s0", "B_fused_s1": "fused_B_s1", "B_fused_s2": "fused_B_s2",
           "A_fused": "fused_A_s0", "C_fused": "fused_C_s0"}
PRIMARY = "B_fused"
OPS = {"detector": {"90": 0.804, "95": 0.297}, "B_fused": {"90": 0.489, "95": 0.246}}   # fixed on LUNA16 (WP4 NOTES 5)
W2_REF = {"full": 0.7478991596638654, "low": 0.7058823529411764}                        # P2 tcia_cpm.csv, S4_f0
W2_TOL = 0.005


def cases_all():
    c = sorted(p.name[:-len("_full.json")] for p in PRE.glob("C*_full.json")); assert len(c) == 50, len(c); return c


def cases_primary():
    excl = set(pd.read_csv(EVAL / "excluded_patients.csv").case)
    c = [x for x in cases_all() if x not in excl]; assert len(c) == 49, len(c); return c


# ---------------------------------------------------------------- inference
def infer():
    import torch
    from lungcad.preprocess import load_volume
    I = _mod("41_wp4_infer")
    CAND.mkdir(parents=True, exist_ok=True)
    det, dcfg, _ = I.load_detector(); s2 = I.load_stage2()
    det_tag = torch.load(I.DET, map_location="cpu", weights_only=False).get("run_tag")
    json.dump({"detector": str(I.DET), "detector_run_tag": det_tag,
               "stage2": {k: {"dir": str(C.STAGE2 / v), "w": float(json.load(open(C.STAGE2 / v / "fusion.json"))["w"])} for k, v in I.S2.items()},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(OUT / "setup.json", "w"), indent=1)
    todo = [f"{c}_{d}" for c in cases_all() for d in DOSES if not (CAND / f"{c}_{d}.parquet").exists()]; t0 = time.time()
    for n, sid in enumerate(todo, 1):
        arr, meta = load_volume(PRE / sid)
        df, _ = I.run_series(sid, arr, meta, det, dcfg, s2)
        tmp = CAND / f"{sid}.parquet.tmp"; df.to_parquet(tmp, index=False); tmp.replace(CAND / f"{sid}.parquet")
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.1f} s/scan", flush=True)


def candidates() -> pd.DataFrame:
    fs = sorted(CAND.glob("C*_*.parquet")); assert len(fs) == 100, f"{len(fs)} of 100 scans inferred"
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).rename(columns={"series": "seriesuid"})


# ---------------------------------------------------------------- measures
def measures(res: list, ops: dict | None) -> dict:
    """res: ScanResult list (one dose). CPM, FPs/scan at 90/95 % sensitivity, and sens/FPs at fixed thresholds."""
    fps, sens, _, cpm = froc_from_results(res)
    out = {"cpm": cpm}
    for s in (90, 95):
        i = np.where(sens >= s / 100 - 1e-12)[0]
        out[f"fps_at_sens{s}"] = float(fps[i[0]]) if len(i) else np.nan
    if ops:
        n_nod = sum(r.n_nodules for r in res); n = len(res)
        for k, t in ops.items():
            out[f"sens_op{k}"] = sum(int((r.tp_probs >= t).sum()) for r in res) / n_nod
            out[f"fps_op{k}"] = sum(int((r.fp_probs >= t).sum()) for r in res) / n
    return out


def scan_results(cands: pd.DataFrame, col: str, uids: list, ev: Path) -> dict:
    d = cands[["seriesuid", "coordX", "coordY", "coordZ", col]].rename(columns={col: "probability"})
    return S.scan_results(d, uids, eval_dir=ev)


def evaluate(cands: pd.DataFrame, ev: Path, n_boot: int, tag: str) -> pd.DataFrame:
    cases = cases_primary()
    cands = T22.ignore_unresolved(cands)
    allu = [f"{c}_{d}" for c in cases for d in DOSES]
    per = {s: scan_results(cands, col, allu, ev) for s, col in SYSTEMS.items()}
    def one(case_list):
        row = {}
        for s in SYSTEMS:
            for d in DOSES:
                m = measures([per[s][f"{c}_{d}"] for c in case_list], OPS.get(s))
                row.update({f"{s}|{d}|{k}": v for k, v in m.items()})
        return row
    point = pd.Series(one(cases)); rng = np.random.RandomState(0); boot = []
    for b in range(n_boot):
        boot.append(one([cases[j] for j in rng.randint(0, len(cases), len(cases))]))
        if (b + 1) % 2000 == 0:
            print(f"  {tag} bootstrap {b + 1}/{n_boot}", flush=True)
    boot = pd.DataFrame(boot)
    rows = []
    def add(name, est, dist):
        dist = np.asarray(dist, float); ok = np.isfinite(dist)
        lo, hi = (np.percentile(dist[ok], [2.5, 97.5]) if ok.mean() > 0.5 else (np.nan, np.nan))
        rows.append({"quantity": name, "estimate": est, "ci_lo": lo, "ci_hi": hi, "boot_defined": float(ok.mean())})
    for k in point.index:
        add(k, point[k], boot[k])
    for s in SYSTEMS:                                          # 10 % minus full
        for m in [k.split("|")[2] for k in point.index if k.startswith(f"{s}|full|")]:
            a, b = f"{s}|low|{m}", f"{s}|full|{m}"; add(f"{s}|low-full|{m}", point[a] - point[b], boot[a] - boot[b])
    for d in DOSES:                                            # B fused minus detector alone
        for m in [k.split("|")[2] for k in point.index if k.startswith(f"{PRIMARY}|{d}|")]:
            a, b = f"{PRIMARY}|{d}|{m}", f"detector|{d}|{m}"
            if b in point.index:
                add(f"B_fused-detector|{d}|{m}", point[a] - point[b], boot[a] - boot[b])
    res = pd.DataFrame(rows); res.to_csv(OUT / f"wp5_results{tag}.csv", index=False)
    return res, per


def froc_figure(per: dict):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cases = cases_primary(); fig, ax = plt.subplots(figsize=(5.2, 3.8))
    sty = {("B_fused", "full"): ("#1f5fa8", "-"), ("B_fused", "low"): ("#1f5fa8", "--"),
           ("detector", "full"): ("#b5651d", "-"), ("detector", "low"): ("#b5651d", "--")}
    for (s, d), (col, ls) in sty.items():
        fps, sens, _, cpm = froc_from_results([per[s][f"{c}_{d}"] for c in cases])
        lab = f"{'B fused' if s == 'B_fused' else 'Detector alone'}, {'full' if d == 'full' else '10 %'} dose (CPM {cpm:.3f})"
        ax.step(np.r_[0, fps], np.r_[0, sens], where="post", color=col, ls=ls, lw=1.4, label=lab)
    ax.set_xscale("log"); ax.set_xlim(0.125, 8); ax.set_ylim(0, 1.0)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())   # log axis: no minor labels over the fixed ticks
    ax.set_xlabel("False positives per scan"); ax.set_ylabel("Sensitivity"); ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, loc="lower right", frameon=False); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"wp5_froc.{ext}", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- known answers
def selftest():
    """W1: the reference lesions as candidates (probability 1) -> sensitivity 1, 0 FP, CPM 1 at both doses."""
    ann = pd.read_csv(EVAL / "annotations" / "annotations.csv")
    c = ann.assign(probability=1.0)[["seriesuid", "coordX", "coordY", "coordZ", "probability"]]
    cases = cases_primary(); ok = True
    for d in DOSES:
        u = [f"{x}_{d}" for x in cases]; r = S.scan_results(c, u, eval_dir=EVAL); res = [r[x] for x in u]
        m = measures(res, {"x": 0.5}); n_nod = sum(x.n_nodules for x in res)
        print(f"W1 {d}: lesions {n_nod}  CPM {m['cpm']:.4f}  sens@0.5 {m['sens_opx']:.4f}  FP/scan {m['fps_opx']:.4f}")
        ok &= n_nod == 34 and abs(m["cpm"] - 1) < 1e-12 and m["sens_opx"] == 1 and m["fps_opx"] == 0
    print("W1", "PASS" if ok else "FAIL"); return ok


def w2(cands: pd.DataFrame) -> bool:
    """W2: the detector alone = P2 S4 fold 0, so its CPM must reproduce P2's tcia_cpm.csv (P2 rule: ignore_unresolved)."""
    cases = cases_primary(); c = T22.ignore_unresolved(cands); res = {}
    for d in DOSES:
        u = [f"{x}_{d}" for x in cases]; r = scan_results(c, "p1", u, EVAL)
        res[d] = float(froc_from_results([r[x] for x in u])[3])
    out = {"cpm": res, "p2": W2_REF, "tol": W2_TOL, "pass": all(abs(res[d] - W2_REF[d]) <= W2_TOL for d in DOSES)}
    json.dump(out, open(OUT / "w2.json", "w"), indent=1); print("W2", json.dumps(out)); return out["pass"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--score", action="store_true")
    ap.add_argument("--n-boot", type=int, default=S.N_BOOT); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        sys.exit("W1 failed")
    T22.check_lesions_in_crop()
    if not a.score:
        infer()
    cands = candidates()
    if not w2(cands):
        sys.exit("W2 failed - results not scored")
    res, per = evaluate(cands, EVAL, a.n_boot, "")
    evaluate(cands, EVAL / "solid_only", a.n_boot, "_solid")
    froc_figure(per)
    json.dump({"primary": PRIMARY, "n_patients": 49, "n_boot": a.n_boot, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(OUT / "wp5_meta.json", "w"), indent=1)
    print(res[res.quantity.str.startswith(("B_fused|", "detector|", "B_fused-detector|", "B_fused|low-full"))].round(4).to_string(index=False))
