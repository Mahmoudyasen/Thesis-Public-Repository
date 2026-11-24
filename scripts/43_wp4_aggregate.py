"""WP4 per-exam scores (plan Section 5): max pooling for every system, FeatureGBM and AttentionMIL out-of-fold, and the
H2 pair (single-scan vs temporal FeatureGBM on after-baseline exams).

    python scripts/43_wp4_aggregate.py [--device cuda|cpu]      # -> work/wp4/exam_scores.parquet (+ meta json)

Refuses to run unless the WP4 plan is committed and the splits are the recorded ones (lungcad.wp4.guard).
Inputs: inference shards of the input chosen by the archive check (work/wp4/input_decision.json), registration results
(work/wp4/registration), the exam table (lungcad.wp4.labels.image_cohort), the splits (nlst/meta/wp4_splits.csv).
Score columns written (exam score; 0 for an exam without candidates):
  max_<col>           max pooling of candidate column <col>: fused_B_s0 (PRIMARY), p1 (detector alone), fused_B_s1,
                      fused_B_s2, fused_Bens (mean of the three B seeds), fused_A_s0, fused_C_s0, p2_B_s0 (stage 2 alone)
  gbm_single          FeatureGBM, single-scan features, trained on all exams of the training folds
  mil                 AttentionMIL over the top-10 candidates (B seed-0 embeddings + [fused score, diameter/10, z_frac])
  gbm_single_after    FeatureGBM single-scan, trained on after-baseline exams only        } H2 pair (same exams,
  gbm_temporal_after  FeatureGBM single-scan + temporal features, after-baseline only     }  only the features differ)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.wp4.aggregators import screen_features, FeatureGBM, AttentionMIL

WP4 = C.WORK / "wp4"; PRIMARY = "fused_B_s0"
MAXPOOL = ["fused_B_s0", "p1", "fused_B_s1", "fused_B_s2", "fused_Bens", "fused_A_s0", "fused_C_s0", "p2_B_s0"]
TEMPORAL_COLS = ["no_prior", "reg_failed", "gap_days", "prior_max_score", "present_prior_of_max", "diam_change_of_max",
                 "score_change_of_max", "n_conf_new", "n_conf_matched"]
TOPK = 10


# ---------------------------------------------------------------- loading
def load_inference(run_dir: Path):
    shards = sorted(run_dir.glob("shard_*.parquet"))
    cands = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
    embs = [np.load(str(f).replace("shard_", "emb_").replace(".parquet", ".npy")) for f in shards]
    emb = np.concatenate([x for x in embs if len(x)])
    assert len(emb) == len(cands), (len(emb), len(cands))
    cands["fused_Bens"] = cands[["fused_B_s0", "fused_B_s1", "fused_B_s2"]].mean(1)
    return cands, emb


def exam_key(df):
    return df["pid"].astype(str) + "_" + df["round"].astype(str)


# ---------------------------------------------------------------- aggregators
def max_pool(cands, exams, col):
    m = cands.groupby("series")[col].max()
    return exams["uid"].map(m).fillna(0.0).values


def single_features(cands, exams):
    f = screen_features(cands.rename(columns={PRIMARY: "score"})[["pid", "round", "score", "dz_mm", "dy_mm", "dx_mm", "z_frac"]])
    out = exams[["pid", "round"]].merge(f, on=["pid", "round"], how="left")
    zero = ["n_cand", "max_score", "mean_top3", "mean_score", "n_gt_0.5", "n_gt_0.8", "score_x_diam"]
    out[zero] = out[zero].fillna(0.0)
    return out


def oof_gbm(feats, y, folds, train_mask=None):
    p = np.full(len(feats), np.nan)
    for k in range(5):
        tr = (folds != k) & (train_mask if train_mask is not None else True); te = folds == k
        if train_mask is not None:
            te = te & train_mask
        m = FeatureGBM().fit(feats[tr].reset_index(drop=True), y[tr]); p[te] = m.predict(feats[te].reset_index(drop=True))
    return p


def bags(cands, emb, exams):
    """Top-10 candidates per exam by the primary score -> (emb (k,d), extra (k,3)) per exam; empty exams get one zero row."""
    d = emb.shape[1]; c = cands.assign(_i=np.arange(len(cands))).sort_values(["series", PRIMARY], ascending=[True, False])
    top = c.groupby("series").head(TOPK); g = {k: v for k, v in top.groupby("series")}; out = []
    for uid in exams["uid"]:
        t = g.get(uid)
        if t is None:
            out.append((np.zeros((1, d), np.float32), np.zeros((1, 3), np.float32))); continue
        diam = t[["dz_mm", "dy_mm", "dx_mm"]].max(1).values / 10.0
        out.append((emb[t._i.values].astype(np.float32), np.c_[t[PRIMARY].values, diam, t.z_frac.values].astype(np.float32)))
    return out


def train_mil(bags_, labels, device, epochs=40, lr=1e-3, seed=0):
    """lungcad.wp4.aggregators.train_mil with a device argument (same model, optimiser, loss, batch 256, 40 epochs)."""
    torch.manual_seed(seed); d_in = bags_[0][0].shape[1]; k = max(b[0].shape[0] for b in bags_); B = len(bags_)
    E = torch.zeros(B, k, d_in); X = torch.zeros(B, k, 3); M = torch.zeros(B, k, dtype=torch.bool)
    for i, (e, x) in enumerate(bags_):
        n = e.shape[0]; E[i, :n] = torch.as_tensor(e); X[i, :n] = torch.as_tensor(x); M[i, :n] = True
    y = torch.as_tensor(labels).float(); m = AttentionMIL(d_in, 3).to(device); E, X, M, y = E.to(device), X.to(device), M.to(device), y.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2); pw = torch.tensor([(1 - y.mean()) / max(y.mean(), 1e-3)], device=device)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(B, device=device)
        for s in range(0, B, 256):
            idx = perm[s:s + 256]; logit, _ = m(E[idx], X[idx], M[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y[idx], pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def predict_mil(m, bags_, device):
    out = []
    for s in range(0, len(bags_), 1024):
        bb = bags_[s:s + 1024]; k = max(b[0].shape[0] for b in bb); d = bb[0][0].shape[1]
        E = torch.zeros(len(bb), k, d); X = torch.zeros(len(bb), k, 3); M = torch.zeros(len(bb), k, dtype=torch.bool)
        for i, (e, x) in enumerate(bb):
            n = e.shape[0]; E[i, :n] = torch.as_tensor(e); X[i, :n] = torch.as_tensor(x); M[i, :n] = True
        out.append(torch.sigmoid(m(E.to(device), X.to(device), M.to(device))[0]).cpu().numpy())
    return np.concatenate(out)


def oof_mil(bags_, y, folds, device):
    p = np.full(len(bags_), np.nan)
    for k in range(5):
        tr = np.nonzero(folds != k)[0]; te = np.nonzero(folds == k)[0]
        m = train_mil([bags_[i] for i in tr], y[tr], device); p[te] = predict_mil(m, [bags_[i] for i in te], device)
    return p


# ---------------------------------------------------------------- main
def aggregate(cands, emb, exams, temporal, device="cpu", with_mil=True) -> pd.DataFrame:
    """exams: one row per exam (pid, round, uid, cancer, baseline, fold). temporal: temporal_exam_features() for the
    after-baseline exams (pid, round + TEMPORAL_COLS). -> exam scores."""
    out = exams[["pid", "round", "uid", "fold"]].copy(); y = exams.cancer.values.astype(int); folds = exams.fold.values
    for col in MAXPOOL:
        out[f"max_{col}"] = max_pool(cands, exams, col)
    feats = single_features(cands, exams).drop(columns=["pid", "round"])
    out["gbm_single"] = oof_gbm(feats, y, folds)
    if with_mil:
        out["mil"] = oof_mil(bags(cands, emb, exams), y, folds, device)
    after = ~exams.baseline.values
    tf = exams[["pid", "round"]].merge(temporal, on=["pid", "round"], how="left")[TEMPORAL_COLS]
    tf["no_prior"] = tf.no_prior.fillna(1)
    out["gbm_single_after"] = oof_gbm(feats, y, folds, train_mask=after)
    out["gbm_temporal_after"] = oof_gbm(pd.concat([feats, tf], axis=1), y, folds, train_mask=after)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = ap.parse_args()
    from lungcad.wp4.guard import require_unblinding_allowed, SPLITS
    from lungcad.wp4.labels import image_cohort
    from lungcad.wp4.temporal import pairs_table, temporal_exam_features
    from lungcad.data.nlst import usable_series
    import importlib.util
    g = require_unblinding_allowed()
    dec = json.load(open(WP4 / "input_decision.json")); run_dir = WP4 / "inference" / f"{dec['input']}_all"
    cands, emb = load_inference(run_dir)
    exams = image_cohort().merge(pd.read_csv(SPLITS), on="pid")
    spec = importlib.util.spec_from_file_location("reg", Path(__file__).with_name("42_wp4_register.py")); reg = importlib.util.module_from_spec(spec); spec.loader.exec_module(reg)
    u = usable_series(); temporal = temporal_exam_features(cands, pairs_table(u), reg.load_results(), exams[~exams.baseline][["pid", "round", "uid"]])
    t0 = time.time(); out = aggregate(cands, emb, exams, temporal, a.device)
    out.to_parquet(WP4 / "exam_scores.parquet", index=False); temporal.to_parquet(WP4 / "temporal_features.parquet", index=False)
    json.dump({**g, "input": dec["input"], "n_exams": len(out), "n_candidates": len(cands), "seconds": time.time() - t0,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(WP4 / "exam_scores.meta.json", "w"), indent=1)
    print("wrote", WP4 / "exam_scores.parquet", len(out), "exams")


if __name__ == "__main__":
    main()
