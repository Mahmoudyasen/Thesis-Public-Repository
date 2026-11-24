"""WP4 — patient-level cancer probability from per-nodule scores: three aggregators compared on identical inputs.

Input: a per-screen candidate table with at least  [pid, round, score, dz_mm, dy_mm, dx_mm, z_frac]  (z_frac = candidate
z-position as a fraction of the lung crop, upper-lobe proxy) and, once WP1 exists, temporal columns
[present_prior, diam_change_mm, vdt_days, attn_change]; and a label table [pid, round, label].

  max_pool         patient score = max candidate score.                                       trivial, hard to beat
  FeatureGBM       hand-crafted per-screen features -> HistGradientBoosting.                  interpretable
  AttentionMIL     gated attention (Ilse et al. 2018) over the top-k candidate embeddings.    the modern answer
Evaluation (evaluate_at_sensitivity): specificity and PPV at the NLST radiologist sensitivity 93.8%, AUC, bootstrap
CIs; reference points NLST (sens 93.8, spec 73.4) and Lung-RADS baseline (sens 84.9, FP rate 12.8).
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

TEMPORAL = ["present_prior", "diam_change_mm", "vdt_days", "attn_change"]
NLST_POINT = {"sensitivity": 0.938, "specificity": 0.734, "ppv": (0.024, 0.044)}
LUNGRADS_POINT = {"sensitivity": 0.849, "specificity": 0.872}


def max_pool(cands: pd.DataFrame) -> pd.DataFrame:
    return cands.groupby(["pid", "round"]).score.max().rename("p").reset_index()


def screen_features(cands: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    rows = []
    for (pid, rnd), g in cands.groupby(["pid", "round"]):
        g = g.sort_values("score", ascending=False); top = g.head(top_k); best = g.iloc[0]
        diam = g[["dz_mm", "dy_mm", "dx_mm"]].max(1)
        f = {"pid": pid, "round": rnd, "n_cand": len(g), "max_score": best.score, "mean_top3": g.score.head(3).mean(), "mean_score": g.score.mean(),
             "n_gt_0.5": int((g.score > 0.5).sum()), "n_gt_0.8": int((g.score > 0.8).sum()), "largest_diam_mm": float(diam.max()),
             "diam_of_max_mm": float(diam.iloc[0]), "upper_lobe_max": float(best.z_frac) if "z_frac" in g else np.nan,
             "score_x_diam": float(best.score * diam.iloc[0])}
        for c in TEMPORAL:
            if c in g: f[c + "_of_max"] = float(best[c]) if pd.notna(best[c]) else np.nan; f[c + "_any"] = float(g[c].fillna(0).max())
        rows.append(f)
    return pd.DataFrame(rows)


class FeatureGBM:
    def __init__(self, **kw):
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=True, random_state=0, **kw)
        self.cols = None

    def fit(self, feats: pd.DataFrame, y):
        self.cols = [c for c in feats.columns if c not in ("pid", "round")]; self.m.fit(feats[self.cols].values, y); return self

    def predict(self, feats: pd.DataFrame):
        return self.m.predict_proba(feats[self.cols].values)[:, 1]


class AttentionMIL(nn.Module):
    """Gated attention MIL over the top-k candidate embeddings (+ scalar score, diameter, temporal features)."""
    def __init__(self, d_in: int = 256, d_extra: int = 3, d_hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in + d_extra, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_v = nn.Sequential(nn.Linear(d_hidden, 64), nn.Tanh()); self.att_u = nn.Sequential(nn.Linear(d_hidden, 64), nn.Sigmoid()); self.att_w = nn.Linear(64, 1)
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, emb, extra, mask):
        """emb (B,k,d_in), extra (B,k,d_extra), mask (B,k) bool for real candidates -> logits (B,), attention (B,k)."""
        h = self.enc(torch.cat([emb, extra], -1))
        a = self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1).masked_fill(~mask, -1e4)
        a = torch.softmax(a, 1)
        return self.head((a.unsqueeze(-1) * h).sum(1)).squeeze(1), a


def train_mil(bags, labels, epochs=40, lr=1e-3, seed=0, d_in=256, d_extra=3):
    """bags: list of (emb (k,d_in), extra (k,d_extra)); labels: array. Simple full-batch AdamW loop with BCE and pos weighting."""
    torch.manual_seed(seed); k = max(b[0].shape[0] for b in bags); B = len(bags)
    E = torch.zeros(B, k, d_in); X = torch.zeros(B, k, d_extra); M = torch.zeros(B, k, dtype=torch.bool)
    for i, (e, x) in enumerate(bags):
        n = e.shape[0]; E[i, :n] = torch.as_tensor(e); X[i, :n] = torch.as_tensor(x); M[i, :n] = True
    y = torch.as_tensor(labels).float(); m = AttentionMIL(d_in, d_extra).cuda(); E, X, M, y = E.cuda(), X.cuda(), M.cuda(), y.cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2); pw = torch.tensor([(1 - y.mean()) / max(y.mean(), 1e-3)]).cuda()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(B, device="cuda")
        for s in range(0, B, 256):
            idx = perm[s:s + 256]; logit, _ = m(E[idx], X[idx], M[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y[idx], pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def evaluate_at_sensitivity(y, p, target_sens=NLST_POINT["sensitivity"], n_boot=2000, seed=0) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    def op(yy, pp):
        fpr, tpr, thr = roc_curve(yy, pp); i = np.argmax(tpr >= target_sens); spec = 1 - fpr[i]
        pred = pp >= thr[i]; tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum())
        return spec, tp / max(1, tp + fp), roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan, float(thr[i])
    spec, ppv, auc, thr = op(y, p); rng = np.random.RandomState(seed); bs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        bs.append(op(y[idx], p[idx])[:3])
    bs = np.array(bs)
    ci = lambda j: [float(np.nanpercentile(bs[:, j], 2.5)), float(np.nanpercentile(bs[:, j], 97.5))]
    return {"target_sensitivity": target_sens, "specificity": float(spec), "specificity_ci95": ci(0), "ppv": float(ppv), "ppv_ci95": ci(1),
            "auc": float(auc), "auc_ci95": ci(2), "threshold": thr, "n": int(len(y)), "n_pos": int(y.sum()),
            "nlst_reference": NLST_POINT, "lungrads_reference": LUNGRADS_POINT, "beats_nlst_specificity": bool(spec > NLST_POINT["specificity"])}
