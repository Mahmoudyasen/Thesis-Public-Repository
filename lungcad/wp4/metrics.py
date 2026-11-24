"""WP4 analysis core (plan Section 7): false-positive rate at matched sensitivity, paired bootstrap over participants.

  s*            = sensitivity of the reference reader (radiologists) on the exams analysed
  thr(system)   = the HIGHEST score threshold at which the system's sensitivity is >= s*  (exam positive iff score >= thr;
                  tied scores enter together)
  FPR_sys       = positives among cancer-absent exams at thr;   Δ = FPR_sys − FPR_ref
Bootstrap: participants are resampled with replacement (all exams of a participant move together); s*, thr and both
FPRs are recomputed in every replicate. Implemented with exam weights = multiplicity of the exam's participant, so the
score sort is done once. Two-sided P = 2·min(#Δ≤0 + 1, #Δ≥0 + 1)/(B + 1), capped at 1.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


class _Sorted:
    """Score order for one system, reused across bootstrap replicates."""
    def __init__(self, score, cancer):
        score = np.asarray(score, float); self.order = np.argsort(-score, kind="stable")
        s = score[self.order]; self.ends = np.r_[np.nonzero(np.diff(s) != 0)[0], len(s) - 1]   # last index of each tie block
        self.thr = s[self.ends]; self.c = np.asarray(cancer, bool)[self.order]

    def fpr_at(self, w, target_sens):
        """w: exam weights in ORIGINAL order -> (fpr, sens, thr) at the highest threshold reaching target_sens."""
        ws = w[self.order]; tp = np.cumsum(ws * self.c)[self.ends]; fp = np.cumsum(ws * ~self.c)[self.ends]
        P, N = tp[-1], fp[-1]
        k = int(np.searchsorted(tp, target_sens * P - EPS * max(1.0, P), side="left"))
        k = min(k, len(tp) - 1)
        return fp[k] / N, tp[k] / P, self.thr[k]


def _ref(w, cancer, ref_pos):
    c = np.asarray(cancer, bool); q = np.asarray(ref_pos, bool)
    return (w * (c & q)).sum() / (w * c).sum(), (w * (~c & q)).sum() / (w * ~c).sum()


def _participant_weights(pid, rng):
    u, inv = np.unique(pid, return_inverse=True)
    counts = np.bincount(rng.randint(0, len(u), len(u)), minlength=len(u))
    return counts[inv].astype(float)


def _p_two(d):
    d = d[np.isfinite(d)]; B = len(d)
    return float(min(1.0, 2 * min(((d <= 0).sum() + 1) / (B + 1), ((d >= 0).sum() + 1) / (B + 1))))


def _ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float("nan"), float("nan")]


def matched_fpr_vs_reader(pid, cancer, score, ref_pos, n_boot=10000, seed=0) -> dict:
    """H1: system vs reference reader on the same exams. Δ = FPR_sys − FPR_ref at the reader's sensitivity."""
    pid = np.asarray(pid); n = len(pid); S = _Sorted(score, cancer); w1 = np.ones(n)
    s_ref, f_ref = _ref(w1, cancer, ref_pos); f_sys, s_sys, thr = S.fpr_at(w1, s_ref)
    rng = np.random.RandomState(seed); D = np.empty(n_boot); FS = np.empty(n_boot); FR = np.empty(n_boot)
    for b in range(n_boot):
        w = _participant_weights(pid, rng); sr, fr = _ref(w, cancer, ref_pos); fs, _, _ = S.fpr_at(w, sr)
        D[b], FS[b], FR[b] = fs - fr, fs, fr
    return {"n_exams": int(n), "n_participants": int(len(np.unique(pid))), "n_cancer": int(np.sum(cancer)),
            "ref_sens": float(s_ref), "ref_fpr": float(f_ref), "sys_fpr": float(f_sys), "sys_sens_at_thr": float(s_sys), "thr": float(thr),
            "delta_fpr": float(f_sys - f_ref), "delta_ci95": _ci(D), "p_two_sided": _p_two(D),
            "sys_fpr_ci95": _ci(FS), "ref_fpr_ci95": _ci(FR), "n_boot": n_boot}


def matched_fpr_two_systems(pid, cancer, score_a, score_b, ref_pos, n_boot=10000, seed=0) -> dict:
    """H2: system A vs system B, both at the reference reader's sensitivity. Δ = FPR_A − FPR_B."""
    pid = np.asarray(pid); n = len(pid); A, Bs = _Sorted(score_a, cancer), _Sorted(score_b, cancer); w1 = np.ones(n)
    s_ref, _ = _ref(w1, cancer, ref_pos); fa, _, ta = A.fpr_at(w1, s_ref); fb, _, tb = Bs.fpr_at(w1, s_ref)
    rng = np.random.RandomState(seed); D = np.empty(n_boot)
    for b in range(n_boot):
        w = _participant_weights(pid, rng); sr, _ = _ref(w, cancer, ref_pos)
        D[b] = A.fpr_at(w, sr)[0] - Bs.fpr_at(w, sr)[0]
    return {"n_exams": int(n), "n_cancer": int(np.sum(cancer)), "ref_sens": float(s_ref), "fpr_a": float(fa), "fpr_b": float(fb),
            "thr_a": float(ta), "thr_b": float(tb), "delta_fpr": float(fa - fb), "delta_ci95": _ci(D), "p_two_sided": _p_two(D), "n_boot": n_boot}


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def verdict(p_holm, ci, alpha=0.05):
    """Plan Section 7 wording."""
    if p_holm < alpha and ci[1] < 0:
        return "lower"
    if ci[0] > 0:
        return "higher"
    return "not distinguishable"


# ---------------------------------------------------------------- K4 known answers
def known_answer_k4() -> dict:
    """Hand-computed toy example + identity checks; all must hold exactly."""
    res = {}
    # 10 exams, 10 participants: 4 cancers (c), reader positive on 3 of them and on 2 of 6 negatives.
    c = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    rd = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0], bool)            # reader: sens 3/4, FPR 2/6
    sc = np.array([.9, .8, .3, .7, .85, .2, .6, .1, .05, .4])      # system
    # sorted desc: .9c .85n .8c .7c .6n .4n .3c ... ; s* = 0.75 -> need 3 of 4 cancers -> first reached at .7
    # positives at thr .7: {.9,.85,.8,.7} -> FP = 1 (the .85) -> FPR 1/6; Δ = 1/6 − 2/6 = −1/6
    r = matched_fpr_vs_reader(np.arange(10), c, sc, rd, n_boot=200)
    res["toy_ref"] = (r["ref_sens"], r["ref_fpr"]) == (0.75, 2 / 6)
    res["toy_thr"] = r["thr"] == 0.7
    res["toy_delta"] = abs(r["delta_fpr"] - (-1 / 6)) < 1e-15
    # ties: two exams at .7 (one cancer, one not) enter together -> FP 2 at the same threshold
    sc2 = sc.copy(); sc2[6] = 0.7
    r2 = matched_fpr_vs_reader(np.arange(10), c, sc2, rd, n_boot=10)
    res["ties_enter_together"] = abs(r2["sys_fpr"] - 2 / 6) < 1e-15 and r2["thr"] == 0.7
    # identity: a system whose score is the reader's decision gives Δ = 0 exactly, also in every replicate
    rng = np.random.RandomState(1); n = 400; pid = rng.randint(0, 150, n); cc = rng.rand(n) < .2; rr = np.where(cc, rng.rand(n) < .9, rng.rand(n) < .25)
    r3 = matched_fpr_vs_reader(pid, cc, rr.astype(float), rr, n_boot=300)
    res["identity_delta_zero"] = r3["delta_fpr"] == 0.0 and r3["delta_ci95"] == [0.0, 0.0]
    # two-systems: same score twice -> Δ = 0 in every replicate
    s5 = rng.rand(n); r5 = matched_fpr_two_systems(pid, cc, s5, s5.copy(), rr, n_boot=100)
    res["two_systems_identical_zero"] = r5["delta_fpr"] == 0.0 and r5["delta_ci95"] == [0.0, 0.0]
    # participant clustering: duplicating every exam under the same pid leaves point estimates unchanged
    r6 = matched_fpr_vs_reader(np.r_[np.arange(10), np.arange(10)], np.r_[c, c], np.r_[sc, sc], np.r_[rd, rd], n_boot=10)
    res["duplicate_exams_same_point"] = abs(r6["delta_fpr"] - r["delta_fpr"]) < 1e-15
    res["holm"] = holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    res["pass"] = all(res.values())
    return res


if __name__ == "__main__":
    k = known_answer_k4(); print(k); print("K4", "PASS" if k["pass"] else "FAIL")
