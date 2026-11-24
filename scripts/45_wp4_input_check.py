"""WP4 Section 3: 1.5 mm archive or native-resolution input — decided label-free on the 1,955-series sample.

    python scripts/45_wp4_input_check.py            # needs inference/archive_sample1955 and inference/native_sample1955
    python scripts/45_wp4_input_check.py --selftest # identical inputs -> agreement 1, Spearman 1; disjoint -> 0

Criterion (fixed in the plan): the archive is accepted only if BOTH
  1. confident-detection agreement between archive and native >= 0.891 (the P2 F7 soft-vs-sharp kernel agreement of the
     S4 detector). P2's rule is reused unchanged (scripts/32_review_stats.match_exam): detector score >= 0.5, greedy
     one-to-one matching within 5 mm; agreement = 2 * matched / (n_archive + n_native), pooled over exams;
  2. Spearman correlation of the per-exam maximum fused score (B seed 0) >= 0.90.
Writes work/wp4/input_decision.json (read by scripts 43/44). No clinical table is read.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C

WP4 = C.WORK / "wp4"; AGREE_MIN, SPEARMAN_MIN = 0.891, 0.90


def _match_exam():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    spec = importlib.util.spec_from_file_location("rs32", Path(__file__).with_name("32_review_stats.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.match_exam


def compare(arch: pd.DataFrame, nat: pd.DataFrame, series: list[str]) -> dict:
    match_exam = _match_exam(); empty = arch.iloc[:0]
    ga = {k: v.rename(columns={"p1": "probability"}) for k, v in arch.groupby("series")}
    gn = {k: v.rename(columns={"p1": "probability"}) for k, v in nat.groupby("series")}
    na = nn = nm = 0
    for s in series:
        e = match_exam(ga.get(s, empty.rename(columns={"p1": "probability"})), gn.get(s, empty.rename(columns={"p1": "probability"})))
        na += e["n_soft"]; nn += e["n_sharp"]; nm += e["n_match"]
    ma = arch.groupby("series").fused_B_s0.max().reindex(series).fillna(0); mn = nat.groupby("series").fused_B_s0.max().reindex(series).fillna(0)
    agree = 2 * nm / max(1, na + nn); rho = float(spearmanr(ma.values, mn.values).statistic)
    return {"n_series": len(series), "confident_archive": int(na), "confident_native": int(nn), "matched": int(nm), "agreement": float(agree),
            "spearman_max_fused": rho, "criterion": {"agreement_min": AGREE_MIN, "spearman_min": SPEARMAN_MIN},
            "input": "archive" if (agree >= AGREE_MIN and rho >= SPEARMAN_MIN) else "native"}


def selftest():
    rng = np.random.RandomState(0); rows = []
    for i in range(50):
        for j in range(rng.poisson(4)):
            rows.append((f"s{i}", *rng.uniform(-100, 100, 3), rng.rand(), rng.rand()))
    a = pd.DataFrame(rows, columns=["series", "coordX", "coordY", "coordZ", "p1", "fused_B_s0"]); s = [f"s{i}" for i in range(50)]
    r1 = compare(a, a.copy(), s); b = a.copy(); b["coordX"] += 500; b["fused_B_s0"] = rng.rand(len(b))
    r2 = compare(a, b, s)
    ok = r1["agreement"] == 1.0 and abs(r1["spearman_max_fused"] - 1) < 1e-12 and r1["input"] == "archive" and r2["agreement"] == 0.0 and r2["input"] == "native"
    print(r1["agreement"], r1["spearman_max_fused"], r2["agreement"], round(r2["spearman_max_fused"], 3)); print("SELFTEST", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest:
        return selftest()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    spec = importlib.util.spec_from_file_location("agg", Path(__file__).with_name("43_wp4_aggregate.py")); agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
    arch, _ = agg.load_inference(WP4 / "inference" / "archive_sample1955"); nat, _ = agg.load_inference(WP4 / "inference" / "native_sample1955")
    done_a = set((WP4 / "inference" / "archive_sample1955" / "done.txt").read_text().split()); done_n = set((WP4 / "inference" / "native_sample1955" / "done.txt").read_text().split())
    series = sorted(done_a & done_n); assert len(series) == 1955, f"only {len(series)} series finished on both inputs"
    r = compare(arch, nat, series) | {"created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(r, open(WP4 / "input_decision.json", "w"), indent=1); print(json.dumps(r, indent=1))
    print(f"DECISION: use the {r['input']} input for WP4 (record it in report/wp4_nlst_prespec_NOTES.md)")


if __name__ == "__main__":
    main()
