"""Figure for the fold-0 score-fusion study (scripts/s6_fusion_fold0.py) -> report/s6_fusion_fold0.png."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C

D = C.WORK / "wp0" / "stage2_dev" / "fusion_fold0"
OUT = Path(__file__).resolve().parents[1] / "report" / "s6_fusion_fold0.png"
COL = {"C_e40": "#2a78d6", "C_e60": "#eb6834"}; LAB = {"C_e40": "C, 40 epochs", "C_e60": "C, 60 epochs"}
INK, MUTE = "#0b0b0b", "#52514e"


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTE)
    ax.tick_params(labelsize=8.5, colors=MUTE); ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)


def main():
    v = json.loads((D / "fusion_val.json").read_text()); t = json.loads((D / "fusion_test.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.1), gridspec_kw={"width_ratios": [1, 1.3]})
    # (a) validation CPM vs fusion weight
    style(a)
    for n, r in v["runs"].items():
        ws = [float(w) for w in r["grid"]]; y = [r["grid"][w]["cpm"] for w in r["grid"]]
        a.plot(ws, y, "-o", color=COL[n], lw=2, ms=6, mec="white", mew=1.2, label=f"{LAB[n]} (chosen w = {r['w_best']})")
        wb = r["w_best"]; a.plot([wb], [r["grid"][str(wb)]["cpm"]], "o", ms=12, mfc="none", mec=COL[n], mew=1.5)
    det = v["runs"]["C_e40"]["grid"]["0.0"]["cpm"]
    a.axhline(det, color=MUTE, ls="--", lw=1); a.text(0.25, det + 0.0006, f"detector alone {det:.3f}", ha="left", va="bottom", fontsize=8, color=MUTE)
    a.set_xlabel("fusion weight w   (p = p₁^(1−w) · p₂^w;  0 = detector, 1 = stage 2)", fontsize=8.5, color=INK)
    a.set_ylabel("validation CPM (fold 0, 89 scans)", fontsize=8.5, color=INK); a.legend(fontsize=8, frameon=False, loc="upper right")
    a.set_ylim(det - 0.003, 0.957); a.set_title("(a) Choosing w on validation", fontsize=9.5, color=INK, loc="left")
    # (b) test: paired ΔCPM vs detector
    style(b); b.grid(False); b.grid(axis="x", color="#e6e5e1", lw=0.6)
    rows = []
    for n, r in t["runs"].items():
        for k in r:
            if k == "w": continue
            name = {"stage2_alone": "stage 2 alone", "lr": "logistic regression (secondary)"}.get(k, f"fused, w = {r['w']} (primary)")
            rows.append((n, name, r[k]))
    for i, (n, name, x) in enumerate(rows[::-1]):
        lo, hi = x["delta_cpm_ci95"]
        b.plot([lo, hi], [i, i], color=COL[n], lw=2, solid_capstyle="round"); b.plot([x["delta_cpm"]], [i], "o", color=COL[n], ms=7, mec="white", mew=1.2)
        b.text(0.068, i, f"{x['delta_cpm']:+.3f} [{lo:+.3f}, {hi:+.3f}]      {x['fp_model@90']:.2f}", va="center", fontsize=7.5, color=MUTE)
    b.set_yticks(range(len(rows))); b.set_yticklabels([f"{LAB[n]}: {name}" for n, name, _ in rows[::-1]], fontsize=8, color=INK)
    b.axvline(0, color=MUTE, lw=1); b.set_xlim(-0.08, 0.175); b.set_xticks([-0.06, -0.03, 0, 0.03, 0.06])
    b.set_xlabel("test \u0394CPM vs. detector alone (paired bootstrap 95 % CI)\ndetector alone: FP/scan @ 90 % = 0.43", fontsize=8.5, color=INK)
    b.set_title("(b) Applied once to the fold-0 test scans", fontsize=9.5, color=INK, loc="left")
    b.set_ylim(-0.5, len(rows) - 0.1); b.text(0.068, len(rows) - 0.5, "\u0394CPM [95 % CI]      FP/scan @ 90 %", fontsize=7.5, color=INK, va="center")
    fig.tight_layout(); fig.savefig(OUT, dpi=200); print(OUT)


if __name__ == "__main__":
    main()
