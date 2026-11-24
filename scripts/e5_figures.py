"""E5 figures: gallery of the 27 missed nodules (axial, 40 mm) with review category, and misses by category x size.

Categories come from a visual review of the three-view crops (views/miss_XX.png) on 4 Oct 2026; one primary
category per miss. Writes report/e5_miss_gallery.png and report/e5_miss_categories.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.luna16_preprocess import stem_for
from lungcad.preprocess import load_volume, world_to_voxel

OUT = C.WORK / "wp0" / "analysis" / "e5_misses"; REP = Path(__file__).resolve().parents[1] / "report"
CAT = {"near": "Localisation near-miss", "wall": "Juxta-pleural / mediastinal / diaphragmatic", "paren": "In abnormal lung (fibrosis, emphysema)",
       "ggn": "Faint ground-glass / subsolid", "vessel": "Vessel-attached / vessel mimic", "noise": "Severe noise / streak artefact",
       "unexpl": "Conspicuous solid, unexplained"}
REVIEW = {1: ("wall", "posterior base, merges with diaphragm/liver dome"), 2: ("paren", "spiculated perihilar mass in fibrotic, honeycomb lung"),
          3: ("paren", "irregular cavitary lesion in honeycomb lung"), 4: ("wall", "sessile on the right heart border"),
          5: ("paren", "irregular apical lesion within scarring/emphysema"), 6: ("ggn", "10 mm pure ground-glass, subpleural"),
          7: ("paren", "subpleural cystic lesion in fibrotic lung; 0.28 box 2.1 r away"), 8: ("ggn", "10 mm ground-glass next to a vessel"),
          9: ("near", "elongated nodule; 0.93 box 1.25 r from centre"), 10: ("wall", "at the costophrenic pleura, 12 mm from crop edge"),
          11: ("vessel", "solid, on a vessel, streak artefacts"), 12: ("ggn", "faint 6 mm ground-glass"),
          13: ("unexpl", "clearly visible isolated solid 6 mm nodule"), 14: ("wall", "paravertebral, against the mediastinum"),
          15: ("vessel", "solid, on a vessel branch, streaks"), 16: ("ggn", "faint 5 mm ground-glass"), 17: ("ggn", "faint 5 mm subsolid, noisy scan"),
          18: ("noise", "very noisy scan with heavy streaks"), 19: ("vessel", "looks like a vessel cross-section"),
          20: ("near", "0.97 box 1.07 r from centre"), 21: ("ggn", "barely visible 4 mm ground-glass, subpleural"), 22: ("ggn", "barely visible 4 mm ground-glass"),
          23: ("vessel", "small solid on a vessel"), 24: ("noise", "faint 4 mm in a noisy scan with motion banding"),
          25: ("ggn", "barely visible 4 mm"), 26: ("wall", "4 mm juxta-pleural, banding artefact"), 27: ("ggn", "barely visible 3.5 mm")}
COLORS = {"lt10": "#86b6ef", "ge10": "#1c5cab"}


def main():
    mi = pd.read_csv(OUT / "misses.csv", dtype={"seriesuid": str})
    mi["category"] = mi.miss_id.map(lambda i: REVIEW[i][0]); mi["review_note"] = mi.miss_id.map(lambda i: REVIEW[i][1])
    mi.to_csv(OUT / "misses_reviewed.csv", index=False)
    # gallery
    fig, axs = plt.subplots(4, 7, figsize=(14, 8.6))
    for ax, (_, m) in zip(axs.flat, mi.iterrows()):
        arr, meta = load_volume(stem_for(m.seriesuid, int(m.subset))); sp = np.array(meta["spacing_zyx"]); shp = np.array(arr.shape)
        v = world_to_voxel(meta, [[m.coordX, m.coordY, m.coordZ]])[0]; vi = np.clip(np.round(v).astype(int), 0, shp - 1)
        h = np.ceil(20 / sp).astype(int); y0, y1 = max(0, vi[1] - h[1]), min(shp[1], vi[1] + h[1]); x0, x1 = max(0, vi[2] - h[2]), min(shp[2], vi[2] + h[2])
        ax.imshow(arr[vi[0], y0:y1, x0:x1], cmap="gray", vmin=-1000, vmax=200, extent=[x0 * sp[2], x1 * sp[2], y1 * sp[1], y0 * sp[1]])
        ax.add_patch(Circle((v[2] * sp[2], v[1] * sp[1]), m.diameter_mm / 2 + 1.5, fill=False, ec="#2bd96b", lw=1.1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"#{m.miss_id}  {m.diameter_mm:.1f} mm  (f{m.fold})\n{CAT[m.category].split(' (')[0].split(' /')[0]}", fontsize=7.5)
    for ax in axs.flat[len(mi):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(REP / "e5_miss_gallery.png", dpi=130); plt.close(fig)
    # category x size
    order = [k for k in CAT if (mi.category == k).any()]; order = sorted(order, key=lambda k: (mi.category == k).sum())
    lt = [((mi.category == k) & (mi.diameter_mm < 10)).sum() for k in order]; ge = [((mi.category == k) & (mi.diameter_mm >= 10)).sum() for k in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.4)); y = np.arange(len(order))
    ax.barh(y, lt, color=COLORS["lt10"], height=0.6, label="< 10 mm", edgecolor="white", linewidth=2)
    ax.barh(y, ge, left=lt, color=COLORS["ge10"], height=0.6, label="≥ 10 mm", edgecolor="white", linewidth=2)
    for i in range(len(order)):
        ax.text(lt[i] + ge[i] + 0.15, i, str(lt[i] + ge[i]), va="center", fontsize=9, color="#0b0b0b")
    ax.set_yticks(y); ax.set_yticklabels([CAT[k] for k in order], fontsize=9); ax.set_xlabel("missed nodules (of 27)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.set_xlim(0, max(np.add(lt, ge)) + 1.2)
    fig.tight_layout(); fig.savefig(REP / "e5_miss_categories.png", dpi=150); plt.close(fig)
    print(mi.groupby("category").agg(n=("miss_id", "size"), ge10=("diameter_mm", lambda d: (d >= 10).sum()), ids=("miss_id", list)))


if __name__ == "__main__":
    main()
