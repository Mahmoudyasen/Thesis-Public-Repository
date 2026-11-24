"""Score the TEST fold's candidates with a trained stage-2 model and run the official scorer (+ bootstrap CI).

    python -m lungcad.stage2.infer --checkpoint work/wp0/stage2/official/C/fold0/seed0/best.pt
Writes test_candidates.csv, froc_test.json/png, and (optionally) test_embeddings.npy for WP4 attention-MIL.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from .. import config as C
from ..data.luna16_index import load_index, get_split
from ..eval.froc import evaluate
from .models import build_model
from .dataset import load_candidates
from .train import score_candidates


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", action="store_true"); ap.add_argument("--own-dir", default=None); ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=False); cfg = ck["cfg"]; tr = cfg["train"]
    model = build_model(ck["model_name"], **{k: v for k, v in ck["model_kwargs"].items() if k not in ("lr", "warmup_iters")}).cuda(); model.load_state_dict(ck["model"])
    index = load_index(); split = get_split(index, ck["fold"]); uids = split[a.split]
    cands = load_candidates(ck["regime"], index, ck["fold"], Path(a.own_dir) if a.own_dir else (C.STAGE1 / "own_candidates"))
    out = Path(a.checkpoint).parent
    if a.embeddings:
        sc, emb = score_candidates(model, cands, uids, tr["eval_batch_size"], tr["workers"], tr["amp"], return_embed=True); np.save(out / f"{a.split}_embeddings.npy", emb)
    else:
        sc = score_candidates(model, cands, uids, tr["eval_batch_size"], tr["workers"], tr["amp"])
    sc.to_csv(out / f"{a.split}_candidates.csv", index=False)
    res = evaluate(sc, uids, out_dir=out, n_bootstrap=a.bootstrap, tag=a.split)
    print(json.dumps({k: res[k] for k in ["n_scans", "n_nodules", "n_candidates", "cpm", "sens_at_fp"] + (["cpm_ci95"] if "cpm_ci95" in res else [])}, indent=1))


if __name__ == "__main__":
    main()
