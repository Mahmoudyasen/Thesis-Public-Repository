"""Stage-2 training: one (model, fold, seed, regime) run.

    python -m lungcad.stage2.train --config configs/stage2.yaml --model C --fold 0 --seed 0 --regime official

Shared recipe (all three models): AdamW, cosine decay, focal loss, AMP, identical augmentation and epochs; the only
difference is the architecture (and C's lower LR with warm-up, as committed in the plan). Selection by validation CPM
computed with the official scorer on the val subset's candidates re-scored by the model.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .. import config as C
from ..augment import dose_kernel_augment
from ..data.luna16_index import load_index, get_split
from ..version import write_run_manifest, run_tag
from ..eval.froc import evaluate
from .models import build_model, FocalLoss
from .patches import make_scales
from .dataset import load_candidates, TrainCandidates, EvalCandidates, collate
from ..preprocess import normalize


def seed_all(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def lr_at(it, base, warmup, total, floor=0.01):
    if it < warmup: return base * (it + 1) / warmup
    p = (it - warmup) / max(1, total - warmup); return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def score_candidates(model, cands: pd.DataFrame, uids, batch_size=128, workers=6, amp=True, return_embed=False):
    """-> DataFrame(seriesuid, coordX, coordY, coordZ, probability, cand_id, label) [+ embedding matrix]."""
    ds = EvalCandidates(cands, uids)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=True)
    model.eval(); probs = []; ids = []; embs = []
    for blk, sp, sub, y, cid in dl:
        x = make_scales(normalize(blk.cuda(non_blocking=True).float())[:, None], sp, augment=False, subvoxel_offset_zyx=sub)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            if return_embed and hasattr(model, "embed"):
                e = model.embed(x); logit = model.head(e).squeeze(1); embs.append(e.float().cpu())
            else:
                logit = model(x)
        probs.append(torch.sigmoid(logit.float()).cpu()); ids.append(cid)
    probs = torch.cat(probs).numpy(); ids = torch.cat(ids).numpy()
    out = ds.c[["seriesuid", "coordX", "coordY", "coordZ", "cand_id", "label"]].copy(); out["probability"] = probs
    assert (out.cand_id.values == ids).all()
    return (out, torch.cat(embs).numpy()) if return_embed else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--model", required=True, choices=["A", "B", "C"])
    ap.add_argument("--fold", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="official", choices=["official", "own"]); ap.add_argument("--own-dir", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--resume", action="store_true"); ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--variant", default="", help="ablation tag: overrides model kwargs, e.g. 'fusion=concat,depth=6,use_scale_embedding=false'")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); mcfg = dict(cfg["models"][a.model]); tr = cfg["train"]
    if a.epochs: tr["epochs"] = a.epochs
    if a.variant:
        for kv in a.variant.split(","):
            k, v = kv.split("="); mcfg[k] = {"true": True, "false": False}.get(v.lower(), int(v) if v.isdigit() else v)
    cfg["_resolved_model"] = mcfg
    name = a.model + (("_" + a.variant.replace("=", "").replace(",", "_")) if a.variant else "")
    out = Path(a.out or (C.STAGE2 / a.regime / name / f"fold{a.fold}" / f"seed{a.seed}")); out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed); torch.backends.cudnn.benchmark = True
    write_run_manifest(out, cfg, {"fold": a.fold, "seed": a.seed, "stage": 2, "model": a.model, "regime": a.regime, "variant": a.variant})
    index = load_index(); split = get_split(index, a.fold)
    cands = load_candidates(a.regime, index, a.fold, Path(a.own_dir) if a.own_dir else (C.STAGE1 / "own_candidates"))
    ds = TrainCandidates(cands, split["train"], pos_repeat=tr["pos_repeat"], neg_per_scan=tr["neg_per_scan"], hard_frac=tr["hard_neg_frac"], seed=a.seed,
                         hard_prior=tr.get("hard_neg_prior", 1.0))
    dl = DataLoader(ds, batch_size=tr["batch_size"], num_workers=tr["workers"], collate_fn=collate, pin_memory=True, persistent_workers=False, drop_last=True)
    model = build_model(a.model, **{k: v for k, v in mcfg.items() if k not in ("lr", "warmup_iters")}).cuda()
    lr = mcfg.get("lr", tr["lr"]); warm = mcfg.get("warmup_iters", tr["warmup_iters"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=tr["amp"]); crit = FocalLoss(tr["focal_alpha"], tr["focal_gamma"])
    iters_per_epoch = len(ds) // tr["batch_size"]; total = tr["epochs"] * iters_per_epoch
    state = {"epoch": 0, "it": 0, "best_cpm": -1.0, "best_epoch": -1}
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cuda", weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
        if (out / "neg_scores.npy").exists(): ds.set_scores(np.load(out / "neg_scores.npy"))
        print("resumed", state)
    tb = SummaryWriter(out / "tb"); log = open(out / "train.log", "a"); aug = cfg.get("augment", {})
    print(f"model {name} params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  samples/epoch {len(ds)}  iters/epoch {iters_per_epoch}  run {run_tag(cfg)}", flush=True)
    val_uids = split["val"][:3] if a.smoke else split["val"]
    if a.smoke: tr["epochs"] = state["epoch"] + 1
    for epoch in range(state["epoch"], tr["epochs"]):
        model.train(); ds.set_epoch(epoch); t0 = time.time(); ls = []; n_pos = 0; n = 0
        for bi, (blk, sp, sub, y, cid) in enumerate(dl):
            if a.smoke and bi >= 3: break
            cur = lr_at(state["it"], lr, warm, total)
            for g in opt.param_groups: g["lr"] = cur
            xb = normalize(blk.cuda(non_blocking=True).float())[:, None]
            # dose/kernel augmentation on the native-spacing block, before the three scales are cut: all scales then see
            # the same simulated noise/kernel, with the same physical meaning as the stage-1 (S4) augmentation
            if aug.get("dose_kernel", False): xb = dose_kernel_augment(xb, **aug.get("dose_kernel_params", {}))
            x = make_scales(xb, sp, augment=True, subvoxel_offset_zyx=sub)
            y = y.cuda()
            with torch.autocast("cuda", dtype=torch.float16, enabled=tr["amp"]):
                logit = model(x); loss = crit(logit.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); state["it"] += 1
            with torch.no_grad():   # online hard-negative bookkeeping
                p = torch.sigmoid(logit.float()).cpu().numpy(); neg = (y.cpu().numpy() == 0)
                ds.scores[cid.numpy()[neg]] = p[neg]
            ls.append(loss.item()); n_pos += int(y.sum()); n += len(y)
            if state["it"] % 20 == 0: tb.add_scalar("train/loss", loss.item(), state["it"]); tb.add_scalar("train/lr", cur, state["it"])
        state["epoch"] = epoch + 1
        msg = f"epoch {epoch+1}/{tr['epochs']} loss {np.mean(ls) if ls else 0:.4f} pos_frac {n_pos/max(1,n):.3f} lr {cur:.2e} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G"
        if (epoch + 1) % tr["val_every"] == 0 or epoch + 1 == tr["epochs"]:
            tv = time.time(); sc = score_candidates(model, cands, val_uids, tr["eval_batch_size"], tr["workers"], tr["amp"])
            res = evaluate(sc, val_uids, n_bootstrap=0, tag=f"val_e{epoch+1}"); cpm = res["cpm"]
            tb.add_scalar("val/cpm", cpm, epoch + 1); msg += f" | val CPM {cpm:.4f} sens@0.125 {res['sens_at_fp'][0]:.3f} sens@8 {res['sens_at_fp'][-1]:.3f} ({time.time()-tv:.0f}s)"
            with open(out / "val_history.jsonl", "a") as f: f.write(json.dumps({"epoch": epoch + 1, **res}) + "\n")
            if cpm > state["best_cpm"]:
                state["best_cpm"], state["best_epoch"] = cpm, epoch + 1
                torch.save({"model": model.state_dict(), "cfg": cfg, "model_name": a.model, "model_kwargs": mcfg, "state": state, "fold": a.fold, "seed": a.seed, "regime": a.regime, "run_tag": run_tag(cfg)}, out / "best.pt"); msg += " *best*"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "cfg": cfg, "model_name": a.model, "model_kwargs": mcfg, "fold": a.fold, "seed": a.seed, "regime": a.regime, "run_tag": run_tag(cfg)}, out / "last.pt")
        np.save(out / "neg_scores.npy", ds.scores)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    with open(out / "done.json", "w") as f: json.dump({"best_cpm": state["best_cpm"], "best_epoch": state["best_epoch"], "run_tag": run_tag(cfg)}, f, indent=1)


if __name__ == "__main__":
    main()
