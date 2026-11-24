"""Stage-2 candidate datasets over the preprocessed LUNA16 archive.

Candidate regimes (both are run, never mixed):
  official : candidates_V2.csv (754,975 candidates, 1,557 positive) — benchmark comparability
  own      : the frozen stage-1 detector's candidates (per fold) — the end-to-end system
Label: 1 iff the candidate lies within one nodule radius of an annotation (official 'class'; recomputed for 'own').

Training sampler (IterableDataset, per epoch, per scan):  all positives x pos_repeat  +  neg_per_scan negatives
drawn fresh every epoch (the "per-epoch renewal of the negative pool"). Online hard-negative mining: half of the
negatives are the highest-scoring negatives seen so far (the training loop records each negative's latest score; the
table reaches the workers at every epoch start because workers are not persistent), the other half are random, which
keeps exploring. hard_prior is the score a never-seen negative carries: 0.0 (default since 4 Oct 2026) = mining from
epoch 2; 1.0 = the original explore-first behaviour (with ~840 negatives per scan, real mining only began after ~29
epochs). Ties are broken randomly, so never-seen negatives aren't picked in file order.
Each worker owns a shard of scans, decompresses each volume once per epoch, and a shuffle buffer mixes scans.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from .. import config as C
from ..preprocess import load_volume, world_to_voxel, load_meta
from ..data.luna16_preprocess import stem_for
from .patches import extract_block


def load_candidates(regime: str, index: pd.DataFrame, fold: int | None = None, own_dir=None) -> pd.DataFrame:
    """-> DataFrame with seriesuid, cand_id, coordX/Y/Z, vz, vy, vx, label, subset."""
    sub = index.set_index("seriesuid").subset
    if regime == "official":
        df = pd.read_csv(C.LUNA_PRE / "candidates_v2_voxel.csv")
        df["label"] = df["class"].astype(int)
    elif regime == "own":
        # own-detector candidates for this fold: test split from the frozen stage-1 model of the same fold,
        # train/val splits from the models of the folds where those scans were held out (leak-free construction is
        # done by scripts/build_own_candidates.py which concatenates all ten test-split candidate files).
        df = pd.read_csv(own_dir / "own_candidates_all.csv")
        ann = pd.read_csv(C.LUNA_RAW / "annotations.csv"); labels = []
        ag = {k: g for k, g in ann.groupby("seriesuid")}
        for uid, g in df.groupby("seriesuid", sort=False):
            a = ag.get(uid)
            lab = np.zeros(len(g), int)
            if a is not None and len(a):
                d = ((g[["coordX", "coordY", "coordZ"]].values[:, None, :] - a[["coordX", "coordY", "coordZ"]].values[None]) ** 2).sum(-1)
                lab = (d <= (a.diameter_mm.values[None] / 2) ** 2).any(1).astype(int)
            labels.append(pd.Series(lab, index=g.index))
        df["label"] = pd.concat(labels).sort_index().values
        vox = []
        for uid, g in df.groupby("seriesuid", sort=False):
            meta = load_meta(stem_for(uid, int(sub[uid]))); v = world_to_voxel(meta, g[["coordX", "coordY", "coordZ"]].values)
            vox.append(pd.DataFrame(v, columns=["vz", "vy", "vx"], index=g.index))
        df = pd.concat([df, pd.concat(vox).sort_index()], axis=1)
        df["cand_id"] = np.arange(len(df))
    else:
        raise ValueError(regime)
    df["subset"] = df.seriesuid.map(sub).astype(int)
    return df


class TrainCandidates(IterableDataset):
    def __init__(self, cands: pd.DataFrame, uids: list[str], pos_repeat: int = 4, neg_per_scan: int = 48, hard_frac: float = 0.5,
                 seed: int = 0, buffer: int = 256, hard_prior: float = 0.0):
        self.c = cands[cands.seriesuid.isin(set(uids))].reset_index(drop=True)
        self.groups = {k: g for k, g in self.c.groupby("seriesuid")}
        self.uids = [u for u in uids if u in self.groups]
        self.pos_repeat, self.neg_per_scan, self.hard_frac, self.seed, self.buffer = pos_repeat, neg_per_scan, hard_frac, seed, buffer
        self.epoch = 0
        self.scores = np.full(int(self.c.cand_id.max()) + 1, hard_prior, np.float32)   # score of a never-seen negative

    def set_epoch(self, e): self.epoch = e

    def set_scores(self, scores: np.ndarray): self.scores = scores

    def __len__(self):
        return sum(int((g.label == 1).sum()) * self.pos_repeat + min(self.neg_per_scan, int((g.label == 0).sum())) for g in self.groups.values())

    def __iter__(self):
        wi = get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.RandomState(self.seed * 100003 + self.epoch * 7919 + wid)
        order = np.array(self.uids)[rng.permutation(len(self.uids))]
        my = [u for i, u in enumerate(order) if i % nw == wid]
        buf = []
        for uid in my:
            g = self.groups[uid]; arr, meta = load_volume(stem_for(uid, int(g.subset.iloc[0]))); sp = np.array(meta["spacing_zyx"], np.float32)
            pos = g[g.label == 1]; neg = g[g.label == 0]
            rows = [pos] * self.pos_repeat
            if len(neg):
                n_hard = int(round(self.neg_per_scan * self.hard_frac)); n_rand = self.neg_per_scan - n_hard
                sc = self.scores[neg.cand_id.values]
                perm = rng.permutation(len(neg))                                    # random tie-break
                hard_idx = perm[np.argsort(-sc[perm], kind="stable")[:n_hard]] if n_hard else np.zeros(0, int)
                rest = np.setdiff1d(np.arange(len(neg)), hard_idx)
                rand_idx = rng.choice(rest, size=min(n_rand, len(rest)), replace=False) if len(rest) else np.zeros(0, int)
                rows.append(neg.iloc[np.concatenate([hard_idx, rand_idx])])
            sel = pd.concat(rows)
            for r in sel.itertuples():
                blk = extract_block(arr, (r.vz, r.vy, r.vx), sp)
                sub = (np.array([r.vz, r.vy, r.vx]) - np.round([r.vz, r.vy, r.vx])) * sp   # mm offset of the true centre
                item = (torch.from_numpy(blk), torch.tensor(sp), torch.tensor(sub, dtype=torch.float32), int(r.label), int(r.cand_id))
                buf.append(item)
                if len(buf) >= self.buffer:
                    j = rng.randint(len(buf)); yield buf.pop(j)
        rng.shuffle(buf)
        for item in buf: yield item


class EvalCandidates(Dataset):
    """All candidates of the given scans, in scan order (deterministic); volumes cached per scan."""
    def __init__(self, cands: pd.DataFrame, uids: list[str]):
        self.c = cands[cands.seriesuid.isin(set(uids))].sort_values(["seriesuid", "cand_id"]).reset_index(drop=True)
        self._uid = None; self._arr = None; self._sp = None

    def __len__(self): return len(self.c)

    def __getitem__(self, i):
        r = self.c.iloc[i]
        if r.seriesuid != self._uid:
            self._arr, meta = load_volume(stem_for(r.seriesuid, int(r.subset))); self._sp = np.array(meta["spacing_zyx"], np.float32); self._uid = r.seriesuid
        blk = extract_block(self._arr, (r.vz, r.vy, r.vx), self._sp)
        sub = (np.array([r.vz, r.vy, r.vx]) - np.round([r.vz, r.vy, r.vx])) * self._sp
        return torch.from_numpy(blk), torch.tensor(self._sp), torch.tensor(sub, dtype=torch.float32), int(r.label), int(r.cand_id)


def collate(batch):
    blk = torch.stack([b[0] for b in batch]); sp = torch.stack([b[1] for b in batch]); sub = torch.stack([b[2] for b in batch])
    return blk, sp, sub, torch.tensor([b[3] for b in batch]), torch.tensor([b[4] for b in batch])


class ScanOrderedSampler(torch.utils.data.Sampler):
    """Contiguous index chunks so each worker keeps its per-scan volume cache warm during evaluation."""
    def __init__(self, n, num_workers, batch_size):
        self.n, self.nw, self.bs = n, max(1, num_workers), batch_size
    def __iter__(self):
        return iter(range(self.n))
    def __len__(self): return self.n
