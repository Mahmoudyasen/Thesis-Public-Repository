"""Candidate patch extraction for stage 2.

CPU side (dataset workers): cut ONE raw int16 block of physical size BLOCK_MM (=104 mm: 96 mm + jitter margin) around
the candidate from the preprocessed volume at native spacing, padded with air (HU_MIN) outside the volume.
GPU side (make_scales): resample that block into the three 48^3 crops at 24/48/96 mm with one affine_grid/grid_sample
per scale. Augmentation is expressed in the sampling grid: random flips (sign), in-plane transpose (axis swap) and
translation jitter (+-2 mm) — identical for all three scales of a sample, as the plan requires (small translations
+-2 voxels, flips, rotations).
"""
from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from .. import config as C
from ..preprocess import normalize

BLOCK_MM = 104.0
SCALES_MM = (24.0, 48.0, 96.0)
OUT = 48


def block_shape(spacing_zyx) -> np.ndarray:
    return np.ceil(BLOCK_MM / np.asarray(spacing_zyx)).astype(int) | 1     # odd sizes -> a centre voxel exists


def extract_block(arr: np.ndarray, center_zyx, spacing_zyx) -> np.ndarray:
    """int16 (bz, by, bx) block centred on the (continuous) voxel coordinate, air-padded."""
    shp = block_shape(spacing_zyx); half = shp // 2
    c = np.round(np.asarray(center_zyx)).astype(int)
    lo = c - half; hi = lo + shp
    out = np.full(shp, int(C.HU_MIN), np.int16)
    src_lo = np.maximum(lo, 0); src_hi = np.minimum(hi, np.array(arr.shape))
    if np.all(src_hi > src_lo):
        dst_lo = src_lo - lo; dst_hi = dst_lo + (src_hi - src_lo)
        out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = arr[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


@torch.no_grad()
def make_scales(block: torch.Tensor, spacing_zyx: torch.Tensor, scales_mm=SCALES_MM, out: int = OUT, augment: bool = False,
                jitter_mm: float = 2.0, subvoxel_offset_zyx: torch.Tensor | None = None) -> torch.Tensor:
    """block: (B,1,bz,by,bx) float in [0,1] on GPU; spacing_zyx: (B,3) mm. -> (B, n_scales, out, out, out).
    subvoxel_offset_zyx: (B,3) mm offset of the true candidate centre from the block centre voxel (optional)."""
    B = block.shape[0]; dev = block.device
    extent = torch.tensor(block.shape[2:], device=dev, dtype=torch.float32)[None] * spacing_zyx.to(dev)   # (B,3) mm, (z,y,x)
    off = torch.zeros(B, 3, device=dev) if subvoxel_offset_zyx is None else subvoxel_offset_zyx.to(dev)
    if augment:
        off = off + (torch.rand(B, 3, device=dev) * 2 - 1) * jitter_mm
        sign = torch.where(torch.rand(B, 3, device=dev) < 0.5, -1.0, 1.0)
        swap = torch.rand(B, device=dev) < 0.5
    else:
        sign = torch.ones(B, 3, device=dev); swap = torch.zeros(B, dtype=torch.bool, device=dev)
    outs = []
    for s in scales_mm:
        # theta maps output normalised coords (x,y,z) -> input normalised coords (x,y,z); torch axis order is (x,y,z)=(W,H,D)
        sc = s / extent                                              # (B,3) in (z,y,x)
        t = 2 * off / extent                                         # translation in normalised input units
        theta = torch.zeros(B, 3, 4, device=dev)
        # rows: input x, y, z ; cols: output x, y, z, 1
        theta[:, 0, 0] = sc[:, 2] * sign[:, 2]; theta[:, 1, 1] = sc[:, 1] * sign[:, 1]; theta[:, 2, 2] = sc[:, 0] * sign[:, 0]
        theta[:, 0, 3] = t[:, 2]; theta[:, 1, 3] = t[:, 1]; theta[:, 2, 3] = t[:, 0]
        if swap.any():   # in-plane transpose: input x <- output y, input y <- output x
            th = theta[swap].clone()
            th[:, 0, 0], th[:, 0, 1] = 0.0, sc[swap, 2] * sign[swap, 2]
            th[:, 1, 1], th[:, 1, 0] = 0.0, sc[swap, 1] * sign[swap, 1]
            theta[swap] = th
        grid = F.affine_grid(theta, (B, 1, out, out, out), align_corners=False)
        outs.append(F.grid_sample(block, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    return torch.cat(outs, 1)


def block_to_tensor(blocks_int16: np.ndarray) -> torch.Tensor:
    """(B, bz, by, bx) int16 numpy -> (B,1,...) float32 [0,1] (still on CPU; move to GPU before make_scales)."""
    return normalize(torch.from_numpy(np.ascontiguousarray(blocks_int16)).float())[:, None]
