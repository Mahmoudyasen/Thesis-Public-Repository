"""WP4 prior -> current registration (plan Section 6): lung-mask driven, rigid then affine, SimpleITK.

Masks: lung_mask_crop = lungcad.preprocess.lung_mask_classical (-320 HU + connected components) on the 1.5 mm archive
volume, with a per-slice border-rule fallback when that mask is < 1 L (lungs touching the 10 mm-margin crop border). The
conversion kept only the lung box, not the lungmask masks, so masks are recomputed. Registration runs on the masks
smoothed with a 2 mm Gaussian (mean squares, 3-level pyramid), first Euler3D, then affine initialised from the rigid result.
The returned transform T maps a CURRENT world point (x, y, z mm) to the PRIOR world frame (SimpleITK fixed -> moving),
so candidate matching is done as T(current candidate) against the prior candidates.
QC: Dice of the prior mask resampled through T onto the current mask; a pair with Dice < 0.85 counts as a failure.
K3 (known_answer_k3): apply a known rigid + affine transform to a real scan, register it back, and require a mean
error <= 1 mm over the lung-surface points.
"""
from __future__ import annotations
import numpy as np, SimpleITK as sitk
from scipy import ndimage
from lungcad.preprocess import load_volume, lung_mask_classical

DICE_MIN = 0.85


def _img(arr, meta, dtype=sitk.sitkFloat32):
    im = sitk.GetImageFromArray(arr.astype(np.float32)); im = sitk.Cast(im, dtype)
    im.SetSpacing(tuple(float(s) for s in meta["spacing_zyx"][::-1])); im.SetOrigin(tuple(float(v) for v in meta["origin_xyz"]))
    im.SetDirection(tuple(float(v) for v in meta["direction"])); return im


def lung_mask_crop(arr, spacing, thresh: float = -320.0, min_litres: float = 1.0):
    """lung_mask_classical; if that gives < min_litres (the archive crop keeps only a 10 mm margin, so the lungs can
    touch the crop border and are then removed as 'outside air'), fall back to the air components >= 10 % of the largest
    WITHOUT the border rule. -> (mask uint8, method)."""
    m = lung_mask_classical(arr, spacing, thresh)
    vox_l = float(np.prod(spacing)) / 1e6
    if m.sum() * vox_l >= min_litres:
        return m, "classical"
    # fallback: remove border-touching air PER AXIAL SLICE (outside-body air touches the border on its own slice; the lungs
    # are lost only on the few slices where they touch the crop edge), then keep 3D components >= 10 % of the largest
    air = arr < thresh
    st2 = np.zeros((3, 3, 3), bool); st2[1] = ndimage.generate_binary_structure(2, 1)
    lab2, n2 = ndimage.label(air, structure=st2)
    if n2 == 0:
        return m, "empty"
    border = np.unique(np.concatenate([lab2[:, 0, :].ravel(), lab2[:, -1, :].ravel(), lab2[:, :, 0].ravel(), lab2[:, :, -1].ravel()]))
    inner = air & ~np.isin(lab2, border[border > 0])
    lab, n = ndimage.label(inner)
    if n == 0:
        return m, "empty"
    sizes = ndimage.sum(inner, lab, index=np.arange(1, n + 1)); keep = np.where(sizes >= 0.10 * sizes.max())[0] + 1
    out = np.isin(lab, keep).astype(np.uint8)
    return out, ("slice_border_rule" if out.sum() * vox_l >= min_litres else "empty")


def mask_image(stem):
    arr, meta = load_volume(stem)
    m, method = lung_mask_crop(arr, meta["spacing_zyx"]); meta = {**meta, "reg_mask_method": method}
    return _img(m, meta), meta


def _smooth(m):
    return sitk.SmoothingRecursiveGaussian(sitk.Cast(m, sitk.sitkFloat32), 2.0)


def _register(fixed, moving, init, kind):
    R = sitk.ImageRegistrationMethod(); R.SetMetricAsMeanSquares(); R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.10, seed=0)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0 if kind == "rigid" else 1.0, minStep=1e-4, numberOfIterations=500,
                                               relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift(); R.SetShrinkFactorsPerLevel([4, 2, 1]); R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn(); R.SetInitialTransform(init, inPlace=False)
    return R.Execute(fixed, moving)


def register_masks(fixed_mask, moving_mask) -> tuple[sitk.Transform, dict]:
    f, m = _smooth(fixed_mask), _smooth(moving_mask)
    init = sitk.CenteredTransformInitializer(f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    rigid = _register(f, m, init, "rigid")
    rig = sitk.Euler3DTransform(sitk.CompositeTransform(rigid).GetNthTransform(0)) if isinstance(rigid, sitk.CompositeTransform) else sitk.Euler3DTransform(rigid)
    aff0 = sitk.AffineTransform(3); aff0.SetMatrix(rig.GetMatrix()); aff0.SetTranslation(rig.GetTranslation()); aff0.SetCenter(rig.GetCenter())
    aff = _register(f, m, aff0, "affine")
    T = sitk.AffineTransform(aff) if not isinstance(aff, sitk.CompositeTransform) else sitk.AffineTransform(aff.GetNthTransform(0))
    return T, {"dice_rigid": dice(fixed_mask, moving_mask, rig), "dice": dice(fixed_mask, moving_mask, T)}


def dice(fixed_mask, moving_mask, T) -> float:
    w = sitk.Resample(moving_mask, fixed_mask, T, sitk.sitkNearestNeighbor, 0.0)
    a = sitk.GetArrayViewFromImage(fixed_mask) > 0.5; b = sitk.GetArrayViewFromImage(w) > 0.5
    return float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))


def register_pair(current_stem, prior_stem) -> tuple[sitk.Transform, dict]:
    fm, _ = mask_image(current_stem); mm, _ = mask_image(prior_stem)
    T, q = register_masks(fm, mm); q["ok"] = q["dice"] >= DICE_MIN
    return T, q


def map_points(T: sitk.Transform, xyz: np.ndarray) -> np.ndarray:
    return np.array([T.TransformPoint(tuple(float(v) for v in p)) for p in np.atleast_2d(xyz)])


def known_answer_k3(stems, tol_mm: float = 1.0, seed: int = 0) -> dict:
    """For each scan: synthetic prior = scan resampled through a known rigid+affine T0 (rotation <= 5 deg, shift <= 10 mm,
    scale +-3 %, shear <= 2 %); register; error = mean |T(p) - T0(p)| over lung-surface points of the current mask."""
    rng = np.random.RandomState(seed); out = []
    for st in stems:
        fm, meta = mask_image(st)
        c = np.array(fm.TransformContinuousIndexToPhysicalPoint([s / 2 for s in fm.GetSize()]))
        ang = np.deg2rad(rng.uniform(-5, 5, 3)); rot = sitk.Euler3DTransform(tuple(c), *ang, tuple(rng.uniform(-10, 10, 3)))
        A = np.array(rot.GetMatrix()).reshape(3, 3) @ (np.eye(3) + np.diag(rng.uniform(-.03, .03, 3)) + np.triu(rng.uniform(-.02, .02, (3, 3)), 1))
        T0 = sitk.AffineTransform(3); T0.SetCenter(tuple(c)); T0.SetMatrix(tuple(A.ravel())); T0.SetTranslation(rot.GetTranslation())
        # synthetic prior P with P(T0(x)) = F(x): resample F through T0^-1 on a grid covering the moved lungs
        pad = sitk.ConstantPad(fm, [40] * 3, [40] * 3, 0.0)
        prior = sitk.Resample(fm, pad, T0.GetInverse(), sitk.sitkNearestNeighbor, 0.0)
        T, q = register_masks(fm, prior)
        a = sitk.GetArrayFromImage(fm) > 0.5; surf = np.argwhere(a & ~ndimage.binary_erosion(a))
        surf = surf[rng.choice(len(surf), min(3000, len(surf)), replace=False)]
        pts = np.array([fm.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in z[::-1])) for z in surf])
        err = np.linalg.norm(map_points(T, pts) - map_points(T0, pts), axis=1)
        out.append({"stem": str(st), "mean_err_mm": float(err.mean()), "p95_err_mm": float(np.percentile(err, 95)), **q})
    return {"cases": out, "pass": all(o["mean_err_mm"] <= tol_mm for o in out)}
