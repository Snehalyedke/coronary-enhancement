"""Quantitative quality metrics (need a ground-truth vessel mask, e.g. from the synthetic set)."""
import numpy as np
import cv2


def auc(score: np.ndarray, gt: np.ndarray, stride: int = 2) -> float:
    """ROC-AUC of a per-pixel score against a boolean mask (rank based)."""
    s = score[::stride, ::stride].ravel().astype(np.float64)
    g = gt[::stride, ::stride].ravel().astype(bool)
    n1, n0 = int(g.sum()), int((~g).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    return float((ranks[g].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_dice(score: np.ndarray, gt: np.ndarray, n_thr: int = 40) -> float:
    """Best Dice over a sweep of thresholds (upper bound of segmentation quality)."""
    g = gt.astype(bool)
    lo, hi = float(score.min()), float(score.max())
    best = 0.0
    for t in np.linspace(lo, hi, n_thr + 2)[1:-1]:
        m = score > t
        inter = np.logical_and(m, g).sum()
        d = 2.0 * inter / max(m.sum() + g.sum(), 1)
        best = max(best, d)
    return float(best)


def cnr(img: np.ndarray, gt: np.ndarray) -> float:
    """Contrast-to-noise ratio: |mean(vessel) - mean(local background)| / std(local background)."""
    g = gt.astype(np.uint8)
    ring = cv2.dilate(g, np.ones((25, 25), np.uint8)) - cv2.dilate(g, np.ones((9, 9), np.uint8))
    f = img.astype(np.float64)
    v, b = f[g > 0], f[ring > 0]
    return float(abs(v.mean() - b.mean()) / (b.std() + 1e-9))


def background_clutter(img: np.ndarray, gt: np.ndarray) -> float:
    """Std-dev of the non-vessel area, normalised by the vessel/background level difference.
    Lower = cleaner background relative to the vessel signal (ribs/spine/noise suppressed)."""
    g = cv2.dilate(gt.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    f = img.astype(np.float64)
    contrast = abs(f[gt > 0].mean() - f[~g].mean()) + 1e-9
    return float(f[~g].std() / contrast)
