"""Synthetic 16-bit coronary angiography generator (legal, dependency-free dataset).

Physics-inspired model:  I = I0 * exp(-A) + Poisson noise, where the attenuation
map A = soft tissue + heart shadow + spine + ribs - lungs + contrast-filled
coronary tree.  A ground-truth vessel mask is produced for quantitative testing.
This is for development / benchmarking only - validate on real angiograms too.
"""
import numpy as np
import cv2


def _blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s)


def _make_background(size, rng):
    h = w = size
    A = np.full((h, w), 0.8, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # lungs: two large low-attenuation lobes
    for cx in (0.25, 0.75):
        lung = np.exp(-(((xx - cx * w) / (0.20 * w)) ** 2 + ((yy - 0.42 * h) / (0.32 * h)) ** 2) ** 1.5)
        A -= 0.38 * lung
    # heart shadow (left-centre)
    heart = np.exp(-(((xx - 0.52 * w) / (0.17 * w)) ** 2 + ((yy - 0.55 * h) / (0.20 * h)) ** 2) ** 1.5)
    A += 0.35 * heart
    # spine with vertebral texture
    spine_x = 0.5 * w + rng.uniform(-0.03, 0.03) * w
    band = np.exp(-((xx - spine_x) / (0.045 * w)) ** 4)
    vert = 0.92 + 0.08 * np.cos(yy / (0.045 * h) * 2 * np.pi)
    A += 0.40 * band * vert
    # ribs: curved oblique bands on both sides
    ribs = np.zeros((h, w), np.float32)
    for side in (-1, 1):
        for k in range(7):
            y0 = (0.10 + 0.12 * k) * h + rng.uniform(-4, 4)
            pts = []
            for t in np.linspace(0, 1, 25):
                x = spine_x + side * t * 0.46 * w
                y = y0 + (0.10 * h) * np.sin(t * np.pi * 0.9) + 0.05 * h * t
                pts.append([x, y])
            cv2.polylines(ribs, [np.array(pts, np.int32)], False, 1.0,
                          thickness=int(0.048 * w), lineType=cv2.LINE_AA)
    A += 0.22 * _blur(ribs, 2.5)
    # diaphragm / abdomen
    A += 0.5 / (1 + np.exp(-(yy - 0.88 * h) / (0.02 * h)))
    return A


def _make_tree(size, rng, shift=(0.0, 0.0)):
    """Random branching coronary tree (LAD/LCX-like and RCA-like); returns float mask 0..1."""
    h = w = size
    m = np.zeros((h, w), np.float32)
    ox, oy = shift

    def branch(x, y, ang, width, steps, depth, bend):
        for i in range(steps):
            ang += rng.normal(0, 0.045) + bend
            nx, ny = x + 3.0 * np.cos(ang), y + 3.0 * np.sin(ang)
            cv2.line(m, (int(x + ox), int(y + oy)), (int(nx + ox), int(ny + oy)), 1.0,
                     thickness=max(1, int(round(width))), lineType=cv2.LINE_AA)
            x, y, width = nx, ny, width * 0.9975
            if depth < 3 and i > 15 and rng.random() < 0.028 and width > 2.6:
                branch(x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.6, 1.1),
                       width * 0.68, int(steps * 0.5), depth + 1, -bend)
            if not (0.05 * w < x < 0.95 * w and 0.18 * h < y < 0.85 * h):
                break

    # left system: main stem -> LAD (down-left, curving) ; right coronary: arcs to the right
    branch(0.52 * w, 0.36 * h, 2.2, 0.011 * w, int(0.30 * w), 0, -0.010)
    branch(0.52 * w, 0.36 * h, 1.2, 0.010 * w, int(0.28 * w), 0, +0.012)
    branch(0.46 * w, 0.40 * h, 0.5, 0.009 * w, int(0.22 * w), 0, +0.008)
    return np.clip(_blur(m, 0.7), 0, 1)


class SyntheticCine:
    """Generate frames of a cine run: static bones/lungs + cyclically moving vessels."""

    def __init__(self, size=512, seed=0, noise_photons=900.0):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.A_bg = _make_background(size, self.rng)
        self.tree_seed = seed + 1
        self.noise_photons = noise_photons
        self._tree_cache = {}

    def _tree(self, phase_idx):
        if phase_idx not in self._tree_cache:
            # same tree every time (same seed), only its position changes
            rng = np.random.default_rng(self.tree_seed)
            dx = 5.0 * np.sin(2 * np.pi * phase_idx / 20.0)
            dy = 3.0 * np.cos(2 * np.pi * phase_idx / 20.0)
            self._tree_cache[phase_idx] = _make_tree(self.size, rng, (dx, dy))
        return self._tree_cache[phase_idx]

    def frame(self, i=0, noisy=True):
        tree = self._tree(i % 20)
        A = self.A_bg + 0.20 * tree
        I = np.exp(-A)                                      # transmission 0..1
        if noisy:
            rng = np.random.default_rng(1000 + i)
            I = rng.poisson(I * self.noise_photons).astype(np.float32) / self.noise_photons
            I += rng.normal(0, 0.004, I.shape).astype(np.float32)
        I = np.clip(I, 0, 1.0)
        raw = (I / np.exp(-0.2) * 60000).clip(0, 65535).astype(np.uint16)
        gt = tree > 0.35
        return raw, gt
