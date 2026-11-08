"""Random geometric + stroke-thickness augmentation, vectorized over a batch.

Drawings made with a mouse differ from scanned pen strokes (thickness, slant,
position), so we train on randomly warped copies. Each warp is a 2x2 affine
matrix A: every output pixel p samples the input at  A^-1 (p - c - t) + c.
"""
import numpy as np

C = 13.5
_YS, _XS = np.mgrid[0:28, 0:28].astype(np.float32)


def affine(X, rng, max_rot=12.0, scale=(0.85, 1.1), max_shear=0.15, max_shift=2.0):
    B = len(X)
    th = np.deg2rad(rng.uniform(-max_rot, max_rot, B))
    s = rng.uniform(*scale, B)
    sh = rng.uniform(-max_shear, max_shear, B)
    tx, ty = rng.uniform(-max_shift, max_shift, (2, B))

    # A = s * R(th) @ [[1, sh], [0, 1]]
    cos, sin = np.cos(th), np.sin(th)
    a, b = s * cos, s * (cos * sh - sin)
    c, d = s * sin, s * (sin * sh + cos)
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det

    px = _XS[None] - C - tx[:, None, None]
    py = _YS[None] - C - ty[:, None, None]
    sx = ia[:, None, None] * px + ib[:, None, None] * py + C
    sy = ic[:, None, None] * px + id_[:, None, None] * py + C
    return bilinear(X, sx, sy)


def bilinear(X, sx, sy):
    B = len(X)
    Xp = np.pad(X, ((0, 0), (1, 1), (1, 1)))  # zero border, so out-of-range samples read 0
    x0, y0 = np.floor(sx), np.floor(sy)
    wx, wy = sx - x0, sy - y0
    x0 = np.clip(x0.astype(np.int64) + 1, 0, 29)
    y0 = np.clip(y0.astype(np.int64) + 1, 0, 29)
    x1, y1 = np.clip(x0 + 1, 0, 29), np.clip(y0 + 1, 0, 29)
    bi = np.arange(B)[:, None, None]
    return (
        Xp[bi, y0, x0] * (1 - wx) * (1 - wy)
        + Xp[bi, y0, x1] * wx * (1 - wy)
        + Xp[bi, y1, x0] * (1 - wx) * wy
        + Xp[bi, y1, x1] * wx * wy
    ).astype(np.float32)


def thickness(X, rng, p_dilate=0.25, p_erode=0.15):
    X = X.copy()
    r = rng.random(len(X))
    dil, ero = r < p_dilate, (r >= p_dilate) & (r < p_dilate + p_erode)
    if dil.any():
        x = X[dil]
        n = np.maximum.reduce([x, np.roll(x, 1, 1), np.roll(x, 1, 2)])
        X[dil] = np.maximum(x, 0.85 * n)
    if ero.any():
        x = X[ero]
        X[ero] = np.minimum.reduce([x, np.roll(x, 1, 1), np.roll(x, 1, 2)])
    return X


def augment(X, rng):
    return np.clip(thickness(affine(X, rng), rng), 0, 1)
