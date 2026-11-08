"""Dataset loading + the canonical image normalization.

`normalize()` is mirrored line-for-line in web/src/preprocess.ts so that what the
network sees during training is exactly what it sees when you draw in the browser.
"""
import gzip
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DIGIT_LABELS = [str(i) for i in range(10)]
LETTER_LABELS = [chr(ord("A") + i) for i in range(26)]

BBOX_THRESHOLD = 0.1  # a pixel counts as "ink" above this value
BOX = 20  # longest side of the glyph after scaling (MNIST convention)
SIZE = 28


def read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if raw[0] != 0 or raw[1] != 0 or raw[2] != 0x08:
        raise ValueError(f"{path}: not an unsigned-byte IDX file")
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(dims)


# ---------------------------------------------------------------- normalization
def area_matrix(n_out, n_in):
    """(n_out x n_in) matrix that resamples a 1-D signal by box-filter overlap.

    Resizing an image is then just two matrix products:  R_y @ img @ R_x.T
    """
    m = np.zeros((n_out, n_in), np.float64)
    scale = n_in / n_out
    for i in range(n_out):
        lo, hi = i * scale, (i + 1) * scale
        for j in range(int(np.floor(lo)), min(n_in, int(np.ceil(hi)))):
            overlap = min(hi, j + 1) - max(lo, j)
            if overlap > 0:
                m[i, j] = overlap
        m[i] /= m[i].sum()
    return m


def normalize(img):
    """img: (H, W) float array in [0, 1] (any size). Returns (28, 28) float32.

    1. crop to the bounding box of the ink
    2. scale so the longer side is 20px (aspect ratio kept)
    3. paste centered into 28x28
    4. shift (integer) so the center of mass is at the center
    """
    ink = img > BBOX_THRESHOLD
    out = np.zeros((SIZE, SIZE), np.float32)
    if not ink.any():
        return out
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    crop = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    h, w = crop.shape
    s = BOX / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = area_matrix(nh, h) @ crop @ area_matrix(nw, w).T
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[top : top + nh, left : left + nw] = small

    total = out.sum()
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    cy = (ys * out).sum() / total
    cx = (xs * out).sum() / total
    dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
    return shift(out, dy, dx)


def shift(img, dy, dx):
    out = np.zeros_like(img)
    h, w = img.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def normalize_all(images):
    return np.stack([normalize(im) for im in images]).astype(np.float32)


# ---------------------------------------------------------------- datasets
def _files(dataset):
    if dataset == "digits":
        p = lambda n: os.path.join(DATA_DIR, f"mnist-{n}.gz")
        return (
            p("train-images-idx3-ubyte"), p("train-labels-idx1-ubyte"),
            p("t10k-images-idx3-ubyte"), p("t10k-labels-idx1-ubyte"),
        )
    p = lambda n: os.path.join(DATA_DIR, f"emnist-letters-{n}.gz")
    return (
        p("train-images-idx3-ubyte"), p("train-labels-idx1-ubyte"),
        p("test-images-idx3-ubyte"), p("test-labels-idx1-ubyte"),
    )


def load(dataset):
    """Returns Xtr (N,28,28) float32, ytr (N,), Xte, yte, labels. Cached as .npz."""
    cache = os.path.join(DATA_DIR, f"{dataset}-normalized.npz")
    labels = DIGIT_LABELS if dataset == "digits" else LETTER_LABELS
    if os.path.exists(cache):
        d = np.load(cache)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"], labels

    tri, trl, tei, tel = _files(dataset)
    Xtr, ytr = read_idx(tri), read_idx(trl).astype(np.int64)
    Xte, yte = read_idx(tei), read_idx(tel).astype(np.int64)
    if dataset == "letters":
        # EMNIST is stored transposed relative to MNIST, labels are 1..26
        Xtr, Xte = Xtr.transpose(0, 2, 1), Xte.transpose(0, 2, 1)
        ytr, yte = ytr - 1, yte - 1

    print(f"normalizing {len(Xtr) + len(Xte)} images (one-time, cached)...")
    Xtr = normalize_all(Xtr.astype(np.float32) / 255.0)
    Xte = normalize_all(Xte.astype(np.float32) / 255.0)
    np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte, labels
