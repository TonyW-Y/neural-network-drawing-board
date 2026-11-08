"""Write preprocessing test fixtures so the browser's normalize() can be checked
against the Python one used for training (web/tests/preprocess.test.ts)."""
import json
import os

import numpy as np

import data

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "tests", "fixtures", "preprocess.json")


def synthetic(rng, h, w):
    """Thick random strokes on a large canvas, like a browser drawing."""
    img = np.zeros((h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for _ in range(3):
        y0, x0, y1, x1 = rng.uniform(0.1, 0.9, 4) * [h, w, h, w]
        for t in np.linspace(0, 1, 40):
            cy, cx = y0 + t * (y1 - y0), x0 + t * (x1 - x0)
            d = np.hypot(ys - cy, xs - cx)
            img = np.maximum(img, np.clip(1 - (d - 6) / 2, 0, 1))
    return img


def main():
    rng = np.random.default_rng(7)
    raw = data.read_idx(os.path.join(data.DATA_DIR, "emnist-letters-test-images-idx3-ubyte.gz"))
    cases = []
    for i in rng.choice(len(raw), 4, replace=False):
        cases.append(raw[i].T.astype(np.float32) / 255.0)
    cases += [synthetic(rng, 280, 280), synthetic(rng, 200, 140), synthetic(rng, 97, 251)]
    out = [
        {
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "input": np.round(img, 6).ravel().tolist(),
            "expected": data.normalize(np.round(img, 6).astype(np.float32)).ravel().tolist(),
        }
        for img in cases
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out)} cases to {OUT}")


if __name__ == "__main__":
    main()
