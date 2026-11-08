"""Train a model and export its weights for the browser.

    python training/train.py --dataset digits
    python training/train.py --dataset letters
"""
import argparse
import base64
import json
import math
import os
import time

import numpy as np

import data
from augment import augment
from mlp import MLP, Adam, cross_entropy

ARCH = {
    "digits": [784, 32, 32, 10],
    "letters": [784, 64, 32, 26],
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "public", "models")


def accuracy(model, X, y):
    return float((model.predict(X) == y).mean())


def train(dataset, epochs, batch, lr, seed):
    Xtr, ytr, Xte, yte, labels = data.load(dataset)
    Xtr_f, Xte_f = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    rng = np.random.default_rng(seed)
    model = MLP(ARCH[dataset], seed=seed)
    opt = Adam(model.W + model.b, lr=lr, wd=1e-4)
    print(f"{dataset}: {' -> '.join(map(str, model.sizes))}  ({model.n_params:,} params)")

    steps_per_epoch = len(Xtr) // batch
    total = epochs * steps_per_epoch
    for ep in range(epochs):
        t0 = time.time()
        perm = rng.permutation(len(Xtr))
        losses = []
        for i in range(steps_per_epoch):
            step = ep * steps_per_epoch + i
            opt.lr = lr * 0.5 * (1 + math.cos(math.pi * step / total))  # cosine decay
            idx = perm[i * batch : (i + 1) * batch]
            # train on half augmented, half clean images
            Xb = Xtr[idx]
            half = batch // 2
            Xb = np.concatenate([augment(Xb[:half], rng), Xb[half:]]).reshape(len(idx), -1)
            acts, zs = model.forward(Xb)
            losses.append(cross_entropy(acts[-1], ytr[idx]))
            gW, gb = model.backward(acts, zs, ytr[idx])
            opt.step(gW + gb)
        print(
            f"epoch {ep + 1:3d}/{epochs}  loss {np.mean(losses):.4f}  "
            f"test acc {accuracy(model, Xte_f, yte):.4f}  ({time.time() - t0:.1f}s)"
        )
    return model, Xte, yte, labels, len(Xtr)


def export(model, dataset, labels, Xte, yte, epochs, train_size):
    # Round weights exactly as the browser will see them, then evaluate that model
    model.W = [np.round(W, 4).astype(np.float32) for W in model.W]
    model.b = [np.round(b, 4).astype(np.float32) for b in model.b]
    acc = accuracy(model, Xte.reshape(len(Xte), -1), yte)

    # A few test images per class for the "sample" button + parity tests
    rng = np.random.default_rng(1)
    per_class = 3 if len(labels) <= 10 else 2
    samples = []
    for c in range(len(labels)):
        for i in rng.choice(np.where(yte == c)[0], per_class, replace=False):
            px = np.round(Xte[i] * 255).astype(np.uint8)
            x = px.reshape(1, -1).astype(np.float32) / 255.0
            _, zs = model.forward(x)
            samples.append({
                "label": int(c),
                "pixels": base64.b64encode(px.tobytes()).decode(),
                "logits": [round(float(v), 5) for v in zs[-1][0]],
            })

    out = {
        "name": dataset,
        "labels": labels,
        "sizes": model.sizes,
        # W stored row-major as (out x in): row j is neuron j's weight vector, so  z = W x + b
        "layers": [
            {"shape": [W.shape[1], W.shape[0]], "W": W.T.ravel().tolist(), "b": b.tolist()}
            for W, b in zip(model.W, model.b)
        ],
        "meta": {
            "params": model.n_params,
            "testAccuracy": round(acc, 4),
            "epochs": epochs,
            "trainSize": train_size,
            "loss": "Cross-entropy",
            "optimizer": "Adam",
            "framework": "NumPy",
        },
        "samples": samples,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"exported {path}  ({os.path.getsize(path) / 1e6:.2f} MB)  test acc (rounded weights) {acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["digits", "letters"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model, Xte, yte, labels, n = train(a.dataset, a.epochs, a.batch, a.lr, a.seed)
    export(model, a.dataset, labels, Xte, yte, a.epochs, n)
