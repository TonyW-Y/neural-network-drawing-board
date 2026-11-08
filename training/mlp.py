"""A multilayer perceptron written as plain matrix algebra.

Forward pass for each layer l:   z_l = a_{l-1} @ W_l + b_l      (batch of row vectors)
                                 a_l = ReLU(z_l)                (hidden)   / softmax (output)
Backward pass is the chain rule, also only matrix products.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        # He initialization: keeps activation variance stable through ReLU layers
        self.W = [rng.normal(0, np.sqrt(2.0 / i), (i, o)).astype(np.float32) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o, np.float32) for o in sizes[1:]]

    @property
    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def forward(self, X):
        acts, zs = [X], []
        a = X
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            a = np.maximum(z, 0) if l < len(self.W) - 1 else softmax(z)
            acts.append(a)
        return acts, zs

    def predict(self, X):
        return self.forward(X)[0][-1].argmax(1)

    def backward(self, acts, zs, y):
        """Gradients of mean cross-entropy loss. y: int labels (B,)."""
        B = len(y)
        delta = acts[-1].copy()
        delta[np.arange(B), y] -= 1  # d(CE)/dz for softmax output
        delta /= B
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = acts[l].T @ delta
            gb[l] = delta.sum(0)
            if l > 0:
                delta = (delta @ self.W[l].T) * (zs[l - 1] > 0)
        return gW, gb


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())


class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.params, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if self.wd and p.ndim == 2:
                g = g + self.wd * p
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
