# Neural Network Drawing Board

**Live demo: https://tonyw-y.github.io/neural-network-drawing-board/**

Draw a digit (0–9) or a letter (A–Z) and watch a small neural network classify it live. Every layer is shown as the matrix–vector product it really is:

```
h₁ = ReLU(W₁ · x  + b₁)     (32×784)·(784) + (32)
h₂ = ReLU(W₂ · h₁ + b₂)     (32×32) ·(32)  + (32)
ŷ  = softmax(W₃ · h₂ + b₃)  (10×32) ·(32)  + (10)
```

Hover any neuron to see its value broken down into the dot product that produced it: one row of the weight matrix, multiplied term by term with the previous layer's vector, summed, plus the bias, then passed through ReLU or softmax.

## Two models

| Model   | Architecture        | Params | Test accuracy | Data                      |
|---------|---------------------|--------|---------------|---------------------------|
| Digits  | 784 → 32 → 32 → 10  | 26,506 | 97.8%         | MNIST (60k)               |
| Letters | 784 → 64 → 32 → 26  | 53,178 | 90.9%         | EMNIST Letters (124.8k)   |

Digits and letters use separate models. A single 36-class model has to tell apart near-identical shapes (0/O, 1/I/L, 5/S, 2/Z), which costs accuracy. Separate models avoid that and keep the output layer small enough to read. Switch between them with the toggle at the top of the page.

## How it works

- **Training** (`training/`): pure NumPy, no ML frameworks. Forward and backward passes are written as matrix products, with softmax + cross-entropy, Adam, cosine learning-rate decay, and random affine and stroke-thickness augmentation so mouse drawings are recognized.
- **Preprocessing**: the drawing is cropped to its bounding box, scaled to 20px, and centered by center of mass in a 28×28 grid, the same way MNIST was made. `web/src/preprocess.ts` is an exact port of `training/data.py`, and the tests check that the two match.
- **Inference** (`web/src/nn.ts`): about 27k multiply-adds per frame, computed in the browser on every pen movement.
- **Visualizer** (`web/src/viz.ts`): only the strongest terms `wᵢⱼ·aᵢ` of each dot product are drawn bright. All other connections form a faint mesh, so 32- and 64-neuron layers stay readable.

## Run it

```bash
cd web
npm install
npm run dev        # http://localhost:5173
npm test           # JS vs Python parity tests
npm run build      # static site in web/dist
```

Deep links: `?model=letters&sample=3&focus=1:5` (focus = `layer:neuron`).

## Retrain the models

```bash
python3 -m venv .venv && .venv/bin/pip install -r training/requirements.txt
# download into training/data/:
#   MNIST:  https://ossci-datasets.s3.amazonaws.com/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz  (saved as mnist-<name>.gz)
#   EMNIST: https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip  (extract emnist-letters-*.gz)
.venv/bin/python training/train.py --dataset digits  --epochs 40
.venv/bin/python training/train.py --dataset letters --epochs 40
.venv/bin/python training/make_fixtures.py   # refresh preprocessing test fixtures
```

The weights are written to `web/public/models/{digits,letters}.json`.
