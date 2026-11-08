// The whole neural network, as linear algebra.
//   z_l = W_l · a_{l-1} + b_l          (matrix-vector product)
//   a_l = ReLU(z_l)  for hidden layers, softmax(z_l) for the output layer

export interface Layer {
  rows: number; // neurons in this layer
  cols: number; // inputs from the previous layer
  W: Float32Array; // row-major (rows x cols): row j = weights of neuron j
  b: Float32Array;
}

export interface Sample {
  label: number;
  pixels: Float32Array; // 784 values in [0, 1]
  logits: number[];
}

export interface ModelMeta {
  params: number;
  testAccuracy: number;
  epochs: number;
  trainSize: number;
  loss: string;
  optimizer: string;
  framework: string;
}

export interface Model {
  name: "digits" | "letters";
  labels: string[];
  sizes: number[];
  layers: Layer[];
  meta: ModelMeta;
  samples: Sample[];
}

export interface Activations {
  /** a[0] = input x, a[l] = output of layer l (post-ReLU / softmax) */
  a: Float32Array[];
  /** z[l] = pre-activation of layer l+1 (z[0] belongs to hidden layer 1) */
  z: Float32Array[];
}

interface RawModel {
  name: Model["name"];
  labels: string[];
  sizes: number[];
  layers: { shape: [number, number]; W: number[]; b: number[] }[];
  meta: ModelMeta;
  samples: { label: number; pixels: string; logits: number[] }[];
}

export function parseModel(raw: RawModel): Model {
  return {
    name: raw.name,
    labels: raw.labels,
    sizes: raw.sizes,
    meta: raw.meta,
    layers: raw.layers.map((l) => ({
      rows: l.shape[0],
      cols: l.shape[1],
      W: Float32Array.from(l.W),
      b: Float32Array.from(l.b),
    })),
    samples: raw.samples.map((s) => {
      const bytes = Uint8Array.from(atob(s.pixels), (c) => c.charCodeAt(0));
      return { label: s.label, logits: s.logits, pixels: Float32Array.from(bytes, (v) => v / 255) };
    }),
  };
}

export async function loadModel(name: Model["name"]): Promise<Model> {
  const res = await fetch(`${import.meta.env.BASE_URL}models/${name}.json`);
  if (!res.ok) throw new Error(`failed to load model ${name}: ${res.status}`);
  return parseModel(await res.json());
}

/** z = W x + b */
export function matVec(layer: Layer, x: Float32Array): Float32Array {
  const { rows, cols, W, b } = layer;
  const z = new Float32Array(rows);
  for (let j = 0; j < rows; j++) {
    let sum = 0;
    const off = j * cols;
    for (let i = 0; i < cols; i++) sum += W[off + i] * x[i];
    z[j] = sum + b[j];
  }
  return z;
}

export const relu = (z: Float32Array) => z.map((v) => (v > 0 ? v : 0));

export function softmax(z: Float32Array): Float32Array {
  let max = -Infinity;
  for (const v of z) max = Math.max(max, v);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / sum);
}

export function forward(model: Model, x: Float32Array): Activations {
  const a = [x];
  const z: Float32Array[] = [];
  model.layers.forEach((layer, l) => {
    const zl = matVec(layer, a[l]);
    z.push(zl);
    a.push(l < model.layers.length - 1 ? relu(zl) : softmax(zl));
  });
  return { a, z };
}

/** The individual terms w_ji * x_i of neuron j's dot product. */
export function contributions(layer: Layer, j: number, x: Float32Array): Float32Array {
  const out = new Float32Array(layer.cols);
  const off = j * layer.cols;
  for (let i = 0; i < layer.cols; i++) out[i] = layer.W[off + i] * x[i];
  return out;
}
