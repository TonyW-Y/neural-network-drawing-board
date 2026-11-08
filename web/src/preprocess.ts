// Mirrors training/data.py `normalize()` exactly, so the drawing is converted
// the same way the training images were:
//   bbox crop -> scale longest side to 20px -> center in 28x28 -> center of mass to middle

const BBOX_THRESHOLD = 0.1;
const BOX = 20;
const SIZE = 28;

/** Python's round(): halves round to the nearest even integer. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Sparse row of the box-filter resampling matrix (see area_matrix in data.py). */
interface AreaRow {
  start: number;
  w: number[];
}

export function areaMatrix(nOut: number, nIn: number): AreaRow[] {
  const scale = nIn / nOut;
  const rows: AreaRow[] = [];
  for (let i = 0; i < nOut; i++) {
    const lo = i * scale;
    const hi = (i + 1) * scale;
    const start = Math.floor(lo);
    const end = Math.min(nIn, Math.ceil(hi));
    const w: number[] = [];
    let sum = 0;
    for (let j = start; j < end; j++) {
      const overlap = Math.max(0, Math.min(hi, j + 1) - Math.max(lo, j));
      w.push(overlap);
      sum += overlap;
    }
    rows.push({ start, w: w.map((v) => v / sum) });
  }
  return rows;
}

/**
 * @param img grayscale values in [0, 1], row-major, size width*height
 * @returns 784 values (28x28), or null if nothing is drawn
 */
export function normalize(img: Float32Array, width: number, height: number): Float32Array | null {
  let r0 = height, r1 = -1, c0 = width, c1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (img[y * width + x] > BBOX_THRESHOLD) {
        if (y < r0) r0 = y;
        if (y > r1) r1 = y;
        if (x < c0) c0 = x;
        if (x > c1) c1 = x;
      }
    }
  }
  if (r1 < 0) return null;

  const h = r1 - r0 + 1;
  const w = c1 - c0 + 1;
  const s = BOX / Math.max(h, w);
  const nh = Math.max(1, roundHalfEven(h * s));
  const nw = Math.max(1, roundHalfEven(w * s));
  const Ry = areaMatrix(nh, h);
  const Rx = areaMatrix(nw, w);

  // tmp = Ry · crop   (nh x w)
  const tmp = new Float64Array(nh * w);
  for (let i = 0; i < nh; i++) {
    const { start, w: wy } = Ry[i];
    for (let k = 0; k < wy.length; k++) {
      const src = (r0 + start + k) * width + c0;
      const f = wy[k];
      for (let x = 0; x < w; x++) tmp[i * w + x] += f * img[src + x];
    }
  }

  // small = tmp · Rx^T, pasted centered into 28x28
  const out = new Float32Array(SIZE * SIZE);
  const top = Math.floor((SIZE - nh) / 2);
  const left = Math.floor((SIZE - nw) / 2);
  for (let i = 0; i < nh; i++) {
    for (let j = 0; j < nw; j++) {
      const { start, w: wx } = Rx[j];
      let v = 0;
      for (let k = 0; k < wx.length; k++) v += tmp[i * w + start + k] * wx[k];
      out[(top + i) * SIZE + left + j] = v;
    }
  }

  let total = 0, sy = 0, sx = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = out[y * SIZE + x];
      total += v;
      sy += y * v;
      sx += x * v;
    }
  }
  const dy = roundHalfEven(13.5 - sy / total);
  const dx = roundHalfEven(13.5 - sx / total);
  return shift(out, dy, dx);
}

function shift(img: Float32Array, dy: number, dx: number): Float32Array {
  const out = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    const ty = y + dy;
    if (ty < 0 || ty >= SIZE) continue;
    for (let x = 0; x < SIZE; x++) {
      const tx = x + dx;
      if (tx < 0 || tx >= SIZE) continue;
      out[ty * SIZE + tx] = img[y * SIZE + x];
    }
  }
  return out;
}
