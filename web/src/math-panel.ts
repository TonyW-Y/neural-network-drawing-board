// "The linear algebra" panel: the three layer equations with live numbers, and a
// breakdown of one neuron's value into the dot product that produced it.

import type { Activations, Model } from "./nn";
import { COLORS, type Focus, LAYER_RGB } from "./viz";

type RGB = [number, number, number];
const SUB = ["₀", "₁", "₂", "₃"];
const SYM = ["x", "h₁", "h₂", "ŷ"];
const CSS_ACCENT = ["var(--c-in)", "var(--c-h1)", "var(--c-h2)", "var(--c-out)"];
const BG: RGB = [8, 11, 20];

const fmt = (v: number, d = 3) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);

export class MathPanel {
  private eqEl = document.getElementById("equations")!;
  private detailEl = document.getElementById("detail")!;
  private key = "";

  setModel(model: Model) {
    const s = model.sizes;
    const rows = [1, 2, 3].map((l) => {
      const prev = SYM[l - 1];
      const fn = l === 3 ? "softmax" : "ReLU";
      return `
        <div class="eq" data-l="${l}" style="--accent:${CSS_ACCENT[l]}">
          <div class="f"><em>${SYM[l]}</em> = ${fn}( W${SUB[l]} · ${prev} + b${SUB[l]} )</div>
          <div class="s">(${s[l]}×${s[l - 1]}) · (${s[l - 1]}) + (${s[l]}) → ${s[l]} values</div>
          <div class="live" data-live="${l}">—</div>
        </div>`;
    });
    this.eqEl.innerHTML = rows.join("");
    this.key = "";
  }

  update(model: Model, acts: Activations | null, focus: Focus) {
    // equations
    this.eqEl.querySelectorAll<HTMLElement>(".eq").forEach((el) => {
      el.classList.toggle("on", focus?.layer === Number(el.dataset.l));
    });
    for (let l = 1; l <= 3; l++) {
      const live = this.eqEl.querySelector<HTMLElement>(`[data-live="${l}"]`)!;
      if (!acts) {
        live.textContent = "waiting for input…";
        continue;
      }
      const a = acts.a[l];
      if (l < 3) {
        const on = a.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);
        live.innerHTML = `active neurons <b>${on}</b> / ${a.length} · ReLU zeroed <b>${a.length - on}</b>`;
      } else {
        const k = a.indexOf(Math.max(...a));
        live.innerHTML = `argmax ŷ = <b>${model.labels[k]}</b> at <b>${(a[k] * 100).toFixed(1)}%</b>`;
      }
    }

    // detail
    if (!acts) {
      this.key = "";
      this.detailEl.innerHTML = `<div class="empty">Draw something, then hover any neuron in the diagram. This panel breaks that neuron's value down into the dot product that produced it: one row of the weight matrix times the previous layer's vector.</div>`;
      return;
    }
    const probs = acts.a[3];
    const f = focus ?? { layer: 3, index: probs.indexOf(Math.max(...probs)) };
    const key = `${model.name}:${f.layer}:${f.index}`;
    if (key !== this.key) {
      this.key = key;
      this.detailEl.style.setProperty("--accent", CSS_ACCENT[f.layer]);
      this.detailEl.innerHTML = f.layer === 1 ? this.layoutL1(f.index) : this.layoutMatrix(model, f.layer, f.index);
    }
    if (f.layer === 1) this.fillL1(model, acts, f.index);
    else this.fillMatrix(model, acts, f.layer, f.index);
  }

  // ---------------------------------------------------------------- layer 1: 28x28 templates
  private layoutL1(j: number) {
    return `
      <div class="detail-title">Neuron <b>h₁[${j}]</b>: row ${j} of W₁ dotted with the 784 input pixels &nbsp;<code>z = w · x + b</code></div>
      <div class="detail-body">
        <figure class="fig"><canvas data-c="w"></canvas><figcaption>w = W₁[${j}, :]</figcaption></figure>
        <span class="op">⊙</span>
        <figure class="fig"><canvas data-c="x"></canvas><figcaption>x (what the net sees)</figcaption></figure>
        <span class="op">=</span>
        <figure class="fig"><canvas data-c="wx"></canvas><figcaption>w ⊙ x (each term)</figcaption></figure>
        <div class="calc" data-calc></div>
      </div>`;
  }

  private fillL1(model: Model, acts: Activations, j: number) {
    const L = model.layers[0];
    const w = L.W.subarray(j * 784, (j + 1) * 784);
    const x = acts.a[0];
    const wx = w.map((v, i) => v * x[i]);
    const S = 112;
    const get = (k: string) => this.detailEl.querySelector<HTMLCanvasElement>(`[data-c="${k}"]`)!;
    heat(get("w"), w, 28, 28, S / 28, { diverging: true });
    heat(get("x"), x, 28, 28, S / 28, { color: COLORS.in, max: 1 });
    heat(get("wx"), wx, 28, 28, S / 28, { diverging: true });
    const sum = wx.reduce((s, v) => s + v, 0);
    const z = acts.z[0][j];
    this.detailEl.querySelector("[data-calc]")!.innerHTML = `
      <div>Σ wᵢxᵢ (784 terms) = <b>${fmt(sum)}</b></div>
      <div>+ bias b₁[${j}] = <b>${fmt(L.b[j])}</b></div>
      <div>z = <b>${fmt(z)}</b></div>
      <div>a = ReLU(z) = max(0, z) = <b class="res">${fmt(acts.a[1][j])}</b></div>
      <div class="legend"><span>−</span><i></i><span>+</span> weight / term sign</div>`;
  }

  // ---------------------------------------------------------------- layers 2, 3: matrix · vector
  private layoutMatrix(model: Model, l: number, j: number) {
    const name = `${SYM[l]}[${l === 3 ? model.labels[j] : j}]`;
    const what = l === 3 ? `the score for “${model.labels[j]}”` : `neuron ${name}`;
    return `
      <div class="detail-title">${l === 3 ? "Output" : "Neuron"} <b>${name}</b>: ${what} is row ${j} of W${SUB[l]} times ${SYM[l - 1]} &nbsp;<code>z = W${SUB[l]}[${j}, :] · ${SYM[l - 1]} + b${SUB[l]}[${j}]</code></div>
      <div class="detail-body">
        <figure class="fig"><canvas data-c="W"></canvas><figcaption>W${SUB[l]}  (${model.sizes[l]}×${model.sizes[l - 1]})</figcaption></figure>
        <span class="op">·</span>
        <figure class="fig"><canvas data-c="a"></canvas><figcaption>${SYM[l - 1]}</figcaption></figure>
        <span class="op">+</span>
        <figure class="fig"><canvas data-c="b"></canvas><figcaption>b${SUB[l]}</figcaption></figure>
        <span class="op">=</span>
        <figure class="fig"><canvas data-c="z"></canvas><figcaption>z</figcaption></figure>
        <figure class="fig"><canvas data-c="terms"></canvas><figcaption>row ${j} ⊙ ${SYM[l - 1]}: the ${model.sizes[l - 1]} terms being summed</figcaption></figure>
        <div class="calc" data-calc></div>
      </div>`;
  }

  private fillMatrix(model: Model, acts: Activations, l: number, j: number) {
    const L = model.layers[l - 1];
    const prev = acts.a[l - 1];
    const z = acts.z[l - 1];
    const cell = Math.max(3, Math.min(9, Math.floor(200 / Math.max(L.rows, L.cols))));
    const get = (k: string) => this.detailEl.querySelector<HTMLCanvasElement>(`[data-c="${k}"]`)!;
    heat(get("W"), L.W, L.rows, L.cols, cell, { diverging: true, highlightRow: j });
    heat(get("a"), prev, L.cols, 1, cell, { color: LAYER_RGB[l - 1], width: 12 });
    heat(get("b"), L.b, L.rows, 1, cell, { diverging: true, width: 12, highlightRow: j });
    heat(get("z"), z, L.rows, 1, cell, { diverging: true, width: 12, highlightRow: j });
    const terms = new Float32Array(L.cols);
    for (let i = 0; i < L.cols; i++) terms[i] = L.W[j * L.cols + i] * prev[i];
    bars(get("terms"), terms, LAYER_RGB[l]);

    const sum = terms.reduce((s, v) => s + v, 0);
    let last: string;
    if (l === 3) {
      const ez = z.map((v) => Math.exp(v));
      const total = ez.reduce((s, v) => s + v, 0);
      last = `<div>softmax: e<sup>z</sup> / Σₖ e<sup>zₖ</sup> = ${fmt(ez[j], 2)} / ${fmt(total, 2)}</div>
              <div>p(“${model.labels[j]}”) = <b class="res">${(acts.a[3][j] * 100).toFixed(2)}%</b></div>`;
    } else {
      last = `<div>a = ReLU(z) = <b class="res">${fmt(acts.a[l][j])}</b></div>`;
    }
    this.detailEl.querySelector("[data-calc]")!.innerHTML = `
      <div>Σ wᵢ·${SYM[l - 1]}ᵢ (${L.cols} terms) = <b>${fmt(sum)}</b></div>
      <div>+ bias = <b>${fmt(L.b[j])}</b></div>
      <div>z = <b>${fmt(z[j])}</b></div>
      ${last}
      <div class="legend"><span>−</span><i></i><span>+</span></div>`;
  }
}

// ---------------------------------------------------------------- canvas helpers
interface HeatOpts {
  diverging?: boolean;
  color?: RGB;
  max?: number;
  highlightRow?: number;
  width?: number; // fixed css width (for column vectors)
}

function setup(canvas: HTMLCanvasElement, w: number, h: number) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }
  const g = canvas.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}

function mix(c: RGB, t: number): string {
  t = Math.min(1, Math.max(0, t));
  return `rgb(${BG[0] + (c[0] - BG[0]) * t},${BG[1] + (c[1] - BG[1]) * t},${BG[2] + (c[2] - BG[2]) * t})`;
}

function heat(canvas: HTMLCanvasElement, v: Float32Array, rows: number, cols: number, cell: number, o: HeatOpts) {
  const cw = o.width ? o.width / cols : cell;
  const w = cols * cw;
  const h = rows * cell;
  const g = setup(canvas, w, h);
  let max = o.max ?? 0;
  if (!max) for (const x of v) max = Math.max(max, Math.abs(x));
  max = max || 1;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = v[r * cols + c] / max;
      const col = o.diverging ? (x >= 0 ? COLORS.out : COLORS.neg) : (o.color ?? COLORS.in);
      // sqrt boosts small values so structure stays visible
      g.fillStyle = mix(col, Math.sqrt(Math.abs(x)));
      g.fillRect(c * cw, r * cell, cw, cell);
    }
  }
  if (o.highlightRow !== undefined) {
    g.fillStyle = "rgba(5,7,12,0.45)";
    g.fillRect(0, 0, w, o.highlightRow * cell);
    g.fillRect(0, (o.highlightRow + 1) * cell, w, h - (o.highlightRow + 1) * cell);
    g.strokeStyle = "rgba(255,255,255,0.95)";
    g.lineWidth = 1.5;
    g.strokeRect(0.75, o.highlightRow * cell + 0.75, w - 1.5, cell - 1.5);
  }
}

function bars(canvas: HTMLCanvasElement, v: Float32Array, pos: RGB) {
  const w = Math.max(200, Math.min(320, v.length * 6));
  const h = 96;
  const g = setup(canvas, w, h);
  g.clearRect(0, 0, w, h);
  let max = 1e-9;
  for (const x of v) max = Math.max(max, Math.abs(x));
  const bw = w / v.length;
  const mid = h / 2;
  g.strokeStyle = "rgba(255,255,255,0.12)";
  g.beginPath();
  g.moveTo(0, mid + 0.5);
  g.lineTo(w, mid + 0.5);
  g.stroke();
  for (let i = 0; i < v.length; i++) {
    const t = v[i] / max;
    const bh = Math.abs(t) * (mid - 4);
    g.fillStyle = mix(t >= 0 ? pos : COLORS.neg, 0.35 + 0.65 * Math.abs(t));
    g.fillRect(i * bw + bw * 0.15, t >= 0 ? mid - bh : mid, bw * 0.7, Math.max(bh, 0.5));
  }
}
