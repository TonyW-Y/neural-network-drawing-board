// Network diagram renderer (canvas 2D).
// Design rules that keep 32/64-neuron layers readable:
//  * every connection is drawn once into a cached, near-invisible "mesh" texture
//  * only the strongest live terms w_ji * a_i of each dot product are drawn bright
//  * nodes are dots whose size/glow encodes activation; values shown only for the top few
//  * hovering a neuron dims everything except its own inputs and outputs

import type { Activations, Model } from "./nn";

export type Focus = { layer: number; index: number } | null; // layer 1..3 (h1, h2, output)
export type ValueMode = "a" | "z";

type RGB = [number, number, number];
export const COLORS = {
  in: [103, 232, 249] as RGB,
  h1: [167, 139, 250] as RGB,
  h2: [232, 121, 249] as RGB,
  out: [251, 113, 133] as RGB,
  gold: [251, 191, 36] as RGB,
  neg: [96, 165, 250] as RGB,
};
export const LAYER_RGB: RGB[] = [COLORS.in, COLORS.h1, COLORS.h2, COLORS.out];
const rgba = (c: RGB, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

const SANS = '"Inter", system-ui, sans-serif';
const MONO = '"JetBrains Mono", ui-monospace, monospace';
const SUB = ["₀", "₁", "₂", "₃"];

interface Edge {
  from: number;
  to: number;
  s: number; // strength 0..1
  pos: boolean; // sign of the term w * a
  seed: number;
}

interface Column {
  x: number;
  ys: number[];
  r: number;
  pitch: number;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const easeOutBack = (t: number) => {
  const c = 1.6;
  return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2);
};

function bez(x1: number, y1: number, x2: number, y2: number, t: number): [number, number] {
  // cubic with horizontal tangents at both ends
  const mx = (x1 + x2) / 2;
  const u = 1 - t;
  const x = u * u * u * x1 + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * x2;
  const y = u * u * u * y1 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y2;
  return [x, y];
}

function topK(values: Float32Array, k: number, filter?: (i: number) => boolean): number[] {
  const idx: number[] = [];
  for (let i = 0; i < values.length; i++) if (values[i] !== 0 && (!filter || filter(i))) idx.push(i);
  idx.sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));
  return idx.slice(0, k);
}

export class NetworkViz {
  onFocus: (f: Focus) => void = () => {};

  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private W = 0;
  private H = 0;
  private model: Model | null = null;
  private acts: Activations | null = null;
  private mode: ValueMode = "a";
  private disp: Float32Array[] = [];
  private presence = 0; // eased 0..1, whether there is an input
  private edges: Edge[][] = [];
  private img = { x: 0, y: 0, s: 0 };
  private cols: Column[] = [];
  private mesh = document.createElement("canvas");
  private inputCanvas = document.createElement("canvas");
  private templateCanvas = document.createElement("canvas");
  private templateFor = "";
  private sprites = new Map<string, HTMLCanvasElement>();
  private hover: Focus = null;
  private pinned: Focus = null;
  private lastFocus = "";
  private t0 = performance.now();
  private last = performance.now();
  private reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    this.inputCanvas.width = this.inputCanvas.height = 28;
    this.templateCanvas.width = this.templateCanvas.height = 28;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.bindPointer();
    document.fonts?.ready.then(() => this.relayout());
  }

  get focus(): Focus {
    return this.hover ?? this.pinned;
  }

  setModel(model: Model) {
    this.model = model;
    this.disp = model.sizes.map((n) => new Float32Array(n));
    this.acts = null;
    this.edges = [];
    this.hover = this.pinned = null;
    this.emitFocus();
    this.t0 = performance.now();
    this.relayout();
    this.drawInput(null);
  }

  setMode(mode: ValueMode) {
    this.mode = mode;
  }

  setActivations(acts: Activations | null) {
    this.acts = acts;
    this.drawInput(acts ? acts.a[0] : null);
    this.edges = acts ? this.computeEdges(acts) : [];
  }

  // ---------------------------------------------------------------- data
  private computeEdges(acts: Activations): Edge[][] {
    const m = this.model!;
    const out: Edge[][] = [[]];
    const probs = acts.a[3];
    const topClasses = new Set(topK(probs, 3));
    for (let l = 1; l < m.sizes.length; l++) {
      const layer = m.layers[l - 1];
      const prev = acts.a[l - 1];
      const cur = acts.a[l];
      const list: Edge[] = [];
      const terms = new Float32Array(layer.cols);
      const k = l === 1 ? 3 : l === 2 ? 4 : 6;
      for (let j = 0; j < layer.rows; j++) {
        const isOut = l === m.sizes.length - 1;
        if (isOut ? !topClasses.has(j) : cur[j] <= 0) continue;
        const off = j * layer.cols;
        for (let i = 0; i < layer.cols; i++) terms[i] = layer.W[off + i] * prev[i];
        const weight = isOut ? Math.sqrt(probs[j]) : 1;
        for (const i of topK(terms, k)) {
          list.push({ from: i, to: j, s: Math.abs(terms[i]) * weight, pos: terms[i] > 0, seed: ((i * 7919 + j * 104729) % 1000) / 1000 });
        }
      }
      const max = list.reduce((mx, e) => Math.max(mx, e.s), 1e-9);
      list.forEach((e) => (e.s /= max));
      out.push(list);
    }
    return out;
  }

  private target(l: number): Float32Array | null {
    if (!this.acts) return null;
    return this.mode === "z" ? this.acts.z[l - 1] : this.acts.a[l];
  }

  // ---------------------------------------------------------------- layout
  private resize() {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.canvas.getBoundingClientRect();
    this.W = r.width;
    this.H = r.height;
    this.canvas.width = Math.round(r.width * this.dpr);
    this.canvas.height = Math.round(r.height * this.dpr);
    this.relayout();
  }

  private relayout() {
    if (!this.model || this.W === 0) return;
    const { W, H } = this;
    const top = 56;
    const bottom = 40;
    const avail = H - top - bottom;
    const narrow = W < 640;
    const s = Math.max(84, Math.min(W * 0.17, avail * 0.4, 168));
    this.img = { x: 4, y: top + (avail - s) / 2, s };

    const left = this.img.x + s;
    const right = W - (narrow ? 70 : 84);
    const span = right - left;
    const xs = [0, left + span * 0.32, left + span * 0.66, right];
    this.cols = this.model.sizes.map((n, l) => {
      const isOut = l === this.model!.sizes.length - 1;
      const pitch = Math.min(avail / n, isOut ? 46 : 26);
      const y0 = top + (avail - pitch * (n - 1)) / 2;
      const r = isOut ? Math.max(4, Math.min(pitch * 0.3, 10)) : Math.max(2.6, Math.min(pitch * 0.32, 7));
      return { x: xs[l], ys: Array.from({ length: n }, (_, i) => y0 + i * pitch), r, pitch };
    });
    this.buildMesh();
  }

  private buildMesh() {
    const c = this.mesh;
    c.width = this.canvas.width;
    c.height = this.canvas.height;
    const g = c.getContext("2d")!;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.lineWidth = 0.6;
    // input fan: from the image's right edge to every h1 neuron
    const { x, y, s } = this.img;
    const h1 = this.cols[1];
    g.strokeStyle = rgba(COLORS.in, 0.035);
    g.beginPath();
    for (const ny of h1.ys) {
      for (let k = 0; k < 7; k++) {
        const sy = y + s * (0.08 + (0.84 * k) / 6);
        this.bezPath(g, x + s, sy, h1.x, ny);
      }
    }
    g.stroke();
    for (let l = 2; l < this.cols.length; l++) {
      const a = this.cols[l - 1];
      const b = this.cols[l];
      g.strokeStyle = rgba(LAYER_RGB[l], 0.045);
      g.beginPath();
      for (const ya of a.ys) for (const yb of b.ys) this.bezPath(g, a.x, ya, b.x, yb);
      g.stroke();
    }
  }

  private bezPath(g: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
    const mx = (x1 + x2) / 2;
    g.moveTo(x1, y1);
    g.bezierCurveTo(mx, y1, mx, y2, x2, y2);
  }

  private pixelPos(i: number): [number, number] {
    const cell = this.img.s / 28;
    return [this.img.x + ((i % 28) + 0.5) * cell, this.img.y + (Math.floor(i / 28) + 0.5) * cell];
  }

  private nodePos(l: number, j: number): [number, number] {
    if (l === 0) return this.pixelPos(j);
    return [this.cols[l].x, this.cols[l].ys[j]];
  }

  // ---------------------------------------------------------------- images
  private drawInput(x: Float32Array | null) {
    const g = this.inputCanvas.getContext("2d")!;
    const im = g.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = x ? x[i] : 0;
      im.data[i * 4] = 8 + v * 200;
      im.data[i * 4 + 1] = 11 + v * 236;
      im.data[i * 4 + 2] = 20 + v * 232;
      im.data[i * 4 + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }

  private drawTemplate(j: number) {
    const key = `${this.model!.name}:${j}`;
    if (this.templateFor === key) return;
    this.templateFor = key;
    const L = this.model!.layers[0];
    let max = 1e-9;
    for (let i = 0; i < 784; i++) max = Math.max(max, Math.abs(L.W[j * 784 + i]));
    const g = this.templateCanvas.getContext("2d")!;
    const im = g.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const w = L.W[j * 784 + i] / max;
      const c = w >= 0 ? COLORS.out : COLORS.neg;
      const t = Math.min(1, Math.abs(w) * 1.3);
      im.data[i * 4] = 8 + (c[0] - 8) * t;
      im.data[i * 4 + 1] = 11 + (c[1] - 11) * t;
      im.data[i * 4 + 2] = 20 + (c[2] - 20) * t;
      im.data[i * 4 + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }

  private sprite(c: RGB): HTMLCanvasElement {
    const key = c.join(",");
    let s = this.sprites.get(key);
    if (!s) {
      s = document.createElement("canvas");
      s.width = s.height = 64;
      const g = s.getContext("2d")!;
      const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      grad.addColorStop(0, rgba(c, 1));
      grad.addColorStop(0.25, rgba(c, 0.45));
      grad.addColorStop(1, rgba(c, 0));
      g.fillStyle = grad;
      g.fillRect(0, 0, 64, 64);
      this.sprites.set(key, s);
    }
    return s;
  }

  // ---------------------------------------------------------------- interaction
  private bindPointer() {
    const c = this.canvas;
    c.addEventListener("pointermove", (e) => {
      const r = c.getBoundingClientRect();
      this.hover = this.hit(e.clientX - r.left, e.clientY - r.top);
      c.style.cursor = this.hover ? "pointer" : "default";
      this.emitFocus();
    });
    c.addEventListener("pointerleave", () => {
      this.hover = null;
      this.emitFocus();
    });
    c.addEventListener("click", (e) => {
      const r = c.getBoundingClientRect();
      const h = this.hit(e.clientX - r.left, e.clientY - r.top);
      const same = h && this.pinned && h.layer === this.pinned.layer && h.index === this.pinned.index;
      this.pinned = same ? null : h;
      this.emitFocus();
    });
  }

  private hit(mx: number, my: number): Focus {
    for (let l = 1; l < this.cols.length; l++) {
      const c = this.cols[l];
      const reach = Math.max(c.r + 6, 10);
      if (Math.abs(mx - c.x) > reach + (l === this.cols.length - 1 ? 30 : 0)) continue;
      const j = Math.round((my - c.ys[0]) / c.pitch);
      if (j < 0 || j >= c.ys.length) continue;
      if (Math.abs(my - c.ys[j]) <= Math.max(c.pitch / 2, c.r + 2)) return { layer: l, index: j };
    }
    return null;
  }

  private emitFocus() {
    const f = this.focus;
    const key = f ? `${f.layer}:${f.index}` : "";
    if (key === this.lastFocus) return;
    this.lastFocus = key;
    this.onFocus(f);
  }

  /** Incoming terms + outgoing weights for the focused neuron. */
  private focusEdges(f: NonNullable<Focus>): { incoming: Edge[]; outgoing: Edge[] } {
    const m = this.model!;
    const acts = this.acts;
    const layer = m.layers[f.layer - 1];
    const prev = acts ? acts.a[f.layer - 1] : null;
    const terms = new Float32Array(layer.cols);
    for (let i = 0; i < layer.cols; i++) {
      const w = layer.W[f.index * layer.cols + i];
      terms[i] = prev ? w * prev[i] : f.layer === 1 ? 0 : w;
    }
    const k = f.layer === 1 ? 48 : 16;
    const inc = topK(terms, k);
    const maxIn = inc.length ? Math.abs(terms[inc[0]]) : 1;
    const incoming = inc.map((i) => ({ from: i, to: f.index, s: Math.abs(terms[i]) / maxIn, pos: terms[i] > 0, seed: (i % 97) / 97 }));

    const outgoing: Edge[] = [];
    if (f.layer < m.layers.length) {
      const next = m.layers[f.layer];
      const a = acts ? acts.a[f.layer][f.index] : 1;
      let max = 1e-9;
      const vals = Array.from({ length: next.rows }, (_, k2) => next.W[k2 * next.cols + f.index] * (a || 1));
      vals.forEach((v) => (max = Math.max(max, Math.abs(v))));
      vals.forEach((v, k2) => outgoing.push({ from: f.index, to: k2, s: Math.abs(v) / max, pos: v > 0, seed: (k2 % 13) / 13 }));
    }
    return { incoming, outgoing };
  }

  // ---------------------------------------------------------------- render
  frame(now: number) {
    if (!this.model || this.cols.length === 0) return;
    const dt = Math.min(64, now - this.last);
    this.last = now;
    const k = this.reduced ? 1 : 1 - Math.exp(-dt / 90);

    this.presence += ((this.acts ? 1 : 0) - this.presence) * k;
    for (let l = 1; l < this.disp.length; l++) {
      const t = this.target(l);
      const d = this.disp[l];
      for (let j = 0; j < d.length; j++) d[j] += ((t ? t[j] : 0) - d[j]) * k;
    }

    const g = this.ctx;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.W, this.H);

    const intro = clamp01((now - this.t0) / 700);
    const focus = this.focus;
    const fe = focus ? this.focusEdges(focus) : null;

    // mesh texture
    g.globalAlpha = intro * (focus ? 0.45 : 1);
    g.drawImage(this.mesh, 0, 0, this.W, this.H);
    g.globalAlpha = 1;

    this.drawHeaders(g, focus);
    this.drawImageBox(g, focus);

    // live edges + pulses
    g.globalCompositeOperation = "lighter";
    const t = now / 1000;
    for (let l = 1; l < this.edges.length; l++) {
      this.drawEdges(g, l, this.edges[l], (focus ? 0.18 : 1) * (l === 1 ? 0.7 : 1), t, !focus);
    }
    if (focus && fe) {
      this.drawEdges(g, focus.layer, fe.incoming, 1, t, true);
      if (focus.layer + 1 < this.cols.length) {
        const out = fe.outgoing.map((e) => ({ ...e, from: e.from }));
        this.drawEdges(g, focus.layer + 1, out, 0.8, t, false);
      }
    }
    g.globalCompositeOperation = "source-over";

    this.drawNodes(g, now, focus, fe);
    this.drawFooter(g);

    if (this.presence < 0.5 && !focus) {
      g.font = `500 13px ${SANS}`;
      g.fillStyle = `rgba(139,147,167,${0.9 * (1 - this.presence * 2) * intro})`;
      g.textAlign = "center";
      g.fillText("Draw on the board — the signal flows left → right", (this.cols[1].x + this.cols[2].x) / 2, this.H - 14);
    }
  }

  private drawEdges(g: CanvasRenderingContext2D, l: number, edges: Edge[], alpha: number, t: number, pulses: boolean) {
    const pres = this.presence;
    for (const e of edges) {
      const [x1, y1] = this.nodePos(l - 1, e.from);
      const [x2, y2] = this.nodePos(l, e.to);
      const c = e.pos ? LAYER_RGB[l] : COLORS.neg;
      const a = (0.08 + 0.62 * e.s) * alpha * pres;
      if (a < 0.01) continue;
      g.strokeStyle = rgba(c, a);
      g.lineWidth = 0.6 + 1.6 * e.s;
      g.beginPath();
      this.bezPath(g, x1, y1, x2, y2);
      g.stroke();
      if (pulses && !this.reduced && e.s > 0.18) {
        const p = (t * 0.55 + e.seed + l * 0.33) % 1;
        const [px, py] = bez(x1, y1, x2, y2, p);
        const R = 3 + 6 * e.s;
        g.globalAlpha = alpha * pres * (0.35 + 0.65 * e.s) * Math.sin(Math.PI * p);
        g.drawImage(this.sprite(c), px - R, py - R, R * 2, R * 2);
        g.globalAlpha = 1;
      }
    }
  }

  private drawHeaders(g: CanvasRenderingContext2D, focus: Focus) {
    const m = this.model!;
    const L = m.sizes.length - 1;
    const names = ["Input", "Hidden", "Hidden", "Output"];
    const syms = ["x", "h₁", "h₂", "ŷ"];
    const fns = ["28 × 28 pixels", "ReLU", "ReLU", "softmax"];
    for (let l = 0; l <= L; l++) {
      const x = l === 0 ? this.img.x + this.img.s / 2 : this.cols[l].x;
      const on = focus?.layer === l;
      g.textAlign = "center";
      g.font = `600 12.5px ${SANS}`;
      g.fillStyle = on ? "#fff" : "rgba(232,235,244,0.85)";
      const head = `${names[l]}  `;
      const symW = (g.measureText(syms[l]).width);
      const headW = g.measureText(head).width;
      const x0 = x - (headW + symW) / 2;
      g.textAlign = "left";
      g.fillText(head, x0, 18);
      g.fillStyle = rgba(LAYER_RGB[l], 1);
      g.fillText(syms[l], x0 + headW, 18);
      g.textAlign = "center";
      g.font = `400 11px ${MONO}`;
      g.fillStyle = "rgba(139,147,167,0.85)";
      g.fillText(`ℝ${sup(m.sizes[l])} · ${fns[l]}`, x, 35);
    }
  }

  private drawFooter(g: CanvasRenderingContext2D) {
    const m = this.model!;
    g.font = `400 11px ${MONO}`;
    g.textAlign = "center";
    for (let l = 1; l < m.sizes.length; l++) {
      const xa = l === 1 ? this.img.x + this.img.s : this.cols[l - 1].x;
      const x = (xa + this.cols[l].x) / 2;
      const y = this.H - 34;
      g.fillStyle = "rgba(139,147,167,0.75)";
      g.fillText(`W${SUB[l]} · ${m.sizes[l]}×${m.sizes[l - 1]}`, x, y);
    }
  }

  private drawImageBox(g: CanvasRenderingContext2D, focus: Focus) {
    const { x, y, s } = this.img;
    g.save();
    roundRect(g, x, y, s, s, 10);
    g.clip();
    g.imageSmoothingEnabled = false;
    if (focus?.layer === 1) {
      // show the neuron's weight template, with the drawing on top of it
      this.drawTemplate(focus.index);
      g.drawImage(this.templateCanvas, x, y, s, s);
      g.globalCompositeOperation = "lighter";
      g.globalAlpha = 0.55;
      g.drawImage(this.inputCanvas, x, y, s, s);
      g.globalAlpha = 1;
      g.globalCompositeOperation = "source-over";
    } else {
      g.drawImage(this.inputCanvas, x, y, s, s);
    }
    // pixel grid
    g.strokeStyle = "rgba(255,255,255,0.045)";
    g.lineWidth = 1;
    g.beginPath();
    const cell = s / 28;
    for (let i = 1; i < 28; i++) {
      g.moveTo(x + i * cell, y);
      g.lineTo(x + i * cell, y + s);
      g.moveTo(x, y + i * cell);
      g.lineTo(x + s, y + i * cell);
    }
    g.stroke();
    g.restore();
    roundRect(g, x + 0.5, y + 0.5, s - 1, s - 1, 10);
    g.strokeStyle = focus?.layer === 1 ? "rgba(251,113,133,0.5)" : "rgba(103,232,249,0.28)";
    g.lineWidth = 1;
    g.stroke();
    if (focus?.layer === 1) {
      g.font = `400 10.5px ${MONO}`;
      g.textAlign = "center";
      g.fillStyle = "rgba(232,235,244,0.75)";
      g.fillText(`weights of h₁[${focus.index}]`, x + s / 2, y + s + 16);
    }
  }

  private drawNodes(g: CanvasRenderingContext2D, now: number, focus: Focus, fe: { incoming: Edge[]; outgoing: Edge[] } | null) {
    const m = this.model!;
    const L = m.sizes.length - 1;
    const probs = this.acts ? this.acts.a[L] : null;
    let best = -1;
    if (probs) best = probs.indexOf(Math.max(...probs));

    const related = new Set<string>();
    if (focus && fe) {
      related.add(`${focus.layer}:${focus.index}`);
      if (focus.layer > 1) fe.incoming.forEach((e) => related.add(`${focus.layer - 1}:${e.from}`));
      fe.outgoing.forEach((e) => related.add(`${focus.layer + 1}:${e.to}`));
    }

    for (let l = 1; l <= L; l++) {
      const col = this.cols[l];
      const d = this.disp[l];
      const isOut = l === L;
      let norm = isOut && this.mode === "a" ? 1 : 1e-6;
      if (norm !== 1) for (const v of d) norm = Math.max(norm, Math.abs(v));
      if (!isOut) norm = Math.max(norm, 0.5);

      const labelSet = new Set(isOut ? [] : topK(d, 5, (i) => Math.abs(d[i]) / norm > 0.12));

      for (let j = 0; j < d.length; j++) {
        const v = d[j];
        const n = clamp01(Math.abs(v) / norm);
        const c = v >= 0 ? LAYER_RGB[l] : COLORS.neg;
        const [x, y] = [col.x, col.ys[j]];
        const p = clamp01((now - this.t0 - (l - 1) * 110 - j * (220 / d.length)) / 380);
        const sc = this.reduced ? 1 : easeOutBack(p);
        const dim = focus && !related.has(`${l}:${j}`) ? 0.3 : 1;
        const r = col.r * (0.85 + 0.35 * n) * sc;
        if (r <= 0.1) continue;

        g.globalAlpha = dim;
        if (n > 0.04) {
          const R = r * (2.2 + 3.5 * n);
          g.globalAlpha = dim * (0.25 + 0.6 * n);
          g.globalCompositeOperation = "lighter";
          g.drawImage(this.sprite(c), x - R, y - R, R * 2, R * 2);
          g.globalCompositeOperation = "source-over";
          g.globalAlpha = dim;
        }
        g.beginPath();
        g.arc(x, y, r, 0, Math.PI * 2);
        g.fillStyle = "#0a0d17";
        g.fill();
        g.fillStyle = rgba(c, 0.1 + 0.9 * n);
        g.fill();
        g.lineWidth = 1;
        g.strokeStyle = rgba(c, 0.35 + 0.55 * n);
        g.stroke();

        if (isOut && j === best && this.presence > 0.3) {
          g.beginPath();
          g.arc(x, y, r + 4.5, 0, Math.PI * 2);
          g.strokeStyle = rgba(COLORS.gold, 0.9 * this.presence);
          g.lineWidth = 1.5;
          g.stroke();
        }
        if (focus && focus.layer === l && focus.index === j) {
          g.beginPath();
          g.arc(x, y, r + 5, 0, Math.PI * 2);
          g.strokeStyle = "rgba(255,255,255,0.9)";
          g.lineWidth = 1.5;
          g.stroke();
        }

        if (isOut) this.drawOutLabel(g, x + col.r + 12, y, m.labels[j], v, n, j === best && this.presence > 0.3, dim);
        else if (labelSet.has(j) || (focus?.layer === l && focus.index === j)) this.drawValue(g, x + r + 7, y, v, dim);
        g.globalAlpha = 1;
      }
    }
  }

  private drawOutLabel(g: CanvasRenderingContext2D, x: number, y: number, label: string, v: number, n: number, best: boolean, dim: number) {
    g.globalAlpha = dim;
    g.textAlign = "left";
    g.textBaseline = "middle";
    const big = this.cols[this.cols.length - 1].pitch > 20;
    g.font = `${best ? 700 : 600} ${big ? 14 : 12}px ${SANS}`;
    g.fillStyle = best ? rgba(COLORS.gold, 1) : `rgba(232,235,244,${0.45 + 0.55 * n})`;
    g.fillText(label, x, y);
    if (big || best || n > 0.15) {
      g.font = `400 ${big ? 11 : 10}px ${MONO}`;
      g.fillStyle = best ? "rgba(253,230,138,0.95)" : `rgba(139,147,167,${0.6 + 0.4 * n})`;
      g.fillText(v.toFixed(3), x + (big ? 16 : 13), y);
    }
    g.textBaseline = "alphabetic";
  }

  private drawValue(g: CanvasRenderingContext2D, x: number, y: number, v: number, dim: number) {
    const text = v.toFixed(3);
    g.font = `500 10px ${MONO}`;
    const w = g.measureText(text).width + 8;
    g.globalAlpha = dim;
    roundRect(g, x, y - 7.5, w, 15, 4);
    g.fillStyle = "rgba(5,7,12,0.82)";
    g.fill();
    g.strokeStyle = "rgba(255,255,255,0.08)";
    g.lineWidth = 1;
    g.stroke();
    g.fillStyle = v < 0 ? rgba(COLORS.neg, 1) : "rgba(232,235,244,0.92)";
    g.textAlign = "left";
    g.textBaseline = "middle";
    g.fillText(text, x + 4, y + 0.5);
    g.textBaseline = "alphabetic";
  }
}

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath();
  g.roundRect(x, y, w, h, r);
}

const SUP: Record<string, string> = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
export const sup = (n: number) => String(n).replace(/\d/g, (d) => SUP[d]);
