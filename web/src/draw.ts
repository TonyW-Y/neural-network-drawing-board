// Drawing board: a glowing display canvas + a hidden 280x280 grayscale "ink"
// canvas that the network actually reads (white strokes on black, like MNIST).

export const INK_SIZE = 280;

type Pt = { x: number; y: number }; // normalized 0..1

export class DrawingBoard {
  brush = 20; // stroke width in ink pixels (of 280)
  onChange: () => void = () => {};
  onStart: () => void = () => {};

  private dctx: CanvasRenderingContext2D;
  private ink = document.createElement("canvas");
  private ictx: CanvasRenderingContext2D;
  private strokes: Pt[][] = [];
  private drawing = false;
  private sample: Float32Array | null = null;
  private renderQueued = false;

  constructor(private canvas: HTMLCanvasElement) {
    this.dctx = canvas.getContext("2d")!;
    this.ink.width = this.ink.height = INK_SIZE;
    this.ictx = this.ink.getContext("2d", { willReadFrequently: true })!;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.bindPointer();
    this.resize();
  }

  get isEmpty() {
    return this.strokes.length === 0 && !this.sample;
  }

  /** Grayscale [0,1] pixels of the ink canvas (row-major 280x280), or null if blank. */
  getImage(): Float32Array | null {
    if (this.strokes.length === 0) return null;
    const data = this.ictx.getImageData(0, 0, INK_SIZE, INK_SIZE).data;
    const out = new Float32Array(INK_SIZE * INK_SIZE);
    for (let i = 0; i < out.length; i++) out[i] = data[i * 4] / 255;
    return out;
  }

  clear() {
    this.strokes = [];
    this.sample = null;
    this.queueRender();
    this.onChange();
  }

  /** Show an already-normalized 28x28 test image. */
  showSample(pixels: Float32Array) {
    this.strokes = [];
    this.sample = pixels;
    this.queueRender();
  }

  private bindPointer() {
    const c = this.canvas;
    const pos = (e: PointerEvent): Pt => {
      const r = c.getBoundingClientRect();
      return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
    };
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      if (this.sample) this.sample = null;
      this.drawing = true;
      c.setPointerCapture(e.pointerId);
      this.strokes.push([pos(e)]);
      this.onStart();
      this.changed();
    });
    c.addEventListener("pointermove", (e) => {
      if (!this.drawing) return;
      const stroke = this.strokes[this.strokes.length - 1];
      const events = e.getCoalescedEvents?.() ?? [e];
      for (const ev of events) stroke.push(pos(ev));
      this.changed();
    });
    const end = () => {
      this.drawing = false;
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
  }

  private changed() {
    this.queueRender();
    this.onChange();
  }

  private resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.render();
  }

  private queueRender() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.render();
    });
  }

  private render() {
    // 1. ink canvas: exactly what the network reads
    const ic = this.ictx;
    ic.fillStyle = "#000";
    ic.fillRect(0, 0, INK_SIZE, INK_SIZE);
    ic.strokeStyle = "#fff";
    ic.fillStyle = "#fff";
    for (const s of this.strokes) strokePath(ic, s, INK_SIZE, this.brush);

    // 2. display canvas: same strokes, neon look
    const d = this.dctx;
    const W = this.canvas.width;
    d.clearRect(0, 0, W, W);
    if (this.sample) {
      this.drawSample(d, W);
      return;
    }
    const lw = (this.brush * W) / INK_SIZE;
    const grad = d.createLinearGradient(0, 0, W, W);
    grad.addColorStop(0, "#a5f3fc");
    grad.addColorStop(0.55, "#c4b5fd");
    grad.addColorStop(1, "#f9a8d4");
    const passes: [number, string | CanvasGradient, number][] = [
      [2.6, "rgba(103,232,249,0.06)", 1],
      [1.7, "rgba(167,139,250,0.14)", 1],
      [1.0, grad, 1],
      [0.35, "rgba(255,255,255,0.7)", 1],
    ];
    for (const [scale, style, alpha] of passes) {
      d.globalAlpha = alpha;
      d.strokeStyle = style;
      d.fillStyle = style;
      for (const s of this.strokes) strokePath(d, s, W, lw * scale);
    }
    d.globalAlpha = 1;
  }

  private drawSample(d: CanvasRenderingContext2D, W: number) {
    const small = document.createElement("canvas");
    small.width = small.height = 28;
    const sc = small.getContext("2d")!;
    const img = sc.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = this.sample![i];
      img.data[i * 4] = 200 + 55 * v;
      img.data[i * 4 + 1] = 225 + 30 * v;
      img.data[i * 4 + 2] = 255;
      img.data[i * 4 + 3] = Math.round(v * 255);
    }
    sc.putImageData(img, 0, 0);
    d.imageSmoothingEnabled = true;
    d.imageSmoothingQuality = "high";
    d.save();
    d.shadowColor = "rgba(103,232,249,0.6)";
    d.shadowBlur = W * 0.04;
    d.drawImage(small, 0, 0, W, W);
    d.restore();
  }
}

/** Smooth stroke through points using quadratic curves between midpoints. */
function strokePath(ctx: CanvasRenderingContext2D, pts: Pt[], size: number, width: number) {
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (pts.length === 1) {
    ctx.beginPath();
    ctx.arc(pts[0].x * size, pts[0].y * size, width / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.beginPath();
  ctx.moveTo(pts[0].x * size, pts[0].y * size);
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = ((pts[i].x + pts[i + 1].x) / 2) * size;
    const my = ((pts[i].y + pts[i + 1].y) / 2) * size;
    ctx.quadraticCurveTo(pts[i].x * size, pts[i].y * size, mx, my);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last.x * size, last.y * size);
  ctx.stroke();
}
