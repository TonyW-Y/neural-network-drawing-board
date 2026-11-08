import "./style.css";
import { DrawingBoard, INK_SIZE } from "./draw";
import { type Activations, type Model, forward, loadModel } from "./nn";
import { MathPanel } from "./math-panel";
import { PredictionCard } from "./prediction";
import { normalize } from "./preprocess";
import { NetworkViz, type ValueMode } from "./viz";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const board = new DrawingBoard($<HTMLCanvasElement>("board"));
const prediction = new PredictionCard();
const viz = new NetworkViz($<HTMLCanvasElement>("net"));
const panel = new MathPanel();
viz.onFocus = (f) => model && panel.update(model, acts, f);

const models = new Map<Model["name"], Promise<Model>>();
const getModel = (name: Model["name"]) => {
  if (!models.has(name)) models.set(name, loadModel(name));
  return models.get(name)!;
};

let model: Model;
let sampleInput: Float32Array | null = null;
let acts: Activations | null = null;
let dirty = true;

function compute() {
  const raw = board.getImage();
  const x = sampleInput ?? (raw ? normalize(raw, INK_SIZE, INK_SIZE) : null);
  acts = x ? forward(model, x) : null;
  prediction.update(model, acts ? acts.a[acts.a.length - 1] : null);
  viz.setActivations(acts);
  panel.update(model, acts, viz.focus);
  $("board-empty").classList.toggle("hidden", !board.isEmpty);
  $("net-tip").textContent = acts
    ? "Hover a neuron to see its dot product · click to pin"
    : "Draw on the board — the signal flows left → right";
}

function loop() {
  if (dirty && model) {
    dirty = false;
    compute();
  }
  viz.frame();
  requestAnimationFrame(loop);
}

$("value-mode").querySelectorAll<HTMLButtonElement>("button").forEach((btn) =>
  btn.addEventListener("click", () => {
    $("value-mode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
    viz.setMode(btn.dataset.mode as ValueMode);
  }),
);

// ------------------------------------------------------------ controls
board.onStart = () => {
  sampleInput = null;
  $("board-wrap").classList.add("active");
};
board.onChange = () => (dirty = true);
window.addEventListener("pointerup", () => $("board-wrap").classList.remove("active"));

$("clear").addEventListener("click", () => {
  sampleInput = null;
  board.clear();
});

function showSample(i: number) {
  const s = model.samples[i % model.samples.length];
  sampleInput = s.pixels;
  board.showSample(s.pixels);
  dirty = true;
}
$("sample").addEventListener("click", () => showSample(Math.floor(Math.random() * model.samples.length)));

const brush = $<HTMLInputElement>("brush");
brush.addEventListener("input", () => {
  board.brush = Number(brush.value);
  $("brush-out").textContent = brush.value;
  board.clear();
});

const toggle = $("model-toggle");
const thumb = toggle.querySelector<HTMLElement>(".seg-thumb")!;
function placeThumb() {
  const btn = toggle.querySelector<HTMLElement>('[aria-selected="true"]')!;
  thumb.style.width = `${btn.offsetWidth}px`;
  thumb.style.transform = `translateX(${btn.offsetLeft - 4}px)`;
}
toggle.querySelectorAll<HTMLButtonElement>("button").forEach((btn) =>
  btn.addEventListener("click", () => selectModel(btn.dataset.model as Model["name"])),
);
window.addEventListener("resize", placeThumb);

async function selectModel(name: Model["name"]) {
  toggle.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.model === name)));
  placeThumb();
  model = await getModel(name);
  sampleInput = null;
  board.clear();
  $("draw-hint").textContent = name === "digits" ? "a digit 0–9" : "a capital or lowercase letter";
  $("arch").innerHTML = `
    <span class="chip"><b>${model.sizes.join(" → ")}</b></span>
    <span class="chip">${model.meta.params.toLocaleString()} params</span>
    <span class="chip"><span class="acc">${(model.meta.testAccuracy * 100).toFixed(1)}%</span> test acc</span>`;
  prediction.setModel(model);
  panel.setModel(model);
  viz.setModel(model);
  dirty = true;
}

// Deep links: ?model=letters&sample=3&focus=1:5  (focus = layer:neuron)
const params = new URLSearchParams(location.search);
const startModel = params.get("model") === "letters" ? "letters" : "digits";

document.fonts?.ready.then(placeThumb);
selectModel(startModel).then(() => {
  placeThumb();
  getModel(startModel === "digits" ? "letters" : "digits"); // prefetch the other model
  if (params.has("sample")) showSample(Number(params.get("sample")) || 0);
  const [fl, fi] = (params.get("focus") ?? "").split(":").map(Number);
  if (fl >= 1 && fl <= 3 && fi >= 0 && fi < model.sizes[fl]) viz.pin({ layer: fl, index: fi });
  requestAnimationFrame(loop);
});
