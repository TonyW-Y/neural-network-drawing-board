import type { Model } from "./nn";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class PredictionCard {
  private glyph = $("glyph");
  private box = $("glyph-box");
  private confVal = $("conf-val");
  private confFill = $("conf-fill");
  private bars = $("bars");
  private meta = $("meta");
  private current = "";

  setModel(model: Model) {
    const m = model.meta;
    const rows: [string, string][] = [
      ["Architecture", model.sizes.join("→")],
      ["Parameters", m.params.toLocaleString()],
      ["Test accuracy", `${(m.testAccuracy * 100).toFixed(1)}%`],
      ["Training images", m.trainSize.toLocaleString()],
      ["Epochs", String(m.epochs)],
      ["Loss", m.loss],
      ["Optimizer", m.optimizer],
      ["Trained with", m.framework],
    ];
    this.meta.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    this.bars.innerHTML = Array.from(
      { length: 3 },
      () => `<li><span class="k">·</span><span class="track"><i style="width:0"></i></span><span class="p">—</span></li>`,
    ).join("");
    this.update(model, null);
  }

  update(model: Model, probs: Float32Array | null) {
    if (!probs) {
      this.setGlyph("?");
      this.box.classList.add("idle");
      this.confVal.textContent = "—";
      this.confFill.style.width = "0%";
      this.bars.querySelectorAll("li").forEach((li) => {
        li.querySelector(".k")!.textContent = "·";
        (li.querySelector(".track i") as HTMLElement).style.width = "0%";
        li.querySelector(".p")!.textContent = "—";
      });
      return;
    }
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
    this.box.classList.remove("idle");
    this.setGlyph(model.labels[order[0]]);
    this.confVal.textContent = `${(probs[order[0]] * 100).toFixed(1)}%`;
    this.confFill.style.width = `${probs[order[0]] * 100}%`;
    this.bars.querySelectorAll("li").forEach((li, r) => {
      const k = order[r];
      li.querySelector(".k")!.textContent = model.labels[k];
      (li.querySelector(".track i") as HTMLElement).style.width = `${probs[k] * 100}%`;
      li.querySelector(".p")!.textContent = `${(probs[k] * 100).toFixed(1)}%`;
    });
  }

  private setGlyph(text: string) {
    if (text === this.current) return;
    this.current = text;
    this.glyph.textContent = text;
    this.glyph.classList.remove("pop");
    void this.glyph.offsetWidth; // restart the animation
    this.glyph.classList.add("pop");
  }
}
