import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { forward, parseModel } from "../src/nn";
import { normalize } from "../src/preprocess";

const root = resolve(__dirname, "..");
const load = (p: string) => JSON.parse(readFileSync(resolve(root, p), "utf8"));

describe.each(["digits", "letters"])("%s model", (name) => {
  const model = parseModel(load(`public/models/${name}.json`));

  it("JS forward pass matches the Python logits", () => {
    for (const s of model.samples) {
      const { z } = forward(model, s.pixels);
      const logits = z[z.length - 1];
      s.logits.forEach((v, k) => expect(logits[k]).toBeCloseTo(v, 3));
    }
  });

  it("classifies its sample test images", () => {
    const correct = model.samples.filter((s) => {
      const p = forward(model, s.pixels).a.at(-1)!;
      return p.indexOf(Math.max(...p)) === s.label;
    }).length;
    expect(correct / model.samples.length).toBeGreaterThan(0.8);
  });
});

describe("preprocess", () => {
  const cases: { width: number; height: number; input: number[]; expected: number[] }[] = load(
    "tests/fixtures/preprocess.json",
  );

  it.each(cases.map((c, i) => [i, c] as const))("case %i matches training/data.py normalize()", (_, c) => {
    const out = normalize(Float32Array.from(c.input), c.width, c.height)!;
    expect(out).not.toBeNull();
    c.expected.forEach((v, k) => expect(out[k]).toBeCloseTo(v, 4));
  });

  it("returns null for an empty canvas", () => {
    expect(normalize(new Float32Array(100), 10, 10)).toBeNull();
  });
});
