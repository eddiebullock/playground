import assert from "node:assert/strict";
import test from "node:test";
import { emotionFromBlendshapes, type Blend } from "./emotion.ts";

function b(name: string, score: number): Blend {
  return { categoryName: name, score };
}

test("still face when nothing is strong", () => {
  const r = emotionFromBlendshapes([b("mouthSmileLeft", 0.05)]);
  assert.equal(r.label, "still");
});

test("smile wins on mouth corners", () => {
  const r = emotionFromBlendshapes([
    b("mouthSmileLeft", 0.7),
    b("mouthSmileRight", 0.65),
    b("mouthFrownLeft", 0.05),
  ]);
  assert.equal(r.label, "smiling");
  assert.match(r.hedge, /might be smiling/);
});

test("raised brows without claiming a feeling", () => {
  const r = emotionFromBlendshapes([
    b("browInnerUp", 0.8),
    b("eyeWideLeft", 0.6),
    b("mouthSmileLeft", 0.05),
  ]);
  assert.equal(r.label, "brows or eyes raised");
  assert.equal(r.hedge.includes("angry"), false);
});
