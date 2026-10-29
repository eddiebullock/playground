import assert from "node:assert/strict";
import test from "node:test";
import { emotionFromBlendshapes, type Blend } from "./emotion.ts";

function b(name: string, score: number): Blend {
  return { categoryName: name, score };
}

test("still face when nothing is strong", () => {
  const r = emotionFromBlendshapes([b("mouthSmileLeft", 0.05)]);
  assert.equal(r.label, "Still");
});

test("smile is one word", () => {
  const r = emotionFromBlendshapes([
    b("mouthSmileLeft", 0.7),
    b("mouthSmileRight", 0.65),
    b("mouthFrownLeft", 0.05),
  ]);
  assert.equal(r.label, "Smile");
});

test("raised brows is two words, not a feeling", () => {
  const r = emotionFromBlendshapes([
    b("browInnerUp", 0.8),
    b("eyeWideLeft", 0.6),
    b("mouthSmileLeft", 0.05),
  ]);
  assert.equal(r.label, "Brows up");
  assert.equal(r.label.includes("angry"), false);
});
