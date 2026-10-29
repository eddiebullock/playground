import assert from "node:assert/strict";
import test from "node:test";
import { gazeFromBlendshapes } from "./gaze.ts";

test("empty blends are unknown", () => {
  assert.equal(gazeFromBlendshapes([]).bin, "unknown");
});

test("look down is screen-ward", () => {
  const r = gazeFromBlendshapes([
    { categoryName: "eyeLookDownLeft", score: 0.5 },
    { categoryName: "eyeLookDownRight", score: 0.5 },
  ]);
  assert.equal(r.bin, "screen");
});

test("centred eyes are camera-ward", () => {
  const r = gazeFromBlendshapes([
    { categoryName: "eyeLookDownLeft", score: 0.05 },
    { categoryName: "eyeLookDownRight", score: 0.05 },
    { categoryName: "eyeLookInLeft", score: 0.04 },
    { categoryName: "eyeLookInRight", score: 0.04 },
  ]);
  assert.equal(r.bin, "camera");
});
