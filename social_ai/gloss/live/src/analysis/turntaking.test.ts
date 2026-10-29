import assert from "node:assert/strict";
import test from "node:test";
import { emptyTurn, rmsToSpeech, stepTurn } from "./turntaking.ts";

test("rms threshold", () => {
  assert.equal(rmsToSpeech(0.04, 0.05), false);
  assert.equal(rmsToSpeech(0.06, 0.05), true);
});

test("them speaking then silence opens the floor", () => {
  let mem = emptyTurn;
  let cue = stepTurn(mem, true, false, 200).cue;
  mem = stepTurn(mem, true, false, 200).mem;
  cue = stepTurn(mem, false, false, 900).cue;
  assert.equal(cue.floorOpen, true);
  assert.match(cue.headline, /Pause/);
});

test("your turn accumulates a timer", () => {
  let mem = emptyTurn;
  const a = stepTurn(mem, false, true, 1000);
  mem = a.mem;
  const b = stepTurn(mem, false, true, 29_000);
  assert.equal(b.cue.longYou, true);
  assert.match(b.cue.headline, /a while/);
});

test("overlap", () => {
  const { cue } = stepTurn(emptyTurn, true, true, 100);
  assert.equal(cue.talk, "both");
});
