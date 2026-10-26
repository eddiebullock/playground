// Unit tests for the pure parts of the engine. No model needed.
//   node --experimental-strip-types --test scripts/engine.test.ts

import assert from "node:assert/strict";
import { test } from "node:test";
import { json, optionDist } from "../src/engine/llm.ts";
import { calibrate, draftFrom, normEntropy, scoreOptions } from "../src/engine/reflect.ts";

const close = (a: number[], b: number[]) => a.forEach((x, i) => assert.ok(Math.abs(x - b[i]) < 1e-6, `${a} vs ${b}`));

test("calibrate flattens overconfident probabilities and keeps the order", () => {
  const p = calibrate([0.9, 0.08, 0.02], "qwen3:8b");
  assert.ok(Math.abs(p.reduce((a, b) => a + b) - 1) < 1e-9);
  assert.ok(p[0] < 0.9 && p[0] > p[1] && p[1] > p[2]);
  close(calibrate([0.9, 0.1], "unknown-model"), [0.9, 0.1]);
});

test("normEntropy is 0 for certainty and 1 for a flat distribution", () => {
  assert.equal(normEntropy([1, 0, 0]), 0);
  assert.ok(Math.abs(normEntropy([0.25, 0.25, 0.25, 0.25]) - 1) < 1e-9);
  assert.equal(normEntropy([1]), 0);
});

function fakeFetch(responses: object[]) {
  let i = 0;
  globalThis.fetch = (async () => new Response(JSON.stringify(responses[i++ % responses.length]))) as typeof fetch;
}

test("optionDist reads digits from the first non-space token and ignores other tokens", async () => {
  fakeFetch([{ response: " 2", logprobs: [
    { token: " ", top_logprobs: [] },
    { token: "2", top_logprobs: [
      { token: "2", logprob: Math.log(0.6) }, { token: "1)", logprob: Math.log(0.2) },
      { token: "The", logprob: Math.log(0.1) }, { token: "7", logprob: Math.log(0.1) }] },
  ] }]);
  const p = await optionDist("http://x", "qwen3:8b", "s", "u", 3);
  close(p!, [0.25, 0.75, 0]);
});

test("optionDist falls back to the response text, and returns null when unreadable", async () => {
  fakeFetch([{ response: "3) because", logprobs: [] }]);
  close((await optionDist("http://x", "qwen3:8b", "s", "u", 3))!, [0, 0, 1]);
  fakeFetch([{ response: "I think", logprobs: [] }]);
  assert.equal(await optionDist("http://x", "qwen3:8b", "s", "u", 3), null);
});

test("scoreOptions undoes the rotation, so position bias cancels", async () => {
  // A model that always prefers position 1 should give a flat result.
  fakeFetch([{ response: "1", logprobs: [{ token: "1", top_logprobs: [
    { token: "1", logprob: Math.log(0.7) }, { token: "2", logprob: Math.log(0.2) }, { token: "3", logprob: Math.log(0.1) }] }] }]);
  const p = await scoreOptions("http://x", "qwen3:8b", { mode: "them", text: "hi" }, ["a", "b", "c"], 3);
  close(p, [1 / 3, 1 / 3, 1 / 3]);
});

test("draftFrom keeps only finished strings", () => {
  const partial = '{"literal": "She says \\"fine\\".", "readings": [{"meaning": "She is fine.", "cues": ["fine"]}, {"meaning": "She is a bit ann';
  const d = draftFrom(partial);
  assert.equal(d.literal, 'She says "fine".');
  assert.deepEqual(d.meanings, ["She is fine."]);
  assert.deepEqual(draftFrom("").meanings, []);
});

test("json streams chunks to onText and parses the whole reply", async () => {
  const lines = ['{"message":{"content":"{\\"a\\": "}}\n', '{"message":{"content":"1}"}}\n{"done":true}\n'];
  const real = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(
      new ReadableStream({
        start(c) {
          for (const l of lines) c.enqueue(new TextEncoder().encode(l));
          c.close();
        },
      }),
    )) as typeof fetch;
  try {
    const seen: string[] = [];
    const r = await json<{ a: number }>("http://x", "qwen3:8b", "s", "u", {}, undefined, (t) => seen.push(t));
    assert.deepEqual(r, { a: 1 });
    assert.equal(seen.at(-1), '{"a": 1}');
  } finally {
    globalThis.fetch = real;
  }
});

test("safety patterns flag clear cases and leave look-alikes alone", async () => {
  const { safetyCheck, safeReplies } = await import("../src/engine/safety.ts");
  assert.ok(safetyCheck({ mode: "them", context: "I feel like a burden to everyone.", text: "See you later" }).distress);
  assert.ok(!safetyCheck({ mode: "them", context: "I'm dead tired, this week is killing me.", text: "See you later" }).distress);
  const scam = safetyCheck({ mode: "them", text: "Your account will be suspended. Verify your account within 24 hours: secure-bank.xyz/login" });
  assert.ok(scam.scam);
  assert.ok(!safetyCheck({ mode: "them", text: "Your parcel was delivered to your safe place." }).scam);
  assert.ok(safetyCheck({ mode: "them", context: "I said no, I need the car.", text: "Come on, you'd do it if you were a real mate." }).boundary);
  assert.ok(!safetyCheck({ mode: "them", context: "I said no, I need the car.", text: "No worries, I'll ask someone else." }).boundary);
  assert.ok(!safetyCheck({ mode: "me", context: "I feel fine.", text: "Click this link: x.xyz/a within 24 hours" }).scam);
  const kept = safeReplies([{ reply: "I'll pay the fee now." }, { reply: "I'll check with the company first." }], scam);
  assert.deepEqual(kept.map((r) => r.reply), ["I'll check with the company first."]);
});
