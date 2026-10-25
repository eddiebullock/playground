// E4 stereotype audit. Does telling Gloss "I am autistic" (or "I get anxious
// about messages") change how it weighs the same readings of the same message?
// Fixed options per scenario: sincere, mildly negative, hostile, joking.
// A good tool should not move: disclosure says nothing about the other
// person's intent.
//
//   node --experimental-strip-types scripts/audit.ts qwen3:8b > ../experiments/e4_audit/qwen3-8b.json

import { scoreOptions, type ReflectInput } from "../src/engine/reflect.ts";
import { DISCLOSURE, S } from "./audit_scenarios.ts";

const base = "http://127.0.0.1:11434";
const model = process.argv[2] ?? "qwen3:4b";
const out: unknown[] = [];
for (const [i, s] of S.entries()) {
  const rec: Record<string, number[]> = {};
  for (const [k, d] of Object.entries(DISCLOSURE)) {
    const input: ReflectInput = { ...s, mode: "them", context: [d, s.context].filter(Boolean).join("\n") };
    rec[k] = await scoreOptions(base, model, input, s.options, 4);
  }
  out.push({ i, text: s.text, ...rec });
  console.error(i, Object.entries(rec).map(([k, p]) => `${k}:${p.map((x) => x.toFixed(2)).join("/")}`).join("  "));
}
console.log(JSON.stringify({ model, scenarios: out }, null, 1));
