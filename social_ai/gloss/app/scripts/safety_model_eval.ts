// Score every safety question with the model and write p(yes) per case, for
// threshold setting (E9b) and testing (E9c). Resumable.
//
//   node --experimental-strip-types scripts/safety_model_eval.ts <set> [model]
//   set: e9 | e9b | e9c | e4 | e10 | audit | e8

import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { safetyScores } from "../src/engine/safety_model.ts";
import { S2 } from "./audit_scenarios_v2.ts";

const BASE = "http://127.0.0.1:11434";
const X = new URL("../../experiments/", import.meta.url).pathname;
const [set, model = "qwen3:8b"] = process.argv.slice(2);
const read = (p: string) => JSON.parse(readFileSync(X + p, "utf8"));
const them = (c: any) => ({ mode: "them" as const, context: c.context, text: c.text });

const cases: { id: string; input: any }[] =
  set === "e9" ? read("e9_safety/items.json")
  : set === "e9b" ? read("e9b_safety_heldout/items.json")
  : set === "e9c" ? read("e9c_safety_test/items.json")
  : set === "e10" ? read("e10_twoway/items.json")
  : set === "e4" ? read("e4_product/items.json").map((c: any) => ({ id: c.id, input: them(c) }))
  : set === "e8" ? read("e8_isarcasm/items.json").map((c: any) => ({ id: c.id, input: them(c) }))
  : S2.map((s, i) => ({ id: `s${i + 1}`, input: them(s) }));

const out = `${X}e9_safety/model_scores/${set}_${model.replace(/:/g, "-")}.jsonl`;
const done = new Set(existsSync(out) ? readFileSync(out, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l).id) : []);
for (const c of cases) {
  if (done.has(c.id)) continue;
  const p = await safetyScores(BASE, model, c.input);
  appendFileSync(out, JSON.stringify({ id: c.id, ...p }) + "\n");
}
console.error(`SAFETY_SCORES_DONE ${set}`);
