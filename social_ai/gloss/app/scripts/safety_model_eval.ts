// Score every safety question with the model and write p(yes) per case, for
// threshold setting (E9b) and testing (E9c). Resumable.
//
//   node --experimental-strip-types scripts/safety_model_eval.ts <set> [model] [twogate|dx]
//   set: e9 | e9b | e9c | e9d | e9e | e4 | e10 | audit | e8
//   dx = distress-only rescore of two-gate (keeps scam/boundary from *_twogate.jsonl)

import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { safetyScores, safetyScoresTwoGate, twoGateQuestions } from "../src/engine/safety_model.ts";
import { optionDist } from "../src/engine/llm.ts";
import { S2 } from "./audit_scenarios_v2.ts";

const BASE = "http://127.0.0.1:11434";
const X = new URL("../../experiments/", import.meta.url).pathname;
const [set, model = "qwen3:8b", variant] = process.argv.slice(2);
const twogate = variant === "twogate" || variant === "dx";
const dx = variant === "dx";
const score = twogate && !dx ? safetyScoresTwoGate : safetyScores;
const read = (p: string) => JSON.parse(readFileSync(X + p, "utf8"));
const them = (c: any) => ({ mode: "them" as const, context: c.context, text: c.text });

const cases: { id: string; input: any }[] =
  set === "e9" ? read("e9_safety/items.json")
  : set === "e9b" ? read("e9b_safety_heldout/items.json")
  : set === "e9c" ? read("e9c_safety_test/items.json")
  : set === "e9d" ? read("e9d_safety_test/items.json")
  : set === "e9e" ? read("e9e_safety_test/items.json")
  : set === "e10" ? read("e10_twoway/items.json")
  : set === "e4" ? read("e4_product/items.json").map((c: any) => ({ id: c.id, input: them(c) }))
  : set === "e8" ? read("e8_isarcasm/items.json").map((c: any) => ({ id: c.id, input: them(c) }))
  : S2.map((s, i) => ({ id: `s${i + 1}`, input: them(s) }));

const tag = dx ? "_twogate_dx" : twogate ? "_twogate" : "";
const out = `${X}e9_safety/model_scores/${set}_${model.replace(/:/g, "-")}${tag}.jsonl`;
const priorTwogate = `${X}e9_safety/model_scores/${set}_${model.replace(/:/g, "-")}_twogate.jsonl`;
const prior = Object.fromEntries(
  existsSync(priorTwogate)
    ? readFileSync(priorTwogate, "utf8").trim().split("\n").filter(Boolean).map((l) => {
        const r = JSON.parse(l);
        return [r.id, r];
      })
    : [],
);
const done = new Set(existsSync(out) ? readFileSync(out, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l).id) : []);
async function pYesDistress(input: any) {
  const q = twoGateQuestions(input).distress;
  if (!q) return 0;
  let yes = 0;
  let used = 0;
  for (const yesFirst of [true, false]) {
    const opts = yesFirst ? "1. Yes\n2. No" : "1. No\n2. Yes";
    const SYSTEM = "You check messages for safety risks. Answer with the number of the best option only. No other text.";
    const p = await optionDist(BASE, model, SYSTEM, `${q}\n\n${opts}`, 2);
    if (!p) continue;
    yes += yesFirst ? p[0] : p[1];
    used++;
  }
  return used ? yes / used : 0;
}
async function scoreOne(input: any) {
  let last: unknown;
  for (let i = 0; i < 4; i++) {
    try {
      if (dx) return { distress: await pYesDistress(input) };
      return await score(BASE, model, input);
    } catch (e) {
      last = e;
      await new Promise((r) => setTimeout(r, 2000 * (i + 1)));
    }
  }
  throw last;
}
for (const c of cases) {
  if (done.has(c.id)) continue;
  const p = await scoreOne(c.input);
  const merged = dx ? { ...(prior[c.id] ?? {}), ...p, id: c.id } : { id: c.id, ...p };
  appendFileSync(out, JSON.stringify(merged) + "\n");
}
console.error(`SAFETY_SCORES_DONE ${set}`);
