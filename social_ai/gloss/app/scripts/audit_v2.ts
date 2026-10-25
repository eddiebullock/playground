// E4 readings audit, v2: 48 scenarios, six disclosures including a placebo
// ("I am left-handed"), any local model, and either scorer variant. Same
// measure as audit.ts: the scorer weighs four fixed readings (sincere, mildly
// negative, hostile, joking) and should not move when only the reader changes.
//
//   node --experimental-strip-types scripts/audit_v2.ts qwen3:8b [stance|words] > ../experiments/e4_audit/v2_<model>_<variant>.json

import { scoreOptions, setScorer, type ReflectInput } from "../src/engine/reflect.ts";
import { DISCLOSURE2, S2 } from "./audit_scenarios_v2.ts";

const base = "http://127.0.0.1:11434";
const model = process.argv[2] ?? "qwen3:8b";
const variant = (process.argv[3] ?? "stance") as "stance" | "words";
setScorer(variant);
const out: unknown[] = [];
for (const [i, s] of S2.entries()) {
  const rec: Record<string, number[]> = {};
  for (const [k, d] of Object.entries(DISCLOSURE2)) {
    const input: ReflectInput = { ...s, mode: "them", context: [d, s.context].filter(Boolean).join("\n") };
    rec[k] = await scoreOptions(base, model, input, s.options, 4);
  }
  out.push({ i, text: s.text, ...rec });
  console.error(i, Object.entries(rec).map(([k, p]) => `${k}:${p.map((x) => x.toFixed(2)).join("/")}`).join("  "));
}
console.log(JSON.stringify({ model, variant, scenarios: out }, null, 1));
