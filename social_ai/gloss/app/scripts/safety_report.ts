// Recall and false positives of the safety check from saved model scores
// (scripts/safety_model_eval.ts), for the model alone and patterns OR model,
// at thresholds 0.3-0.9. E9c is held back until --e9c is passed, after
// thresholds are fixed on E9b and the ordinary-input sets.
//
//   node --experimental-strip-types scripts/safety_report.ts [model] [--e9c] [--e9d] [--twogate]

import { existsSync, readFileSync } from "node:fs";
import { safetyCheck } from "../src/engine/safety.ts";
import { S2 } from "./audit_scenarios_v2.ts";

const X = new URL("../../experiments/", import.meta.url).pathname;
const args = process.argv.slice(2);
const includeE9c = args.includes("--e9c");
const includeE9d = args.includes("--e9d");
const twogate = args.includes("--twogate");
const model = (args.find((a) => !a.startsWith("--") ) ?? "qwen3:8b").replace(/:/g, "-");
const KINDS = ["distress", "scam", "boundary"] as const;
const THS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
const ORDINARY = ["audit", "e10", "e4", "e8"] as const;
const read = (p: string) => JSON.parse(readFileSync(X + p, "utf8"));
const them = (c: any) => ({ mode: "them" as const, context: c.context, text: c.text });
const scores = (set: string): Record<string, any> => {
  const f = `${X}e9_safety/model_scores/${set}_${model}${twogate ? "_twogate" : ""}.jsonl`;
  if (!existsSync(f)) return {};
  return Object.fromEntries(readFileSync(f, "utf8").trim().split("\n").filter(Boolean).map((l) => [JSON.parse(l).id, JSON.parse(l)]));
};
const E9_CAT: Record<string, string[]> = { boundary_pressure: ["boundary"], scam: ["scam"], distress_self: ["distress"] };

const sets: Record<string, { id: string; input: any; flags?: Record<string, boolean> }[]> = {
  e9: read("e9_safety/items.json").map((c: any) => ({
    ...c, flags: Object.fromEntries(KINDS.map((k) => [k, (E9_CAT[c.category] ?? []).includes(k)])),
  })),
  e9b: read("e9b_safety_heldout/items.json"),
  ...(includeE9c && existsSync(X + "e9c_safety_test/items.json")
    ? { e9c: read("e9c_safety_test/items.json") } : {}),
  ...(includeE9d && existsSync(X + "e9d_safety_test/items.json")
    ? { e9d: read("e9d_safety_test/items.json") } : {}),
  audit: S2.map((s, i) => ({ id: `s${i + 1}`, input: them(s) })),
  e10: read("e10_twoway/items.json"),
  e4: read("e4_product/items.json").map((c: any) => ({ id: c.id, input: them(c) })),
  ...(existsSync(X + "e8_isarcasm/items.json")
    ? { e8: read("e8_isarcasm/items.json").map((c: any) => ({ id: c.id, input: them(c) })) } : {}),
};

for (const [name, cases] of Object.entries(sets)) {
  const sc = scores(name);
  const scored = cases.filter((c) => sc[c.id]);
  if (!scored.length) continue;
  console.log(`\n${name} (scored ${scored.length}/${cases.length})`);
  for (const k of KINDS) {
    const pat = (c: any) => !!safetyCheck(c.input)[k];
    const mod = (c: any, t: number) => (sc[c.id][k] ?? 0) >= t;
    const row = (label: string, f: (c: any) => boolean) => {
      if (scored[0].flags) {
        const pos = scored.filter((c) => c.flags![k]);
        const neg = scored.filter((c) => !c.flags![k]);
        return `${label} recall ${pos.filter(f).length}/${pos.length} fp ${neg.filter(f).length}/${neg.length}`;
      }
      return `${label} flagged ${scored.filter(f).length}`;
    };
    const parts = [row("patterns", pat), ...THS.map((t) => row(`model>=${t}`, (c) => mod(c, t))),
      ...THS.map((t) => row(`either>=${t}`, (c) => pat(c) || mod(c, t)))];
    console.log(`  ${k}\n    ${parts.join("\n    ")}`);
  }
}

// Pre-registered selection (DECISIONS "E9c frozen"): per flag, the lowest
// threshold with combined (patterns OR model) false positives <= 1/60 on E9b
// and <= 2 on the scored ordinary inputs. If none qualify, the model check
// is not adopted for that flag (threshold 1.0).
const e9bCases = sets.e9b.filter((c) => scores("e9b")[c.id]);
const ordinary = ORDINARY.flatMap((name) => {
  const sc = scores(name);
  return (sets[name] ?? []).filter((c) => sc[c.id]).map((c) => ({ ...c, _sc: sc[c.id], _set: name }));
});
console.log(`\nthreshold rule (E9b scored ${e9bCases.length}/80; ordinary scored ${ordinary.length}/384)`);
for (const k of KINDS) {
  const e9bNeg = e9bCases.filter((c) => !c.flags![k]);
  const scE9b = scores("e9b");
  const hit = (c: { input: any; _sc?: any }, t: number, sc: any) =>
    !!safetyCheck(c.input)[k] || (sc[k] ?? 0) >= t;
  const ok: number[] = [];
  for (const t of THS) {
    const fpE9b = e9bNeg.filter((c) => hit(c, t, scE9b[c.id])).length;
    const fpOrd = ordinary.filter((c) => hit(c, t, c._sc)).length;
    const pass = fpE9b <= 1 && fpOrd <= 2;
    if (pass) ok.push(t);
    console.log(`  ${k} t=${t} e9b-fp ${fpE9b}/${e9bNeg.length} ordinary-fp ${fpOrd}/${ordinary.length}${pass ? "  QUALIFIES" : ""}`);
  }
  const chosen = ok.length ? Math.min(...ok) : 2;
  console.log(`  ${k} chosen ${chosen}${ok.length ? "" : " (none qualify; model not adopted)"}`);
}
