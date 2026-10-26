// Detection rates of the fixed safety layer (src/engine/safety.ts). No model
// needed. Sets with expected flags: E9 (development; the patterns were written
// after seeing its failures) and E9b (held out, written by a separate agent
// without access to the patterns). Sets that should raise nothing: E4 product
// items, E10 drafts, the 48 audit scenarios; E8 tweets are reported but may
// contain genuine flags.
//
//   node --experimental-strip-types scripts/safety_eval.ts

import { existsSync, readFileSync } from "node:fs";
import { safetyCheck, type SafetyInput } from "../src/engine/safety.ts";
import { S2 } from "./audit_scenarios_v2.ts";

const X = new URL("../../experiments/", import.meta.url).pathname;
const load = (p: string) => (existsSync(X + p) ? JSON.parse(readFileSync(X + p, "utf8")) : null);
const KINDS = ["distress", "scam", "boundary"] as const;
type Kind = (typeof KINDS)[number];

function labelled(name: string, cases: { id: string; input: SafetyInput; flags: Record<Kind, boolean> }[]) {
  console.log(`\n${name} (n=${cases.length})`);
  for (const k of KINDS) {
    const pos = cases.filter((c) => c.flags[k]);
    const neg = cases.filter((c) => !c.flags[k]);
    const tp = pos.filter((c) => safetyCheck(c.input)[k]);
    const fp = neg.filter((c) => safetyCheck(c.input)[k]);
    console.log(`  ${k.padEnd(9)} recall ${tp.length}/${pos.length}  false positives ${fp.length}/${neg.length}`);
    const miss = pos.filter((c) => !safetyCheck(c.input)[k]).map((c) => c.id);
    if (miss.length) console.log(`    missed: ${miss.join(" ")}`);
    if (fp.length) console.log(`    false:  ${fp.map((c) => c.id).join(" ")}`);
  }
}

function unlabelled(name: string, inputs: { id: string; input: SafetyInput }[]) {
  const hits = inputs.map((c) => ({ id: c.id, s: safetyCheck(c.input) })).filter((c) => Object.keys(c.s).length);
  console.log(`\n${name} (n=${inputs.length}, expected none): flagged ${hits.length}`);
  for (const h of hits) console.log(`    ${h.id}: ${Object.keys(h.s).join(",")}`);
}

const E9_CAT: Record<string, Kind[]> = {
  boundary_pressure: ["boundary"], coercion_or_control: [], scam: ["scam"], distress_self: ["distress"],
};
const e9 = load("e9_safety/items.json");
if (e9)
  labelled("E9 development", e9.map((c: any) => ({
    id: c.id, input: c.input,
    flags: Object.fromEntries(KINDS.map((k) => [k, (E9_CAT[c.category] ?? []).includes(k)])),
  })));
const e9b = load("e9b_safety_heldout/items.json");
if (e9b) labelled("E9b held out", e9b);

const e4 = load("e4_product/items.json");
if (e4) unlabelled("E4 product items", e4.map((c: any) => ({ id: c.id, input: { mode: "them", context: c.context, text: c.text } })));
const e10 = load("e10_twoway/items.json");
if (e10) unlabelled("E10 drafts", e10.map((c: any) => ({ id: c.id, input: c.input })));
unlabelled("Audit scenarios", S2.map((s, i) => ({ id: `s${i + 1}`, input: { mode: "them", context: s.context, text: s.text } })));
const e8 = load("e8_isarcasm/items.json");
if (e8) unlabelled("E8 tweets (may hold real flags)", e8.map((c: any) => ({ id: c.id, input: { mode: "them", context: c.context, text: c.text } })));
