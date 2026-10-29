// Latency of Reflect as the user feels it: time until the literal line, the
// first reading and all readings appear (they stream in while being written),
// then the total once they are weighed. Runs on the first n E4 items.
//
//   node --experimental-strip-types scripts/latency.ts qwen3:8b 20 > ../experiments/e4_product/latency.jsonl

import { readFileSync } from "node:fs";
import { reflect } from "../src/engine/reflect.ts";

const BASE = "http://127.0.0.1:11434";
const model = process.argv[2] ?? "qwen3:8b";
const n = Number(process.argv[3] ?? 20);
type Item = { id: string; person: string | null; context: string; text: string };
const items: Item[] = JSON.parse(readFileSync(new URL("../../experiments/e4_product/items.json", import.meta.url), "utf8"));

// Warm the model so the first item does not pay the load time.
await reflect(BASE, model, { mode: "them", context: "", text: "See you tomorrow." });

for (const it of items.slice(0, n)) {
  const t0 = Date.now();
  const at: Record<string, number> = {};
  const r = await reflect(BASE, model, { mode: "them", person: it.person ?? undefined, context: it.context, text: it.text }, undefined, (step, d) => {
    const t = Date.now() - t0;
    if (step === "writing" && d?.literal && at.literal === undefined) at.literal = t;
    if (step === "writing" && d && d.meanings.length > 0 && at.first === undefined) at.first = t;
    if (step === "weighing" && at.written === undefined) at.written = t;
  });
  const rec = { id: it.id, ...at, total: Date.now() - t0, chars: r.readings.reduce((a, x) => a + x.meaning.length, 0) };
  console.log(JSON.stringify(rec));
  console.error(rec);
}
