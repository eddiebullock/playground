// Run Reflect (and, with --replies, ideas for replying) on a set of cases
// and write one JSON line per case. Used for E9 (safety red team) and E10
// (two-way drafts). Resumable.
//
//   node --experimental-strip-types scripts/run_set.ts <items.json> <out.jsonl> [--replies] [model]

import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { reflect, waysToReply, type ReflectInput } from "../src/engine/reflect.ts";

const BASE = "http://127.0.0.1:11434";
const [itemsPath, outPath, ...rest] = process.argv.slice(2);
const replies = rest.includes("--replies");
const model = rest.find((a) => !a.startsWith("--")) ?? "qwen3:8b";
const items: { id: string; input: ReflectInput }[] = JSON.parse(readFileSync(itemsPath, "utf8"));
const done = new Set(existsSync(outPath) ? readFileSync(outPath, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l).id) : []);
for (const it of items) {
  if (done.has(it.id)) continue;
  const r = await reflect(BASE, model, it.input);
  const rec: Record<string, unknown> = { id: it.id, model, ...r };
  if (replies && it.input.mode === "them") rec.replies = await waysToReply(BASE, model, it.input, r);
  appendFileSync(outPath, JSON.stringify(rec) + "\n");
  console.error(`${it.id} ${r.clarity} ${(r.ms / 1000).toFixed(0)}s`);
}
console.error("RUN_SET_DONE");
