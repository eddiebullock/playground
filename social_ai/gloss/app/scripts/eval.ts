// E4 product eval: run the full Reflect pipeline on the Hu et al. stories
// (experiments/e4_product/items.json), then map every generated reading to the
// story's original options with a separate judge call, so a reading can be
// graded against the intended meaning.
//
//   npm run eval -- qwen3:8b            # writes experiments/e4_product/results/<model>.jsonl
//
// Resumable: items already in the output are skipped.
// Options (environment): E4_TAG=v2 adds a suffix to the output name;
// E4_HALF=even|odd runs only items with even or odd item numbers (odd is the
// development half, even is held out); E4_JUDGE=0 skips the judge, for runs
// that will be hand-labelled (the judge agreed with hand labels on only 0.72
// of readings, docs/DECISIONS.md). E4_WIDE=6 writes up to 6 readings and
// shows the top 4 (setWidth). E4_ITEMS=<path to items.json> runs another item
// set (E8: experiments/e8_isarcasm/items.json); results go next to it.

import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { optionDist } from "../src/engine/llm.ts";
import { reflect, setWidth } from "../src/engine/reflect.ts";

type Item = {
  id: string;
  task: string;
  person?: string | null;
  channel?: string;
  context: string;
  text: string;
  options: string[];
  labels: string[];
  gold: string;
};

const BASE = "http://127.0.0.1:11434";
const model = process.argv[2] ?? "qwen3:8b";
const judge = process.argv[3] ?? "qwen3:8b";
const itemsPath = process.env.E4_ITEMS ? new URL(process.env.E4_ITEMS, `file://${process.cwd()}/`) : new URL("../../experiments/e4_product/items.json", import.meta.url);
const dir = new URL("./", itemsPath);
const items: Item[] = JSON.parse(readFileSync(itemsPath, "utf8"));
if (process.env.E4_WIDE) setWidth(Number(process.env.E4_WIDE));
mkdirSync(new URL("results/", dir), { recursive: true });
const tag = process.env.E4_TAG ? `_${process.env.E4_TAG}` : "";
const half = process.env.E4_HALF;
const useJudge = process.env.E4_JUDGE !== "0";
const out = new URL(`results/${model.replace(/[:/]/g, "-")}${tag}.jsonl`, dir);
const done = new Set(
  existsSync(out) ? readFileSync(out, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l).id) : [],
);

const JUDGE_SYSTEM = "You compare the meaning of statements. Answer with the number only.";

/**
 * Which original option (index) a reading means, or -1 for none. Options are
 * rotated and averaged to cancel position bias; "none" always comes last.
 * Without rotation and the note on "me", qwen3:8b chose "none" for clear
 * matches (smoke test, 2026-10-25).
 */
async function mapReading(it: Item, reading: string): Promise<{ k: number; p: number }> {
  const all = [...it.options, "None of the above."];
  const n = it.options.length;
  const acc = new Array<number>(all.length).fill(0);
  for (let rot = 0; rot < n; rot++) {
    const order = it.options.map((_, i) => (i + rot) % n).concat([n]);
    const user = `Story: ${it.context} "${it.text}"\n\nIn the statement below, "I" and "me" mean the person the remark was said to.\nStatement: ${reading}\n\nWhich option is closest to what the statement says the speaker meant? Choose the last option only if none of them is close.\n${order
      .map((k, i) => `${i + 1}) ${all[k]}`)
      .join("\n")}`;
    const p = (await optionDist(BASE, judge, JUDGE_SYSTEM, user, all.length)) ?? all.map(() => 1 / all.length);
    order.forEach((k, pos) => (acc[k] += p[pos] / n));
  }
  const k = acc.indexOf(Math.max(...acc));
  return { k: k === n ? -1 : k, p: acc[k] };
}

let n = 0;
for (const it of items) {
  if (done.has(it.id)) continue;
  const num = Number(it.id.split(":")[1]);
  if (half && (num % 2 === 0) !== (half === "even")) continue;
  const r = await reflect(BASE, model, { mode: "them", person: it.person ?? undefined, context: it.context, text: it.text, channel: it.channel });
  const readings = [];
  for (const x of r.readings) {
    const m = useJudge && it.options ? await mapReading(it, x.meaning) : { k: -1, p: 0 };
    readings.push({ ...x, label: m.k >= 0 ? it.labels[m.k] : "None", judge_p: m.p });
  }
  const rec = { id: it.id, task: it.task, gold: it.gold, model, judge, clarity: r.clarity, spread: r.spread, abstain: r.abstain, ms: r.ms, ms_gen: r.ms_gen, literal: r.literal, check: r.check, boundary: r.boundary, readings };
  appendFileSync(out, JSON.stringify(rec) + "\n");
  n++;
  console.error(`${it.id} ${r.clarity} top=${readings[0].label} (${readings[0].p.toFixed(2)}) ${(r.ms / 1000).toFixed(0)}s`);
}
console.error(`done ${n}`);
