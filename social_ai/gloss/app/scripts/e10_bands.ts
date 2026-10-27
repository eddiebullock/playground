// Fit me-mode clarity bands on the frozen E10 Reflect run.
// Odd ids = development, even = test. Prints the pre-registered grid.
//
//   node --experimental-strip-types scripts/e10_bands.ts

import { readFileSync } from "node:fs";

const items: { id: string; group: string }[] = JSON.parse(
  readFileSync(new URL("../../experiments/e10_twoway/items.json", import.meta.url), "utf8"),
);
const byId = Object.fromEntries(items.map((i) => [i.id, i]));
const rows: { id: string; readings: { p: number }[]; clarity: string }[] = readFileSync(
  new URL("../../experiments/e10_twoway/results_qwen3-8b.jsonl", import.meta.url),
  "utf8",
)
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l));

function band(top: number, clearAt: number, abstainAt: number) {
  if (top >= clearAt) return "clear";
  if (top >= abstainAt) return "some";
  return "high";
}

function stats(subset: typeof rows, clearAt: number, abstainAt: number) {
  const clear = subset.filter((r) => byId[r.id]?.group === "clear");
  const amb = subset.filter((r) => byId[r.id]?.group === "ambiguous");
  const top = (r: (typeof rows)[0]) => r.readings[0]?.p ?? 0;
  const fa = clear.filter((r) => band(top(r), clearAt, abstainAt) === "high").length;
  const miss = amb.filter((r) => band(top(r), clearAt, abstainAt) === "clear").length;
  const clearOnClear = clear.filter((r) => band(top(r), clearAt, abstainAt) === "clear").length;
  return { fa, nClear: clear.length, miss, nAmb: amb.length, clearOnClear };
}

const odd = rows.filter((r) => Number(r.id.split(":")[1]) % 2 === 1);
const even = rows.filter((r) => Number(r.id.split(":")[1]) % 2 === 0);
const ship = stats(rows, 0.7, 0.5);
console.log(`shipped 0.7/0.5  FA ${ship.fa}/${ship.nClear}  miss ${ship.miss}/${ship.nAmb}  clear-on-clear ${ship.clearOnClear}/${ship.nClear}`);
console.log(`odd shipped     FA ${stats(odd, 0.7, 0.5).fa}/12  miss ${stats(odd, 0.7, 0.5).miss}/12`);

const ok: { a: number; c: number }[] = [];
for (let a = 0.15; a <= 0.5 + 1e-9; a += 0.05) {
  for (let c = 0.35; c <= 0.7 + 1e-9; c += 0.05) {
    const aa = Math.round(a * 100) / 100;
    const cc = Math.round(c * 100) / 100;
    if (cc <= aa) continue;
    const s = stats(odd, cc, aa);
    if (s.fa <= 2 && s.miss <= 3) ok.push({ a: aa, c: cc });
  }
}
ok.sort((x, y) => y.a - x.a || y.c - x.c);
const chosen = ok[0];
console.log(`qualifying on odd: ${ok.length}; chosen abstain ${chosen?.a} clear ${chosen?.c}`);
if (chosen) {
  const t = stats(even, chosen.c, chosen.a);
  const pass = t.fa <= 2 && t.miss <= 3;
  console.log(`even FA ${t.fa}/12  miss ${t.miss}/12  clear-on-clear ${t.clearOnClear}/12  ${pass ? "PASS" : "FAIL"}`);
}
