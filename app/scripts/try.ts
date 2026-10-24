import { reflect } from "../src/engine/reflect.ts";
const model = process.argv[2] ?? "qwen3:4b";
const cases = [
  { mode: "them" as const, person: "Simon", relation: "friend", channel: "in person",
    context: "In a shop, I try on a dress. It is far too long for me. I ask Simon: \"Does this dress fit me?\"",
    text: "Wow! That must be custom made! It's clearly the perfect size and length for you." },
  { mode: "them" as const, person: "Priya", relation: "manager", channel: "Slack",
    context: "I sent my report at 4pm, a day before the deadline.",
    text: "Thanks, got it. Will read tomorrow." },
  { mode: "them" as const, person: "Sam", relation: "colleague", channel: "email",
    context: "I missed the team meeting this morning because my train was cancelled. I emailed to say sorry.",
    text: "No worries at all, we managed just fine without you." },
];
for (const c of cases) {
  const r = await reflect("http://127.0.0.1:11434", model, c);
  console.log(`\n# ${c.text}\n${r.clarity} spread=${r.spread.toFixed(2)} abstain=${r.abstain} ${r.ms}ms`);
  for (const x of r.readings) console.log(`  ${x.p.toFixed(2)} ${x.literal ? "[lit]" : "     "} ${x.meaning}  {${x.cues.join("; ")}}`);
  console.log("  check:", r.check);
}
