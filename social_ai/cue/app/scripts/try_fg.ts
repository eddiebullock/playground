// Development cases drawn from situations the focus groups described
// (docs/research/04_focus_groups.md, paraphrased and changed). A development
// set for prompt work, not an evaluation.
//
//   node --experimental-strip-types scripts/try_fg.ts qwen3:8b

import { reflect, waysToReply, type ReflectInput } from "../src/engine/reflect.ts";

const base = "http://127.0.0.1:11434";
const model = process.argv[2] ?? "qwen3:8b";
const cases: (ReflectInput & { tag: string })[] = [
  { tag: "invitation", mode: "them", person: "Sophie", relation: "someone I just met", channel: "in person",
    context: "We talked for twenty minutes at a community event and found we both like old films. As she was leaving:",
    text: "We should get coffee sometime!" },
  { tag: "pressure", mode: "them", person: "Linda", relation: "neighbour", channel: "in person",
    context: "Linda invited me to a dinner party on Friday. I said Friday does not work for me because I will be too tired after work.",
    text: "Oh come on, you'll come, won't you? It's only a few hours and everyone's expecting you." },
  { tag: "blow-up", mode: "them", person: "Mark", relation: "acquaintance", channel: "in person",
    context: "Mark has just moved house. At a group tea I asked his son about the move and told him about when I moved years ago. Mark cut in loudly.",
    text: "We've heard all this before." },
  { tag: "reschedule", mode: "them", person: "the surgery", relation: "", channel: "phone call",
    context: "My GP phone appointment was booked for Thursday 14th. The receptionist rang to say the doctor is ill and they need to move it.",
    text: "So when would suit you?" },
  { tag: "white-lie", mode: "them", person: "Amy", relation: "friend", channel: "in person",
    context: "I showed Amy my new haircut. I am not sure it suits me. Amy hesitated for a moment.",
    text: "Oh, it looks great!" },
  { tag: "me-clear", mode: "me", person: "Priya", relation: "manager", channel: "work chat (Slack, Teams)",
    context: "Priya asked when I can send the draft report.",
    text: "Friday.", intended: "I will send it on Friday." },
];

for (const c of cases) {
  const r = await reflect(base, model, c);
  console.log(`\n# [${c.tag}] ${c.text}\n${r.clarity} spread=${r.spread.toFixed(2)} ${(r.ms / 1000).toFixed(0)}s`);
  for (const x of r.readings) console.log(`  ${x.p.toFixed(2)} ${x.literal ? "[lit]" : "     "} ${x.meaning}  {${x.cues.join("; ")}}`);
  console.log("  check:", r.check);
  if (r.boundary) console.log("  boundary:", r.boundary);
  if (c.mode === "them" && ["invitation", "pressure"].includes(c.tag)) {
    for (const x of await waysToReply(base, model, c, r)) console.log(`  reply: ${x.reply}  -> ${x.might_land}`);
  }
}
