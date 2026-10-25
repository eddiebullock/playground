import { reflect } from "../src/engine/reflect.ts";
const r = await reflect("http://127.0.0.1:11434", process.argv[2] ?? "qwen3:4b", {
  mode: "me", person: "Jo", relation: "colleague", channel: "work chat (Slack, Teams)",
  context: "Jo asked in the team channel if anyone wants to join for drinks after work on Friday.",
  text: "No.", intended: "I don't want to come this time. Nothing against Jo, I just don't like pubs.",
});
console.log(r.clarity, r.spread.toFixed(2), r.ms + "ms");
for (const x of r.readings) console.log(`  ${x.p.toFixed(2)} ${x.literal ? "[mine]" : "      "} ${x.meaning}`);
console.log("  add:", r.check);
