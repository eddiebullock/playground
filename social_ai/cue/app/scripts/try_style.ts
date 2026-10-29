import { styleNote } from "../src/engine/reflect.ts";
for (const m of process.argv.slice(2))
  console.log(m, "->", await styleNote("http://127.0.0.1:11434", m, { relation: "colleague",
    style: "I write short messages and get straight to the point. I rarely use emojis or exclamation marks. If I say no to plans it's usually about the place, not the people." }));
