// UI screenshots with a canned local model, so layout can be checked without
// Ollama: light, dark and phone widths. Needs the dev server running.
//
//   npm run dev &   node scripts/shots.cjs <out-prefix> [url]

const { firefox } = require(process.env.HOME + "/playground/social_ai/node_modules/playwright");

const out = process.argv[2] ?? "shot";
const url = process.argv[3] ?? "http://127.0.0.1:5173/";

const GEN = {
  literal: "Sophie says the two of you should have coffee at some point.",
  readings: [
    { meaning: "Sophie would genuinely like to meet for coffee and hopes you will.", cues: ["you talked for twenty minutes", "you share an interest"] },
    { meaning: "Sophie enjoyed talking and is being friendly, without a firm plan to meet.", cues: ["'sometime' has no date", "said while leaving"] },
    { meaning: "Sophie is hinting that you could be the one to suggest a time.", cues: ["she left it open"] },
  ],
  check_question: "I'd like that. Would next Saturday morning work for you?",
  boundary: "",
};
const REPLIES = {
  replies: [
    { reply: "I'd like that. Are you free next Saturday?", might_land: "Sophie would likely take it as warm and easy to say yes to." },
    { reply: "Yes, let's. Shall I message you?", might_land: "It keeps it open and lets her choose a time." },
    { reply: "(Nothing now; message her in a day or two.)", might_land: "Fine either way. A short message later still reads as friendly." },
  ],
};

async function fake(route) {
  const u = route.request().url();
  if (u.endsWith("/api/tags")) return route.fulfill({ json: { models: [{ name: "qwen3:8b" }, { name: "qwen3:4b" }] } });
  if (u.endsWith("/api/chat")) {
    const body = JSON.parse(route.request().postData());
    const isReplies = body.messages[1].content.includes("ways I could respond");
    const content = JSON.stringify(isReplies ? REPLIES : GEN);
    if (!body.stream) return route.fulfill({ json: { message: { content } } });
    // Streamed: NDJSON in a few chunks, as Ollama sends it.
    const parts = content.match(/[\s\S]{1,40}/g).map((c) => JSON.stringify({ message: { content: c } }));
    return route.fulfill({ contentType: "application/x-ndjson", body: parts.join("\n") + '\n{"done":true}\n' });
  }
  if (u.endsWith("/api/generate")) {
    const n = (JSON.parse(route.request().postData()).prompt.match(/^\d\) /gm) || []).length || 3;
    const lp = [Math.log(0.55), Math.log(0.3), Math.log(0.1), Math.log(0.05)].slice(0, n);
    return route.fulfill({
      json: { response: "1", logprobs: [{ token: "1", top_logprobs: lp.map((l, i) => ({ token: String(i + 1), logprob: l })) }] },
    });
  }
  return route.continue();
}

(async () => {
  const browser = await firefox.launch();
  for (const [name, opts] of [
    ["light", { viewport: { width: 1200, height: 1000 }, colorScheme: "light" }],
    ["dark", { viewport: { width: 1200, height: 1000 }, colorScheme: "dark" }],
    ["phone", { viewport: { width: 390, height: 844 }, colorScheme: "light" }],
  ]) {
    const ctx = await browser.newContext(opts);
    await ctx.addInitScript(() => {
      localStorage.setItem("gloss.seenAbout", "1");
      localStorage.setItem("gloss.readFirst", "1");
      localStorage.setItem("gloss.people", JSON.stringify({ sophie: "Met once at the film club; easy to talk to" }));
    });
    const page = await ctx.newPage();
    await page.route("http://127.0.0.1:11434/**", fake);
    await page.goto(url);
    await page.getByPlaceholder("A name or initial").fill("Sophie");
    await page.locator("select").first().selectOption("someone I just met");
    await page.locator("textarea").first().fill("We talked for twenty minutes at a community event and found we both like old films. As she was leaving:");
    await page.locator("textarea").nth(1).fill("We should get coffee sometime!");
    await page.getByLabel(/Your reading first/).fill("She was just being polite.");
    await page.getByRole("button", { name: "Read it" }).click();
    await page.getByText("Ways you could reply").waitFor({ state: "detached" }).catch(() => {});
    await page.getByRole("button", { name: /Ideas for replying/ }).click();
    await page.getByText("Ways you could reply, if you want to").waitFor();
    await page.screenshot({ path: `${out}-${name}.png`, fullPage: true });
    if (name === "phone") {
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: `${out}-${name}-top.png` });
    }
    await ctx.close();
  }
  await browser.close();
})();
