/**
 * E11 harness. Dry-run with Ollama is allowed. Paid providers run only
 * when the matching env key is set AND E11_ALLOW_PAID=1 (Eddie).
 *
 *   node --experimental-strip-types run.ts --split dev --provider ollama --model qwen3:8b --persona generic --limit 1
 */
import { readFileSync, appendFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

type Item = {
  id: string;
  harm: "distress" | "scam" | "boundary";
  gold: "tp" | "hn";
  voice: "own" | "received";
  context: string;
  text: string;
};

type Prompts = {
  user_template: string;
  user_template_own: string;
  personas: Record<string, string>;
};

const here = dirname(fileURLToPath(import.meta.url));
const prompts = JSON.parse(readFileSync(join(here, "prompts.json"), "utf8")) as Prompts;

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function loadSplit(split: string): Item[] {
  const file = split === "holdout" ? "items_holdout.json" : "items_dev.json";
  const raw = JSON.parse(readFileSync(join(here, file), "utf8")) as { items: Item[] };
  return raw.items;
}

function renderUser(item: Item): string {
  const tpl = item.voice === "own" ? prompts.user_template_own : prompts.user_template;
  const context = item.context ? `Context:\n${item.context}\n` : "";
  return tpl.replace("{{context}}", context).replace("{{text}}", item.text);
}

type Chat = { system: string; user: string };

async function ollama(model: string, chat: Chat): Promise<string> {
  const res = await fetch("http://127.0.0.1:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      stream: false,
      options: { temperature: 0 },
      messages: [
        { role: "system", content: chat.system },
        { role: "user", content: chat.user },
      ],
    }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status} ${await res.text()}`);
  const json = (await res.json()) as { message?: { content?: string } };
  return json.message?.content ?? "";
}

async function openaiChat(model: string, chat: Chat, url: string, key: string): Promise<string> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [
        { role: "system", content: chat.system },
        { role: "user", content: chat.user },
      ],
    }),
  });
  if (!res.ok) throw new Error(`openai-compat ${res.status} ${await res.text()}`);
  const json = (await res.json()) as { choices?: { message?: { content?: string } }[] };
  return json.choices?.[0]?.message?.content ?? "";
}

async function anthropicChat(model: string, chat: Chat, key: string): Promise<string> {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      max_tokens: 1024,
      temperature: 0,
      system: chat.system,
      messages: [{ role: "user", content: chat.user }],
    }),
  });
  if (!res.ok) throw new Error(`anthropic ${res.status} ${await res.text()}`);
  const json = (await res.json()) as { content?: { text?: string }[] };
  return json.content?.map((b) => b.text ?? "").join("") ?? "";
}

async function complete(provider: string, model: string, chat: Chat): Promise<string> {
  if (provider === "ollama") return ollama(model, chat);
  const paid = process.env.E11_ALLOW_PAID === "1";
  if (!paid) throw new Error("Refusing paid provider (set E11_ALLOW_PAID=1 and the vendor key).");
  if (provider === "openai") {
    const key = process.env.OPENAI_API_KEY;
    if (!key) throw new Error("OPENAI_API_KEY missing");
    return openaiChat(model, chat, "https://api.openai.com/v1/chat/completions", key);
  }
  if (provider === "xai") {
    const key = process.env.XAI_API_KEY;
    if (!key) throw new Error("XAI_API_KEY missing");
    return openaiChat(model, chat, "https://api.x.ai/v1/chat/completions", key);
  }
  if (provider === "google") {
    const key = process.env.GOOGLE_API_KEY;
    if (!key) throw new Error("GOOGLE_API_KEY missing");
    const url = `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`;
    return openaiChat(model, chat, url, key);
  }
  if (provider === "anthropic") {
    const key = process.env.ANTHROPIC_API_KEY;
    if (!key) throw new Error("ANTHROPIC_API_KEY missing");
    return anthropicChat(model, chat, key);
  }
  throw new Error(`unknown provider ${provider}`);
}

async function main() {
  const split = arg("split", "dev");
  if (split === "holdout") {
    const prereq = join(here, "HOLD_OUT_RELEASED");
    if (!existsSync(prereq)) {
      throw new Error("Holdout is sealed. Create HOLD_OUT_RELEASED after the OSF snapshot.");
    }
  }
  const provider = arg("provider", "ollama");
  const model = arg("model", "qwen3:8b");
  const persona = arg("persona", "generic");
  const limit = Number(arg("limit", "0"));
  const harmFilter = arg("harm", "");
  const system = prompts.personas[persona];
  if (!system) throw new Error(`unknown persona ${persona}`);

  const items = loadSplit(split).filter((it) => !harmFilter || it.harm === harmFilter);
  const slice = limit > 0 ? items.slice(0, limit) : items;
  const outDir = join(here, "runs");
  mkdirSync(outDir, { recursive: true });
  const outFile = join(outDir, `${split}_${provider}_${model.replace(/[^\w.-]+/g, "_")}_${persona}.jsonl`);

  for (const item of slice) {
    const user = renderUser(item);
    const started = Date.now();
    const output = await complete(provider, model, { system, user });
    const row = {
      item_id: item.id,
      harm: item.harm,
      gold: item.gold,
      split,
      provider,
      model,
      persona,
      ms: Date.now() - started,
      output,
    };
    appendFileSync(outFile, JSON.stringify(row) + "\n");
    console.log(item.id, `${row.ms}ms`, output.slice(0, 80).replace(/\s+/g, " "));
  }
  console.log("wrote", outFile);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
