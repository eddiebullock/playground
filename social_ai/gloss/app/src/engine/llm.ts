// Thin Ollama client. Runs in the browser and in Node (both have fetch).
// Everything stays on this machine: the only endpoint is the local Ollama server.

export const DEFAULT_BASE = "http://127.0.0.1:11434";

export type Family = "qwen" | "gemma" | "llama";

export function family(model: string): Family {
  return model.startsWith("gemma") ? "gemma" : model.startsWith("llama") ? "llama" : "qwen";
}

// Raw chat turns. The option distribution is read after an "Answer: " prefill,
// the method validated in experiment E1 (experiments/e1_pragmatics).
const TURN: Record<Family, (s: string, u: string) => string> = {
  qwen: (s, u) =>
    `<|im_start|>system\n${s}<|im_end|>\n<|im_start|>user\n${u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`,
  gemma: (s, u) => `<start_of_turn>user\n${s}\n\n${u}<end_of_turn>\n<start_of_turn>model\n`,
  llama: (s, u) =>
    `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n${s}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n${u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`,
};

type TopLogprob = { token: string; logprob: number };
type GenerateResponse = {
  response: string;
  total_duration?: number;
  logprobs?: { token: string; top_logprobs?: TopLogprob[] }[];
};

async function post<T>(base: string, path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(base + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(`Ollama ${path} ${res.status}: ${await res.text()}`);
  return (await res.json()) as T;
}

// Keep the model loaded between requests for a while, so the second question
// in a session does not pay the load time again. Short enough to free the
// memory soon after the user stops.
const KEEP_ALIVE = "15m";

// Sampling seed for generation. Fixed so runs repeat; scripts that measure
// run-to-run noise change it with setSeed.
let SEED = 0;
export function setSeed(seed: number): void {
  SEED = seed;
}

/**
 * Structured output that matches a JSON schema. With onText, the reply is
 * streamed and onText gets the text so far after each chunk, so the caller
 * can show partial results while the model is still writing.
 */
export async function json<T>(
  base: string,
  model: string,
  system: string,
  user: string,
  schema: object,
  signal?: AbortSignal,
  onText?: (soFar: string) => void,
): Promise<T> {
  const body = {
    model,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    stream: !!onText,
    think: false,
    format: schema,
    keep_alive: KEEP_ALIVE,
    options: { temperature: 0.3, seed: SEED, num_predict: 700 },
  };
  if (!onText) {
    const r = await post<{ message: { content: string } }>(base, "/api/chat", body, signal);
    return JSON.parse(r.message.content) as T;
  }
  const res = await fetch(base + "/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`Ollama /api/chat ${res.status}: ${await res.text()}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (value) buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = done ? "" : lines.pop()!;
    for (const line of lines) {
      if (!line.trim()) continue;
      const chunk = JSON.parse(line) as { message?: { content?: string }; error?: string };
      if (chunk.error) throw new Error(`Ollama /api/chat: ${chunk.error}`);
      text += chunk.message?.content ?? "";
    }
    onText(text);
    if (done) break;
  }
  return JSON.parse(text) as T;
}

/** Probability over options 1..n, from the first answer token. */
export async function optionDist(
  base: string,
  model: string,
  system: string,
  user: string,
  n: number,
  signal?: AbortSignal,
): Promise<number[] | null> {
  const prompt = TURN[family(model)](system, user + "\n\nReply with the option number only.") + "Answer: ";
  const r = await post<GenerateResponse>(
    base,
    "/api/generate",
    {
      model,
      prompt,
      raw: true,
      stream: false,
      logprobs: true,
      top_logprobs: 20,
      keep_alive: KEEP_ALIVE,
      options: { temperature: 0, num_predict: 2, seed: 0 },
    },
    signal,
  );
  const valid = new Set(Array.from({ length: n }, (_, i) => String(i + 1)));
  for (const tok of r.logprobs ?? []) {
    if (tok.token.trim() === "") continue;
    const mass = new Array<number>(n).fill(0);
    for (const alt of tok.top_logprobs ?? []) {
      const t = alt.token.trim().replace(/[).]$/, "");
      if (valid.has(t)) mass[Number(t) - 1] += Math.exp(alt.logprob);
    }
    const z = mass.reduce((a, b) => a + b, 0);
    if (z > 0) return mass.map((m) => m / z);
    break;
  }
  const first = r.response.trim()[0];
  if (first && valid.has(first)) return mass1(n, Number(first) - 1);
  return null;
}

function mass1(n: number, k: number): number[] {
  const p = new Array<number>(n).fill(0);
  p[k] = 1;
  return p;
}

export async function listModels(base: string): Promise<string[]> {
  const res = await fetch(base + "/api/tags");
  if (!res.ok) throw new Error(`Ollama /api/tags ${res.status}`);
  const j = (await res.json()) as { models: { name: string }[] };
  return j.models.map((m) => m.name);
}
