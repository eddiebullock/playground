// Thin Ollama client. Runs in the browser and in Node (both have fetch).
// Everything stays on this machine: the only endpoint is the local Ollama server.

export const DEFAULT_BASE = "http://127.0.0.1:11434";

export type Family = "qwen" | "gemma";

export function family(model: string): Family {
  return model.startsWith("gemma") ? "gemma" : "qwen";
}

// Raw chat turns. The option distribution is read after an "Answer: " prefill,
// the method validated in experiment E1 (experiments/e1_pragmatics).
const TURN: Record<Family, (s: string, u: string) => string> = {
  qwen: (s, u) =>
    `<|im_start|>system\n${s}<|im_end|>\n<|im_start|>user\n${u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`,
  gemma: (s, u) => `<start_of_turn>user\n${s}\n\n${u}<end_of_turn>\n<start_of_turn>model\n`,
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

/** Structured output that matches a JSON schema. */
export async function json<T>(
  base: string,
  model: string,
  system: string,
  user: string,
  schema: object,
  signal?: AbortSignal,
): Promise<T> {
  const r = await post<{ message: { content: string } }>(
    base,
    "/api/chat",
    {
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      stream: false,
      think: false,
      format: schema,
      options: { temperature: 0.3, seed: 0, num_predict: 700 },
    },
    signal,
  );
  return JSON.parse(r.message.content) as T;
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
