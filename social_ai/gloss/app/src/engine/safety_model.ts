// Model-based second line for the safety layer. The fixed patterns in
// safety.ts are precise but missed most held-out cases (E9b recall 0.25-0.60,
// 2026-10-26), so each flag is also asked of the local model as a yes/no
// question, scored from first-token probabilities over both option orders
// (deterministic, no generated text). A flag is raised if the patterns fire or
// p(yes) reaches the threshold. Thresholds are set on E9b and tested on E9c.

import { optionDist } from "./llm.ts";
import { BOUNDARY_NOTE, DISTRESS_NOTE, SCAM_NOTE, safetyCheck, type Safety, type SafetyInput } from "./safety.ts";

const SYSTEM = "You check messages for safety risks. Answer with the number of the best option only. No other text.";

// Set on E9b + 384 ordinary inputs (DECISIONS 2026-10-26). 2 means the model
// check is not adopted for that flag (p(yes) cannot reach it).
export const THRESHOLD = { distress: 0.3, scam: 2, boundary: 2 };

function questions(input: SafetyInput): Partial<Record<keyof Safety, string>> {
  const self = [input.context, input.own, input.about].filter((s) => s?.trim()).join("\n");
  const q: Partial<Record<keyof Safety, string>> = {};
  if (self)
    q.distress = `A person wrote this about their own situation:
"${self}"

Does anything they wrote suggest they may be in serious emotional distress or at risk: hopelessness, not wanting to live or to be here, thoughts of suicide or self-harm, feeling worthless or like a burden?`;
  if (input.mode === "them") {
    q.scam = `A person received this message:
"${input.text}"
${self ? `What they know about it: "${self}"\n` : ""}
Is this message likely to be a scam, phishing or fraud attempt (for example asking for money, gift cards, card details, passwords or codes, or to use a link, often with urgency, or pretending to be a company, bank, official body or someone they know)?`;
    if (self)
      q.boundary = `What a person wrote about the situation:
"${self}"
The message they then received:
"${input.text}"

Had the person already said no, or made clear they do not want something, and does the message push, pressure, guilt-trip or ignore that no?`;
  }
  return q;
}

/** p(yes) for each question that applies, averaged over both option orders. */
export async function safetyScores(base: string, model: string, input: SafetyInput, signal?: AbortSignal) {
  const out: Partial<Record<keyof Safety, number>> = {};
  for (const [k, q] of Object.entries(questions(input)) as [keyof Safety, string][]) {
    let yes = 0;
    let used = 0;
    for (const yesFirst of [true, false]) {
      const opts = yesFirst ? "1. Yes\n2. No" : "1. No\n2. Yes";
      const p = await optionDist(base, model, SYSTEM, `${q}\n\n${opts}`, 2, signal);
      if (!p) continue;
      yes += yesFirst ? p[0] : p[1];
      used++;
    }
    out[k] = used ? yes / used : 0;
  }
  return out;
}

const NOTE = { distress: DISTRESS_NOTE, scam: SCAM_NOTE, boundary: BOUNDARY_NOTE };

/** Patterns OR model. */
export async function safetyFull(base: string, model: string, input: SafetyInput, signal?: AbortSignal): Promise<Safety> {
  const s = safetyCheck(input);
  const p = await safetyScores(base, model, input, signal);
  for (const k of ["distress", "scam", "boundary"] as const) {
    if (!s[k] && (p[k] ?? 0) >= THRESHOLD[k]) s[k] = NOTE[k];
  }
  return s;
}
