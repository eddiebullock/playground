// Model-based second line for the safety layer. The fixed patterns in
// safety.ts are precise but missed most held-out cases (E9b recall 0.25-0.60,
// 2026-10-26), so each flag is also asked of the local model as a yes/no
// question, scored from first-token probabilities over both option orders
// (deterministic, no generated text). A flag is raised if the patterns fire or
// p(yes) reaches the threshold. Thresholds are set on E9b and tested on E9c.

import { optionDist } from "./llm.ts";
import { BOUNDARY_NOTE, DISTRESS_NOTE, SCAM_NOTE, safetyCheck, type Safety, type SafetyInput } from "./safety.ts";

const SYSTEM = "You check messages for safety risks. Answer with the number of the best option only. No other text.";

// v1 single-question check, set on E9b + 384 ordinary inputs then tested on
// E9c (failed overall). 2 means the model check is not adopted.
export const THRESHOLD = { distress: 0.3, scam: 2, boundary: 2 };

// Two-gate scores (safetyScoresTwoGate). Distress cut re-checked after the
// E9b cue expansion (still 0.3). Wired into safetyFull after E9e passed.
export const THRESHOLD_TWOGATE = { distress: 0.3, scam: 0.5, boundary: 0.3 };

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

async function pYes(base: string, model: string, q: string, signal?: AbortSignal) {
  let yes = 0;
  let used = 0;
  for (const yesFirst of [true, false]) {
    const opts = yesFirst ? "1. Yes\n2. No" : "1. No\n2. Yes";
    const p = await optionDist(base, model, SYSTEM, `${q}\n\n${opts}`, 2, signal);
    if (!p) continue;
    yes += yesFirst ? p[0] : p[1];
    used++;
  }
  return used ? yes / used : 0;
}

/** p(yes) for each question that applies, averaged over both option orders. */
export async function safetyScores(base: string, model: string, input: SafetyInput, signal?: AbortSignal) {
  const out: Partial<Record<keyof Safety, number>> = {};
  for (const [k, q] of Object.entries(questions(input)) as [keyof Safety, string][]) {
    out[k] = await pYes(base, model, q, signal);
  }
  return out;
}

/** Two-gate wording (DECISIONS 2026-10-26). scam/boundary are min of two questions. */
export function twoGateQuestions(input: SafetyInput) {
  const self = [input.context, input.own, input.about].filter((s) => s?.trim()).join("\n");
  const q: Record<string, string> = {};
  if (self)
    q.distress = `A person wrote this about their own situation:
"${self}"

Does anything they wrote suggest they may be in serious emotional distress or at risk: hopelessness, not wanting to live or to be here, thoughts of suicide or self-harm, feeling worthless or like a burden, saying there is no point in anything, writing a goodbye letter, or giving away things they would not need?`;
  if (input.mode === "them") {
    const known = self ? `What they know about it: "${self}"\n` : "";
    q.scam_ask = `A person received this message:
"${input.text}"
${known}
Does this message ask them to pay money, send gift cards, share card details, passwords or codes, or use a link?`;
    q.scam_cue = `A person received this message:
"${input.text}"
${known}
Does this message use urgency or a deadline, pretend to be a company, bank, official body or someone they know, or pressure them to act now?`;
    if (self) {
      q.boundary_no = `What a person wrote about the situation:
"${self}"

Had they already said no, or made clear they do not want something?`;
      q.boundary_push = `The message they then received:
"${input.text}"

Does this message push, pressure, guilt-trip or ignore a no?`;
    }
  }
  return q;
}

const scoreCache = new Map<string, Promise<Record<string, number>>>();
function scoreKey(input: SafetyInput) {
  return JSON.stringify({ mode: input.mode, text: input.text, context: input.context, own: input.own, about: input.about });
}

export async function safetyScoresTwoGate(base: string, model: string, input: SafetyInput, signal?: AbortSignal) {
  const key = `${model}|${scoreKey(input)}`;
  const hit = scoreCache.get(key);
  if (hit) return hit;
  const work = (async () => {
    const q = twoGateQuestions(input);
    const out: Record<string, number> = {};
    for (const [k, text] of Object.entries(q)) out[k] = await pYes(base, model, text, signal);
    if (out.scam_ask != null && out.scam_cue != null) out.scam = Math.min(out.scam_ask, out.scam_cue);
    if (out.boundary_no != null && out.boundary_push != null) out.boundary = Math.min(out.boundary_no, out.boundary_push);
    return out;
  })();
  scoreCache.set(key, work);
  work.catch(() => {
    if (scoreCache.get(key) === work) scoreCache.delete(key);
  });
  if (scoreCache.size > 80) scoreCache.delete(scoreCache.keys().next().value!);
  return work;
}

const NOTE = { distress: DISTRESS_NOTE, scam: SCAM_NOTE, boundary: BOUNDARY_NOTE };

/** Patterns OR two-gate model. */
export async function safetyFull(base: string, model: string, input: SafetyInput, signal?: AbortSignal): Promise<Safety> {
  const s = safetyCheck(input);
  const p = await safetyScoresTwoGate(base, model, input, signal);
  for (const k of ["distress", "scam", "boundary"] as const) {
    if (!s[k] && (p[k] ?? 0) >= THRESHOLD_TWOGATE[k]) s[k] = NOTE[k];
  }
  return s;
}
