// Gloss Reflect: possible readings of what someone meant, with an honest
// ambiguity rating.
//
// Two steps:
// 1. Generate 2-4 candidate readings. The first is always the sincere,
//    literal one, so Gloss never assumes a hidden meaning.
// 2. Score the candidates as a multiple-choice question from first-token
//    probabilities, averaged over rotated option orders to cancel position
//    bias. This is the mechanism measured in E1, so the confidence shown to
//    the user has a measured calibration behind it.

import { json, optionDist } from "./llm.ts";

export type Mode = "them" | "me";

export type ReflectInput = {
  mode: Mode;
  /** The line or message being read. */
  text: string;
  /** What led up to it: earlier messages, or what was happening. */
  context?: string;
  /** Who said it (them) or who it is for (me). */
  person?: string;
  /** e.g. "manager", "friend", "partner", "someone I just met". */
  relation?: string;
  /** e.g. "text message", "email", "in person", "video call". */
  channel?: string;
  /** Me mode: what the user means by it. */
  intended?: string;
};

export type Reading = { meaning: string; cues: string[]; p: number; literal: boolean };

export type Clarity = "clear" | "some" | "high";

export type ReflectResult = {
  literal: string;
  readings: Reading[];
  clarity: Clarity;
  /** Normalised entropy of the reading distribution, 0 (certain) to 1 (flat). */
  spread: number;
  /** True when Gloss should not pick a favourite. */
  abstain: boolean;
  check: string;
  model: string;
  ms: number;
};

// Per-model temperature for the stance scorer, fitted by 5-fold CV on the E1
// items (experiments/e1_pragmatics/tempscale.py). Raw option probabilities are
// overconfident; scaling brings expected calibration error for qwen3:8b from
// 0.14 to 0.04. These were fitted on test items, not on Gloss's own generated
// readings, so whether they transfer is checked in Study 5 (diary ratings by
// clarity band).
const TEMPERATURE: Record<string, number> = { "qwen3:8b": 3.4, "qwen3:4b": 1.8 };

// Thresholds on the calibrated top reading. With qwen3:8b and the stance
// scorer on the E1 items, >= 0.7 covered 71% of items at 0.91 accuracy
// (humans 0.86).
export const CLEAR_AT = 0.7;
export const ABSTAIN_BELOW = 0.5;

export function calibrate(p: number[], model: string): number[] {
  const t = TEMPERATURE[model] ?? 1;
  const q = p.map((x) => Math.max(x, 1e-9) ** (1 / t));
  const z = q.reduce((a, b) => a + b, 0);
  return q.map((x) => x / z);
}

const SYSTEM_GEN = [
  "You help people understand messages and conversations.",
  "You offer possible readings of what someone meant. Many messages mean exactly what they say, so never assume a hidden meaning.",
  "Write in plain, concrete English. Be brief. Do not judge either person.",
  "Never give advice about body language, eye contact, or how to look or seem.",
].join(" ");

// Pragmatic-stance scorer (E1b). On the E1 items with qwen3:8b it lifted
// accuracy 0.69 -> 0.78 and irony 0.51 -> 0.85. On sincere minimal pairs it cost
// 0.98 -> 0.90, mostly on remarks that sound conventionally sarcastic
// (docs/DECISIONS.md).
const SYSTEM_SCORE =
  "People often mean something other than their literal words: irony, sarcasm, politeness, hints, jokes, " +
  "white lies. They also often mean exactly what they say. Use the situation to decide what the speaker most " +
  "plausibly meant. Answer with the number of the best option only. No other text.";

type Generated = {
  literal: string;
  readings: { meaning: string; cues: string[] }[];
  check_question: string;
};

const SCHEMA = {
  type: "object",
  properties: {
    literal: { type: "string" },
    readings: {
      type: "array",
      minItems: 2,
      maxItems: 4,
      items: {
        type: "object",
        properties: {
          meaning: { type: "string" },
          cues: { type: "array", items: { type: "string" }, maxItems: 3 },
        },
        required: ["meaning", "cues"],
      },
    },
    check_question: { type: "string" },
  },
  required: ["literal", "readings", "check_question"],
};

function scene(input: ReflectInput): string {
  const who = input.person?.trim() || (input.mode === "them" ? "They" : "I");
  const lines: string[] = [
    `Roles: "I" and "me" are the person asking for help. ${input.mode === "them" ? who : input.person || "They"} is the other person.`,
  ];
  if (input.relation) lines.push(`Relationship: ${input.mode === "them" ? `${who} is my ${input.relation}` : `${input.person || "They"} is my ${input.relation}`}.`);
  if (input.channel) lines.push(`Where: ${input.channel}.`);
  if (input.context?.trim()) lines.push(`What came before:\n${input.context.trim()}`);
  if (input.mode === "them") lines.push(`${who} said: "${input.text.trim()}"`);
  else lines.push(`I want to send this to ${input.person || "them"}: "${input.text.trim()}"`);
  return lines.join("\n\n");
}

function generationPrompt(input: ReflectInput): string {
  const s = scene(input);
  if (input.mode === "them") {
    return `${s}

Give:
- literal: one sentence saying what the words themselves say.
- readings: 2 to 4 different things they could have meant. The FIRST reading is the plain one: they meant exactly what they said, in its ordinary, neutral or kind sense, stated concretely. Every other reading must change what I would understand or do next (for example: they are annoyed, they are joking, they are hinting at a request). Never give two readings that mean nearly the same thing. If the situation does not genuinely support another reading, give only one alternative. Each reading is one sentence starting with "${input.person?.trim() || "They"}", says what they meant about me or the situation, and has up to 3 short cues: words or facts from the situation that support it.
- check_question: one short, friendly, direct question I could ask them to check what they meant.`;
  }
  const intended = input.intended?.trim();
  return `${s}
${intended ? `\nWhat I mean by it: ${intended}\n` : ""}
Give:
- literal: one sentence saying what my words themselves say.
- readings: 2 to 4 different ways ${input.person || "they"} could read my message, most likely first. The FIRST reading must be the meaning I intend${intended ? "" : " (my words taken sincerely)"}. Then add readings a reasonable person might actually take from it, about the content or about the tone (for example warm, neutral, brief, curt, annoyed, joking). Different people read short or direct messages differently, so include a tone reading when the message is short or direct. Each reading is one sentence starting with "They might think" and has up to 3 short cues: the words in my message, or what is missing from it, that could lead to that reading.
- check_question: one short sentence I could add, if I want to, that makes my meaning explicit in my own direct style.`;
}

function scoringPrompt(input: ReflectInput, options: string[]): string {
  const q =
    input.mode === "them"
      ? `What did ${input.person?.trim() || "they"} most likely want to convey?`
      : `How will ${input.person?.trim() || "they"} most likely read this message?`;
  return `Scenario:\n${scene(input)}\n\n${q}\nOptions:\n${options.map((o, i) => `${i + 1}) ${o}`).join("\n")}`;
}

export function normEntropy(p: number[]): number {
  if (p.length < 2) return 0;
  const h = -p.reduce((a, x) => (x > 0 ? a + x * Math.log(x) : a), 0);
  return h / Math.log(p.length);
}

/** Score options from first-token probabilities, averaged over cyclic rotations. */
export async function scoreOptions(
  base: string,
  model: string,
  input: ReflectInput,
  options: string[],
  rotations = Math.min(3, options.length),
  signal?: AbortSignal,
): Promise<number[]> {
  const n = options.length;
  const acc = new Array<number>(n).fill(0);
  let used = 0;
  for (let r = 0; r < rotations; r++) {
    const order = options.map((_, i) => (i + r) % n);
    const p = await optionDist(base, model, SYSTEM_SCORE, scoringPrompt(input, order.map((i) => options[i])), n, signal);
    if (!p) continue;
    order.forEach((orig, pos) => (acc[orig] += p[pos]));
    used++;
  }
  if (!used) return new Array<number>(n).fill(1 / n);
  return acc.map((x) => x / used);
}

export async function reflect(
  base: string,
  model: string,
  input: ReflectInput,
  signal?: AbortSignal,
  onStep?: (step: "writing" | "weighing") => void,
): Promise<ReflectResult> {
  const t0 = Date.now();
  onStep?.("writing");
  const gen = await json<Generated>(base, model, SYSTEM_GEN, generationPrompt(input), SCHEMA, signal);
  const raw = gen.readings.filter((r) => r.meaning?.trim()).slice(0, 4);
  onStep?.("weighing");
  const p = calibrate(await scoreOptions(base, model, input, raw.map((r) => r.meaning), undefined, signal), model);
  const readings = raw
    .map((r, i) => ({ meaning: r.meaning.trim(), cues: r.cues ?? [], p: p[i], literal: i === 0 }))
    .sort((a, b) => b.p - a.p);
  const top = readings[0]?.p ?? 0;
  const clarity: Clarity = top >= CLEAR_AT ? "clear" : top >= ABSTAIN_BELOW ? "some" : "high";
  return {
    literal: gen.literal,
    readings,
    clarity,
    spread: normEntropy(readings.map((r) => r.p)),
    abstain: top < ABSTAIN_BELOW,
    check: gen.check_question,
    model,
    ms: Date.now() - t0,
  };
}

const SCHEMA_NOTE = {
  type: "object",
  properties: { note: { type: "string" } },
  required: ["note"],
};

/**
 * A short note, in the user's own voice, that explains how they communicate.
 * Optional and user-initiated. Disclosure of communication style has been
 * found to improve rapport between autistic and non-autistic people
 * (Crompton et al., Nature Human Behaviour 2025).
 */
export async function styleNote(
  base: string,
  model: string,
  input: { style: string; relation?: string; channel?: string },
  signal?: AbortSignal,
): Promise<string> {
  const user = `Write a short, friendly note (one or two sentences, first person) that I could share with ${
    input.relation ? `a ${input.relation}` : "people I message"
  }${input.channel ? ` on ${input.channel}` : ""} to explain how I communicate.

How I communicate, in my words: ${input.style}

Rewrite it as a message to them: speak to the reader as "you", and say what my style does and does not mean for them (for example "if I reply briefly, it doesn't mean I'm annoyed with you"). Keep my meaning and my directness. Do not apologise for it, do not mention any diagnosis unless I did, and do not promise to change. Plain words. Do not copy my description word for word.`;
  const r = await json<{ note: string }>(base, model, SYSTEM_GEN, user, SCHEMA_NOTE, signal);
  return r.note.trim();
}
