export type Talker = "them" | "you" | "both" | "silence";

export type TurnMemory = {
  youMs: number;
  themMs: number;
  silenceMs: number;
  lastNonSilence: Talker;
};

export type TurnCue = {
  talk: Talker;
  headline: string;
  detail: string;
  floorOpen: boolean;
  longYou: boolean;
  youMs: number;
  themMs: number;
  silenceMs: number;
};

export const emptyTurn: TurnMemory = {
  youMs: 0,
  themMs: 0,
  silenceMs: 0,
  lastNonSilence: "silence",
};

export const LONG_YOU_MS = 12_000;

export function rmsToSpeech(rms: number, threshold: number): boolean {
  return rms >= threshold;
}

function clock(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}:${String(r).padStart(2, "0")}` : `${s} s`;
}

export function stepTurn(
  mem: TurnMemory,
  themSpeech: boolean,
  youSpeech: boolean,
  dtMs: number,
  longYouMs = 12_000,
  floorMs = 800,
): { mem: TurnMemory; cue: TurnCue } {
  let talk: Talker = "silence";
  if (themSpeech && youSpeech) talk = "both";
  else if (themSpeech) talk = "them";
  else if (youSpeech) talk = "you";

  let youMs = mem.youMs;
  let themMs = mem.themMs;
  let silenceMs = mem.silenceMs;
  let lastNonSilence = mem.lastNonSilence;

  if (talk === "you") {
    youMs += dtMs;
    themMs = 0;
    silenceMs = 0;
    lastNonSilence = "you";
  } else if (talk === "them") {
    themMs += dtMs;
    youMs = 0;
    silenceMs = 0;
    lastNonSilence = "them";
  } else if (talk === "both") {
    youMs += dtMs;
    themMs += dtMs;
    silenceMs = 0;
    lastNonSilence = "both";
  } else {
    silenceMs += dtMs;
  }

  const floorOpen = talk === "silence" && lastNonSilence === "them" && silenceMs >= floorMs;
  const longYou = (talk === "you" || talk === "both") && youMs >= longYouMs;

  let headline = "Quiet";
  let detail = "No one is speaking.";
  if (talk === "them") {
    headline = "They are speaking";
    detail = `Hold. They have had the floor for ${clock(themMs)}.`;
  } else if (talk === "you") {
    headline = longYou ? "Too long" : "You are speaking";
    detail = longYou
      ? `${clock(youMs)}. Pause so they can come in.`
      : clock(youMs);
  } else if (talk === "both") {
    headline = longYou ? "Too long" : "You are both speaking";
    detail = longYou ? `${clock(youMs)}. Pause so they can come in.` : "They have started as well.";
  } else if (floorOpen) {
    headline = "Pause — you could start";
    detail = "They have stopped. A reply would fit here.";
  }

  return {
    mem: { youMs, themMs, silenceMs, lastNonSilence },
    cue: { talk, headline, detail, floorOpen, longYou, youMs, themMs, silenceMs },
  };
}
