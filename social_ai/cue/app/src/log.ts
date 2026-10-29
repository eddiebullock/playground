// Local-only record of readings the user chose to rate. Used as the study
// diary: the participant exports it and decides what to share.

import type { ReflectInput, ReflectResult } from "./engine/reflect.ts";

export type Verdict = "right" | "partly" | "wrong" | "unsure";

export type Entry = {
  at: string;
  input: ReflectInput;
  result: ReflectResult;
  verdict: Verdict;
  note: string;
};

const KEY = "cue.entries";
const OLD_KEY = "gloss.entries";

export function loadEntries(): Entry[] {
  try {
    const raw = localStorage.getItem(KEY) ?? localStorage.getItem(OLD_KEY) ?? "[]";
    return JSON.parse(raw) as Entry[];
  } catch {
    return [];
  }
}

/** Adds or updates the entry for the same result; returns the new count. */
export function addEntry(entry: Entry): number {
  const all = loadEntries().filter((e) => !(e.result.ms === entry.result.ms && e.input.text === entry.input.text));
  all.push(entry);
  try {
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    /* storage unavailable */
  }
  return all.length;
}

export function clearEntries() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable */
  }
}

export function exportEntries() {
  const blob = new Blob([JSON.stringify(loadEntries(), null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `cue-entries-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}
