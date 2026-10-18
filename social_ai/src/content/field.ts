import type { CueLevel } from "../types";

export type FieldCue = {
  id: string;
  start: number;
  end: number;
  level: CueLevel;
  title: string;
  detail: string;
  /** Optional reminder. Shown only while eye-contact prompts are on. */
  eyeContact?: string;
  status: string;
  eyeStatus?: string;
};

export type FieldJump = {
  label: string;
  t: number;
};

/** [time, x, y]: eye midpoint as a percent of the frame, from on-device face detection at 4 fps. */
export type FacePoint = [number, number, number];

/** Someone else is speaking. Look-here follows their face along the track. */
export type FieldSpeech = {
  start: number;
  end: number;
  track: FacePoint[];
};

/** [time, x, y, faceWidth]: all percent of the frame. */
export type FocusPoint = [number, number, number, number];

/** One person held in focus. Moving to the next span is a cut, never a sweep across the room. */
export type FocusSpan = {
  start: number;
  end: number;
  track: FocusPoint[];
};

export type RapportLevel = "good" | "mixed" | "unsure";

export type FieldRapport = {
  start: number;
  end: number;
  level: RapportLevel;
};

/** A pause where the talking bar drops, so there is a place to speak. */
export type FieldOpening = {
  start: number;
  end: number;
  prompt: string;
};

export type FieldClip = {
  id: "joke" | "intro";
  title: string;
  src: string;
  duration: number;
  cues: FieldCue[];
  jumps: FieldJump[];
  speech: FieldSpeech[];
  openings: FieldOpening[];
  rapport: FieldRapport[];
  /** Whose face stays sharp in the focus view. Covers the whole clip, pauses included. */
  focus?: FocusSpan[];
};

/** Two colleagues joking about work with each other. The joke is not about the camera. */
export const JOKE_CLIP: FieldClip = {
  id: "joke",
  title: "Work joke",
  src: "field/joke.mp4",
  duration: 18.7,
  jumps: [
    { label: "Hello", t: 1.2 },
    { label: "Your turn", t: 3.3 },
    { label: "Between them", t: 8.2 },
    { label: "A gap", t: 13.6 },
    { label: "Rapport", t: 16.4 },
  ],
  speech: [
    {
      start: 0.35,
      end: 2.75,
      track: [[0.38, 62.1, 52.6], [0.62, 60.5, 52.8], [0.88, 59.7, 53.3], [1.12, 59.8, 51.9], [1.38, 58.9, 50.6], [1.62, 59.6, 50.3], [1.88, 58.8, 50.5], [2.12, 60.5, 49.5], [2.38, 65.4, 51.0], [2.62, 70.0, 50.0]],
    },
    {
      start: 4.05,
      end: 5.35,
      track: [[4.12, 51.3, 51.2], [4.38, 35.9, 50.8], [4.62, 33.6, 54.5], [4.88, 36.7, 54.4], [5.12, 41.0, 51.3], [5.38, 42.5, 51.3]],
    },
    {
      start: 6.95,
      end: 7.8,
      track: [[6.88, 41.3, 50.7], [7.12, 38.5, 50.3], [7.38, 37.4, 49.7], [7.62, 35.4, 49.5]],
    },
    {
      start: 7.8,
      end: 9.25,
      track: [[7.88, 59.8, 49.3], [8.12, 60.8, 49.4], [8.38, 61.7, 50.4], [8.62, 61.7, 50.5], [8.88, 62.7, 50.8], [9.12, 69.0, 51.4]],
    },
    {
      start: 9.25,
      end: 12.9,
      track: [[9.12, 24.8, 51.9], [9.38, 29.8, 48.4], [9.62, 35.8, 46.1], [9.88, 40.2, 42.3], [10.12, 33.3, 42.4], [10.38, 10.7, 46.2], [10.62, 13.8, 50.0], [10.88, 37.4, 49.7], [11.12, 48.9, 47.5], [11.38, 57.9, 46.2], [11.62, 63.1, 46.6], [11.88, 64.1, 47.3], [12.12, 61.3, 48.1], [12.38, 58.4, 46.6], [12.62, 57.0, 47.1], [12.88, 54.7, 46.5]],
    },
    {
      start: 14.5,
      end: 15.75,
      track: [[14.38, 44.2, 43.7], [14.62, 53.4, 45.0], [14.88, 60.4, 46.3], [15.12, 61.9, 47.2], [15.38, 63.3, 48.1], [15.62, 62.9, 49.2]],
    },
    {
      start: 15.75,
      end: 18.5,
      track: [[15.88, 90.2, 49.3], [16.12, 83.0, 49.5], [16.38, 78.9, 49.0], [16.62, 78.0, 48.8], [16.88, 77.6, 48.3], [17.12, 77.4, 47.9], [17.38, 72.5, 48.0], [17.62, 62.6, 48.6], [17.88, 50.9, 50.0], [18.12, 34.6, 52.7], [18.38, 22.7, 54.5]],
    },
  ],
  rapport: [{ start: 0, end: 18.7, level: "good" }],
  focus: [
    {
      start: 0,
      end: 7.8,
      track: [[0.12, 59.7, 53.9, 16.6], [0.38, 62.1, 52.6, 18.9], [0.62, 60.5, 52.8, 22.9], [0.88, 59.7, 53.3, 27.8], [1.12, 59.8, 51.9, 27.2], [1.38, 58.9, 50.6, 24.1], [1.62, 59.6, 50.3, 22.5], [1.88, 58.8, 50.5, 21.5], [2.12, 60.5, 49.5, 22.8], [2.38, 65.4, 51.0, 23.2], [2.62, 70.0, 50.0, 22.5], [2.88, 70.7, 51.0, 20.7], [3.12, 70.6, 51.9, 18.6], [3.38, 71.0, 51.8, 19.1], [3.62, 69.2, 51.3, 19.1], [3.88, 62.4, 52.4, 18.2], [4.12, 51.3, 51.2, 19.8], [4.38, 35.9, 50.8, 20.9], [4.62, 33.6, 54.5, 18.3], [4.88, 36.7, 54.4, 19.3], [5.12, 41.0, 51.3, 17.4], [5.38, 42.5, 51.3, 19.0], [5.62, 42.8, 51.1, 18.3], [5.88, 43.9, 51.4, 17.4], [6.12, 44.6, 51.5, 18.3], [6.38, 45.4, 51.5, 17.7], [6.62, 44.2, 51.3, 17.7], [6.88, 41.3, 50.7, 17.9], [7.12, 38.5, 50.3, 17.6], [7.38, 37.4, 49.7, 18.0], [7.62, 35.4, 49.5, 17.0]],
    },
    {
      start: 7.8,
      end: 9.25,
      track: [[7.88, 59.8, 49.3, 12.9], [8.12, 60.8, 49.4, 13.2], [8.38, 61.7, 50.4, 13.2], [8.62, 61.7, 50.5, 12.9], [8.88, 62.7, 50.8, 12.4], [9.12, 69.0, 51.4, 9.9]],
    },
    {
      start: 9.25,
      end: 15.75,
      track: [[9.12, 24.8, 51.9, 27.2], [9.38, 29.8, 48.4, 26.2], [9.62, 35.8, 46.1, 26.7], [9.88, 40.2, 42.3, 29.1], [10.12, 33.3, 42.4, 33.3], [10.38, 10.7, 46.2, 42.2], [10.62, 13.8, 50.0, 44.3], [10.88, 37.4, 49.7, 40.0], [11.12, 48.9, 47.5, 39.2], [11.38, 57.9, 46.2, 36.8], [11.62, 63.1, 46.6, 34.3], [11.88, 64.1, 47.3, 30.0], [12.12, 61.3, 48.1, 26.7], [12.38, 58.4, 46.6, 21.2], [12.62, 57.0, 47.1, 19.4], [12.88, 54.7, 46.5, 16.3], [13.12, 50.6, 47.4, 16.8], [13.38, 42.3, 46.0, 18.0], [13.62, 38.4, 43.5, 18.4], [13.88, 29.9, 40.5, 22.3], [14.12, 31.5, 40.6, 26.0], [14.38, 44.2, 43.7, 25.0], [14.62, 53.4, 45.0, 26.1], [14.88, 60.4, 46.3, 26.8], [15.12, 61.9, 47.2, 26.9], [15.38, 63.3, 48.1, 26.7], [15.62, 62.9, 49.2, 26.8]],
    },
    {
      start: 15.75,
      end: 18.7,
      track: [[15.88, 90.2, 49.3, 15.8], [16.12, 83.0, 49.5, 15.3], [16.38, 78.9, 49.0, 14.9], [16.62, 78.0, 48.8, 14.6], [16.88, 77.6, 48.3, 14.8], [17.12, 77.4, 47.9, 14.8], [17.38, 72.5, 48.0, 14.6], [17.62, 62.6, 48.6, 14.5], [17.88, 50.9, 50.0, 13.2], [18.12, 34.6, 52.7, 13.8], [18.38, 22.7, 54.5, 14.8], [18.62, 28.9, 55.1, 14.0]],
    },
  ],
  openings: [
    { start: 2.75, end: 4.05, prompt: "She has paused. A short hello back can fit here." },
    { start: 5.35, end: 6.95, prompt: "A pause in their joke. A short word can fit here." },
    { start: 12.9, end: 14.5, prompt: "She is looking over. You can say something short." },
  ],
  cues: [
    {
      id: "joke-hello",
      start: 0.4,
      end: 5.2,
      level: "calm",
      title: "Warm hello",
      detail: "She is smiling at you. This looks friendly.",
      eyeContact: "A brief look toward her can be enough.",
      status: "She smiles hello. Gloss reads the tone as friendly.",
      eyeStatus: "She smiles hello. Gloss adds a reminder that a brief look can be enough.",
    },
    {
      id: "joke-between",
      start: 5.2,
      end: 13.2,
      level: "calm",
      title: "Joke between them",
      detail: "They may be joking about work, with each other. It does not seem to be about you.",
      status: "The joke is between them, about work. Gloss says it does not seem to be about you.",
    },
    {
      id: "joke-rapport",
      start: 13.2,
      end: 18.7,
      level: "calm",
      title: "Good rapport",
      detail: "They are easy together. You can stay with it. Nothing else is needed.",
      status: "They are easy together. Gloss says nothing else is needed.",
    },
  ],
};

/** Arriving among new people. Friendly, then a joke up close, then someone returns to work. */
export const INTRO_CLIP: FieldClip = {
  id: "intro",
  title: "Introductions",
  src: "field/intro.mp4",
  duration: 19.13,
  jumps: [
    { label: "Hello", t: 0.6 },
    { label: "Let them talk", t: 5.2 },
    { label: "Your turn", t: 6.9 },
    { label: "Closest", t: 9.2 },
    { label: "Back to work", t: 15.2 },
  ],
  speech: [
    {
      start: 1.25,
      end: 1.55,
      track: [[1.12, 73.2, 39.4], [1.38, 78.9, 40.4]],
    },
    {
      start: 1.55,
      end: 4.0,
      track: [[1.38, 42.8, 62.0], [1.62, 69.5, 64.1], [1.88, 42.2, 65.8], [2.12, 35.4, 67.3], [2.62, 36.5, 68.3], [2.88, 39.0, 66.0], [3.12, 46.2, 64.5], [3.38, 53.5, 63.0], [3.62, 63.4, 61.4], [3.88, 73.4, 60.3]],
    },
    {
      start: 4.0,
      end: 5.5,
      track: [[3.88, 7.4, 61.1], [4.12, 18.8, 60.1], [4.38, 30.9, 59.6], [4.62, 44.0, 59.4], [4.88, 47.3, 59.8], [5.12, 34.5, 60.8], [5.38, 11.4, 62.1]],
    },
    {
      start: 5.5,
      end: 6.45,
      track: [[5.38, 76.1, 62.5], [5.62, 61.7, 62.6], [5.88, 58.3, 62.2], [6.12, 64.2, 61.5], [6.38, 79.4, 61.0]],
    },
    {
      start: 7.35,
      end: 9.75,
      track: [[7.38, 31.5, 49.5], [7.62, 35.5, 48.9], [7.88, 37.3, 48.9], [8.12, 36.7, 48.4], [8.38, 34.8, 48.1], [8.62, 32.5, 46.8], [8.88, 33.0, 47.5], [9.12, 34.5, 48.3], [9.38, 32.0, 51.0], [9.62, 15.1, 51.4]],
    },
    {
      start: 9.75,
      end: 11.25,
      track: [[9.88, 43.1, 58.8], [10.12, 22.5, 58.9], [10.38, 15.0, 58.6], [10.62, 16.2, 57.7], [10.88, 24.7, 56.8], [11.12, 37.3, 55.5]],
    },
    {
      start: 11.25,
      end: 12.55,
      track: [[11.38, 18.4, 46.1], [11.62, 35.4, 46.1], [11.88, 41.8, 46.0], [12.12, 46.8, 43.5], [12.38, 38.5, 43.2]],
    },
    { start: 12.55, end: 13.75, track: [] },
    {
      start: 13.75,
      end: 15.75,
      track: [[13.88, 60.3, 52.2], [14.12, 66.4, 49.4], [14.38, 57.1, 45.6], [14.62, 49.4, 42.5], [14.88, 44.9, 38.9], [15.12, 44.0, 36.3]],
    },
    {
      start: 16.5,
      end: 17.6,
      track: [[16.62, 16.0, 48.9], [16.88, 32.3, 53.8], [17.12, 43.0, 56.0], [17.38, 50.7, 54.6], [17.62, 58.6, 53.9]],
    },
  ],
  rapport: [
    { start: 0, end: 13.8, level: "good" },
    { start: 13.8, end: 16.5, level: "mixed" },
    { start: 16.5, end: 19.13, level: "good" },
  ],
  openings: [
    { start: 0.2, end: 1.25, prompt: "A short hello can fit here." },
    { start: 6.45, end: 7.35, prompt: "They have paused. You can say hello." },
  ],
  cues: [
    {
      id: "intro-hello",
      start: 0.4,
      end: 4.2,
      level: "calm",
      title: "New people",
      detail: "They look friendly. A short hello is enough. This is going well.",
      eyeContact: "A brief look can be enough.",
      status: "New people, and they look friendly. A short hello is enough.",
      eyeStatus: "New people look friendly. Gloss adds a reminder that a brief look can be enough.",
    },
    {
      id: "intro-wait",
      start: 4.2,
      end: 8.6,
      level: "calm",
      title: "They are talking",
      detail: "You can let this bit finish. There is no need to jump in.",
      status: "They are talking. Gloss suggests letting that bit finish.",
    },
    {
      id: "intro-closest",
      start: 8.6,
      end: 13.8,
      level: "calm",
      title: "Joke with you",
      detail: "The person closest is being playful. Rapport here looks good.",
      eyeContact: "A brief look toward the person closest can be enough.",
      status: "The person closest is playful. Gloss says the rapport looks good.",
      eyeStatus: "The person closest is playful. Gloss adds a reminder that a brief look can be enough.",
    },
    {
      id: "intro-work",
      start: 13.8,
      end: 19.13,
      level: "headsup",
      title: "Back to work",
      detail: "He has turned to his screen. He may not want a distraction now.",
      status: "He turns back to his screen. Gloss suggests leaving him to work.",
    },
  ],
};

export function fieldCueAt(clip: FieldClip, t: number, quiet: boolean): FieldCue | null {
  if (quiet) return null;
  return clip.cues.find((cue) => t >= cue.start && t < cue.end) ?? null;
}

export function fieldSpeechAt(clip: FieldClip, t: number): FieldSpeech | null {
  return clip.speech.find((beat) => t >= beat.start && t < beat.end) ?? null;
}

/** Linear interpolation along a [time, ...values] track, holding the ends. */
function sample(track: number[][], t: number): number[] {
  if (t <= track[0][0]) return track[0].slice(1);
  for (let i = 1; i < track.length; i += 1) {
    if (t <= track[i][0]) {
      const a = track[i - 1];
      const b = track[i];
      const k = (t - a[0]) / (b[0] - a[0]);
      return a.slice(1).map((value, index) => value + (b[index + 1] - value) * k);
    }
  }
  return track[track.length - 1].slice(1);
}

/** Where Look here sits now. Null while nobody is tracked, so the marker never guesses. */
export function fieldLookAt(clip: FieldClip, t: number): { x: number; y: number } | null {
  const track = fieldSpeechAt(clip, t)?.track;
  if (!track || track.length === 0) return null;
  const [x, y] = sample(track, t);
  return { x, y };
}

/** The face kept sharp in the focus view: eye midpoint and face width, percent of the frame. */
export function fieldFocusAt(clip: FieldClip, t: number): { x: number; y: number; w: number } | null {
  const span = clip.focus?.find((item) => t >= item.start && t < item.end);
  if (!span || span.track.length === 0) return null;
  const [x, y, w] = sample(span.track, t);
  return { x, y, w };
}

export function fieldRapportAt(clip: FieldClip, t: number): RapportLevel {
  return clip.rapport.find((span) => t >= span.start && t < span.end)?.level ?? "unsure";
}

export function fieldOpeningAt(clip: FieldClip, t: number): FieldOpening | null {
  return clip.openings.find((gap) => t >= gap.start && t < gap.end) ?? null;
}

export function fieldStatus(clip: FieldClip, t: number, quiet: boolean, eyeContact: boolean): string {
  if (quiet) return "Quiet mode. The clip keeps going. The suggestion is paused.";
  const opening = fieldOpeningAt(clip, t);
  if (opening) return "The bar has gone. " + opening.prompt;
  if (fieldSpeechAt(clip, t)) {
    return eyeContact
      ? "Someone else is talking. Look here sits on their face."
      : "Someone else is talking. The colour bar stays across the top.";
  }
  const cue = fieldCueAt(clip, t, false);
  if (!cue) return "Gloss is watching. A note appears if something may help.";
  if (eyeContact && cue.eyeStatus) return cue.eyeStatus;
  return cue.status;
}
