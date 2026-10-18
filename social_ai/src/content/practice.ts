import type { Cue } from "../types";

export type WatchLine = {
  start: number;
  end: number;
  speaker: string;
  text: string;
};

export type WatchScene = {
  id: string;
  duration: number;
  lines: WatchLine[];
  cues: Cue[];
};

export const YOUTUBE_WATCH: WatchScene = {
  id: "practice-youtube",
  duration: 24,
  lines: [
    {
      start: 1,
      end: 8.4,
      speaker: "Mara",
      text: "I absolutely love a cancelled train. Best morning I have had all year.",
    },
    {
      start: 8.8,
      end: 16.2,
      speaker: "Mara",
      text: "Before I carry on: a company paid me to say this next bit.",
    },
    {
      start: 16.8,
      end: 22.4,
      speaker: "Mara",
      text: "Right. Back to the station.",
    },
  ],
  cues: [
    {
      id: "yt-comedy",
      start: 2.2,
      end: 8.6,
      level: "calm",
      kind: "tone",
      kindLabel: "Tone",
      title: "Exaggeration for comedy",
      detail: "The praise may not be meant literally.",
      aboutId: "mara",
    },
    {
      id: "yt-advert",
      start: 9.2,
      end: 16.4,
      level: "calm",
      kind: "advert",
      kindLabel: "Advert",
      title: "This part is an advert",
      detail: "She says a company paid for this mention.",
      aboutId: "mara",
    },
  ],
};

export const NETFLIX_WATCH: WatchScene = {
  id: "practice-netflix",
  duration: 26,
  lines: [
    {
      start: 1.2,
      end: 8.2,
      speaker: "Helen",
      text: "You have always been the reliable one. Someone has to be.",
    },
  ],
  cues: [
    {
      id: "nf-unkind",
      start: 2.4,
      end: 9,
      level: "headsup",
      kind: "tone",
      kindLabel: "Tone",
      title: "Smiling, but the words are unkind",
      detail: "This may be passive-aggressive.",
      aboutId: "helen",
    },
    {
      id: "nf-tension",
      start: 9.4,
      end: 17.2,
      level: "headsup",
      kind: "tone",
      kindLabel: "Between them",
      title: "Possible tension",
      detail: "These two may be holding something back.",
      aboutId: "helen",
    },
    {
      id: "nf-recap",
      start: 18.2,
      end: 26,
      level: "calm",
      kind: "topic",
      kindLabel: "Recap",
      title: "2 moments you might want to revisit",
      detail: "The line about being reliable, and the look across the table.",
      aboutId: "helen",
    },
  ],
};

export function watchAt(scene: WatchScene, t: number, quiet: boolean) {
  const line = scene.lines.find((item) => t >= item.start && t < item.end) ?? null;
  const cue = quiet ? null : (scene.cues.find((item) => t >= item.start && t < item.end) ?? null);
  return { line, cue };
}
