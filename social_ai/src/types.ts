export type AvatarId = "priya" | "sam" | "jordan" | "alex";

export type CueLevel = "calm" | "headsup" | "uncertain";

export type CueKind = "tone" | "idiom" | "turn" | "topic" | "uncertain" | "advert";

export type EarconId = "headsup" | "idiom" | "pause" | "topic";

export type Person = {
  id: string;
  name: string;
  role: string;
  avatar: AvatarId;
};

export type Line = {
  id: string;
  speakerId: string;
  start: number;
  end: number;
  text: string;
};

export type Cue = {
  id: string;
  start: number;
  end: number;
  level: CueLevel;
  kind: CueKind;
  /** Short label beside the icon, e.g. "Tone". */
  kindLabel: string;
  title: string;
  detail: string;
  /** Whose tile the colour cue refers to. */
  aboutId: string;
  audio?: {
    earcon: EarconId;
    /** Plain ASCII, sent to the speech synthesiser. */
    speech?: string;
    caption: string;
    /** Scene time when the on-screen audio indicator hides. */
    until: number;
  };
  haptic?: {
    label: string;
    /** Pulse times, seconds after the cue starts. */
    pulses: number[];
  };
};

export type AgendaItem = {
  id: string;
  label: string;
  /** Scene time when this item becomes current. */
  at: number;
};

export type Scene = {
  id: string;
  title: string;
  duration: number;
  /** Scene time when quiet mode engages in the playthrough. */
  quietAt: number;
  meetingTitle: string;
  clock: string;
  people: Person[];
  agenda: AgendaItem[];
  lines: Line[];
  cues: Cue[];
};

export type ViewMode = "desktop" | "glasses";

export type SceneParams = {
  scene: string;
  view: ViewMode;
  play: boolean;
  hold: boolean;
  t: number;
  quiet: boolean;
  explainer: boolean;
  reduced: boolean;
  motion: boolean;
};
