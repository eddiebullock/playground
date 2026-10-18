import type { Scene } from "../types";

/**
 * Scene A — a work video meeting on the fictional app Harbor.
 * Times are seconds from the start of the playthrough.
 * Cues are suggestions about other people, never about Alex's body.
 */
export const SCENE_A: Scene = {
  id: "A",
  title: "Work video meeting",
  duration: 34.5,
  quietAt: 31,
  meetingTitle: "Delivery check-in",
  clock: "10:41",
  people: [
    { id: "priya", name: "Priya", role: "Manager", avatar: "priya" },
    { id: "sam", name: "Sam", role: "Design", avatar: "sam" },
    { id: "jordan", name: "Jordan", role: "Finance", avatar: "jordan" },
    { id: "alex", name: "Alex", role: "You", avatar: "alex" },
  ],
  agenda: [
    { id: "launch", label: "Launch date", at: 0 },
    { id: "review", label: "Friday review", at: 14 },
    { id: "budget", label: "Contractor budget", at: 26.4 },
  ],
  lines: [
    {
      id: "l1",
      speakerId: "priya",
      start: 0.6,
      end: 4.8,
      text: "The client moved the launch. We are going to miss Friday.",
    },
    {
      id: "l2",
      speakerId: "sam",
      start: 5.2,
      end: 8.8,
      text: "Oh great, that\u2019s exactly what we needed.",
    },
    {
      id: "l3",
      speakerId: "priya",
      start: 9.2,
      end: 13.6,
      text: "Alright. Let\u2019s take this offline and sort the date there.",
    },
    {
      id: "l4",
      speakerId: "priya",
      start: 14.0,
      end: 17.6,
      text: "Alex, can your team still make a Friday review?",
    },
    {
      id: "l5",
      speakerId: "alex",
      start: 21.4,
      end: 25.2,
      text: "Friday is tight. Monday would be safer for us.",
    },
    {
      id: "l6",
      speakerId: "jordan",
      start: 25.8,
      end: 29.6,
      text: "Then the next item is the contractor budget.",
    },
  ],
  cues: [
    {
      id: "sarcasm",
      start: 6.2,
      end: 10.8,
      level: "headsup",
      kind: "tone",
      kindLabel: "Tone",
      title: "Possibly sarcastic",
      detail: "Tone doesn\u2019t match the words",
      aboutId: "sam",
      audio: {
        earcon: "headsup",
        speech: "Possibly sarcastic.",
        caption: "Possibly sarcastic",
        until: 9.4,
      },
    },
    {
      id: "idiom",
      start: 10.8,
      end: 15.6,
      level: "calm",
      kind: "idiom",
      kindLabel: "Idiom",
      title: "Idiom",
      detail: "\u201CTake this offline\u201D means discuss later, privately",
      aboutId: "priya",
      audio: {
        earcon: "idiom",
        caption: "Idiom",
        until: 12.6,
      },
    },
    {
      id: "pause",
      start: 18.0,
      end: 21.4,
      level: "headsup",
      kind: "turn",
      kindLabel: "Turn-taking",
      title: "Pause",
      detail: "Priya may be waiting for your answer",
      aboutId: "priya",
      haptic: {
        label: "Haptic \u00b7 two short pulses",
        pulses: [0.15, 0.55],
      },
    },
    {
      id: "topic",
      start: 27.0,
      end: 31.0,
      level: "calm",
      kind: "topic",
      kindLabel: "Topic",
      title: "Topic changed",
      detail: "Now discussing the budget",
      aboutId: "jordan",
      audio: {
        earcon: "topic",
        caption: "Topic changed",
        until: 28.8,
      },
    },
  ],
};

export const HONESTY_TAG = "Concept prototype \u2014 simulated output";

export const EXPLAINER = {
  title: "What is this?",
  lead: "This is Gloss, Alex\u2019s personal assistant for conversations.",
  paragraphs: [
    "It may offer a short note when tone, an idiom, or a pause could be easy to miss. Notes are suggestions. They are not instructions, and they are not a judgement of anyone here.",
    "Nothing is recorded. What the assistant hears and sees is processed on this device and is not stored.",
    "You can ignore it.",
  ],
  footer: "Alex can show this card, or send the same words.",
};

export const PRIVACY_LINE = "Nothing is recorded \u00b7 processed on device";
