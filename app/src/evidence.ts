// What the app tells people about its own accuracy. Every number here comes
// from docs/DECISIONS.md; update both together.

export const EVIDENCE = {
  model: "qwen3:8b",
  updated: "2026-10-25",
  points: [
    {
      head: "On published test stories, it picks the intended meaning about 4 times in 5.",
      body: "On 169 short stories with irony, hints, white lies and figures of speech (Hu et al. 2023), it chose the meaning the authors intended 78% of the time. 374 people answering the same stories averaged 86%.",
    },
    {
      head: "When it says \"Fairly clear\", it is usually right.",
      body: "On those stories, about 7 in 10 came out \"Fairly clear\", and 91% of those were right. The rest are shown as \"Somewhat open\" or \"Hard to tell\" instead of a confident guess.",
    },
    {
      head: "It does not assume a hidden meaning.",
      body: "When the same remarks were rewritten to be sincere, it took them at face value 90% of the time. The plain reading is always on the list.",
    },
    {
      head: "It is weak on TV-style sarcasm from text alone.",
      body: "On sarcasm from sitcom scripts, with no tone of voice, it did no better than chance. Tone of voice and the people involved matter, and you know them; it does not.",
    },
  ],
} as const;
