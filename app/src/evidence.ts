// What the app tells people about its own accuracy. Every number here comes
// from docs/DECISIONS.md; update both together.

export const EVIDENCE = {
  model: "qwen3:8b",
  updated: "2026-10-25",
  points: [
    {
      head: "Its first reading matches the intended meaning about half the time.",
      body: "On 52 short stories with irony, hints, white lies and figures of speech that were held back for testing (Hu et al. 2023), the meaning the authors intended was Gloss's top reading 56% of the time, and one of its readings 77% of the time. That is why it shows several readings, not one answer. People choosing from the authors' options averaged 86%.",
    },
    {
      head: "When it says \"Fairly clear\", it is right about 7 times in 10.",
      body: "About 4 in 10 stories came out \"Fairly clear\", and the top reading was right on 18 of those 25. The rest are shown as \"Somewhat open\" or \"Hard to tell\" instead of a confident guess.",
    },
    {
      head: "It keeps the plain reading in view.",
      body: "When the same remarks were rewritten to be sincere, the plain reading was on the list every time (12 of 12) and came first 9 times in 12.",
    },
    {
      head: "It is weakest on kind white lies and figures of speech.",
      body: "It put the intended meaning first on only 4 in 10 of those stories. It often takes a kind white lie at face value, and it can miss what a figure of speech says about the person or thing it describes.",
    },
    {
      head: "It is weak on TV-style sarcasm from text alone.",
      body: "On sarcasm from sitcom scripts, with no tone of voice, it did no better than chance. Tone of voice and the people involved matter, and you know them; it does not.",
    },
  ],
} as const;
