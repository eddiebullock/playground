export type Blend = { categoryName: string; score: number };

export type EmotionRead = {
  label: string;
  hedge: string;
  cues: string[];
  confidence: "low" | "some" | "higher";
};

export function blendScore(blends: Blend[], name: string): number {
  return blends.find((b) => b.categoryName === name)?.score ?? 0;
}

function pair(blends: Blend[], left: string, right: string): number {
  return Math.max(blendScore(blends, left), blendScore(blends, right));
}

type Recipe = { key: string; value: number; label: string; cues: string[] };

export function emotionFromBlendshapes(blends: Blend[]): EmotionRead {
  const smile = pair(blends, "mouthSmileLeft", "mouthSmileRight");
  const frown = pair(blends, "mouthFrownLeft", "mouthFrownRight");
  const browUp = blendScore(blends, "browInnerUp");
  const browDown = pair(blends, "browDownLeft", "browDownRight");
  const jaw = blendScore(blends, "jawOpen");
  const eyeWide = pair(blends, "eyeWideLeft", "eyeWideRight");
  const press = pair(blends, "mouthPressLeft", "mouthPressRight");

  const recipes: Recipe[] = [
    { key: "smile", value: smile, label: "smiling", cues: ["mouth corners lifted"] },
    {
      key: "frown",
      value: frown * 0.7 + browDown * 0.5 + press * 0.2,
      label: "downturned or tense",
      cues: [
        frown > 0.25 ? "mouth corners down" : "",
        browDown > 0.25 ? "brows lowered" : "",
        press > 0.3 ? "lips pressed" : "",
      ].filter(Boolean),
    },
    {
      key: "raised",
      value: browUp * 0.6 + eyeWide * 0.5,
      label: "brows or eyes raised",
      cues: [
        browUp > 0.3 ? "inner brows up" : "",
        eyeWide > 0.3 ? "eyes widened" : "",
      ].filter(Boolean),
    },
    {
      key: "open",
      value: jaw,
      label: "mouth open",
      cues: ["jaw dropped"],
    },
  ];

  recipes.sort((a, b) => b.value - a.value);
  const top = recipes[0];
  const second = recipes[1]?.value ?? 0;
  const margin = top.value - second;

  if (top.value < 0.22) {
    return {
      label: "still",
      hedge: "This face looks relatively still.",
      cues: ["no strong mouth or brow shape"],
      confidence: "low",
    };
  }

  const confidence: EmotionRead["confidence"] =
    margin > 0.18 && top.value > 0.45 ? "higher" : margin > 0.1 ? "some" : "low";

  const cues = top.cues.length ? top.cues : [`strongest cue: ${top.key}`];
  return {
    label: top.label,
    hedge: `This face looks like it might be ${top.label}.`,
    cues,
    confidence,
  };
}
