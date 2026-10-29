import { blendScore, type Blend } from "./emotion.ts";

export type GazeBin = "camera" | "screen" | "away" | "unknown";

export type GazeRead = {
  bin: GazeBin;
  headline: string;
  detail: string;
};

function pair(blends: Blend[], a: string, b: string): number {
  return (blendScore(blends, a) + blendScore(blends, b)) / 2;
}

/** User-facing camera: are they looking toward the lens, down at the screen, or away. */
export function gazeFromBlendshapes(blends: Blend[]): GazeRead {
  if (!blends.length) {
    return {
      bin: "unknown",
      headline: "No face on your camera yet",
      detail: "Sit in frame if you want this guess.",
    };
  }
  const down = pair(blends, "eyeLookDownLeft", "eyeLookDownRight");
  const up = pair(blends, "eyeLookUpLeft", "eyeLookUpRight");
  const inward = pair(blends, "eyeLookInLeft", "eyeLookInRight");
  const outward = pair(blends, "eyeLookOutLeft", "eyeLookOutRight");
  const sideways = Math.max(inward, outward);
  const peak = Math.max(down, up, sideways);

  if (down > 0.32 && down >= up && down >= sideways) {
    return {
      bin: "screen",
      headline: "Looking toward the screen",
      detail: "They will see you looking down, not at them. The camera is the ring.",
    };
  }
  if (peak < 0.22) {
    return {
      bin: "camera",
      headline: "Looking toward the camera",
      detail: "This is the direction that reads as looking at them.",
    };
  }
  return {
    bin: "away",
    headline: "Looking away from the camera",
    detail: "Fine if you want it. They will not read this as being looked at.",
  };
}
