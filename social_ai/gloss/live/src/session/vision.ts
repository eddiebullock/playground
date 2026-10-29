import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { Blend } from "../analysis/emotion.ts";

export type FaceBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  blends: Blend[];
};

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;

async function makeLandmarker(): Promise<FaceLandmarker> {
  fileset ??= await FilesetResolver.forVisionTasks(WASM);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 4,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
}

export async function createLandmarker(): Promise<FaceLandmarker> {
  return makeLandmarker();
}

export function facesFrom(
  landmarker: FaceLandmarker | null,
  video: HTMLVideoElement | null,
  ts: number,
): FaceBox[] {
  if (!landmarker || !video || video.readyState < 2) return [];
  const out = landmarker.detectForVideo(video, ts);
  const boxes: FaceBox[] = [];
  const dets = out.faceLandmarks ?? [];
  for (let i = 0; i < dets.length; i++) {
    const pts = dets[i];
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    const cats = out.faceBlendshapes?.[i]?.categories ?? [];
    boxes.push({
      x: minX,
      y: minY,
      w: Math.max(0, maxX - minX),
      h: Math.max(0, maxY - minY),
      blends: cats.map((c) => ({ categoryName: c.categoryName, score: c.score })),
    });
  }
  boxes.sort((a, b) => b.w * b.h - a.w * a.h);
  return boxes;
}

export function largest(faces: FaceBox[]): FaceBox | null {
  return faces[0] ?? null;
}
