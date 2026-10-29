import { useCallback, useEffect, useRef, useState } from "react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { emotionFromBlendshapes, type EmotionRead } from "../analysis/emotion.ts";
import { gazeFromBlendshapes, type GazeRead } from "../analysis/gaze.ts";
import { emptyTurn, stepTurn, type TurnCue, type TurnMemory } from "../analysis/turntaking.ts";
import { attachMeter, type Meter } from "./audio.ts";
import { shareCall, shareCamera, shareMic, stopStream } from "./capture.ts";
import { createLandmarker, facesFrom, largest, type FaceBox } from "./vision.ts";

export type TabId = "eye" | "turn" | "emotion";

export type LiveSnap = {
  callFaces: FaceBox[];
  userFaces: FaceBox[];
  them: Meter;
  you: Meter;
  turn: TurnCue;
  emotion: EmotionRead;
  gaze: GazeRead;
  callAudio: boolean;
  visionReady: boolean;
  visionError: string;
};

const quietTurn: TurnCue = {
  talk: "silence",
  headline: "Quiet",
  detail: "Share the call and your mic to hear turns.",
  floorOpen: false,
  longYou: false,
  youMs: 0,
  themMs: 0,
  silenceMs: 0,
};

const idleEmotion: EmotionRead = {
  label: "none",
  hedge: "Share the call to read a face shape.",
  cues: [],
  confidence: "low",
};

const idleGaze: GazeRead = {
  bin: "unknown",
  headline: "Camera off",
  detail: "Optional: share your camera to guess whether you are looking at the lens.",
};

const idle: LiveSnap = {
  callFaces: [],
  userFaces: [],
  them: { rms: 0, speaking: false, hasAudio: false },
  you: { rms: 0, speaking: false, hasAudio: false },
  turn: quietTurn,
  emotion: idleEmotion,
  gaze: idleGaze,
  callAudio: false,
  visionReady: false,
  visionError: "",
};

export function useLive() {
  const [callStream, setCallStream] = useState<MediaStream | null>(null);
  const [camStream, setCamStream] = useState<MediaStream | null>(null);
  const [micStream, setMicStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState("");
  const [snap, setSnap] = useState<LiveSnap>(idle);
  const callVideo = useRef<HTMLVideoElement | null>(null);
  const camVideo = useRef<HTMLVideoElement | null>(null);
  const callLm = useRef<FaceLandmarker | null>(null);
  const userLm = useRef<FaceLandmarker | null>(null);
  const themMeter = useRef<ReturnType<typeof attachMeter> | null>(null);
  const youMeter = useRef<ReturnType<typeof attachMeter> | null>(null);
  const mem = useRef<TurnMemory>(emptyTurn);
  const lastTs = useRef(0);
  const lastUi = useRef(0);

  useEffect(() => {
    let gone = false;
    createLandmarker()
      .then((a) => {
        if (gone) return;
        callLm.current = a;
        return createLandmarker();
      })
      .then((b) => {
        if (gone || !b) return;
        userLm.current = b;
        setSnap((s) => ({ ...s, visionReady: true }));
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        setSnap((s) => ({ ...s, visionError: msg }));
      });
    return () => {
      gone = true;
    };
  }, []);

  useEffect(() => {
    themMeter.current?.stop();
    themMeter.current = attachMeter(callStream);
    setSnap((s) => ({
      ...s,
      callAudio: Boolean(callStream && callStream.getAudioTracks().length > 0),
    }));
    return () => themMeter.current?.stop();
  }, [callStream]);

  useEffect(() => {
    youMeter.current?.stop();
    youMeter.current = attachMeter(micStream);
    return () => youMeter.current?.stop();
  }, [micStream]);

  useEffect(() => {
    let raf = 0;
    const loop = (ts: number) => {
      raf = requestAnimationFrame(loop);
      const dt = lastTs.current ? ts - lastTs.current : 16;
      lastTs.current = ts;
      const them = themMeter.current?.tick() ?? idle.them;
      const you = youMeter.current?.tick() ?? idle.you;
      const stepped = stepTurn(mem.current, them.speaking, you.speaking, dt);
      mem.current = stepped.mem;
      const callFaces = facesFrom(callLm.current, callVideo.current, ts);
      const userFaces = facesFrom(userLm.current, camVideo.current, ts);
      const other = largest(callFaces);
      const self = largest(userFaces);
      if (ts - lastUi.current < 80) return;
      lastUi.current = ts;
      setSnap((s) => ({
        ...s,
        callFaces,
        userFaces,
        them,
        you,
        turn: stepped.cue,
        emotion: other ? emotionFromBlendshapes(other.blends) : idleEmotion,
        gaze: self ? gazeFromBlendshapes(self.blends) : idleGaze,
        callAudio: them.hasAudio,
      }));
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  const startCall = useCallback(async () => {
    setError("");
    try {
      const stream = await shareCall();
      stopStream(callStream);
      setCallStream(stream);
      stream.getVideoTracks()[0]?.addEventListener("ended", () => setCallStream(null));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [callStream]);

  const startCam = useCallback(async () => {
    setError("");
    try {
      const stream = await shareCamera();
      stopStream(camStream);
      setCamStream(stream);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [camStream]);

  const startMic = useCallback(async () => {
    setError("");
    try {
      const stream = await shareMic();
      stopStream(micStream);
      setMicStream(stream);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [micStream]);

  const stopAll = useCallback(() => {
    stopStream(callStream);
    stopStream(camStream);
    stopStream(micStream);
    setCallStream(null);
    setCamStream(null);
    setMicStream(null);
    mem.current = emptyTurn;
  }, [callStream, camStream, micStream]);

  return {
    snap,
    error,
    callStream,
    camStream,
    micStream,
    callVideo,
    camVideo,
    startCall,
    startCam,
    startMic,
    stopAll,
  };
}
