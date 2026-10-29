import { useCallback, useEffect, useRef, useState } from "react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { emotionFromBlendshapes, type EmotionRead } from "../analysis/emotion.ts";
import { gazeFromBlendshapes, type GazeRead } from "../analysis/gaze.ts";
import { emptyTurn, stepTurn, type TurnCue, type TurnMemory } from "../analysis/turntaking.ts";
import { attachElementMeter, attachMeter, type Meter } from "./audio.ts";
import { createSpeechRecorder, revokeClips, type Clip, type SpeechRecorder } from "./record.ts";
import {
  shareCall,
  shareCamera,
  shareMeetingTab,
  shareMic,
  stopStream,
} from "./capture.ts";
import { openMeeting, parseMeetingLink, type MeetKind } from "./meet.ts";
import { createLandmarker, facesFrom, largest, type FaceBox } from "./vision.ts";

export type TabId = "eye" | "turn" | "emotion";
export type LiveSource = "none" | "file" | "display";

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
  detail: "Open a video, or share a call, to hear turns.",
  floorOpen: false,
  longYou: false,
  youMs: 0,
  themMs: 0,
  silenceMs: 0,
};

const idleEmotion: EmotionRead = {
  label: "None",
  hedge: "None",
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
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [source, setSource] = useState<LiveSource>("none");
  const [error, setError] = useState("");
  const [snap, setSnap] = useState<LiveSnap>(idle);
  const [clips, setClips] = useState<Clip[]>([]);
  const [recording, setRecording] = useState(false);
  const callVideo = useRef<HTMLVideoElement | null>(null);
  const camVideo = useRef<HTMLVideoElement | null>(null);
  const callLm = useRef<FaceLandmarker | null>(null);
  const userLm = useRef<FaceLandmarker | null>(null);
  const themMeter = useRef<ReturnType<typeof attachMeter> | null>(null);
  const youMeter = useRef<ReturnType<typeof attachMeter> | null>(null);
  const fileTap = useRef<MediaStream | null>(null);
  const fileUrlRef = useRef<string | null>(null);
  const fileMeterOn = useRef(false);
  const mem = useRef<TurnMemory>(emptyTurn);
  const lastTs = useRef(0);
  const lastUi = useRef(0);
  const micRef = useRef<MediaStream | null>(null);
  const rec = useRef<SpeechRecorder | null>(null);
  const recStarted = useRef(0);
  micRef.current = micStream;

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
    if (source === "file") return;
    themMeter.current?.stop();
    themMeter.current = attachMeter(callStream);
    setSnap((s) => ({
      ...s,
      callAudio: Boolean(callStream && callStream.getAudioTracks().length > 0),
    }));
    return () => themMeter.current?.stop();
  }, [callStream, source]);

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
      if (you.speaking && !rec.current && micRef.current) {
        try {
          rec.current = createSpeechRecorder(micRef.current);
          recStarted.current = ts;
          setRecording(true);
        } catch {
          rec.current = null;
        }
      }
      if (!you.speaking && rec.current) {
        const handle = rec.current;
        rec.current = null;
        setRecording(false);
        const durationMs = Math.max(0, ts - recStarted.current);
        void handle.stop().then((blob) => {
          if (blob.size < 200) return;
          const url = URL.createObjectURL(blob);
          setClips((c) => [
            ...c,
            {
              id: crypto.randomUUID(),
              startedAt: new Date().toISOString(),
              durationMs,
              url,
            },
          ]);
        });
      }
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
        callAudio: fileMeterOn.current || them.hasAudio,
      }));
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      if (rec.current) {
        void rec.current.stop();
        rec.current = null;
      }
    };
  }, []);

  const clearFile = useCallback(() => {
    if (fileUrlRef.current?.startsWith("blob:")) URL.revokeObjectURL(fileUrlRef.current);
    fileUrlRef.current = null;
    setFileUrl(null);
    setFileName("");
    stopStream(fileTap.current);
    fileTap.current = null;
    fileMeterOn.current = false;
  }, []);

  const startDisplay = useCallback(
    async (preferTab = false) => {
      setError("");
      try {
        const stream = preferTab ? await shareMeetingTab() : await shareCall(false);
        stopStream(callStream);
        clearFile();
        setSource("display");
        setCallStream(stream);
        mem.current = emptyTurn;
        stream.getVideoTracks()[0]?.addEventListener("ended", () => {
          setCallStream(null);
          setSource("none");
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [callStream, clearFile],
  );

  const tapFileAudio = useCallback((el: HTMLVideoElement) => {
    if (fileMeterOn.current) return;
    fileMeterOn.current = true;
    el.muted = false;
    el.volume = 1;
    themMeter.current?.stop();
    stopStream(fileTap.current);
    fileTap.current = null;
    themMeter.current = attachElementMeter(el);
    setSnap((s) => ({ ...s, callAudio: true }));
  }, []);

  const startFile = useCallback(
    (file: File) => {
      setError("");
      if (!file.type.startsWith("video/") && !/\.(mp4|webm|mov|m4v|mkv)$/i.test(file.name)) {
        setError("Pick a video file (mp4, webm, mov).");
        return;
      }
      stopStream(callStream);
      setCallStream(null);
      clearFile();
      const url = URL.createObjectURL(file);
      fileUrlRef.current = url;
      setFileUrl(url);
      setFileName(file.name);
      setSource("file");
      mem.current = emptyTurn;
    },
    [callStream, clearFile],
  );

  const startDemo = useCallback(() => {
    setError("");
    stopStream(callStream);
    setCallStream(null);
    clearFile();
    setFileUrl("/demo.mp4");
    setFileName("demo.mp4");
    setSource("file");
    mem.current = emptyTurn;
  }, [callStream, clearFile]);

  const startMeeting = useCallback(
    async (kind: MeetKind, raw: string) => {
      const parsed = parseMeetingLink(kind, raw);
      if ("error" in parsed) {
        setError(parsed.error);
        return;
      }
      setError("");
      openMeeting(parsed.href);
    },
    [],
  );

  const shareMeeting = useCallback(async () => {
    await startDisplay(true);
  }, [startDisplay]);

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

  const clearClips = useCallback(() => {
    setClips((c) => {
      revokeClips(c);
      return [];
    });
  }, []);

  const stopAll = useCallback(() => {
    if (rec.current) {
      void rec.current.stop();
      rec.current = null;
    }
    setRecording(false);
    stopStream(callStream);
    stopStream(camStream);
    stopStream(micStream);
    stopStream(fileTap.current);
    fileTap.current = null;
    clearFile();
    clearClips();
    setCallStream(null);
    setCamStream(null);
    setMicStream(null);
    setSource("none");
    mem.current = emptyTurn;
  }, [callStream, camStream, micStream, clearFile, clearClips]);

  return {
    snap,
    error,
    source,
    callStream,
    camStream,
    micStream,
    fileUrl,
    fileName,
    callVideo,
    camVideo,
    startCall: () => startDisplay(false),
    startFile,
    startDemo,
    startMeeting,
    shareMeeting,
    startCam,
    startMic,
    tapFileAudio,
    clips,
    recording,
    clearClips,
    stopAll,
  };
}
