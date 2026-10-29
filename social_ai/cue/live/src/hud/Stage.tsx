import { useEffect, useRef, useState, type PointerEvent } from "react";
import type { FaceBox } from "../session/vision.ts";
import type { LiveSource } from "../session/useLive.ts";
import { FaceBlur } from "./FaceBlur.tsx";

export type Marker = { x: number; y: number };

type Props = {
  source: LiveSource;
  callStream: MediaStream | null;
  camStream: MediaStream | null;
  fileUrl: string | null;
  fileName: string;
  callVideo: { current: HTMLVideoElement | null };
  camVideo: { current: HTMLVideoElement | null };
  faces: FaceBox[];
  marker: Marker;
  onMarker: (m: Marker) => void;
  onFilePlaying: (el: HTMLVideoElement) => void;
  showMarker: boolean;
  showFaces: boolean;
  blur: number;
  blurFace: boolean;
};

function clock(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const s = Math.floor(sec);
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function Stage({
  source,
  callStream,
  camStream,
  fileUrl,
  fileName,
  callVideo,
  camVideo,
  faces,
  marker,
  onMarker,
  onFilePlaying,
  showMarker,
  showFaces,
  blur,
  blurFace,
}: Props) {
  const stage = useRef<HTMLDivElement | null>(null);
  const drag = useRef(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [paused, setPaused] = useState(true);

  useEffect(() => {
    const el = callVideo.current;
    if (!el) return;
    if (fileUrl) {
      el.srcObject = null;
      el.src = fileUrl;
      el.muted = false;
      el.loop = true;
      void el.play().catch(() => undefined);
      return;
    }
    el.removeAttribute("src");
    el.srcObject = callStream;
    el.muted = true;
    el.loop = false;
    if (callStream) void el.play().catch(() => undefined);
  }, [fileUrl, callStream, callVideo]);

  useEffect(() => {
    const el = camVideo.current;
    if (!el) return;
    el.srcObject = camStream;
    if (camStream) void el.play().catch(() => undefined);
  }, [camStream, camVideo]);

  useEffect(() => {
    const el = callVideo.current;
    if (!el || source !== "file") return;
    const onTime = () => setTime(el.currentTime);
    const onMeta = () => setDuration(el.duration || 0);
    const onPlay = () => {
      setPaused(false);
      onFilePlaying(el);
    };
    const onPause = () => setPaused(true);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("loadedmetadata", onMeta);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    return () => {
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("loadedmetadata", onMeta);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
    };
  }, [source, fileUrl, callVideo, onFilePlaying]);

  function localPoint(e: PointerEvent<HTMLDivElement>): Marker {
    const box = stage.current?.getBoundingClientRect();
    if (!box) return marker;
    return {
      x: Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)),
      y: Math.min(1, Math.max(0, (e.clientY - box.top) / box.height)),
    };
  }

  const other = faces[0];
  const faceCx = other ? other.x + other.w / 2 : 0.5;
  const faceCy = other ? other.y + other.h / 2 : 0.4;
  const hasPicture = Boolean(fileUrl || callStream);

  return (
    <div className="stage-wrap">
      <div
        className="stage"
        ref={stage}
        onPointerDown={(e) => {
          if (!showMarker) return;
          drag.current = true;
          (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
          onMarker(localPoint(e));
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          onMarker(localPoint(e));
        }}
        onPointerUp={() => {
          drag.current = false;
        }}
      >
        <video
          ref={(el) => { callVideo.current = el; }}
          className={`stage-video${blurFace && blur > 0 ? " stage-video-under" : ""}`}
          playsInline
          muted={source !== "file"}
        />
        <FaceBlur video={callVideo} faces={faces} blur={blur} active={blurFace && blur > 0} />
        {!hasPicture && (
          <div className="stage-empty">
            <p>Open a video file, or share a Zoom / Teams tab.</p>
            <p className="muted">
              File playback is analysed as it plays. For a live call, open Zoom or Teams
              in Chrome, then share that tab and tick share audio.
            </p>
          </div>
        )}
        {showFaces &&
          faces.map((f, i) => (
            <div
              key={i}
              className={`face-box${i === 0 ? " face-box-main" : ""}`}
              style={{
                left: `${f.x * 100}%`,
                top: `${f.y * 100}%`,
                width: `${f.w * 100}%`,
                height: `${f.h * 100}%`,
              }}
            />
          ))}
        {showMarker && source === "display" && (
          <>
            <svg className="stage-lines" viewBox="0 0 1 1" preserveAspectRatio="none">
              {other && (
                <line
                  x1={faceCx}
                  y1={faceCy}
                  x2={marker.x}
                  y2={marker.y}
                  stroke="currentColor"
                  strokeWidth="0.004"
                  strokeDasharray="0.02 0.02"
                />
              )}
            </svg>
            <div
              className="cam-mark"
              style={{ left: `${marker.x * 100}%`, top: `${marker.y * 100}%` }}
            >
              <span>Camera</span>
            </div>
          </>
        )}
        <video ref={(el) => { camVideo.current = el; }} className={`pip${camStream ? "" : " pip-off"}`} playsInline muted />
      </div>
      {source === "file" && fileUrl && (
        <div className="playback">
          <button
            type="button"
            onClick={() => {
              const el = callVideo.current;
              if (!el) return;
              if (el.paused) void el.play();
              else el.pause();
            }}
          >
            {paused ? "Play" : "Pause"}
          </button>
          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.1}
            value={Math.min(time, duration || 0)}
            onChange={(e) => {
              const el = callVideo.current;
              if (!el) return;
              el.currentTime = Number(e.target.value);
            }}
            aria-label="Position in the video"
          />
          <span className="playback-time">
            {clock(time)} / {clock(duration)}
          </span>
          <span className="playback-name" title={fileName}>
            {fileName}
          </span>
        </div>
      )}
    </div>
  );
}
