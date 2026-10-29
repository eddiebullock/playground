import { useEffect, useRef, type PointerEvent } from "react";
import type { FaceBox } from "../session/vision.ts";

export type Marker = { x: number; y: number };

type Props = {
  callStream: MediaStream | null;
  camStream: MediaStream | null;
  callVideo: { current: HTMLVideoElement | null };
  camVideo: { current: HTMLVideoElement | null };
  faces: FaceBox[];
  marker: Marker;
  onMarker: (m: Marker) => void;
  showMarker: boolean;
  showFaces: boolean;
};

export function Stage({
  callStream,
  camStream,
  callVideo,
  camVideo,
  faces,
  marker,
  onMarker,
  showMarker,
  showFaces,
}: Props) {
  const stage = useRef<HTMLDivElement | null>(null);
  const drag = useRef(false);

  useEffect(() => {
    const el = callVideo.current;
    if (!el) return;
    el.srcObject = callStream;
    if (callStream) void el.play().catch(() => undefined);
  }, [callStream, callVideo]);

  useEffect(() => {
    const el = camVideo.current;
    if (!el) return;
    el.srcObject = camStream;
    if (camStream) void el.play().catch(() => undefined);
  }, [camStream, camVideo]);

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

  return (
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
      <video ref={(el) => { callVideo.current = el; }} className="stage-video" playsInline muted />
      {!callStream && (
        <div className="stage-empty">
          <p>Share the call or the video window.</p>
          <p className="muted">Chrome tab + share audio for turn-taking. A Zoom window is usually picture only.</p>
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
      {showMarker && (
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
  );
}
