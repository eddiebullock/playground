import { useEffect, useRef } from "react";
import type { CSSProperties } from "react";
import type { FieldClip, FieldCue, FieldOpening } from "../content/field";
import { fieldCueAt, fieldFocusAt, fieldLookAt, fieldOpeningAt, fieldRapportAt, fieldSpeechAt } from "../content/field";
import { LookHere, RapportFrame, TalkingBar, YourTurn } from "../components/CallCues";
import { LevelChip } from "../components/LevelChip";

type Props = {
  clip: FieldClip;
  t: number;
  playing: boolean;
  seekId: number;
  quiet: boolean;
  sound: boolean;
  eyeContact: boolean;
  reduced: boolean;
  /** Glasses view with no text notes: only turn-taking, Look here and the rapport border. */
  simple?: boolean;
  /** Blur everything except the face of the person Gloss is following. Replaces Look here. */
  focus?: boolean;
  wordmark?: boolean;
  onTime: (time: number) => void;
  onEnded: () => void;
};

type Spot = { x: number; y: number; below: boolean };

export function FieldWatch({ clip, t, playing, seekId, quiet, sound, eyeContact, reduced, simple = false, focus = false, wordmark = true, onTime, onEnded }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cue = fieldCueAt(clip, t, quiet);
  const speech = quiet ? null : fieldSpeechAt(clip, t);
  const opening = quiet ? null : fieldOpeningAt(clip, t);
  const look = quiet ? null : fieldLookAt(clip, t);
  const lastSpot = useRef<Spot | null>(null);
  if (look && speech) {
    const avgY = speech.track.reduce((sum, point) => sum + point[2], 0) / speech.track.length;
    lastSpot.current = { x: look.x, y: look.y, below: avgY <= 52 };
  }
  // In a pause the note stays by the person who just spoke, since that is who the user would answer.
  const anchor = !eyeContact || quiet ? null : look ? lastSpot.current : opening ? lastSpot.current : null;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (Math.abs(video.currentTime - t) > 0.3) video.currentTime = t;
    if (!playing) {
      video.pause();
      return;
    }
    const pending = video.play();
    if (pending) pending.catch(() => undefined);
    let raf = 0;
    const tick = () => {
      onTime(video.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // t is applied only when the user seeks or the clip changes. Playback then owns the clock.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, seekId, clip.src]);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "#0e1218",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: simple ? 18 : 0,
        boxSizing: "border-box",
      }}
    >
      <div
        data-cue={cue?.id ?? (quiet ? "quiet" : "none")}
        data-eye={look && eyeContact ? "true" : "false"}
        data-talking={speech ? "true" : "false"}
        data-turn={opening ? "true" : "false"}
        data-t={t.toFixed(2)}
        style={{
          position: "relative",
          height: "100%",
          maxHeight: "100%",
          aspectRatio: "3 / 4",
          maxWidth: "100%",
          overflow: "hidden",
          background: "#000",
          borderRadius: simple ? 36 : 0,
        }}
      >
        <video
          ref={videoRef}
          key={clip.src}
          src={clip.src}
          muted={!sound}
          playsInline
          onTimeUpdate={(event) => {
            if (!playing) onTime(event.currentTarget.currentTime);
          }}
          onEnded={onEnded}
          style={{ width: "100%", height: "100%", objectFit: "cover", background: "#000" }}
        />
        {focus ? <FocusVeil face={quiet ? null : fieldFocusAt(clip, t)} /> : null}
        <TalkingBar on={Boolean(speech)} reduced={reduced} rounded={false} />
        {look && eyeContact && !focus ? <LookHere x={look.x} y={look.y} /> : null}

        {simple ? (
          <>
            <YourTurn on={Boolean(opening)} top={24} />
            <RapportFrame level={quiet ? "unsure" : fieldRapportAt(clip, t)} reduced={reduced} radius={36} />
          </>
        ) : (
          <>
            {wordmark ? (
              <div
                style={{
                  position: "absolute",
                  top: speech ? 70 : 18,
                  left: 18,
                  zIndex: 6,
                  color: "#F7F5F0",
                  fontFamily: "var(--serif)",
                  fontSize: 22,
                  lineHeight: 1,
                  textShadow: SHADOW,
                }}
              >
                Gloss
              </div>
            ) : null}
            <div
              data-note={anchor ? "face" : "bottom"}
              style={{
                position: "absolute",
                ...noteSpot(anchor),
                transition: reduced ? "none" : "left 0.5s ease-out, top 0.5s ease-out, bottom 0.5s ease-out",
                zIndex: 6,
                borderRadius: 16,
                padding: anchor ? "8px 10px 10px" : "12px 14px 14px",
                background: "transparent",
                border: "1px solid rgba(247, 245, 240, 0.55)",
                color: "#F7F5F0",
                textShadow: SHADOW,
              }}
            >
              {quiet ? (
                <QuietNote />
              ) : cue ? (
                <CueNote cue={cue} opening={opening} compact={Boolean(anchor)} />
              ) : (
                <p style={{ margin: 0, fontSize: 16, lineHeight: 1.4, color: "#F7F5F0" }}>
                  Watching. A note appears if something may help.
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Blurs and softens the whole view, leaving an oval clear over one face.
 * The frame is 3:4, so a face w% of the width is about w% of the height tall.
 */
function FocusVeil({ face }: { face: { x: number; y: number; w: number } | null }) {
  const mask = face
    ? `radial-gradient(${face.w * 0.8}% ${face.w * 0.92}% at ${face.x}% ${face.y + face.w * 0.15}%, transparent 0%, transparent 72%, black 100%)`
    : undefined;
  return (
    <div
      aria-hidden="true"
      data-focus={face ? "face" : "none"}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 2,
        pointerEvents: "none",
        backdropFilter: "blur(5px) brightness(0.9) saturate(0.85)",
        WebkitBackdropFilter: "blur(5px) brightness(0.9) saturate(0.85)",
        maskImage: mask,
        WebkitMaskImage: mask,
      }}
    />
  );
}

/** Keeps light text readable on a see-through note over moving video. */
const SHADOW = "0 1px 2px rgba(0,0,0,0.95), 0 0 6px rgba(0,0,0,0.85), 0 0 14px rgba(0,0,0,0.6)";

const NOTE_WIDTH = 64;

/** Places the note under the speaker's chin, or above their head when the face is low in frame, so it never covers the face. */
function noteSpot(anchor: Spot | null): CSSProperties {
  if (!anchor) return { left: 18, right: 18, bottom: 18 };
  const left = Math.min(100 - NOTE_WIDTH - 3, Math.max(3, anchor.x - NOTE_WIDTH / 2));
  const base: CSSProperties = { left: `${left}%`, width: `${NOTE_WIDTH}%` };
  if (!anchor.below) return { ...base, top: "auto", bottom: `calc(${100 - anchor.y}% + 12%)` };
  return { ...base, top: `calc(${anchor.y}% + 14%)`, bottom: "auto" };
}

function CueNote({ cue, opening, compact }: { cue: FieldCue; opening: FieldOpening | null; compact: boolean }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: compact ? 6 : 8 }}>
      <LevelChip level={opening ? "calm" : cue.level} light />
      <h2 style={{ margin: 0, fontFamily: "var(--serif)", fontWeight: 400, fontSize: compact ? 22 : 28, lineHeight: 1.15 }}>
        {opening ? "Your turn" : cue.title}
      </h2>
      <p style={{ margin: 0, fontSize: compact ? 15 : 17, lineHeight: 1.4, fontWeight: 500 }}>
        {opening ? opening.prompt : cue.detail}
      </p>
    </div>
  );
}

function QuietNote() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: "0.04em" }}>Quiet</div>
      <h2 style={{ margin: 0, fontFamily: "var(--serif)", fontWeight: 400, fontSize: 28, lineHeight: 1.15 }}>Suggestions are paused</h2>
    </div>
  );
}
