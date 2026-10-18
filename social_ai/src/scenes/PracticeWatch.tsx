import type { CSSProperties } from "react";
import type { Cue } from "../types";
import type { WatchLine, WatchScene } from "../content/practice";
import { watchAt } from "../content/practice";
import { GlossPanel } from "../components/GlossPanel";

type Variant = "youtube" | "netflix";

type Props = {
  variant: Variant;
  scene: WatchScene;
  t: number;
  quiet: boolean;
  reduced: boolean;
  frozen: boolean;
  customStill: string | null;
  onExplain: () => void;
  onToggleQuiet: () => void;
};

const clip: CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "cover",
};

export function PracticeWatch({
  variant,
  scene,
  t,
  quiet,
  reduced,
  frozen,
  customStill,
  onExplain,
  onToggleQuiet,
}: Props) {
  const { line, cue } = watchAt(scene, t, quiet);
  const close = variant === "netflix" && t >= 9.4 && t < 18.2;

  return (
    <div style={{ width: 1920, height: 1080, position: "relative", background: variant === "youtube" ? "#0f0f0f" : "#141414", overflow: "hidden", color: "#f1f1f1", fontFamily: "var(--sans)" }}>
      {variant === "youtube" ? (
        <YouTubeChrome
          t={t}
          duration={scene.duration}
          line={line}
          cue={cue}
          reduced={reduced}
          customStill={customStill}
        />
      ) : (
        <NetflixChrome
          t={t}
          duration={scene.duration}
          line={line}
          close={close}
          reduced={reduced}
          customStill={customStill}
        />
      )}
      <div
        style={
          variant === "youtube"
            ? { position: "absolute", left: 1268, top: 448, width: 624, height: 600 }
            : { position: "absolute", right: 36, bottom: 118, width: 460, height: 360 }
        }
      >
        <GlossPanel
          cue={cue}
          audioOn={false}
          quiet={quiet}
          reduced={reduced}
          frozen={frozen}
          localT={t}
          onExplain={onExplain}
          onToggleQuiet={onToggleQuiet}
          compact
        />
      </div>
    </div>
  );
}

function YouTubeChrome({
  t,
  duration,
  line,
  cue,
  reduced,
  customStill,
}: {
  t: number;
  duration: number;
  line: WatchLine | null;
  cue: Cue | null;
  reduced: boolean;
  customStill: string | null;
}) {
  const progress = Math.min(t / duration, 1);
  const shown = progress * (8 * 60 + 40);
  const paid = cue?.id === "yt-advert";

  return (
    <>
      <header
        style={{
          height: 64,
          display: "flex",
          alignItems: "center",
          gap: 18,
          padding: "0 24px",
          background: "#0f0f0f",
        }}
      >
        <MenuMark />
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            aria-hidden="true"
            style={{
              width: 32,
              height: 22,
              borderRadius: 6,
              background: "#ff0000",
              display: "grid",
              placeItems: "center",
            }}
          >
            <span
              style={{
                width: 0,
                height: 0,
                borderTop: "5px solid transparent",
                borderBottom: "5px solid transparent",
                borderLeft: "8px solid #fff",
                marginLeft: 2,
              }}
            />
          </span>
          <span style={{ fontWeight: 700, fontSize: 22, letterSpacing: "-0.04em" }}>YouTube</span>
        </div>
        <div
          style={{
            marginLeft: 48,
            width: 560,
            height: 40,
            borderRadius: 999,
            border: "1px solid #303030",
            background: "#121212",
            display: "flex",
            alignItems: "center",
            padding: "0 18px",
            color: "#888",
            fontSize: 16,
          }}
        >
          Search
        </div>
      </header>

      <div style={{ position: "absolute", left: 28, top: 80, width: 1220, height: 620, background: "#000", borderRadius: 12, overflow: "hidden" }}>
        <Picture
          reduced={reduced}
          customStill={customStill}
          poster="practice/youtube.jpg"
          src="practice/youtube.mp4"
        />
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 92, background: "linear-gradient(transparent, rgba(0,0,0,0.72))" }} />
        {paid ? (
          <div
            style={{
              position: "absolute",
              top: 16,
              left: 16,
              background: "rgba(15,15,15,0.88)",
              color: "#f1f1f1",
              borderRadius: 6,
              padding: "6px 10px",
              fontSize: 14,
              fontWeight: 600,
            }}
          >
            Includes paid promotion
          </div>
        ) : null}
        {line ? (
          <div
            style={{
              position: "absolute",
              left: 24,
              right: 24,
              bottom: 52,
              display: "flex",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                maxWidth: 980,
                background: "rgba(14, 18, 28, 0.92)",
                color: "#F7F5F0",
                borderRadius: 12,
                padding: "10px 14px",
                fontSize: 22,
                fontWeight: 600,
                lineHeight: 1.35,
                textAlign: "center",
              }}
            >
              {line.text}
            </div>
          </div>
        ) : null}
        <div style={{ position: "absolute", left: 12, right: 12, bottom: 10, display: "flex", alignItems: "center", gap: 12 }}>
          <PlayGlyph />
          <span style={{ fontSize: 13, fontVariantNumeric: "tabular-nums", color: "#eee" }}>
            {clock(shown)} / 8:40
          </span>
          <div style={{ flex: 1, height: 4, borderRadius: 99, background: "rgba(255,255,255,0.28)", overflow: "hidden" }}>
            <div style={{ width: `${progress * 100}%`, height: "100%", background: "#ff0000" }} />
          </div>
        </div>
      </div>

      <div style={{ position: "absolute", left: 36, top: 716, width: 1100 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>
          The trains are a mess, and I am thrilled
        </h1>
        <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ width: 40, height: 40, borderRadius: 999, background: "#3d2a24" }} />
          <div>
            <div style={{ fontWeight: 600, fontSize: 16 }}>Mara Ellis</div>
            <div style={{ color: "#aaa", fontSize: 13 }}>214K subscribers</div>
          </div>
        </div>
      </div>

      <aside style={{ position: "absolute", left: 1268, top: 80, width: 624 }}>
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Up next</div>
        <RecRow img="practice/thumb-train.jpg" title="The 7:42 never comes" meta="Mara Ellis · 18 minutes" />
        <RecRow img="practice/thumb-cook.jpg" title="What I cook on a quiet Sunday" meta="Mara Ellis · 11 minutes" />
        <RecRow img="practice/thumb-pack.jpg" title="Packing for a wet week" meta="Mara Ellis · 9 minutes" />
      </aside>
    </>
  );
}

function NetflixChrome({
  t,
  duration,
  line,
  close,
  reduced,
  customStill,
}: {
  t: number;
  duration: number;
  line: WatchLine | null;
  close: boolean;
  reduced: boolean;
  customStill: string | null;
}) {
  const progress = Math.min(t / duration, 1);
  const shown = progress * (46 * 60 + 12);

  return (
    <>
      <Picture
        reduced={reduced}
        customStill={customStill}
        poster="practice/netflix.jpg"
        src="practice/netflix.mp4"
        zoom={close}
      />
      <div style={{ position: "absolute", inset: 0, background: "linear-gradient(180deg, rgba(0,0,0,0.55) 0%, transparent 18%, transparent 62%, rgba(0,0,0,0.82) 100%)" }} />
      <header style={{ position: "absolute", top: 0, left: 0, right: 360, height: 72, display: "flex", alignItems: "center", gap: 28, padding: "0 48px" }}>
        <span style={{ color: "#e50914", fontWeight: 800, letterSpacing: "0.12em", fontSize: 28 }}>NETFLIX</span>
        <span style={{ fontSize: 16, fontWeight: 600 }}>Home</span>
        <span style={{ fontSize: 16, color: "#b3b3b3" }}>Series</span>
        <span style={{ fontSize: 16, color: "#b3b3b3" }}>Films</span>
      </header>
      <div style={{ position: "absolute", left: 56, bottom: 132, width: 820 }}>
        <div style={{ fontFamily: "var(--serif)", fontSize: 64, lineHeight: 1, fontWeight: 700 }}>Keel Street</div>
        <div style={{ marginTop: 10, fontSize: 20, color: "#e5e5e5" }}>Series 1 · Episode 3 · After lunch</div>
        {line ? (
          <div
            style={{
              marginTop: 22,
              fontSize: 28,
              fontWeight: 600,
              lineHeight: 1.35,
              textShadow: "0 2px 10px rgba(0,0,0,0.85)",
            }}
          >
            {line.text}
          </div>
        ) : (
          <div style={{ marginTop: 22, fontSize: 18, color: "#c8c8c8" }}>
            {close ? "They hold the look." : "Sunday lunch, continued."}
          </div>
        )}
      </div>
      <div style={{ position: "absolute", left: 48, right: 48, bottom: 36 }}>
        <div style={{ height: 4, borderRadius: 99, background: "rgba(255,255,255,0.28)", overflow: "hidden" }}>
          <div style={{ width: `${progress * 100}%`, height: "100%", background: "#e50914" }} />
        </div>
        <div style={{ marginTop: 10, display: "flex", justifyContent: "space-between", fontSize: 14, color: "#ddd" }}>
          <span>{clock(shown)}</span>
          <span>46:12</span>
        </div>
      </div>
    </>
  );
}

function Picture({
  reduced,
  customStill,
  poster,
  src,
  zoom = false,
}: {
  reduced: boolean;
  customStill: string | null;
  poster: string;
  src: string;
  zoom?: boolean;
}) {
  const frame: CSSProperties = {
    ...clip,
    transform: zoom ? "scale(1.28)" : "scale(1)",
    transformOrigin: "center 42%",
    transition: "transform 420ms ease",
  };
  if (customStill) {
    return <img data-clip="still" src={customStill} alt="" style={frame} />;
  }
  if (reduced) {
    return <img data-clip="still" src={poster} alt="" style={frame} />;
  }
  return <video data-clip="talk" src={src} poster={poster} autoPlay muted loop playsInline style={frame} />;
}

function RecRow({ img, title, meta }: { img: string; title: string; meta: string }) {
  return (
    <div style={{ display: "flex", gap: 12, marginBottom: 14 }}>
      <img src={img} alt="" style={{ width: 168, height: 94, objectFit: "cover", borderRadius: 8, flex: "0 0 auto" }} />
      <div>
        <div style={{ fontSize: 15, fontWeight: 600, lineHeight: 1.3 }}>{title}</div>
        <div style={{ marginTop: 6, color: "#aaa", fontSize: 13 }}>{meta}</div>
      </div>
    </div>
  );
}

function MenuMark() {
  return (
    <span aria-hidden="true" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ width: 18, height: 2, background: "#fff" }} />
      <span style={{ width: 18, height: 2, background: "#fff" }} />
      <span style={{ width: 18, height: 2, background: "#fff" }} />
    </span>
  );
}

function PlayGlyph() {
  return (
    <span
      aria-hidden="true"
      style={{
        width: 0,
        height: 0,
        borderTop: "7px solid transparent",
        borderBottom: "7px solid transparent",
        borderLeft: "11px solid #fff",
      }}
    />
  );
}

function clock(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60);
  const remain = whole % 60;
  return `${minutes}:${remain.toString().padStart(2, "0")}`;
}
