import { AnimatePresence, motion } from "framer-motion";
import { PRIVACY_LINE } from "../content/sceneA";
import type { Cue } from "../types";
import { AudioIcon, CueKindIcon, LockIcon, PauseIcon, QuestionIcon } from "./Icons";
import { LevelChip } from "./LevelChip";
import { HapticMark } from "./HapticMark";

type Props = {
  cue: Cue | null;
  audioOn: boolean;
  quiet: boolean;
  reduced: boolean;
  frozen: boolean;
  localT: number;
  onExplain: () => void;
  onToggleQuiet: () => void;
  compact?: boolean;
};

export function GlossPanel({
  cue,
  audioOn,
  quiet,
  reduced,
  frozen,
  localT,
  onExplain,
  onToggleQuiet,
  compact = false,
}: Props) {
  const pad = compact ? 18 : 26;
  return (
    <section
      aria-label="Gloss suggestions"
      style={{
        height: "100%",
        background: "#F7F5F0",
        color: "#14213D",
        borderRadius: compact ? 18 : 22,
        boxShadow: "0 18px 50px rgba(8, 12, 20, 0.28)",
        padding: pad,
        display: "flex",
        flexDirection: "column",
        gap: compact ? 12 : 16,
        overflow: "hidden",
      }}
    >
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <div style={{ fontFamily: "var(--serif)", fontSize: compact ? 22 : 28, lineHeight: 1.1 }}>Gloss</div>
        <button
          type="button"
          onClick={onExplain}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            border: "1px solid rgba(20,33,61,0.18)",
            background: "transparent",
            color: "#14213D",
            borderRadius: 999,
            padding: "6px 10px",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          <QuestionIcon size={16} />
          What is this?
        </button>
      </header>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          color: "#4A5568",
          fontSize: compact ? 13 : 15,
          lineHeight: 1.35,
        }}
      >
        <LockIcon size={15} color="#4A5568" />
        <span>{PRIVACY_LINE}</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ color: "#4A5568", fontSize: 14 }}>Suggestions</span>
        <button
          type="button"
          onClick={onToggleQuiet}
          aria-pressed={quiet}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            border: "none",
            background: quiet ? "#E7E2D8" : "#E4EEF3",
            color: "#14213D",
            borderRadius: 999,
            padding: "6px 10px 6px 8px",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          <span
            aria-hidden="true"
            style={{
              width: 28,
              height: 16,
              borderRadius: 999,
              background: quiet ? "#8A94A6" : "#3A7CA5",
              position: "relative",
              display: "inline-block",
            }}
          >
            <span
              style={{
                position: "absolute",
                top: 2,
                left: quiet ? 2 : 14,
                width: 12,
                height: 12,
                borderRadius: 999,
                background: "#F7F5F0",
              }}
            />
          </span>
          {quiet ? "Cues paused" : "Cues on"}
        </button>
      </div>

      <div style={{ flex: 1, minHeight: 0 }}>
        {frozen || reduced ? (
          <Body cue={cue} audioOn={audioOn} quiet={quiet} localT={localT} reduced={reduced} compact={compact} />
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={quiet ? "quiet" : (cue?.id ?? "rest")}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.28, ease: "easeOut" }}
              style={{ height: "100%" }}
            >
              <Body cue={cue} audioOn={audioOn} quiet={quiet} localT={localT} reduced={reduced} compact={compact} />
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </section>
  );
}

function Body({
  cue,
  audioOn,
  quiet,
  localT,
  reduced,
  compact,
}: {
  cue: Cue | null;
  audioOn: boolean;
  quiet: boolean;
  localT: number;
  reduced: boolean;
  compact: boolean;
}) {
  if (quiet) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingTop: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#14213D", fontWeight: 600, fontSize: 15 }}>
          <PauseIcon />
          Quiet mode
        </div>
        <h2 style={{ margin: 0, fontFamily: "var(--serif)", fontWeight: 400, fontSize: compact ? 26 : 32, lineHeight: 1.2 }}>
          Suggestions are paused
        </h2>
        <p style={{ margin: 0, color: "#4A5568", fontSize: compact ? 16 : 18, lineHeight: 1.45 }}>
          Turn cues back on whenever you want. Nothing else changes.
        </p>
      </div>
    );
  }

  if (!cue) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingTop: 8 }}>
        <h2 style={{ margin: 0, fontFamily: "var(--serif)", fontWeight: 400, fontSize: compact ? 26 : 32, lineHeight: 1.25 }}>
          No suggestion right now
        </h2>
        <p style={{ margin: 0, color: "#4A5568", fontSize: compact ? 16 : 18, lineHeight: 1.45, maxWidth: 360 }}>
          A note appears here only when something may be worth a second look.
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: compact ? 10 : 14, paddingTop: 4 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <LevelChip level={cue.level} />
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "#14213D", fontSize: 15, fontWeight: 600 }}>
          <CueKindIcon kind={cue.kind} color="#14213D" />
          {cue.kindLabel}
        </span>
      </div>
      <h2 style={{ margin: 0, fontFamily: "var(--serif)", fontWeight: 400, fontSize: compact ? 26 : 32, lineHeight: 1.2 }}>
        {cue.title}
      </h2>
      <p style={{ margin: 0, color: "#4A5568", fontSize: compact ? 16 : 18, lineHeight: 1.45 }}>{cue.detail}</p>
      {audioOn && cue.audio ? (
        <div
          style={{
            marginTop: 4,
            borderRadius: 14,
            background: "#EFECE4",
            padding: "12px 14px",
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600, fontSize: 15 }}>
            <AudioIcon size={18} color="#14213D" />
            Audio cue playing
          </div>
          <div style={{ color: "#4A5568", fontSize: 16, paddingLeft: 26 }}>&ldquo;{cue.audio.caption}&rdquo;</div>
        </div>
      ) : null}
      {cue.haptic ? (
        <div
          style={{
            marginTop: 4,
            borderRadius: 14,
            background: "#EFECE4",
            padding: "12px 14px",
            display: "flex",
            alignItems: "center",
            gap: 12,
          }}
        >
          <HapticMark level={cue.level} localT={localT} pulses={cue.haptic.pulses} reduced={reduced} />
          <div>
            <div style={{ fontWeight: 600, fontSize: 15 }}>{cue.haptic.label}</div>
            <div style={{ color: "#4A5568", fontSize: 14, marginTop: 2 }}>Wristband, optional</div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
