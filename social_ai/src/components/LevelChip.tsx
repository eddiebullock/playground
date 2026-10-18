import type { CueLevel } from "../types";

export const LEVELS: Record<
  CueLevel,
  { label: string; swatch: string; glow: string }
> = {
  calm: { label: "Calm", swatch: "#3A7CA5", glow: "rgba(58, 124, 165, 0.55)" },
  headsup: { label: "Heads-up", swatch: "#E07A2F", glow: "rgba(224, 122, 47, 0.55)" },
  uncertain: { label: "Uncertain", swatch: "#8A94A6", glow: "rgba(138, 148, 166, 0.55)" },
};

export function LevelChip({ level, light = false }: { level: CueLevel; light?: boolean }) {
  const meta = LEVELS[level];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        color: light ? "#F7F5F0" : "#14213D",
        fontFamily: "var(--sans)",
        fontSize: 15,
        fontWeight: 600,
        letterSpacing: "0.04em",
        lineHeight: 1,
      }}
    >
      <span
        aria-hidden="true"
        style={{
          width: 12,
          height: 12,
          borderRadius: 999,
          background: meta.swatch,
          boxShadow: `0 0 0 3px ${light ? "rgba(247,245,240,0.16)" : "rgba(20,33,61,0.08)"}`,
          flex: "0 0 auto",
        }}
      />
      {meta.label}
    </span>
  );
}
