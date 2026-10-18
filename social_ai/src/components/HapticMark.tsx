import type { CueLevel } from "../types";
import { LEVELS } from "./LevelChip";

type Props = {
  level: CueLevel;
  localT: number;
  pulses: number[];
  reduced: boolean;
};

export function HapticMark({ level, localT, pulses, reduced }: Props) {
  const amp = reduced ? 0.72 : pulseAmp(localT, pulses);
  const swatch = LEVELS[level].swatch;
  return (
    <svg width="78" height="48" viewBox="0 0 78 48" aria-hidden="true">
      <path d="M8 18h14c2 0 3 2 3 6s-1 6-3 6H8c-2 0-3-2-3-6s1-6 3-6Z" fill="#14213D" />
      <rect x="18" y="14" width="22" height="20" rx="5" fill="#F7F5F0" stroke="#14213D" strokeWidth="1.7" />
      <path d="M40 18h12c2 0 3 2 3 6s-1 6-3 6H40" fill="#14213D" />
      <circle cx="25" cy="24" r="2.1" fill={swatch} opacity={0.45 + amp * 0.55} />
      <circle cx="33" cy="24" r="2.1" fill={swatch} opacity={0.3 + amp} />
      <path
        d="M58 24c3.5-5 7-7 11-7"
        fill="none"
        stroke={swatch}
        strokeWidth="1.8"
        strokeLinecap="round"
        opacity={0.35 + amp * 0.4}
      />
      <path
        d="M58 24c5-8 10-11 16-11"
        fill="none"
        stroke={swatch}
        strokeWidth="1.8"
        strokeLinecap="round"
        opacity={0.2 + amp * 0.8}
      />
    </svg>
  );
}

function pulseAmp(localT: number, pulses: number[]) {
  let amp = 0;
  for (const pulse of pulses) {
    const d = localT - pulse;
    if (d >= 0 && d < 0.42) {
      amp = Math.max(amp, Math.sin((d / 0.42) * Math.PI));
    }
  }
  return amp;
}
