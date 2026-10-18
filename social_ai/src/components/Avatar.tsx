import type { AvatarId } from "../types";

type Palette = {
  skin: string;
  skinDeep: string;
  hair: string;
  cloth: string;
  clothDeep: string;
  ink: string;
};

const PALETTES: Record<AvatarId, Palette> = {
  priya: {
    skin: "#A56B45",
    skinDeep: "#8A5636",
    hair: "#1C1410",
    cloth: "#1E3350",
    clothDeep: "#15263C",
    ink: "#2A1C16",
  },
  sam: {
    skin: "#E0B090",
    skinDeep: "#C8926E",
    hair: "#3A2A22",
    cloth: "#3F5C4E",
    clothDeep: "#2E463A",
    ink: "#3A2A22",
  },
  jordan: {
    skin: "#6B3E2E",
    skinDeep: "#552F22",
    hair: "#140E0C",
    cloth: "#7A3E34",
    clothDeep: "#5C2E27",
    ink: "#1A100E",
  },
  alex: {
    skin: "#C48A62",
    skinDeep: "#A87350",
    hair: "#2A211C",
    cloth: "#355E78",
    clothDeep: "#27485C",
    ink: "#2A211C",
  },
};

/**
 * Flat editorial busts. Hair is drawn behind the head so each
 * silhouette stays a simple shape, not a portrait of a real person.
 */
export function Avatar({ id }: { id: AvatarId }) {
  const tone = PALETTES[id];
  return (
    <svg viewBox="0 0 240 280" width="100%" height="100%" aria-hidden="true">
      <Shoulders tone={tone} id={id} />
      <Hair id={id} color={tone.hair} />
      <ellipse cx="120" cy="118" rx="58" ry="66" fill={tone.skin} />
      {id === "priya" ? <circle cx="168" cy="78" r="18" fill={tone.hair} /> : null}
      <Face tone={tone} glasses={id === "sam"} />
      {id === "priya" ? <circle cx="172" cy="132" r="3.5" fill="#E07A2F" /> : null}
      <Neckline tone={tone} id={id} />
    </svg>
  );
}

function Shoulders({ tone, id }: { tone: Palette; id: AvatarId }) {
  const neck = id === "jordan" ? 46 : 40;
  return (
    <g>
      <path
        d="M18 280 C48 214 78 196 120 196 C162 196 192 214 222 280 Z"
        fill={tone.cloth}
      />
      <path
        d="M78 214 C92 196 148 196 162 214 L154 250 L86 250 Z"
        fill={tone.skin}
      />
      <rect x={120 - neck / 2} y="168" width={neck} height="48" rx="16" fill={tone.skin} />
    </g>
  );
}

function Hair({ id, color }: { id: AvatarId; color: string }) {
  if (id === "priya") {
    return <ellipse cx="120" cy="86" rx="62" ry="48" fill={color} />;
  }
  if (id === "sam") {
    return <ellipse cx="118" cy="78" rx="60" ry="42" fill={color} />;
  }
  if (id === "jordan") {
    return <ellipse cx="120" cy="80" rx="70" ry="52" fill={color} />;
  }
  return <ellipse cx="120" cy="128" rx="74" ry="92" fill={color} />;
}

function Face({ tone, glasses }: { tone: Palette; glasses: boolean }) {
  return (
    <g>
      <path
        d="M86 104 Q96 98 108 104"
        fill="none"
        stroke={tone.ink}
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <path
        d="M132 104 Q144 98 154 104"
        fill="none"
        stroke={tone.ink}
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <ellipse cx="96" cy="116" rx="4.2" ry="4.6" fill={tone.ink} />
      <ellipse cx="144" cy="116" rx="4.2" ry="4.6" fill={tone.ink} />
      <path
        d="M120 118 Q124 132 120 136"
        fill="none"
        stroke={tone.skinDeep}
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <path
        d="M104 150 Q120 158 136 150"
        fill="none"
        stroke={tone.ink}
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      {glasses ? (
        <g fill="none" stroke="#243044" strokeWidth="2.6">
          <circle cx="96" cy="116" r="14" />
          <circle cx="144" cy="116" r="14" />
          <path d="M110 116 H130" />
          <path d="M82 114 H86" />
          <path d="M154 114 H160" />
        </g>
      ) : null}
    </g>
  );
}

function Neckline({ tone, id }: { tone: Palette; id: AvatarId }) {
  if (id === "priya") {
    return (
      <g>
        <path d="M78 230 L120 268 L162 230 L168 280 L72 280 Z" fill={tone.clothDeep} />
        <path d="M108 214 L120 236 L132 214" fill="none" stroke={tone.clothDeep} strokeWidth="8" />
      </g>
    );
  }
  if (id === "jordan") {
    return (
      <g>
        <path d="M96 210 L120 248 L144 210 L150 236 L120 268 L90 236 Z" fill={tone.clothDeep} />
      </g>
    );
  }
  return <path d="M92 214 Q120 246 148 214 L156 250 L84 250 Z" fill={tone.clothDeep} />;
}
