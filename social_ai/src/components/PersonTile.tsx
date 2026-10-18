import type { Cue, Person } from "../types";
import { LEVELS } from "./LevelChip";

type Props = {
  person: Person;
  speaking: boolean;
  cue: Cue | null;
  width: number;
  height: number;
  reduced: boolean;
  /** Push the cue chip down when a bar covers the top of the tile. */
  chipOffset?: number;
};

export function PersonTile({ person, speaking, cue, width, height, reduced, chipOffset = 14 }: Props) {
  const glow = cue ? LEVELS[cue.level] : null;
  return (
    <article
      style={{
        width,
        height,
        borderRadius: 18,
        background: "#121820",
        position: "relative",
        overflow: "hidden",
        boxShadow: glow
          ? `0 0 0 3px ${glow.swatch}, 0 0 28px ${glow.glow}`
          : speaking
            ? "0 0 0 2px #C5DDD2"
            : "0 0 0 1px rgba(247,245,240,0.06)",
      }}
    >
      <Clip person={person} speaking={speaking} reduced={reduced} />
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: 0,
          height: 92,
          background: "linear-gradient(transparent, rgba(12, 16, 24, 0.88))",
        }}
      />
      {glow ? (
        <div style={{ position: "absolute", top: chipOffset, left: 14 }}>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              background: "#F7F5F0",
              color: "#14213D",
              borderRadius: 999,
              padding: "6px 10px",
              fontSize: 14,
              fontWeight: 600,
            }}
          >
            <span
              aria-hidden="true"
              style={{ width: 10, height: 10, borderRadius: 999, background: glow.swatch }}
            />
            {glow.label}
          </span>
        </div>
      ) : null}
      <div
        style={{
          position: "absolute",
          left: 16,
          right: 16,
          bottom: 14,
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <div>
          <div style={{ fontSize: 20, fontWeight: 600, color: "#F7F5F0", lineHeight: 1.2 }}>{person.name}</div>
          <div style={{ fontSize: 15, color: "#D6DEE8", marginTop: 2 }}>{person.role}</div>
        </div>
        {speaking ? (
          <div style={{ fontSize: 14, fontWeight: 600, color: "#F7F5F0", paddingBottom: 2 }}>Speaking</div>
        ) : (
          <div style={{ width: 8 }} />
        )}
      </div>
    </article>
  );
}

const clipStyle = {
  position: "absolute" as const,
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "cover" as const,
  objectPosition: "center center",
};

function Clip({ person, speaking, reduced }: { person: Person; speaking: boolean; reduced: boolean }) {
  const poster = `clips/${person.avatar}.jpg`;
  if (reduced) {
    return <img data-clip="still" src={poster} alt="" style={clipStyle} />;
  }
  return (
    <video
      data-clip={speaking ? "talk" : "idle"}
      src={`clips/${person.avatar}.mp4`}
      poster={poster}
      autoPlay
      muted
      loop
      playsInline
      style={clipStyle}
    />
  );
}
