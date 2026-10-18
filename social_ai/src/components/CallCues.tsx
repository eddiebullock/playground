import type { Person } from "../types";

const FACE: Record<string, { x: number; y: number }> = {
  priya: { x: 50, y: 51 },
  sam: { x: 50, y: 31 },
  jordan: { x: 52, y: 36 },
  alex: { x: 48, y: 44 },
};

export function TalkingBar({ on, reduced, rounded = true }: { on: boolean; reduced: boolean; rounded?: boolean }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 4,
        height: on ? 58 : 0,
        opacity: on ? 1 : 0,
        overflow: "hidden",
        pointerEvents: "none",
        transition: reduced ? "none" : "height 0.35s ease, opacity 0.28s ease",
      }}
    >
      <div
        style={{
          height: 58,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 16,
          background: "linear-gradient(90deg, #3A7CA5 0%, #6FA8C9 28%, #E07A2F 62%, #F2B06A 82%, #3A7CA5 100%)",
          borderRadius: rounded ? "18px 18px 0 0" : 0,
        }}
      >
        <span
          style={{
            background: "#F7F5F0",
            color: "#14213D",
            borderRadius: 999,
            padding: "7px 14px",
            fontSize: 18,
            fontWeight: 700,
            lineHeight: 1,
          }}
        >
          They{"\u2019"}re talking
        </span>
        <span aria-hidden="true" style={{ display: "inline-flex", alignItems: "flex-end", gap: 4, height: 22 }}>
          {[0, 0.15, 0.3, 0.08].map((delay, index) => (
            <span
              key={index}
              className="talk-beat"
              style={{
                width: 5,
                height: 22,
                borderRadius: 999,
                background: "#F7F5F0",
                animationDelay: `${delay}s`,
              }}
            />
          ))}
        </span>
      </div>
    </div>
  );
}

export const RAPPORT: Record<"good" | "mixed" | "unsure", { label: string; colour: string; glow: string }> = {
  good: { label: "Good rapport", colour: "#3FA37A", glow: "rgba(63, 163, 122, 0.55)" },
  mixed: { label: "Mixed signals", colour: "#E07A2F", glow: "rgba(224, 122, 47, 0.55)" },
  unsure: { label: "Not sure", colour: "#8A94A6", glow: "rgba(138, 148, 166, 0.5)" },
};

/** Colour around the whole view. A word sits on it too, so colour is never the only signal. */
export function RapportFrame({
  level,
  reduced,
  radius = 0,
  scale = 1,
}: {
  level: "good" | "mixed" | "unsure";
  reduced: boolean;
  radius?: number;
  scale?: number;
}) {
  const meta = RAPPORT[level];
  const fade = reduced ? "none" : "border-color 0.6s ease, box-shadow 0.6s ease";
  return (
    <>
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 7,
          pointerEvents: "none",
          borderRadius: radius,
          border: `${10 * scale}px solid ${meta.colour}`,
          boxShadow: `inset 0 0 ${36 * scale}px ${meta.glow}`,
          transition: fade,
        }}
      />
      <div
        data-rapport={level}
        style={{
          position: "absolute",
          left: "50%",
          bottom: 22 * scale,
          transform: "translateX(-50%)",
          zIndex: 8,
          display: "inline-flex",
          alignItems: "center",
          gap: 8 * scale,
          background: "#F7F5F0",
          color: "#14213D",
          borderRadius: 999,
          padding: `${6 * scale}px ${12 * scale}px`,
          fontSize: 15 * scale,
          fontWeight: 700,
          lineHeight: 1,
          boxShadow: "0 6px 16px rgba(0,0,0,0.28)",
          whiteSpace: "nowrap",
        }}
      >
        <span
          aria-hidden="true"
          style={{ width: 11 * scale, height: 11 * scale, borderRadius: 999, background: meta.colour, transition: fade }}
        />
        {meta.label}
      </div>
    </>
  );
}

/** Shown only in the gap after others stop. The bar leaving is the main signal. */
export function YourTurn({ on, scale = 1, top = 16 }: { on: boolean; scale?: number; top?: number }) {
  return (
    <div
      aria-hidden={!on}
      style={{
        position: "absolute",
        top,
        left: "50%",
        transform: `translateX(-50%) translateY(${on ? 0 : -8}px)`,
        opacity: on ? 1 : 0,
        transition: "opacity 0.3s ease, transform 0.3s ease",
        zIndex: 8,
        background: "#F7F5F0",
        color: "#14213D",
        borderRadius: 999,
        padding: `${7 * scale}px ${14 * scale}px`,
        fontSize: 16 * scale,
        fontWeight: 700,
        lineHeight: 1,
        boxShadow: "0 6px 16px rgba(0,0,0,0.28)",
        pointerEvents: "none",
      }}
    >
      Your turn
    </div>
  );
}

export function LookHere({ person, x, y }: { person?: Person; x?: number; y?: number }) {
  const face = x != null && y != null ? { x, y } : FACE[person?.id ?? ""] ?? { x: 50, y: 40 };
  const align = face.x > 76 ? "flex-end" : face.x < 24 ? "flex-start" : "center";
  return (
    <div
      style={{
        position: "absolute",
        left: `${face.x}%`,
        top: `${face.y}%`,
        zIndex: 5,
        transform: "translate(-50%, -18px)",
        display: "flex",
        flexDirection: "column",
        alignItems: align,
        pointerEvents: "none",
      }}
    >
      <span
        aria-hidden="true"
        style={{
          width: 86,
          height: 36,
          borderRadius: 999,
          border: "3px solid #F7F5F0",
          boxShadow: "0 0 0 3px rgba(224, 122, 47, 0.95), 0 6px 16px rgba(0,0,0,0.35)",
        }}
      />
      <span
        style={{
          marginTop: 8,
          background: "#F7F5F0",
          color: "#14213D",
          borderRadius: 999,
          padding: "6px 12px",
          fontSize: 16,
          fontWeight: 700,
          lineHeight: 1,
          boxShadow: "0 6px 16px rgba(0,0,0,0.28)",
        }}
      >
        Look here
      </span>
    </div>
  );
}
