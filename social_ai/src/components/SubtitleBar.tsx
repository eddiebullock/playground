import type { Line, Person } from "../types";

export function SubtitleBar({
  line,
  speaker,
  width,
}: {
  line: Line | null;
  speaker: Person | null;
  width: number;
}) {
  return (
    <div style={{ width, minHeight: 64, display: "flex", alignItems: "center" }}>
      {line && speaker ? (
        <div
          style={{
            maxWidth: width,
            background: "rgba(14, 18, 28, 0.92)",
            color: "#F7F5F0",
            borderRadius: 14,
            padding: "12px 16px",
            fontSize: 20,
            lineHeight: 1.35,
            boxShadow: "0 10px 24px rgba(0,0,0,0.18)",
          }}
        >
          <span style={{ fontWeight: 600 }}>{speaker.name}</span>
          <span style={{ color: "#D6DEE8" }}> · </span>
          <span>{line.text}</span>
        </div>
      ) : (
        <div style={{ color: "#D6DEE8", fontSize: 16, paddingLeft: 4 }}>No one speaking</div>
      )}
    </div>
  );
}
