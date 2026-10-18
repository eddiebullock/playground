import { HONESTY_TAG } from "../content/sceneA";

export function HonestyTag() {
  return (
    <div
      style={{
        position: "absolute",
        top: 18,
        right: 18,
        zIndex: 60,
        maxWidth: 420,
        padding: "8px 12px",
        borderRadius: 999,
        background: "rgba(20, 33, 61, 0.92)",
        color: "#F7F5F0",
        fontFamily: "var(--sans)",
        fontSize: 14,
        fontWeight: 500,
        letterSpacing: "0.01em",
        lineHeight: 1.2,
        boxShadow: "0 8px 24px rgba(0,0,0,0.18)",
      }}
    >
      {HONESTY_TAG}
    </div>
  );
}
