import { EXPLAINER } from "../content/sceneA";

export function ExplainerCard({ onClose }: { onClose: () => void }) {
  return (
    <div
      role="dialog"
      aria-label="What is this?"
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 40,
        background: "rgba(20, 33, 61, 0.46)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <article
        style={{
          width: 760,
          background: "#F7F5F0",
          color: "#14213D",
          borderRadius: 24,
          padding: "40px 44px 32px",
          boxShadow: "0 24px 70px rgba(8, 12, 20, 0.32)",
        }}
      >
        <h2 style={{ margin: 0, fontFamily: "var(--serif)", fontWeight: 400, fontSize: 40, lineHeight: 1.15 }}>
          {EXPLAINER.title}
        </h2>
        <p style={{ margin: "18px 0 0", fontSize: 20, lineHeight: 1.45 }}>{EXPLAINER.lead}</p>
        {EXPLAINER.paragraphs.map((paragraph) => (
          <p key={paragraph} style={{ margin: "14px 0 0", color: "#4A5568", fontSize: 18, lineHeight: 1.5 }}>
            {paragraph}
          </p>
        ))}
        <div
          style={{
            marginTop: 26,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <p style={{ margin: 0, color: "#4A5568", fontSize: 15 }}>{EXPLAINER.footer}</p>
          <button
            type="button"
            onClick={onClose}
            style={{
              border: "none",
              background: "#14213D",
              color: "#F7F5F0",
              borderRadius: 999,
              padding: "10px 18px",
              fontSize: 15,
              fontWeight: 600,
            }}
          >
            Close
          </button>
        </div>
      </article>
    </div>
  );
}
