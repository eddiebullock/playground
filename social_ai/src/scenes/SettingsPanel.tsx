type Props = {
  eyeContact: boolean;
  onToggleEyeContact: () => void;
};

export function SettingsPanel({ eyeContact, onToggleEyeContact }: Props) {
  return (
    <aside
      style={{
        width: "min(360px, 100%)",
        overflow: "auto",
        background: "#F7F5F0",
        color: "#14213D",
        padding: "22px 20px 24px",
        display: "flex",
        flexDirection: "column",
        gap: 18,
        borderRadius: 18,
        boxShadow: "0 16px 40px rgba(0,0,0,0.28)",
      }}
    >
      <div>
        <h2 style={{ margin: 0, fontFamily: "var(--serif)", fontWeight: 400, fontSize: 32, lineHeight: 1.15 }}>Settings</h2>
        <p style={{ margin: "8px 0 0", fontSize: 15, lineHeight: 1.45, color: "#4A5568" }}>
          You decide what Gloss may mention. The note on the picture updates as you change this.
        </p>
      </div>
      <div
        style={{
          border: "1px solid rgba(20, 33, 61, 0.14)",
          borderRadius: 16,
          padding: "16px 16px 18px",
          display: "flex",
          flexDirection: "column",
          gap: 12,
          background: "#fff",
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.04em", color: "#4A5568" }}>What to mention</div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 600, lineHeight: 1.3 }}>Eye contact reminders</div>
            <p style={{ margin: "4px 0 0", fontSize: 14, lineHeight: 1.4, color: "#4A5568" }}>
              A short reminder when someone is looking at you. It is a suggestion.
            </p>
          </div>
          <button
            type="button"
            aria-pressed={eyeContact}
            onClick={onToggleEyeContact}
            style={{
              flex: "0 0 auto",
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              border: "1px solid #14213D",
              background: eyeContact ? "#14213D" : "#F7F5F0",
              color: eyeContact ? "#F7F5F0" : "#14213D",
              borderRadius: 999,
              padding: "8px 12px",
              font: "600 14px var(--sans)",
            }}
          >
            <span
              aria-hidden="true"
              style={{
                width: 10,
                height: 10,
                borderRadius: 999,
                background: eyeContact ? "#3A7CA5" : "#8A94A6",
                boxShadow: "0 0 0 2px rgba(247,245,240,0.35)",
              }}
            />
            {eyeContact ? "On" : "Off"}
          </button>
        </div>
      </div>
      <p style={{ margin: 0, fontSize: 14, lineHeight: 1.45, color: "#4A5568" }}>
        {eyeContact
          ? "The reminder is included on a hello, and when the person closest is being playful."
          : "The reminder is left out. The rest of the note stays."}
      </p>
    </aside>
  );
}
