import { useState } from "react";
export default function DisclaimerBanner({ onDismiss }) {
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) return null;
  return (
    <div style={{
      position: "fixed",
      top: 68,
      left: 0,
      right: 0,
      zIndex: 999,
      background: "rgba(251,191,36,0.06)",
      borderBottom: "1px solid rgba(251,191,36,0.2)",
      padding: "10px 24px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 16,
      animation: "fadeIn 0.4s ease",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1 }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
            stroke="#fbbf24" strokeWidth="1.5" strokeLinejoin="round"/>
          <line x1="12" y1="9" x2="12" y2="13" stroke="#fbbf24" strokeWidth="1.5" strokeLinecap="round"/>
          <line x1="12" y1="17" x2="12.01" y2="17" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        <p style={{ fontSize: 12, color: "rgba(251,191,36,0.9)", lineHeight: 1.5 }}>
          <strong>Research tool only.</strong> DermAI is not a medical device and does not provide clinical diagnosis.
          Always consult a qualified dermatologist for any skin concern.
        </p>
      </div>
      <button
        onClick={() => setDismissed(true)}
        style={{
          background: "none", border: "none",
          color: "rgba(251,191,36,0.5)",
          cursor: "pointer", padding: 4, flexShrink: 0,
          transition: "color 0.2s",
        }}
        onMouseEnter={e => e.currentTarget.style.color = "rgba(251,191,36,0.9)"}
        onMouseLeave={e => e.currentTarget.style.color = "rgba(251,191,36,0.5)"}
        aria-label="Dismiss"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
      </button>
    </div>
  );
}
