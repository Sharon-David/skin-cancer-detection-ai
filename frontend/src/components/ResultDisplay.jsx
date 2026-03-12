import HeatmapOverlay from "./HeatmapOverlay";

function ConfidenceBar({ value, color }) {
  return (
    <div style={{
      width: "100%", height: 4, borderRadius: 2,
      background: "rgba(255,255,255,0.06)", overflow: "hidden", marginTop: 8,
    }}>
      <div style={{
        height: "100%",
        width: `${Math.min(100, Math.max(0, value))}%`,
        background: color,
        borderRadius: 2,
        transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
      }}/>
    </div>
  );
}

function RiskBadge({ level }) {
  const cfg = {
    High:    { bg: "rgba(248,113,113,0.12)", border: "rgba(248,113,113,0.3)",  dot: "#f87171", text: "#f87171" },
    Medium:  { bg: "rgba(251,191,36,0.12)",  border: "rgba(251,191,36,0.3)",   dot: "#fbbf24", text: "#fbbf24" },
    Low:     { bg: "rgba(52,211,153,0.12)",  border: "rgba(52,211,153,0.3)",   dot: "#34d399", text: "#34d399" },
    Unknown: { bg: "rgba(148,163,184,0.12)", border: "rgba(148,163,184,0.3)",  dot: "#94a3b8", text: "#94a3b8" },
  };
  const c = cfg[level] || cfg.Unknown;
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "5px 12px", borderRadius: 20,
      background: c.bg, border: `1px solid ${c.border}`,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: c.dot, display: "inline-block" }}/>
      <span style={{ fontSize: 11, fontWeight: 700, color: c.text, letterSpacing: "0.05em" }}>
        {level} Risk
      </span>
    </div>
  );
}

export default function ResultDisplay({ result, previewImage }) {
  if (!result) return null;

  const {
    prediction, confidence, prob_malignant, prob_benign,
    risk_level, recommendation, disclaimer,
    gradcam_image, uncertainty, is_ood, ood_reasons,
  } = result;

  const probMalignant = typeof prob_malignant === "number" ? prob_malignant : 0;
  const probBenign    = typeof prob_benign    === "number" ? prob_benign    : 100 - probMalignant;
  const confidenceVal = typeof confidence     === "number" ? confidence     : 0;
  const predColor     = prediction === "Malignant" ? "#f87171" : "#34d399";

  // OOD — image not recognised as skin lesion
  if (is_ood) {
    return (
      <div style={{
        padding: 24, borderRadius: "var(--radius-lg)",
        background: "rgba(251,191,36,0.06)",
        border: "1px solid rgba(251,191,36,0.2)",
        animation: "fadeUp 0.4s ease both",
      }}>
        <p style={{ fontWeight: 700, color: "#fbbf24", marginBottom: 8 }}>
          ⚠ Image not recognised
        </p>
        <p style={{ fontSize: 13, color: "var(--grey)", marginBottom: 12 }}>
          {recommendation}
        </p>
        {ood_reasons?.map((r, i) => (
          <p key={i} style={{ fontSize: 12, color: "rgba(251,191,36,0.6)" }}>— {r}</p>
        ))}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeUp 0.4s ease both" }}>

      {/* Main prediction card */}
      <div style={{
        padding: 24, borderRadius: "var(--radius-lg)",
        background: "var(--navy-mid)",
        border: `1px solid ${prediction === "Malignant" ? "rgba(248,113,113,0.2)" : "rgba(52,211,153,0.2)"}`,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: "var(--grey)", textTransform: "uppercase" }}>
            PREDICTION
          </p>
          <RiskBadge level={risk_level} />
        </div>

        <h2 style={{
          fontFamily: "var(--font-display)",
          fontSize: "clamp(2rem, 4vw, 3rem)",
          color: predColor, marginBottom: 20,
          letterSpacing: "-0.02em",
        }}>
          {prediction}
        </h2>

        {/* Confidence */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 12, color: "var(--grey)" }}>Model confidence</span>
            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--white)" }}>
              {confidenceVal.toFixed(1)}%
            </span>
          </div>
          <ConfidenceBar value={confidenceVal} color={predColor} />
        </div>

        {/* Prob bars */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          {[
            { label: "Malignant probability", value: probMalignant, color: "#f87171" },
            { label: "Benign probability",    value: probBenign,    color: "#34d399" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{
              padding: "14px 16px", borderRadius: "var(--radius-md)",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.06)",
            }}>
              <p style={{ fontSize: 11, color: "var(--grey)", marginBottom: 6 }}>{label}</p>
              <p style={{ fontSize: 22, fontWeight: 700, color, marginBottom: 6 }}>
                {value.toFixed(1)}%
              </p>
              <ConfidenceBar value={value} color={color} />
            </div>
          ))}
        </div>
      </div>

      {/* Uncertainty card */}
      {uncertainty && (
        <div style={{
          padding: "16px 20px", borderRadius: "var(--radius-md)",
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
        }}>
          <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: "var(--grey)", textTransform: "uppercase", marginBottom: 10 }}>
            UNCERTAINTY ESTIMATION
          </p>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
            <div>
              <p style={{ fontSize: 12, color: "var(--grey)", marginBottom: 2 }}>
                Std deviation across {uncertainty.samples} MC samples
              </p>
              <p style={{ fontSize: 13, color: "var(--white)" }}>{uncertainty.label}</p>
            </div>
            <span style={{
              fontSize: 20, fontWeight: 700,
              color: uncertainty.std > 15 ? "#f87171" : uncertainty.std > 8 ? "#fbbf24" : "#34d399",
            }}>
              ±{uncertainty.std.toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Grad-CAM heatmap */}
      {gradcam_image && previewImage && (
        <div style={{
          padding: "16px 20px", borderRadius: "var(--radius-md)",
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
        }}>
          <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: "var(--grey)", textTransform: "uppercase", marginBottom: 12 }}>
            GRAD-CAM EXPLANATION
          </p>
          <p style={{ fontSize: 12, color: "var(--grey)", marginBottom: 12 }}>
            Highlighted regions influenced the prediction most.
          </p>
          <HeatmapOverlay
            originalImage={previewImage}
            heatmapBase64={gradcam_image}
          />
        </div>
      )}

      {/* Recommendation */}
      <div style={{
        padding: "16px 20px", borderRadius: "var(--radius-md)",
        background: "rgba(251,191,36,0.04)",
        border: "1px solid rgba(251,191,36,0.1)",
        display: "flex", gap: 12, alignItems: "flex-start",
      }}>
        <span style={{ fontSize: 16, flexShrink: 0 }}>💡</span>
        <div>
          <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: "#fbbf24", textTransform: "uppercase", marginBottom: 6 }}>
            RECOMMENDATION
          </p>
          <p style={{ fontSize: 13, color: "var(--white-dim)", lineHeight: 1.6 }}>{recommendation}</p>
        </div>
      </div>

      {/* Disclaimer */}
      <p style={{ fontSize: 11, color: "rgba(148,163,184,0.5)", display: "flex", gap: 6, alignItems: "flex-start" }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0, marginTop: 1 }}>
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5"/>
          <path d="M12 8v4M12 16h.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        {disclaimer}
      </p>
    </div>
  );
}
