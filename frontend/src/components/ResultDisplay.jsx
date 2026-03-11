import HeatmapOverlay from "./HeatmapOverlay";

function RiskBadge({ riskLevel }) {
  const config = {
    High:   { bg: "rgba(248,113,113,0.12)", border: "rgba(248,113,113,0.3)", color: "#f87171", dot: "#f87171", label: "High Risk" },
    Medium: { bg: "rgba(251,191,36,0.1)",  border: "rgba(251,191,36,0.3)",  color: "#fbbf24", dot: "#fbbf24", label: "Medium Risk" },
    Low:    { bg: "rgba(52,211,153,0.1)",  border: "rgba(52,211,153,0.3)",  color: "#34d399", dot: "#34d399", label: "Low Risk" },
  };
  const c = config[riskLevel] || config.Low;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "5px 12px",
      borderRadius: 100,
      background: c.bg,
      border: `1px solid ${c.border}`,
      color: c.color,
      fontSize: 12, fontWeight: 700,
      letterSpacing: "0.04em",
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: c.dot }}/>
      {c.label}
    </span>
  );
}

function ConfidenceBar({ value, color }) {
  return (
    <div style={{
      height: 6, borderRadius: 3,
      background: "rgba(255,255,255,0.06)",
      overflow: "hidden",
    }}>
      <div style={{
        height: "100%",
        width: `${value.toFixed(1)}%`,
        borderRadius: 3,
        background: color,
        transition: "width 1s cubic-bezier(0.4, 0, 0.2, 1)",
        boxShadow: `0 0 8px ${color}66`,
      }}/>
    </div>
  );
}

export default function ResultDisplay({ result, previewImage }) {
  if (!result) return null;

  const {
    prediction,
    confidence,
    prob_malignant,
    risk_level,
    recommendation,
    disclaimer,
    gradcam_image,
  } = result;

  const isMalignant = prediction === "Malignant";
  const probMalignant = typeof prob_malignant === "number" ? prob_malignant : 0;
  const probBenign = 100 - probMalignant;
  const confidenceVal = typeof confidence === "number" ? confidence : 0;

  const predColor = isMalignant ? "#f87171" : "#34d399";

  return (
    <div style={{ animation: "fadeUp 0.5s ease both" }}>

      {/* Main prediction card */}
      <div style={{
        borderRadius: "var(--radius-lg)",
        background: "var(--navy-mid)",
        border: `1px solid ${isMalignant ? "rgba(248,113,113,0.2)" : "rgba(52,211,153,0.2)"}`,
        overflow: "hidden",
        marginBottom: 16,
        boxShadow: isMalignant
          ? "0 4px 32px rgba(248,113,113,0.08)"
          : "0 4px 32px rgba(52,211,153,0.08)",
      }}>
        {/* Top strip */}
        <div style={{
          height: 3,
          background: isMalignant
            ? "linear-gradient(90deg, #f87171, #ef4444)"
            : "linear-gradient(90deg, #34d399, #10b981)",
        }}/>

        <div style={{ padding: "24px 24px 20px" }}>
          {/* Header row */}
          <div style={{
            display: "flex", alignItems: "flex-start",
            justifyContent: "space-between", marginBottom: 20,
          }}>
            <div>
              <p style={{
                fontSize: 11, fontWeight: 700, letterSpacing: "0.1em",
                textTransform: "uppercase", color: "var(--grey)",
                marginBottom: 6,
              }}>PREDICTION</p>
              <h2 style={{
                fontFamily: "var(--font-display)",
                fontSize: "clamp(1.8rem, 3vw, 2.4rem)",
                color: predColor,
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}>{prediction}</h2>
            </div>
            <RiskBadge riskLevel={risk_level} />
          </div>

          {/* Confidence */}
          <div style={{ marginBottom: 20 }}>
            <div style={{
              display: "flex", justifyContent: "space-between",
              marginBottom: 8,
            }}>
              <span style={{ fontSize: 12, color: "var(--grey)", fontWeight: 500 }}>
                Model confidence
              </span>
              <span style={{ fontSize: 13, fontWeight: 700, color: "var(--white)" }}>
                {confidenceVal.toFixed(1)}%
              </span>
            </div>
            <ConfidenceBar value={confidenceVal} color={predColor} />
          </div>

          {/* Probability breakdown */}
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12,
          }}>
            {[
              { label: "Malignant probability", value: probMalignant, color: "#f87171" },
              { label: "Benign probability",    value: probBenign,    color: "#34d399" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                padding: "14px 16px",
                borderRadius: "var(--radius-md)",
                background: "rgba(0,0,0,0.2)",
                border: "1px solid rgba(255,255,255,0.05)",
              }}>
                <p style={{ fontSize: 11, color: "var(--grey)", marginBottom: 6, fontWeight: 500 }}>
                  {label}
                </p>
                <p style={{ fontSize: 22, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>
                  {value.toFixed(1)}%
                </p>
                <div style={{ marginTop: 8 }}>
                  <ConfidenceBar value={value} color={color} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Heatmap */}
      {gradcam_image && previewImage && (
        <div style={{ marginBottom: 16 }}>
          <HeatmapOverlay
            originalImage={previewImage}
            heatmapBase64={gradcam_image}
          />
        </div>
      )}

      {/* Recommendation */}
      {recommendation && (
        <div style={{
          padding: "16px 18px",
          borderRadius: "var(--radius-md)",
          background: "rgba(15,184,169,0.06)",
          border: "1px solid rgba(15,184,169,0.2)",
          marginBottom: 12,
          display: "flex", gap: 12, alignItems: "flex-start",
        }}>
          <span style={{ fontSize: 18, flexShrink: 0, marginTop: 1 }}>💡</span>
          <div>
            <p style={{
              fontSize: 11, fontWeight: 700, color: "var(--teal)",
              letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 4,
            }}>Recommendation</p>
            <p style={{ fontSize: 13, color: "var(--white-dim)", lineHeight: 1.6 }}>
              {recommendation}
            </p>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div style={{
        padding: "12px 16px",
        borderRadius: "var(--radius-md)",
        background: "rgba(255,255,255,0.02)",
        border: "1px solid rgba(255,255,255,0.05)",
        display: "flex", gap: 10, alignItems: "flex-start",
      }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0, marginTop: 1 }}>
          <circle cx="12" cy="12" r="10" stroke="var(--grey)" strokeWidth="1.5"/>
          <path d="M12 8v4M12 16h.01" stroke="var(--grey)" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        <p style={{ fontSize: 11, color: "var(--grey)", lineHeight: 1.6 }}>
          {disclaimer || "This tool is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist."}
        </p>
      </div>
    </div>
  );
}
