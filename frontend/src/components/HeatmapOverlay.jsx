import { useState } from "react";

export default function HeatmapOverlay({ originalImage, heatmapBase64 }) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.6);

  if (!heatmapBase64) return null;

  const heatmapSrc = heatmapBase64.startsWith("data:")
    ? heatmapBase64
    : `data:image/png;base64,${heatmapBase64}`;

  return (
    <div style={{
      borderRadius: "var(--radius-lg)",
      overflow: "hidden",
      border: "1px solid rgba(255,255,255,0.07)",
      background: "var(--navy-mid)",
    }}>
      {/* Header */}
      <div style={{
        padding: "14px 18px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%",
            background: "var(--teal)", display: "inline-block",
          }}/>
          <span style={{
            fontSize: 12, fontWeight: 700,
            letterSpacing: "0.08em", textTransform: "uppercase",
            color: "var(--teal)",
          }}>Grad-CAM Explanation</span>
        </div>

        {/* Toggle */}
        <button
          onClick={() => setShowOverlay(!showOverlay)}
          style={{
            fontSize: 11, fontWeight: 600,
            padding: "4px 10px", borderRadius: "var(--radius-sm)",
            background: showOverlay ? "rgba(15,184,169,0.1)" : "rgba(255,255,255,0.05)",
            border: `1px solid ${showOverlay ? "rgba(15,184,169,0.3)" : "rgba(255,255,255,0.1)"}`,
            color: showOverlay ? "var(--teal)" : "var(--grey)",
            cursor: "pointer",
            transition: "var(--transition)",
            letterSpacing: "0.04em",
          }}>
          {showOverlay ? "Heatmap ON" : "Heatmap OFF"}
        </button>
      </div>

      {/* Image stack */}
      <div style={{ position: "relative", lineHeight: 0 }}>
        <img
          src={originalImage}
          alt="Original"
          style={{ width: "100%", display: "block", maxHeight: 300, objectFit: "contain" }}
        />
        {showOverlay && (
          <img
            src={heatmapSrc}
            alt="Grad-CAM heatmap"
            style={{
              position: "absolute", inset: 0,
              width: "100%", height: "100%",
              objectFit: "contain",
              opacity: opacity,
              mixBlendMode: "screen",
              transition: "opacity 0.2s ease",
            }}
          />
        )}
      </div>

      {/* Opacity slider */}
      {showOverlay && (
        <div style={{
          padding: "14px 18px",
          borderTop: "1px solid rgba(255,255,255,0.06)",
          display: "flex", alignItems: "center", gap: 12,
        }}>
          <span style={{ fontSize: 11, color: "var(--grey)", whiteSpace: "nowrap", fontWeight: 500 }}>
            Overlay
          </span>
          <input
            type="range" min="0.1" max="1" step="0.05"
            value={opacity}
            onChange={e => setOpacity(parseFloat(e.target.value))}
            style={{
              flex: 1, appearance: "none",
              height: 3, borderRadius: 2,
              background: `linear-gradient(to right, var(--teal) ${opacity * 100}%, rgba(255,255,255,0.1) ${opacity * 100}%)`,
              cursor: "pointer", outline: "none",
            }}
          />
          <span style={{ fontSize: 11, color: "var(--grey)", minWidth: 32, textAlign: "right" }}>
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}

      {/* Legend */}
      <div style={{
        padding: "10px 18px",
        borderTop: "1px solid rgba(255,255,255,0.06)",
        display: "flex", alignItems: "center", gap: 6,
      }}>
        <div style={{
          height: 8, flex: 1, borderRadius: 4,
          background: "linear-gradient(to right, #1a1aff, #00ffff, #00ff00, #ffff00, #ff0000)",
        }}/>
        <div style={{
          display: "flex", justifyContent: "space-between",
          fontSize: 10, color: "var(--grey)", marginTop: 4,
          width: "100%", position: "absolute",
          left: 18, right: 18,
          bottom: 4,
        }}>
        </div>
      </div>
      <div style={{
        padding: "0 18px 10px",
        display: "flex", justifyContent: "space-between",
        fontSize: 10, color: "var(--grey)", fontWeight: 500,
      }}>
        <span>Low attention</span>
        <span>High attention</span>
      </div>
    </div>
  );
}
