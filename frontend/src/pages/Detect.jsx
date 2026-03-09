import { useState } from "react";
import ImageUpload from "../components/ImageUpload";
import ResultDisplay from "../components/ResultDisplay";
import { predictImage } from "../services/api";

function StepIndicator({ step, current }) {
  const done = current > step;
  const active = current === step;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        width: 28, height: 28, borderRadius: "50%",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 12, fontWeight: 700,
        background: done ? "var(--teal)" : active ? "rgba(15,184,169,0.15)" : "rgba(255,255,255,0.05)",
        border: `2px solid ${done || active ? "var(--teal)" : "rgba(255,255,255,0.1)"}`,
        color: done ? "var(--navy)" : active ? "var(--teal)" : "var(--grey)",
        transition: "all 0.3s ease", flexShrink: 0,
      }}>
        {done
          ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
          : step}
      </div>
    </div>
  );
}

export default function Detect() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [step, setStep]       = useState(1);

  const handleFileSelect = (f) => {
    setFile(f); setResult(null); setError(null); setStep(1);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setStep(2);
    try {
      const data = await predictImage(file);
      setResult(data); setStep(3);
    } catch (err) {
      setError(err?.response?.data?.detail || "Analysis failed. Please check the backend is running and try again.");
      setStep(1);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null); setPreview(null); setResult(null); setError(null); setStep(1);
  };

  const steps = [{ n: 1, label: "Upload" }, { n: 2, label: "Analyse" }, { n: 3, label: "Results" }];

  return (
    <div style={{ paddingTop: 68, minHeight: "100vh" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 40px 80px" }}>

        {/* Header */}
        <div style={{ marginBottom: 48 }}>
          <p style={{
            fontSize: 11, fontWeight: 700, letterSpacing: "0.12em",
            textTransform: "uppercase", color: "var(--teal)", marginBottom: 10,
          }}>LESION ANALYSIS</p>
          <h1 style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(1.8rem, 3.5vw, 2.8rem)",
            letterSpacing: "-0.03em", lineHeight: 1.1, marginBottom: 12,
            animation: "fadeUp 0.5s ease both",
          }}>Skin lesion detection</h1>
          <p style={{
            color: "var(--grey)", fontSize: 15, maxWidth: 520,
            animation: "fadeUp 0.5s ease 0.1s both",
          }}>
            Upload a dermoscopy or smartphone image. The model will classify it as malignant or benign and highlight regions of interest.
          </p>

          {/* Steps */}
          <div style={{ display: "flex", alignItems: "center", gap: 0, marginTop: 28, animation: "fadeUp 0.5s ease 0.2s both" }}>
            {steps.map(({ n, label }, i) => (
              <div key={n} style={{ display: "flex", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <StepIndicator step={n} current={step} />
                  <span style={{
                    fontSize: 12, fontWeight: 500,
                    color: step === n ? "var(--white)" : step > n ? "var(--teal)" : "var(--grey)",
                    transition: "color 0.3s ease",
                  }}>{label}</span>
                </div>
                {i < steps.length - 1 && (
                  <div style={{
                    height: 1, width: 40, margin: "0 8px",
                    background: step > n ? "var(--teal)" : "rgba(255,255,255,0.1)",
                    transition: "background 0.3s ease",
                  }}/>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Two-column layout — fluid, not fixed px */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 32,
          alignItems: "start",
        }} className="detect-grid">

          {/* Left */}
          <div>
            <div style={{
              padding: "28px",
              borderRadius: "var(--radius-lg)",
              background: "var(--navy-mid)",
              border: "1px solid rgba(255,255,255,0.06)",
              marginBottom: 16,
            }}>
              <p style={{
                fontSize: 11, fontWeight: 700, letterSpacing: "0.08em",
                textTransform: "uppercase", color: "var(--grey)", marginBottom: 16,
              }}>IMAGE</p>
              <ImageUpload onFileSelect={handleFileSelect} isLoading={loading} />
            </div>

            {error && (
              <div style={{
                padding: "14px 16px", borderRadius: "var(--radius-md)",
                background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.25)",
                marginBottom: 16, display: "flex", gap: 10, alignItems: "flex-start",
                animation: "fadeUp 0.3s ease both",
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0, marginTop: 1 }}>
                  <circle cx="12" cy="12" r="10" stroke="#f87171" strokeWidth="1.5"/>
                  <path d="M15 9l-6 6M9 9l6 6" stroke="#f87171" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
                <p style={{ fontSize: 13, color: "#f87171", lineHeight: 1.5 }}>{error}</p>
              </div>
            )}

            {file && !result && (
              <button onClick={handleAnalyze} disabled={loading} style={{
                width: "100%", padding: "15px 24px", borderRadius: "var(--radius-md)",
                background: loading ? "rgba(15,184,169,0.3)" : "var(--teal)",
                color: loading ? "rgba(255,255,255,0.5)" : "var(--navy)",
                fontSize: 15, fontWeight: 700, letterSpacing: "0.02em",
                cursor: loading ? "not-allowed" : "pointer", transition: "var(--transition)",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                animation: "fadeUp 0.3s ease both",
              }}
              onMouseEnter={e => { if (!loading) { e.currentTarget.style.background = "var(--teal-dim)"; e.currentTarget.style.boxShadow = "var(--shadow-teal)"; }}}
              onMouseLeave={e => { if (!loading) { e.currentTarget.style.background = "var(--teal)"; e.currentTarget.style.boxShadow = "none"; }}}>
                {loading ? (
                  <><div style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid rgba(255,255,255,0.2)", borderTopColor: "white", animation: "spin 0.7s linear infinite" }}/>Analysing...</>
                ) : (
                  <><svg width="16" height="16" viewBox="0 0 24 24" fill="none"><circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/><path d="M21 21l-4.35-4.35" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>Run Analysis</>
                )}
              </button>
            )}

            {result && (
              <button onClick={handleReset} style={{
                width: "100%", padding: "13px 24px", borderRadius: "var(--radius-md)",
                background: "transparent", color: "var(--grey)", fontSize: 14, fontWeight: 600,
                border: "1px solid rgba(255,255,255,0.1)", cursor: "pointer",
                transition: "var(--transition)", marginTop: 8,
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.25)"; e.currentTarget.style.color = "var(--white)"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"; e.currentTarget.style.color = "var(--grey)"; }}>
                Analyse another image
              </button>
            )}
          </div>

          {/* Right */}
          <div>
            {result ? (
              <ResultDisplay result={result} previewImage={preview} />
            ) : (
              <div style={{
                padding: "48px 32px", borderRadius: "var(--radius-lg)",
                background: "rgba(255,255,255,0.01)", border: "1px dashed rgba(255,255,255,0.06)",
                display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "center", textAlign: "center", gap: 16, minHeight: 320,
              }}>
                <div style={{
                  width: 56, height: 56, borderRadius: "50%",
                  background: "rgba(255,255,255,0.04)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2" stroke="var(--grey)" strokeWidth="1.5" strokeLinecap="round"/>
                    <rect x="9" y="3" width="6" height="4" rx="1" stroke="var(--grey)" strokeWidth="1.5"/>
                    <path d="M9 12h6M9 16h4" stroke="var(--grey)" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                </div>
                <div>
                  <p style={{ fontSize: 14, color: "var(--grey)", marginBottom: 4 }}>Results will appear here</p>
                  <p style={{ fontSize: 12, color: "rgba(148,163,184,0.5)" }}>Upload an image and click Run Analysis</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @media (max-width: 800px) {
          .detect-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}