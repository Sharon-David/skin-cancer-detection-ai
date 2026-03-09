import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";

function StatCard({ value, label, delay }) {
  return (
    <div style={{
      textAlign: "center",
      animation: `fadeUp 0.6s ease ${delay}s both`,
    }}>
      <div style={{
        fontFamily: "var(--font-display)",
        fontSize: "clamp(2rem, 4vw, 3rem)",
        color: "var(--teal)",
        lineHeight: 1,
        marginBottom: 6,
      }}>{value}</div>
      <div style={{
        fontSize: 13,
        color: "var(--grey)",
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        fontWeight: 500,
      }}>{label}</div>
    </div>
  );
}

function FeatureCard({ icon, title, desc, delay }) {
  return (
    <div style={{
      padding: "32px 28px",
      borderRadius: "var(--radius-lg)",
      background: "var(--navy-mid)",
      border: "1px solid rgba(255,255,255,0.06)",
      transition: "var(--transition)",
      animation: `fadeUp 0.6s ease ${delay}s both`,
      cursor: "default",
    }}
    onMouseEnter={e => {
      e.currentTarget.style.borderColor = "rgba(15,184,169,0.3)";
      e.currentTarget.style.transform = "translateY(-4px)";
      e.currentTarget.style.boxShadow = "var(--shadow-teal)";
    }}
    onMouseLeave={e => {
      e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)";
      e.currentTarget.style.transform = "translateY(0)";
      e.currentTarget.style.boxShadow = "none";
    }}>
      <div style={{
        width: 48, height: 48, borderRadius: "var(--radius-md)",
        background: "rgba(15,184,169,0.12)",
        display: "flex", alignItems: "center", justifyContent: "center",
        marginBottom: 20,
        fontSize: 22,
      }}>{icon}</div>
      <h3 style={{
        fontFamily: "var(--font-display)",
        fontSize: 20, marginBottom: 10,
        color: "var(--white)",
      }}>{title}</h3>
      <p style={{
        fontSize: 14, lineHeight: 1.7,
        color: "var(--grey)",
      }}>{desc}</p>
    </div>
  );
}

export default function Home() {
  return (
    <div style={{ paddingTop: 118 }}>

      {/* Hero */}
      <section style={{
        minHeight: "92vh",
        display: "flex", alignItems: "center",
        position: "relative", overflow: "hidden",
        padding: "80px 24px",
      }}>
        {/* Background orbs */}
        <div style={{
          position: "absolute", top: "10%", right: "5%",
          width: 600, height: 600, borderRadius: "50%",
          background: "radial-gradient(circle, rgba(15,184,169,0.08) 0%, transparent 70%)",
          pointerEvents: "none",
        }}/>
        <div style={{
          position: "absolute", bottom: "5%", left: "-10%",
          width: 500, height: 500, borderRadius: "50%",
          background: "radial-gradient(circle, rgba(15,100,184,0.06) 0%, transparent 70%)",
          pointerEvents: "none",
        }}/>

        <div style={{ maxWidth: 1200, margin: "0 auto", width: "100%" }}>
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 64, alignItems: "center",
          }} className="hero-grid">

            {/* Left */}
            <div>
              <div style={{
                display: "inline-flex", alignItems: "center", gap: 8,
                padding: "6px 14px",
                borderRadius: 100,
                background: "rgba(15,184,169,0.1)",
                border: "1px solid rgba(15,184,169,0.25)",
                marginBottom: 28,
                animation: "fadeUp 0.5s ease 0.1s both",
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: "var(--teal)",
                  animation: "pulse-teal 2s infinite",
                  display: "inline-block",
                }}/>
                <span style={{ fontSize: 12, color: "var(--teal)", fontWeight: 600, letterSpacing: "0.06em" }}>
                  AI-POWERED DERMATOLOGY
                </span>
              </div>

              <h1 style={{
                fontFamily: "var(--font-display)",
                fontSize: "clamp(2.4rem, 5vw, 3.8rem)",
                lineHeight: 1.1,
                letterSpacing: "-0.03em",
                marginBottom: 24,
                animation: "fadeUp 0.5s ease 0.2s both",
              }}>
                Early detection<br/>
                <span style={{ color: "var(--teal)" }}>saves lives.</span>
              </h1>

              <p style={{
                fontSize: 17, lineHeight: 1.75,
                color: "var(--grey)",
                maxWidth: 480,
                marginBottom: 40,
                animation: "fadeUp 0.5s ease 0.3s both",
              }}>
                Dermoscopy analysis powered by deep learning trained across three clinical datasets. Designed with fairness across all skin tones.
              </p>

              <div style={{
                display: "flex", gap: 12, flexWrap: "wrap",
                animation: "fadeUp 0.5s ease 0.4s both",
              }}>
                <Link to="/detect" style={{
                  padding: "14px 32px",
                  borderRadius: "var(--radius-md)",
                  background: "var(--teal)",
                  color: "var(--navy)",
                  fontWeight: 700,
                  fontSize: 15,
                  letterSpacing: "0.02em",
                  transition: "var(--transition)",
                  display: "inline-flex", alignItems: "center", gap: 8,
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = "var(--teal-dim)";
                  e.currentTarget.style.boxShadow = "var(--shadow-teal)";
                  e.currentTarget.style.transform = "translateY(-2px)";
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = "var(--teal)";
                  e.currentTarget.style.boxShadow = "none";
                  e.currentTarget.style.transform = "translateY(0)";
                }}>
                  Analyze a lesion
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M5 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </Link>
                <Link to="/about" style={{
                  padding: "14px 32px",
                  borderRadius: "var(--radius-md)",
                  background: "transparent",
                  color: "var(--white-dim)",
                  fontWeight: 500,
                  fontSize: 15,
                  border: "1px solid rgba(255,255,255,0.12)",
                  transition: "var(--transition)",
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = "rgba(255,255,255,0.3)";
                  e.currentTarget.style.color = "var(--white)";
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)";
                  e.currentTarget.style.color = "var(--white-dim)";
                }}>
                  How it works
                </Link>
              </div>
            </div>

            {/* Right — visual */}
            <div style={{
              animation: "fadeIn 0.8s ease 0.3s both",
              display: "flex", justifyContent: "center",
            }}>
              <div style={{
                width: "100%", maxWidth: 420, aspectRatio: "1",
                borderRadius: "var(--radius-xl)",
                background: "var(--navy-mid)",
                border: "1px solid rgba(255,255,255,0.07)",
                display: "flex", alignItems: "center", justifyContent: "center",
                position: "relative", overflow: "hidden",
                boxShadow: "var(--shadow-lg), var(--shadow-teal)",
              }}>
                {/* Decorative rings */}
                {[280, 220, 160].map((size, i) => (
                  <div key={i} style={{
                    position: "absolute",
                    width: size, height: size,
                    borderRadius: "50%",
                    border: `1px solid rgba(15,184,169,${0.06 + i * 0.04})`,
                  }}/>
                ))}
                {/* Centre icon */}
                <div style={{
                  width: 96, height: 96,
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, rgba(15,184,169,0.25), rgba(15,184,169,0.05))",
                  border: "1px solid rgba(15,184,169,0.3)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  position: "relative", zIndex: 1,
                  boxShadow: "0 0 40px rgba(15,184,169,0.2)",
                }}>
                  <svg width="44" height="44" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2a10 10 0 100 20A10 10 0 0012 2z" stroke="var(--teal)" strokeWidth="1.5" opacity="0.4"/>
                    <circle cx="12" cy="12" r="4" fill="var(--teal)" opacity="0.8"/>
                    <path d="M12 6v2M12 16v2M6 12H4M20 12h-2" stroke="var(--teal)" strokeWidth="1.5" strokeLinecap="round" opacity="0.5"/>
                  </svg>
                </div>
                {/* Corner labels */}
                {[
                  { label: "HAM10000", top: "12%", left: "8%" },
                  { label: "DDI Dataset", top: "12%", right: "8%" },
                  { label: "PAD-UFES", bottom: "12%", left: "8%" },
                  { label: "Grad-CAM XAI", bottom: "12%", right: "8%" },
                ].map(({ label, ...pos }) => (
                  <div key={label} style={{
                    position: "absolute", ...pos,
                    fontSize: 10, fontWeight: 600,
                    color: "var(--teal)",
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    background: "rgba(15,184,169,0.08)",
                    padding: "4px 8px",
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid rgba(15,184,169,0.15)",
                  }}>{label}</div>
                ))}
              </div>
            </div>
          </div>

          {/* Stats strip */}
          <div style={{
            marginTop: 80,
            padding: "40px 48px",
            borderRadius: "var(--radius-lg)",
            background: "var(--navy-mid)",
            border: "1px solid rgba(255,255,255,0.06)",
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 32,
            animation: "fadeUp 0.6s ease 0.5s both",
          }} className="stats-grid">
            <StatCard value="10k+" label="Training Images" delay={0.5}/>
            <StatCard value="3" label="Clinical Datasets" delay={0.6}/>
            <StatCard value="7" label="Lesion Classes" delay={0.7}/>
            <StatCard value="I–VI" label="Fitzpatrick Coverage" delay={0.8}/>
          </div>
        </div>
      </section>

      {/* Features */}
      <section style={{ padding: "100px 24px", background: "rgba(0,0,0,0.15)" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 64 }}>
            <p style={{
              fontSize: 12, fontWeight: 700, letterSpacing: "0.12em",
              textTransform: "uppercase", color: "var(--teal)",
              marginBottom: 12,
            }}>CAPABILITIES</p>
            <h2 style={{
              fontFamily: "var(--font-display)",
              fontSize: "clamp(1.8rem, 3.5vw, 2.8rem)",
              letterSpacing: "-0.02em",
            }}>
              Built for clinical reliability
            </h2>
          </div>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 24,
          }} className="features-grid">
            <FeatureCard
              icon="🔬"
              title="Deep Lesion Analysis"
              desc="EfficientNetB0 trained on HAM10000 dermoscopy images, fine-tuned for binary malignant vs. benign classification with calibrated confidence scores."
              delay={0.1}
            />
            <FeatureCard
              icon="⚖️"
              title="Fairness by Design"
              desc="Domain-adapted on the DDI dataset to ensure consistent performance across all Fitzpatrick skin tones — from Type I to Type VI."
              delay={0.2}
            />
            <FeatureCard
              icon="🗺️"
              title="Explainable AI"
              desc="Grad-CAM heatmaps highlight the exact regions driving each prediction, giving clinicians a transparent view into the model's reasoning."
              delay={0.3}
            />
            <FeatureCard
              icon="📱"
              title="Real-World Ready"
              desc="Sequential fine-tuning on PAD-UFES-20 adapts the model for smartphone photography — not just clinical dermoscope images."
              delay={0.4}
            />
            <FeatureCard
              icon="🛡️"
              title="Safety Guardrails"
              desc="Out-of-distribution detection flags low-quality, blurry, or non-dermatological images before inference, preventing misleading results."
              delay={0.5}
            />
            <FeatureCard
              icon="⚡"
              title="Sub-second Inference"
              desc="FastAPI backend serves predictions with Grad-CAM overlays in under one second. Designed for responsive, real-time clinical workflows."
              delay={0.6}
            />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ padding: "100px 24px", textAlign: "center" }}>
        <div style={{ maxWidth: 600, margin: "0 auto" }}>
          <h2 style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(1.8rem, 3.5vw, 2.6rem)",
            letterSpacing: "-0.02em",
            marginBottom: 16,
            animation: "fadeUp 0.6s ease 0.1s both",
          }}>
            Ready to analyze?
          </h2>
          <p style={{
            color: "var(--grey)", fontSize: 16, marginBottom: 36,
            animation: "fadeUp 0.6s ease 0.2s both",
          }}>
            Upload a dermoscopy or smartphone image and receive a prediction with confidence score and visual explanation in seconds.
          </p>
          <Link to="/detect" style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "16px 40px",
            borderRadius: "var(--radius-md)",
            background: "var(--teal)",
            color: "var(--navy)",
            fontWeight: 700, fontSize: 16,
            transition: "var(--transition)",
            animation: "fadeUp 0.6s ease 0.3s both",
          }}
          onMouseEnter={e => {
            e.currentTarget.style.transform = "translateY(-3px)";
            e.currentTarget.style.boxShadow = "var(--shadow-teal)";
          }}
          onMouseLeave={e => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "none";
          }}>
            Start Analysis
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M5 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </Link>
        </div>
      </section>

      <style>{`
        @media (max-width: 900px) {
          .hero-grid { grid-template-columns: 1fr !important; }
          .features-grid { grid-template-columns: 1fr 1fr !important; }
          .stats-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
        @media (max-width: 600px) {
          .features-grid { grid-template-columns: 1fr !important; }
          .stats-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
      `}</style>
    </div>
  );
}
