import { useEffect } from "react";

const TEAM = [
  {
    name:   "Sharon David",
    role:   "ML Engineer & Full Stack Developer",
    bio:    "BTech Computer Engineering student specialising in medical AI, deep learning, and responsible ML systems.",
    links:  {
      linkedin: "https://www.linkedin.com/in/sharon-david-tech/",
      github:   "https://github.com/Sharon-David",
    },
  },
];

const TECH = [
  {
    category: "Model & Training",
    items: [
      { name: "EfficientNetB0",       desc: "Backbone — lightweight, high accuracy"              },
      { name: "TensorFlow / Keras",   desc: "Training framework"                                  },
      { name: "Albumentations",       desc: "Advanced augmentation pipeline"                      },
      { name: "Sequential Fine-Tuning", desc: "HAM10000 → DDI → PAD-UFES domain adaptation"      },
      { name: "Monte Carlo Dropout",  desc: "Uncertainty estimation across 20 forward passes"     },
    ],
  },
  {
    category: "Explainability & Safety",
    items: [
      { name: "Grad-CAM",             desc: "Visual explanation of model decisions"               },
      { name: "OOD Detection",        desc: "Rejects non-skin images before inference"            },
      { name: "Fairness Audit",       desc: "Per skin tone AUC across Fitzpatrick scale I–VI"     },
    ],
  },
  {
    category: "Backend & MLOps",
    items: [
      { name: "FastAPI",              desc: "Async REST API with automatic docs"                  },
      { name: "MLflow",               desc: "Experiment tracking & model registry"                },
      { name: "Prometheus",           desc: "API metrics — latency, confidence drift, OOD rate"   },
      { name: "Grafana",              desc: "Real-time monitoring dashboards"                     },
      { name: "Docker Compose",       desc: "One-command deployment of full stack"                },
      { name: "GitHub Actions",       desc: "CI/CD — auto-evaluate on every push"                 },
    ],
  },
  {
    category: "Frontend",
    items: [
      { name: "React + Vite",         desc: "Fast, modern UI framework"                           },
      { name: "Axios",                desc: "HTTP client with timeout handling"                   },
    ],
  },
];

const DATASETS = [
  {
    name:    "HAM10000",
    images:  "10,015",
    type:    "Dermoscopy",
    source:  "ISIC Archive",
    purpose: "Foundation model training",
    auc:     "0.769",
    color:   "var(--teal)",
  },
  {
    name:    "DDI",
    images:  "656",
    type:    "Clinical photos",
    source:  "Stanford",
    purpose: "Fairness fine-tuning (skin tones I–VI)",
    auc:     "In progress",
    color:   "#60a5fa",
  },
  {
    name:    "PAD-UFES-20",
    images:  "2,298",
    type:    "Smartphone photos",
    source:  "Brazil",
    purpose: "Real-world domain adaptation",
    auc:     "Pending",
    color:   "#a78bfa",
  },
];

export default function About() {
  useEffect(() => { window.scrollTo(0, 0); }, []);

  return (
    <main style={{
      paddingTop: 118,
      minHeight: "100vh",
      background: "var(--navy)",
      color: "var(--white)",
    }}>
      <div style={{ maxWidth: 900, margin: "0 auto", padding: "48px 24px" }}>

        {/* Hero */}
        <div style={{ textAlign: "center", marginBottom: 64 }}>
          <h1 style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2.5rem, 5vw, 3.5rem)",
            letterSpacing: "-0.02em",
            marginBottom: 16,
          }}>
            About <span style={{ color: "var(--teal)" }}>DermAI</span>
          </h1>
          <p style={{
            fontSize: "clamp(1rem, 2vw, 1.15rem)",
            color: "var(--white-dim)",
            maxWidth: 620,
            margin: "0 auto",
            lineHeight: 1.7,
          }}>
            A research-grade skin lesion classification system built with fairness,
            explainability, and responsible AI at its core.
          </p>
        </div>

        {/* Mission */}
        <Section title="Mission">
          <p style={{ color: "var(--white-dim)", lineHeight: 1.8, fontSize: 15 }}>
            Skin cancer is the most common cancer worldwide — and also one of the most
            treatable when caught early. DermAI is a research prototype exploring how deep
            learning can assist clinicians with early detection, while remaining transparent
            about its limitations and fair across diverse patient populations.
          </p>
          <p style={{ color: "var(--white-dim)", lineHeight: 1.8, fontSize: 15, marginTop: 12 }}>
            Unlike most academic projects, DermAI explicitly audits performance across
            Fitzpatrick skin tones I–VI using the DDI dataset — addressing a known bias in
            dermatology AI systems that historically underperform on darker skin tones.
          </p>
          <div style={{
            marginTop: 20,
            padding: "14px 18px",
            borderRadius: "var(--radius-md)",
            background: "rgba(248,113,113,0.06)",
            border: "1px solid rgba(248,113,113,0.15)",
            fontSize: 13,
            color: "#fca5a5",
          }}>
            ⚠ DermAI is a research tool only. It is not a medical device and does not
            provide clinical diagnoses. Always consult a qualified dermatologist.
          </div>
        </Section>

        {/* Datasets */}
        <Section title="Training Datasets">
          <div style={{ display: "grid", gap: 16 }}>
            {DATASETS.map(d => (
              <div key={d.name} style={{
                padding: "18px 20px",
                borderRadius: "var(--radius-md)",
                background: "var(--navy-mid)",
                border: `1px solid rgba(255,255,255,0.06)`,
                display: "grid",
                gridTemplateColumns: "1fr auto",
                gap: 12,
                alignItems: "center",
              }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: "50%",
                      background: d.color, display: "inline-block", flexShrink: 0,
                    }}/>
                    <span style={{ fontWeight: 700, fontSize: 15 }}>{d.name}</span>
                    <span style={{
                      fontSize: 11, padding: "2px 8px", borderRadius: 20,
                      background: "rgba(255,255,255,0.05)",
                      color: "var(--grey)", fontWeight: 500,
                    }}>{d.type}</span>
                  </div>
                  <p style={{ fontSize: 13, color: "var(--grey)", marginLeft: 18 }}>
                    {d.images} images · {d.source} · {d.purpose}
                  </p>
                </div>
                <div style={{ textAlign: "right" }}>
                  <p style={{ fontSize: 10, color: "var(--grey)", marginBottom: 2 }}>Val AUC</p>
                  <p style={{ fontSize: 18, fontWeight: 700, color: d.color }}>{d.auc}</p>
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* Model architecture */}
        <Section title="Model Architecture">
          <div style={{
            padding: "20px 24px",
            borderRadius: "var(--radius-md)",
            background: "var(--navy-mid)",
            border: "1px solid rgba(255,255,255,0.06)",
            fontFamily: "monospace",
            fontSize: 13,
            lineHeight: 2,
            color: "var(--white-dim)",
          }}>
            <div>Input (224×224×3)</div>
            <div style={{ color: "var(--teal)", marginLeft: 16 }}>↓ EfficientNetB0 backbone (ImageNet pretrained)</div>
            <div style={{ marginLeft: 32 }}>↓ GlobalAveragePooling2D</div>
            <div style={{ marginLeft: 32 }}>↓ BatchNormalization</div>
            <div style={{ marginLeft: 32 }}>↓ Dropout(0.4)</div>
            <div style={{ marginLeft: 32 }}>↓ Dense(256, relu, L2)</div>
            <div style={{ marginLeft: 32 }}>↓ BatchNormalization</div>
            <div style={{ marginLeft: 32 }}>↓ Dropout(0.3)</div>
            <div style={{ marginLeft: 32 }}>↓ Dense(64, relu)</div>
            <div style={{ marginLeft: 32 }}>↓ Dropout(0.2)</div>
            <div style={{ color: "#f87171", marginLeft: 32 }}>↓ Dense(1, sigmoid) → P(malignant)</div>
          </div>
          <div style={{
            marginTop: 16, display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12,
          }}>
            {[
              { label: "Loss",        value: "Binary crossentropy"   },
              { label: "Optimizer",   value: "Adam"                  },
              { label: "Batch size",  value: "8"                     },
              { label: "Image size",  value: "224 × 224"             },
              { label: "Fine-tuning", value: "Progressive unfreezing"},
              { label: "Weighting",   value: "√(neg/pos) ≈ 2.03×"   },
            ].map(({ label, value }) => (
              <div key={label} style={{
                padding: "12px 14px",
                borderRadius: "var(--radius-sm)",
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.06)",
              }}>
                <p style={{ fontSize: 10, color: "var(--grey)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</p>
                <p style={{ fontSize: 13, fontWeight: 600 }}>{value}</p>
              </div>
            ))}
          </div>
        </Section>

        {/* Tech stack */}
        <Section title="Technology Stack">
          <div style={{ display: "grid", gap: 24 }}>
            {TECH.map(({ category, items }) => (
              <div key={category}>
                <p style={{
                  fontSize: 11, fontWeight: 700, letterSpacing: "0.1em",
                  color: "var(--teal)", textTransform: "uppercase", marginBottom: 12,
                }}>{category}</p>
                <div style={{ display: "grid", gap: 8 }}>
                  {items.map(({ name, desc }) => (
                    <div key={name} style={{
                      display: "flex", alignItems: "center", gap: 12,
                      padding: "10px 14px",
                      borderRadius: "var(--radius-sm)",
                      background: "rgba(255,255,255,0.02)",
                      border: "1px solid rgba(255,255,255,0.05)",
                    }}>
                      <span style={{
                        width: 6, height: 6, borderRadius: "50%",
                        background: "var(--teal)", flexShrink: 0,
                      }}/>
                      <span style={{ fontWeight: 600, fontSize: 13, minWidth: 180 }}>{name}</span>
                      <span style={{ fontSize: 12, color: "var(--grey)" }}>{desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* Team */}
        <Section title="Developer">
          {TEAM.map(({ name, role, bio, links }) => (
            <div key={name} style={{
              padding: "24px",
              borderRadius: "var(--radius-lg)",
              background: "var(--navy-mid)",
              border: "1px solid rgba(255,255,255,0.06)",
            }}>
              <div style={{
                width: 48, height: 48, borderRadius: "50%",
                background: "var(--teal)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontWeight: 700, fontSize: 20, marginBottom: 14,
              }}>
                {name[0]}
              </div>
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>{name}</h3>
              <p style={{ fontSize: 13, color: "var(--teal)", marginBottom: 10 }}>{role}</p>
              <p style={{ fontSize: 13, color: "var(--white-dim)", lineHeight: 1.7, marginBottom: 16 }}>{bio}</p>
              <div style={{ display: "flex", gap: 12 }}>
                {Object.entries(links).map(([platform, url]) => (
                  <a key={platform} href={url} target="_blank" rel="noopener noreferrer"
                    style={{
                      fontSize: 12, fontWeight: 600,
                      padding: "6px 14px", borderRadius: "var(--radius-sm)",
                      background: "rgba(15,184,169,0.1)",
                      border: "1px solid rgba(15,184,169,0.25)",
                      color: "var(--teal)",
                      textDecoration: "none",
                      textTransform: "capitalize",
                    }}>
                    {platform}
                  </a>
                ))}
              </div>
            </div>
          ))}
        </Section>

      </div>
    </main>
  );
}

function Section({ title, children }) {
  return (
    <section style={{ marginBottom: 56 }}>
      <h2 style={{
        fontFamily: "var(--font-display)",
        fontSize: "clamp(1.4rem, 3vw, 1.8rem)",
        letterSpacing: "-0.01em",
        marginBottom: 24,
        paddingBottom: 12,
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}>
        {title}
      </h2>
      {children}
    </section>
  );
}
