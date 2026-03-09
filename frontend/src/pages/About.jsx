export default function About() {
  const datasets = [
    {
      name: "HAM10000",
      role: "Foundation training",
      desc: "10,000+ dermoscopy images across 7 lesion classes. Primary training corpus for the EfficientNetB0 backbone.",
      tag: "Step 1",
      color: "var(--teal)",
    },
    {
      name: "DDI",
      role: "Fairness adaptation",
      desc: "Diverse Dermatology Images — covers Fitzpatrick skin tones I–VI. Fine-tuning ensures equitable performance across skin tones.",
      tag: "Step 2",
      color: "#a78bfa",
    },
    {
      name: "PAD-UFES-20",
      role: "Real-world adaptation",
      desc: "Smartphone photographs from clinical practice. Fine-tuning adapts the model beyond dermoscope-only images.",
      tag: "Step 3",
      color: "#fbbf24",
    },
  ];

  const pipeline = [
    { n: "01", title: "Image Preprocessing", desc: "Resize to 224×224, ImageNet normalisation. Albumentations augmentation during training (flips, colour jitter, noise)." },
    { n: "02", title: "EfficientNetB0 Backbone", desc: "Pre-trained on ImageNet. Global average pooling → BatchNorm → Dropout → Dense(256) → Dense(64) → Sigmoid." },
    { n: "03", title: "Focal Loss Training", desc: "Binary focal loss (γ=2, α=0.25) handles class imbalance better than standard BCE by down-weighting easy examples." },
    { n: "04", title: "Sequential Fine-Tuning", desc: "Phase 1: frozen base, head warmup. Phase 2: top 50 layers unfrozen at LR=5e-6 to prevent catastrophic forgetting." },
    { n: "05", title: "Grad-CAM Explanation", desc: "Class activation maps from the final convolutional layer highlight which regions drove the prediction." },
    { n: "06", title: "OOD Detection", desc: "Heuristic checks (resolution, brightness) and model uncertainty flags images where the model should not be trusted." },
  ];

  return (
    <div style={{ paddingTop: 118 }}>
      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "60px 24px 100px" }}>

        {/* Header */}
        <div style={{ marginBottom: 64, animation: "fadeUp 0.5s ease both" }}>
          <p style={{
            fontSize: 11, fontWeight: 700, letterSpacing: "0.12em",
            textTransform: "uppercase", color: "var(--teal)", marginBottom: 12,
          }}>ABOUT</p>
          <h1 style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2rem, 4vw, 3rem)",
            letterSpacing: "-0.03em", lineHeight: 1.1, marginBottom: 20,
          }}>
            How DermAI works
          </h1>
          <p style={{
            fontSize: 16, color: "var(--grey)", lineHeight: 1.8, maxWidth: 680,
          }}>
            DermAI is a research-grade skin lesion classification system built with sequential domain adaptation across three public clinical datasets. It is designed to be accurate, explainable, and equitable.
          </p>
        </div>

        {/* Datasets */}
        <section style={{ marginBottom: 80 }}>
          <h2 style={{
            fontFamily: "var(--font-display)",
            fontSize: "1.6rem", marginBottom: 28,
            letterSpacing: "-0.02em",
          }}>Training pipeline</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {datasets.map(({ name, role, desc, tag, color }, i) => (
              <div key={name} style={{
                display: "grid",
                gridTemplateColumns: "auto 1fr",
                gap: 20,
                padding: "24px",
                borderRadius: "var(--radius-lg)",
                background: "var(--navy-mid)",
                border: "1px solid rgba(255,255,255,0.06)",
                animation: `fadeUp 0.5s ease ${i * 0.1}s both`,
              }}>
                <div style={{
                  width: 48, height: 48, borderRadius: "var(--radius-md)",
                  background: `rgba(${color === "var(--teal)" ? "15,184,169" : color === "#a78bfa" ? "167,139,250" : "251,191,36"}, 0.1)`,
                  border: `1px solid ${color}33`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 11, fontWeight: 800,
                  color, letterSpacing: "0.04em",
                  flexShrink: 0,
                }}>{tag}</div>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                    <h3 style={{ fontSize: 16, fontWeight: 700, color: "var(--white)" }}>{name}</h3>
                    <span style={{
                      fontSize: 11, padding: "2px 8px", borderRadius: 100,
                      background: `${color}18`, color,
                      fontWeight: 600, border: `1px solid ${color}33`,
                    }}>{role}</span>
                  </div>
                  <p style={{ fontSize: 13, color: "var(--grey)", lineHeight: 1.6 }}>{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Technical pipeline */}
        <section style={{ marginBottom: 80 }}>
          <h2 style={{
            fontFamily: "var(--font-display)",
            fontSize: "1.6rem", marginBottom: 28, letterSpacing: "-0.02em",
          }}>Technical details</h2>
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16,
          }} className="pipeline-grid">
            {pipeline.map(({ n, title, desc }, i) => (
              <div key={n} style={{
                padding: "20px 22px",
                borderRadius: "var(--radius-md)",
                background: "var(--navy-mid)",
                border: "1px solid rgba(255,255,255,0.05)",
                animation: `fadeUp 0.5s ease ${i * 0.07}s both`,
              }}>
                <div style={{
                  fontSize: 10, fontWeight: 800, color: "var(--teal)",
                  letterSpacing: "0.1em", marginBottom: 8,
                }}>{n}</div>
                <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 6 }}>{title}</h3>
                <p style={{ fontSize: 12, color: "var(--grey)", lineHeight: 1.7 }}>{desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Limitations */}
        <section>
          <div style={{
            padding: "28px 32px",
            borderRadius: "var(--radius-lg)",
            background: "rgba(251,191,36,0.04)",
            border: "1px solid rgba(251,191,36,0.15)",
          }}>
            <h2 style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.4rem", marginBottom: 16,
              letterSpacing: "-0.02em",
              color: "#fbbf24",
            }}>Limitations & responsible use</h2>
            <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                "This system is a research prototype, not a certified medical device.",
                "Performance may degrade on image types not represented in training data.",
                "Always seek professional dermatological evaluation for any skin concern.",
                "The model outputs probabilities, not definitive diagnoses.",
              ].map((item, i) => (
                <li key={i} style={{
                  display: "flex", gap: 10, alignItems: "flex-start",
                  fontSize: 13, color: "rgba(251,191,36,0.7)", lineHeight: 1.6,
                }}>
                  <span style={{ color: "#fbbf24", flexShrink: 0, marginTop: 2 }}>—</span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </section>
      </div>

      <style>{`
        @media (max-width: 680px) {
          .pipeline-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
