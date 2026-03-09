export default function Resources() {
  const resources = [
    {
      category: "Datasets",
      items: [
        {
          title: "HAM10000",
          desc: "Human Against Machine with 10000 training images. The primary dermoscopy dataset used for baseline training.",
          link: "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
        },
        {
          title: "DDI — Diverse Dermatology Images",
          desc: "1,000+ images labeled with Fitzpatrick skin tone. Used for fairness adaptation in this project.",
          link: "https://ddi-dataset.github.io",
        },
        {
          title: "PAD-UFES-20",
          desc: "Smartphone images from a Brazilian clinical practice. Six skin lesion types across 1,641 patients.",
          link: "https://data.mendeley.com/datasets/zr7vgbcyr2/1",
        },
      ],
    },
    {
      category: "Key Papers",
      items: [
        {
          title: "EfficientNet: Rethinking Model Scaling",
          desc: "Tan & Le (2019). The architecture backbone used in this project.",
          link: "https://arxiv.org/abs/1905.11946",
        },
        {
          title: "Focal Loss for Dense Object Detection",
          desc: "Lin et al. (2017). The loss function used to handle class imbalance during training.",
          link: "https://arxiv.org/abs/1708.02002",
        },
        {
          title: "Grad-CAM: Visual Explanations from Deep Networks",
          desc: "Selvaraju et al. (2017). The explainability technique used to generate attention heatmaps.",
          link: "https://arxiv.org/abs/1610.02391",
        },
      ],
    },
    {
      category: "Clinical Context",
      items: [
        {
          title: "ABCDE Rule for Melanoma",
          desc: "Asymmetry, Border, Colour, Diameter, Evolution — the clinical framework for visual lesion assessment.",
          link: "https://www.aad.org/public/diseases/skin-cancer/find/at-risk/abcdes",
        },
        {
          title: "Fitzpatrick Scale",
          desc: "A six-point skin phototype classification used to assess skin tone diversity in dermatological AI.",
          link: "https://en.wikipedia.org/wiki/Fitzpatrick_scale",
        },
        {
          title: "AI Bias in Dermatology — Review",
          desc: "Overview of equity challenges and mitigation strategies in AI-based dermatological tools.",
          link: "https://www.nature.com/articles/s41746-022-00592-y",
        },
      ],
    },
  ];

  return (
    <div style={{ paddingTop: 118 }}>
      <div style={{ maxWidth: 900, margin: "0 auto", padding: "60px 24px 100px" }}>

        <div style={{ marginBottom: 56, animation: "fadeUp 0.5s ease both" }}>
          <p style={{
            fontSize: 11, fontWeight: 700, letterSpacing: "0.12em",
            textTransform: "uppercase", color: "var(--teal)", marginBottom: 12,
          }}>RESOURCES</p>
          <h1 style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(1.8rem, 3.5vw, 2.8rem)",
            letterSpacing: "-0.03em",
          }}>
            Reference materials
          </h1>
        </div>

        {resources.map(({ category, items }, ci) => (
          <section key={category} style={{ marginBottom: 56 }}>
            <h2 style={{
              fontSize: 12, fontWeight: 700,
              letterSpacing: "0.1em", textTransform: "uppercase",
              color: "var(--grey)", marginBottom: 20,
              paddingBottom: 12,
              borderBottom: "1px solid rgba(255,255,255,0.06)",
            }}>{category}</h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {items.map(({ title, desc, link }, i) => (
                <a
                  key={title}
                  href={link}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    display: "block",
                    padding: "18px 20px",
                    borderRadius: "var(--radius-md)",
                    background: "var(--navy-mid)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    transition: "var(--transition)",
                    animation: `fadeUp 0.5s ease ${(ci * 3 + i) * 0.06}s both`,
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.borderColor = "rgba(15,184,169,0.3)";
                    e.currentTarget.style.transform = "translateX(4px)";
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.05)";
                    e.currentTarget.style.transform = "translateX(0)";
                  }}
                >
                  <div style={{
                    display: "flex", justifyContent: "space-between",
                    alignItems: "flex-start", gap: 12,
                  }}>
                    <div>
                      <h3 style={{
                        fontSize: 14, fontWeight: 600,
                        color: "var(--white)", marginBottom: 4,
                      }}>{title}</h3>
                      <p style={{ fontSize: 12, color: "var(--grey)", lineHeight: 1.6 }}>
                        {desc}
                      </p>
                    </div>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0, marginTop: 2 }}>
                      <path d="M7 17L17 7M17 7H7M17 7v10" stroke="var(--teal)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                </a>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
