import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer style={{
      width: "100%",
      borderTop: "1px solid rgba(255,255,255,0.05)",
      padding: "40px 0",
      background: "rgba(0,0,0,0.2)",
    }}>
      <div style={{
        width: "100%",
        maxWidth: 1200,
        margin: "0 auto",
        padding: "0 40px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: 20,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: "50%",
            background: "linear-gradient(135deg, #0fb8a9, #0a6b63)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="4" fill="white"/>
            </svg>
          </div>
          <span style={{ fontFamily: "var(--font-display)", fontSize: 16, color: "var(--white)" }}>
            Derm<span style={{ color: "var(--teal)" }}>AI</span>
          </span>
          <span style={{ fontSize: 11, color: "var(--grey)", marginLeft: 4 }}>
            Research prototype · Not for clinical use
          </span>
        </div>

        <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
          {[
            { to: "/", label: "Home" },
            { to: "/detect", label: "Detect" },
            { to: "/about", label: "About" },
            { to: "/resources", label: "Resources" },
          ].map(({ to, label }) => (
            <Link key={to} to={to} style={{
              fontSize: 12, color: "var(--grey)", fontWeight: 500,
              transition: "color 0.2s",
            }}
            onMouseEnter={e => e.currentTarget.style.color = "var(--white)"}
            onMouseLeave={e => e.currentTarget.style.color = "var(--grey)"}>
              {label}
            </Link>
          ))}
        </div>

        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <a href="https://www.linkedin.com/in/sharon-david-tech/" target="_blank" rel="noopener noreferrer"
            style={{
              width: 34, height: 34, borderRadius: "50%",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.08)",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.2s", color: "var(--grey)",
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = "rgba(15,184,169,0.15)";
              e.currentTarget.style.borderColor = "var(--teal)";
              e.currentTarget.style.color = "var(--teal)";
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = "rgba(255,255,255,0.05)";
              e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
              e.currentTarget.style.color = "var(--grey)";
            }}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
              <path d="M16 8a6 6 0 016 6v7h-4v-7a2 2 0 00-2-2 2 2 0 00-2 2v7h-4v-7a6 6 0 016-6zM2 9h4v12H2z"/>
              <circle cx="4" cy="4" r="2"/>
            </svg>
          </a>

          <a href="https://github.com/Sharon-David" target="_blank" rel="noopener noreferrer"
            style={{
              width: 34, height: 34, borderRadius: "50%",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.08)",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.2s", color: "var(--grey)",
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = "rgba(15,184,169,0.15)";
              e.currentTarget.style.borderColor = "var(--teal)";
              e.currentTarget.style.color = "var(--teal)";
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = "rgba(255,255,255,0.05)";
              e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
              e.currentTarget.style.color = "var(--grey)";
            }}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
            </svg>
          </a>
        </div>
      </div>
    </footer>
  );
}