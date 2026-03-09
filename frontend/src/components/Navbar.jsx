import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => setMenuOpen(false), [location]);

  const links = [
    { to: "/", label: "Home" },
    { to: "/detect", label: "Detect" },
    { to: "/about", label: "About" },
    { to: "/resources", label: "Resources" },
  ];

  return (
    <nav style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 1000,
      transition: "all 0.3s ease",
      background: scrolled
        ? "rgba(10, 22, 40, 0.95)"
        : "transparent",
      backdropFilter: scrolled ? "blur(12px)" : "none",
      borderBottom: scrolled ? "1px solid rgba(255,255,255,0.06)" : "none",
    }}>
      <div style={{
        maxWidth: 1200, margin: "0 auto", padding: "0 24px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        height: 68,
      }}>
        {/* Logo */}
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: "50%",
            background: "linear-gradient(135deg, #0fb8a9, #0a6b63)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 16px rgba(15,184,169,0.4)",
          }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"
                fill="white" opacity="0.3"/>
              <circle cx="12" cy="12" r="4" fill="white"/>
            </svg>
          </div>
          <span style={{
            fontFamily: "var(--font-display)",
            fontSize: 20,
            letterSpacing: "-0.02em",
            color: "var(--white)",
          }}>
            Derm<span style={{ color: "var(--teal)" }}>AI</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}
             className="desktop-nav">
          {links.map(({ to, label }) => {
            const active = location.pathname === to;
            return (
              <Link key={to} to={to} style={{
                padding: "8px 16px",
                borderRadius: "var(--radius-sm)",
                fontSize: 14,
                fontWeight: 500,
                letterSpacing: "0.01em",
                color: active ? "var(--teal)" : "var(--grey)",
                background: active ? "rgba(15,184,169,0.08)" : "transparent",
                transition: "var(--transition)",
                position: "relative",
              }}
              onMouseEnter={e => {
                if (!active) e.currentTarget.style.color = "var(--white)";
              }}
              onMouseLeave={e => {
                if (!active) e.currentTarget.style.color = "var(--grey)";
              }}>
                {label}
              </Link>
            );
          })}
          <Link to="/detect" style={{
            marginLeft: 12,
            padding: "9px 20px",
            borderRadius: "var(--radius-sm)",
            fontSize: 14,
            fontWeight: 600,
            color: "var(--navy)",
            background: "var(--teal)",
            transition: "var(--transition)",
            letterSpacing: "0.02em",
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = "var(--teal-dim)";
            e.currentTarget.style.boxShadow = "var(--shadow-teal)";
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = "var(--teal)";
            e.currentTarget.style.boxShadow = "none";
          }}>
            Analyze Now
          </Link>
        </div>

        {/* Mobile hamburger */}
        <button onClick={() => setMenuOpen(!menuOpen)} style={{
          display: "none", background: "none",
          color: "var(--white)", padding: 4,
        }} className="hamburger">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            {menuOpen
              ? <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              : <path d="M3 8h18M3 12h18M3 16h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            }
          </svg>
        </button>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div style={{
          background: "var(--navy-mid)",
          borderTop: "1px solid rgba(255,255,255,0.06)",
          padding: "12px 24px 20px",
          animation: "fadeIn 0.2s ease",
        }}>
          {links.map(({ to, label }) => (
            <Link key={to} to={to} style={{
              display: "block", padding: "12px 0",
              fontSize: 15, fontWeight: 500,
              color: location.pathname === to ? "var(--teal)" : "var(--white-dim)",
              borderBottom: "1px solid rgba(255,255,255,0.04)",
            }}>
              {label}
            </Link>
          ))}
        </div>
      )}

      <style>{`
        @media (max-width: 768px) {
          .desktop-nav { display: none !important; }
          .hamburger { display: block !important; }
        }
      `}</style>
    </nav>
  );
}
