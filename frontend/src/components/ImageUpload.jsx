import { useState, useRef, useCallback } from "react";

export default function ImageUpload({ onFileSelect, isLoading }) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
    onFileSelect(file);
  }, [onFileSelect]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const onInputChange = (e) => handleFile(e.target.files[0]);

  const clearImage = (e) => {
    e.stopPropagation();
    setPreview(null);
    setFileName(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div
      onClick={() => !preview && inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      style={{
        borderRadius: "var(--radius-lg)",
        border: `2px dashed ${dragOver
          ? "var(--teal)"
          : preview
            ? "rgba(15,184,169,0.4)"
            : "rgba(255,255,255,0.12)"}`,
        background: dragOver
          ? "rgba(15,184,169,0.05)"
          : preview
            ? "var(--navy-mid)"
            : "rgba(255,255,255,0.02)",
        cursor: preview ? "default" : "pointer",
        transition: "all 0.25s ease",
        overflow: "hidden",
        position: "relative",
        minHeight: preview ? "auto" : 220,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={onInputChange}
        style={{ display: "none" }}
        disabled={isLoading}
      />

      {preview ? (
        // Image preview
        <div style={{ position: "relative" }}>
          <img
            src={preview}
            alt="Preview"
            style={{
              width: "100%",
              maxHeight: 360,
              objectFit: "contain",
              display: "block",
              borderRadius: "calc(var(--radius-lg) - 2px)",
            }}
          />
          {/* Overlay info */}
          <div style={{
            position: "absolute", bottom: 0, left: 0, right: 0,
            background: "linear-gradient(transparent, rgba(10,22,40,0.9))",
            padding: "32px 16px 16px",
            display: "flex", alignItems: "center", justifyContent: "space-between",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="var(--teal)">
                <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-1.1 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
              </svg>
              <span style={{ fontSize: 12, color: "var(--white-dim)", fontWeight: 500 }}>
                {fileName}
              </span>
            </div>
            {!isLoading && (
              <button
                onClick={clearImage}
                style={{
                  background: "rgba(255,255,255,0.1)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--white-dim)",
                  fontSize: 11,
                  fontWeight: 600,
                  padding: "4px 10px",
                  cursor: "pointer",
                  letterSpacing: "0.04em",
                  transition: "var(--transition)",
                }}
                onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.2)"}
                onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.1)"}
              >
                Change
              </button>
            )}
          </div>

          {/* Loading overlay */}
          {isLoading && (
            <div style={{
              position: "absolute", inset: 0,
              background: "rgba(10,22,40,0.7)",
              display: "flex", alignItems: "center", justifyContent: "center",
              borderRadius: "calc(var(--radius-lg) - 2px)",
              backdropFilter: "blur(4px)",
            }}>
              <div style={{
                width: 40, height: 40, borderRadius: "50%",
                border: "3px solid rgba(15,184,169,0.2)",
                borderTopColor: "var(--teal)",
                animation: "spin 0.8s linear infinite",
              }}/>
            </div>
          )}
        </div>
      ) : (
        // Drop zone
        <div style={{
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          padding: "48px 24px", gap: 16,
          transition: "var(--transition)",
        }}>
          <div style={{
            width: 64, height: 64, borderRadius: "var(--radius-lg)",
            background: dragOver ? "rgba(15,184,169,0.15)" : "rgba(255,255,255,0.04)",
            border: `1px solid ${dragOver ? "rgba(15,184,169,0.4)" : "rgba(255,255,255,0.08)"}`,
            display: "flex", alignItems: "center", justifyContent: "center",
            transition: "var(--transition)",
          }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" stroke={dragOver ? "var(--teal)" : "var(--grey)"} strokeWidth="1.5" strokeLinecap="round"/>
              <polyline points="17 8 12 3 7 8" stroke={dragOver ? "var(--teal)" : "var(--grey)"} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="12" y1="3" x2="12" y2="15" stroke={dragOver ? "var(--teal)" : "var(--grey)"} strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>

          <div style={{ textAlign: "center" }}>
            <p style={{
              fontSize: 15, fontWeight: 500,
              color: dragOver ? "var(--teal)" : "var(--white-dim)",
              marginBottom: 6, transition: "var(--transition)",
            }}>
              {dragOver ? "Drop to upload" : "Drop image here or click to browse"}
            </p>
            <p style={{ fontSize: 12, color: "var(--grey)" }}>
              JPEG, PNG, WEBP · Max 10MB
            </p>
          </div>

          <div style={{
            display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center",
            marginTop: 4,
          }}>
            {["Dermoscopy", "Smartphone photo", "Clinical image"].map(tag => (
              <span key={tag} style={{
                fontSize: 11, padding: "3px 10px",
                borderRadius: 100,
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "var(--grey)",
                fontWeight: 500,
              }}>{tag}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
