"""
main.py — FastAPI app with Prometheus monitoring middleware
"""
import time
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from backend.api.routes import router

# ── Prometheus metrics ────────────────────────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True

    REQUEST_COUNT = Counter(
        "dermai_requests_total",
        "Total API requests",
        ["method", "endpoint", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "dermai_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    )
    PREDICTION_CONFIDENCE = Histogram(
        "dermai_prediction_confidence",
        "Distribution of model confidence scores",
        buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    )
    MALIGNANT_PREDICTIONS = Counter(
        "dermai_malignant_predictions_total",
        "Total malignant predictions"
    )
    BENIGN_PREDICTIONS = Counter(
        "dermai_benign_predictions_total",
        "Total benign predictions"
    )
    OOD_REJECTIONS = Counter(
        "dermai_ood_rejections_total",
        "Total out-of-distribution image rejections"
    )
    ACTIVE_REQUESTS = Gauge(
        "dermai_active_requests",
        "Currently active requests"
    )
    print("✓ Prometheus metrics enabled")

except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠ prometheus_client not installed — metrics disabled")
    print("  Run: pip install prometheus-client --break-system-packages")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DermAI API",
    description="Skin lesion analysis API with fairness-aware ML, Grad-CAM explainability, and uncertainty estimation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    if not PROMETHEUS_AVAILABLE:
        return await call_next(request)

    endpoint = request.url.path
    method   = request.method

    ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        response = await call_next(request)
        status   = response.status_code
    except Exception as e:
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
        raise e

    duration = time.time() - start
    ACTIVE_REQUESTS.dec()
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    return response

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# ── Prometheus scrape endpoint ────────────────────────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics():
    if not PROMETHEUS_AVAILABLE:
        return PlainTextResponse("prometheus_client not installed", status_code=503)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":   "ok",
        "message":  "DermAI API is running",
        "docs":     "/docs",
        "metrics":  "/metrics",
        "version":  "1.0.0",
    }
