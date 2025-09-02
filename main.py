from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import os

APP_NAME = "SmartPicks Backend"
ALLOWED_ORIGINS = (
    os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
)

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------- Health check ----------
@app.get("/health")
def health():
    return {"status": "ok", "time": now_iso()}

@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "time": now_iso()}

@app.get("/metrics")
def metrics():
    # Minimal placeholder so imports never fail
    return {"pairs_returned_total": 0}

# ---------- Odds endpoints (minimal, non-blocking stubs) ----------
# Your iOS app calls these; return empty lists quickly so the UI doesn't spin if providers are down
@app.get("/api/odds")
def api_odds(league: str | None = None, sport: str | None = None, date: str | None = None, market: str | None = None):
    return []

@app.get("/api/upcoming")
def api_upcoming(league: str | None = None, sport: str | None = None, market: str | None = None):
    return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
