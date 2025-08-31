# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional, List, Dict, Any
import os
import uvicorn
from datetime import datetime, timezone, timedelta

app = FastAPI(title="SmartPicks backend v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for testing; tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

def _iso_now_plus(minutes: int = 60) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat(timespec="seconds").replace("+00:00","Z")

@app.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok", status_code=200)

def _demo_event(league_code: str) -> Dict[str, Any]:
    # minimal example matching your iOS decoder
    return {
        "id": "evt_demo_1",
        "sport_key": league_code or "demo",
        "sport_title": (league_code or "Demo").upper(),
        "commence_time": _iso_now_plus(90),
        "home_team": "Home Team",
        "away_team": "Away Team",
        "bookmakers": [{
            "key": "fanduel",
            "title": "FanDuel",
            "last_update": _iso_now_plus(),
            "markets": [{
                "key": "player_points",        # your app maps this to "Points"
                "outcomes": [
                    {"name": "Over",  "price": -115, "point": 24.5, "description": "Player A"},
                    {"name": "Under", "price": -105, "point": 24.5, "description": "Player A"}
                ]
            }]
        }]
    }

@app.get("/api/upcoming")
def upcoming(
    league: Optional[str] = Query(None, description="nba, nfl, mlb, nhl, ncaa_football, etc."),
    sport: Optional[str]  = Query(None, description="accept for leniency e.g. baseball_mlb"),
    market: Optional[str] = Query(None, description="player_props, moneyline, etc."),
) -> JSONResponse:
    league_code = (league or "").strip().lower() or (sport or "").strip().lower()
    market = (market or "player_props").strip().lower()
    print(f"[UPCOMING] league={league_code!r} market={market!r}")
    # Return 1 demo event so the app renders rows
    return JSONResponse([_demo_event(league_code)], status_code=200)

@app.get("/api/odds")
def odds(
    league: Optional[str] = Query(None),
    sport: Optional[str]  = Query(None),
    date: Optional[str]   = Query(None, description="YYYY-MM-DD"),
    market: Optional[str] = Query(None),
) -> JSONResponse:
    league_code = (league or "").strip().lower() or (sport or "").strip().lower()
    market = (market or "player_props").strip().lower()
    date_str = (date or "").strip()
    print(f"[ODDS] league={league_code!r} date={date_str!r} market={market!r}")
    # Don’t 422 on bad dates; ignore and just return demo data for now
    return JSONResponse([_demo_event(league_code)], status_code=200)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
