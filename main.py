# main.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional, List, Dict, Any
import os
import uvicorn
from datetime import datetime

app = FastAPI(title="SmartPicks backend v1")

# ---- Models your iOS client decodes ----
# (We just return dicts; no strict pydantic models so we never 422.)
# APIEvent:
#   id, sport_key, sport_title, commence_time, home_team, away_team, bookmakers:[{ key, title, last_update, markets:[{ key, outcomes:[{ name, price, point, description }] }] }]

def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

@app.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok", status_code=200)

@app.get("/api/upcoming")
def upcoming(
    league: Optional[str] = Query(None, description="nba, nfl, mlb, nhl, ncaa_football, etc."),
    sport: Optional[str] = Query(None, description="accept for leniency e.g. baseball_mlb"),
    market: Optional[str] = Query(None, description="player_props, moneyline, etc."),
) -> JSONResponse:
    # Be lenient: accept league OR sport; normalize to league-ish
    league_code = (league or "").strip().lower() or (sport or "").strip().lower()
    market = (market or "player_props").strip().lower()

    print(f"[UPCOMING] league={league_code!r} market={market!r}")

    # TODO: plug real provider here and build events list.
    # For now return an empty list so the app can render without errors.
    events: List[Dict[str, Any]] = []

    return JSONResponse(events, status_code=200)

@app.get("/api/odds")
def odds(
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    market: Optional[str] = Query(None),
) -> JSONResponse:
    league_code = (league or "").strip().lower() or (sport or "").strip().lower()
    market = (market or "player_props").strip().lower()
    date_str = (date or "").strip()

    print(f"[ODDS] league={league_code!r} date={date_str!r} market={market!r}")

    # Validate date gently (do not 422)
    if date_str:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"[ODDS] bad date '{date_str}', proceeding with empty result")

    # TODO: fetch real odds and map to the shape below.
    # Minimal sample (commented) if you want to see a row render:
    # events = [{
    #   "id": "evt_demo_1",
    #   "sport_key": league_code or "demo",
    #   "sport_title": (league_code or "Demo").upper(),
    #   "commence_time": _iso_now(),
    #   "home_team": "Home Team",
    #   "away_team": "Away Team",
    #   "bookmakers": [{
    #       "key": "fanduel",
    #       "title": "FanDuel",
    #       "last_update": _iso_now(),
    #       "markets": [{
    #           "key": "player_points",
    #           "outcomes": [
    #               {"name":"Over", "price": -115, "point": 24.5, "description":"Player A"},
    #               {"name":"Under","price": -105, "point": 24.5, "description":"Player A"}
    #           ]
    #       }]
    #   }]
    # }]
    events: List[Dict[str, Any]] = []

    return JSONResponse(events, status_code=200)

# ---- Local run (Render uses the CMD below) ----
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
