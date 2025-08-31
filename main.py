# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Accept both names and values the app may send
SPORT_ALIASES = {
    "nba": "nba", "basketball_nba": "nba",
    "nfl": "nfl", "americanfootball_nfl": "nfl",
    "ncaa_football": "cfb", "americanfootball_ncaaf": "cfb", "cfb": "cfb",
    "mlb": "mlb", "baseball_mlb": "mlb",
    "nhl": "nhl", "icehockey_nhl": "nhl",
}

def norm_sport(league: Optional[str], sport: Optional[str]) -> Optional[str]:
    v = (league or sport or "").lower()
    return SPORT_ALIASES.get(v)

@app.get("/api/odds")
@app.get("/odds")
def odds(
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    market: str = Query("player_props"),
    date: Optional[str] = Query(None),   # yyyy-mm-dd
):
    s = norm_sport(league, sport)
    if not s:
        # don’t 404 — return empty array the app can handle
        return []

    # TODO: actually fetch real data here
    # Shape MUST match the app’s APIEvent DTO
    sample = [{
        "id": f"{s}-event-1",
        "sport_key": s,
        "sport_title": s.upper(),
        "commence_time": "2025-08-30T23:59:00Z",
        "home_team": "Home Team",
        "away_team": "Away Team",
        "bookmakers": [{
            "key": "fanduel",
            "title": "FanDuel",
            "last_update": "2025-08-30T12:00:00Z",
            "markets": [{
                "key": "player_points",
                "outcomes": [
                    {"name": "Over", "price": -110, "point": 24.5, "description": "Star Player"},
                    {"name": "Under","price": -110, "point": 24.5, "description": "Star Player"}
                ]
            }]
        }]
    }]

    # if you don’t have data for given market/date, return [] not 404
    return sample

@app.get("/api/upcoming")
@app.get("/upcoming")
def upcoming(
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    market: str = Query("player_props"),
):
    # simply forward to the same handler without date filter
    return odds(league=league, sport=sport, market=market, date=None)
