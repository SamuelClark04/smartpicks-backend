# main.py
import os
from datetime import datetime, timezone
from fastapi import FastAPI, Query, HTTPException
import httpx

app = FastAPI(title="SmartPicks Backend", version="1.0")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
ODDS_BASE = "https://api.the-odds-api.com/v4"

# ---- Health & root -----------------------------------------------------------
@app.get("/")
def root():
    return {"service": "smartpicks-backend", "ok": True}

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

# ---- Helpers ----------------------------------------------------------------
LEAGUE_MAP = {
    # the iOS app sends league codes like these:
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "ncaa_football": "americanfootball_ncaaf",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
}

def resolve_league(league_or_sport: str) -> str:
    """Allow both league=mlb and sport=baseball_mlb."""
    key = league_or_sport.lower()
    if key in LEAGUE_MAP:         # league form
        return LEAGUE_MAP[key]
    # already a sport key (e.g. baseball_mlb)? accept as-is
    return key

async def fetch_events_from_oddsapi(sport_key: str, date: str | None, market: str):
    if not ODDS_API_KEY:
        # No key set -> behave gracefully with empty list
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": market,   # "player_props" or "h2h" etc
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    # TheOddsAPI doesn’t support explicit calendar date filtering directly for all endpoints.
    # We’ll call the sport events endpoint and let the client filter by date.
    url = f"{ODDS_BASE}/sports/{sport_key}/odds"

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        if r.status_code == 404:
            return []  # nothing for that sport/market
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()

# ---- Endpoints the app calls -------------------------------------------------
@app.get("/api/odds")
async def api_odds(
    league: str | None = Query(None),
    sport: str | None = Query(None),
    date: str | None = Query(None),          # YYYY-MM-DD
    market: str = Query("player_props"),
):
    league_param = league or sport
    if not league_param:
        raise HTTPException(422, detail="Provide ?league= or ?sport=")
    sport_key = resolve_league(league_param)
    data = await fetch_events_from_oddsapi(sport_key, date, market)
    return data

@app.get("/api/upcoming")
async def api_upcoming(
    league: str | None = Query(None),
    sport: str | None = Query(None),
    market: str = Query("player_props"),
):
    league_param = league or sport
    if not league_param:
        raise HTTPException(422, detail="Provide ?league= or ?sport=")
    sport_key = resolve_league(league_param)
    data = await fetch_events_from_oddsapi(sport_key, None, market)
    return data
