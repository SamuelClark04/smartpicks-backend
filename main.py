# main.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import os, httpx

app = FastAPI(title="SmartPicks backend v2")

PROVIDER_KEY = os.getenv("ODDS_API_KEY", "").strip()

# map both league-style and sport-style codes to a canonical provider key
CANON = {
    # NBA
    "nba": "basketball_nba",
    "basketball_nba": "basketball_nba",
    # NFL
    "nfl": "americanfootball_nfl",
    "americanfootball_nfl": "americanfootball_nfl",
    # CFB
    "ncaa_football": "americanfootball_ncaaf",
    "cfb": "americanfootball_ncaaf",
    "americanfootball_ncaaf": "americanfootball_ncaaf",
    # MLB
    "mlb": "baseball_mlb",
    "baseball_mlb": "baseball_mlb",
    # NHL
    "nhl": "icehockey_nhl",
    "icehockey_nhl": "icehockey_nhl",
}

def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

@app.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok", status_code=200)

def _normalize_league(league: Optional[str], sport: Optional[str]) -> str:
    val = (league or "").strip().lower() or (sport or "").strip().lower()
    return CANON.get(val, val)

def _mock_events(canon: str, market: str, date: Optional[str]) -> List[Dict[str, Any]]:
    # Small deterministic mock so your app UI always has picks while we finish wiring a provider.
    title = {
        "basketball_nba": "NBA",
        "americanfootball_nfl": "NFL",
        "americanfootball_ncaaf": "NCAAF",
        "baseball_mlb": "MLB",
        "icehockey_nhl": "NHL",
    }.get(canon, canon.upper() or "DEMO")

    evt_id = f"evt_{canon or 'demo'}_{(date or 'upcoming').replace('-', '')}"
    line = 24.5 if "basketball" in canon else 1.5 if "baseball" in canon else 75.5
    market_key = (
        "player_points" if "basketball" in canon else
        "player_total_bases" if "baseball" in canon else
        "player_rush_yards" if "americanfootball" in canon else
        "player_shots_on_goal" if "icehockey" in canon else
        "player_points"
    )

    return [{
        "id": evt_id,
        "sport_key": canon or "demo",
        "sport_title": title,
        "commence_time": _iso_now(),
        "home_team": "Home Team",
        "away_team": "Away Team",
        "bookmakers": [{
            "key": "fanduel",
            "title": "FanDuel",
            "last_update": _iso_now(),
            "markets": [{
                "key": market if market else market_key,
                "outcomes": [
                    {"name": "Over",  "price": -115, "point": line, "description": "Player A"},
                    {"name": "Under", "price": -105, "point": line, "description": "Player A"},
                ],
            }],
        }],
    }]

async def _fetch_provider(canon: str, market: str, date: Optional[str]) -> List[Dict[str, Any]]:
    """Example provider call (The Odds API v4). If it fails, return []."""
    if not PROVIDER_KEY or not canon:
        return []

    # NOTE: This is a minimal example. Adjust the endpoint/params to your provider.
    # The Odds API example (events/odds vary by plan/market availability).
    # We’ll call the "odds" endpoint and then map to our app’s schema.
    base = "https://api.the-odds-api.com/v4/sports"
    params = {
        "apiKey": PROVIDER_KEY,
        "regions": "us",
        "markets": "player_props" if market == "player_props" else "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    url = f"{base}/{canon}/odds"
    # Optional date filter: providers differ; if unsupported, we still return upcoming
    # (You can refine this once you pick a provider/plan that supports prop markets by date.)
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                print(f"[PROVIDER] {r.status_code} {r.text[:200]}")
                return []
            raw = r.json()
    except Exception as e:
        print(f"[PROVIDER] error: {e}")
        return []

    events: List[Dict[str, Any]] = []
    now_iso = _iso_now()

    for e in raw if isinstance(raw, list) else []:
        # Provider payloads vary; the mapping below is defensive.
        ev_id = str(e.get("id") or e.get("event_id") or e.get("commence_time", now_iso))
        home = e.get("home_team") or e.get("homeTeam") or "Home Team"
        away = e.get("away_team") or e.get("awayTeam") or "Away Team"
        commence = e.get("commence_time") or e.get("startTime") or now_iso

        books_out = []
        for bk in e.get("bookmakers", []):
            markets_out = []
            for m in bk.get("markets", []):
                key = m.get("key") or m.get("market_key") or "player_points"
                outs = []
                for o in m.get("outcomes", []):
                    # normalise outcome fields
                    outs.append({
                        "name": o.get("name") or o.get("label") or "Over",
                        "price": o.get("price") or o.get("odds") or -110,
                        "point": o.get("point") if o.get("point") is not None else o.get("line"),
                        "description": o.get("description") or o.get("player") or o.get("participant"),
                    })
                if outs:
                    markets_out.append({"key": key, "outcomes": outs})
            if markets_out:
                books_out.append({
                    "key": (bk.get("key") or bk.get("title") or "book").lower().replace(" ", "_"),
                    "title": bk.get("title") or bk.get("key") or "Book",
                    "last_update": bk.get("last_update") or now_iso,
                    "markets": markets_out,
                })

        if books_out:
            events.append({
                "id": ev_id,
                "sport_key": canon,
                "sport_title": canon.upper(),
                "commence_time": commence,
                "home_team": home,
                "away_team": away,
                "bookmakers": books_out,
            })

    return events

@app.get("/api/upcoming")
async def upcoming(
    league: Optional[str] = Query(None, description="nba, nfl, mlb, nhl, ncaa_football, etc."),
    sport: Optional[str]  = Query(None, description="accepts basketball_nba, baseball_mlb, etc."),
    market: Optional[str] = Query("player_props"),
) -> JSONResponse:
    canon = _normalize_league(league, sport)
    market = (market or "player_props").strip().lower()

    events = await _fetch_provider(canon, market, date=None)
    if not events:
        events = _mock_events(canon, market, date=None)
    return JSONResponse(events, status_code=200)

@app.get("/api/odds")
async def odds(
    league: Optional[str] = Query(None),
    sport: Optional[str]  = Query(None),
    date: Optional[str]   = Query(None, description="YYYY-MM-DD"),
    market: Optional[str] = Query("player_props"),
) -> JSONResponse:
    canon = _normalize_league(league, sport)
    market = (market or "player_props").strip().lower()
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            # be lenient—just ignore invalid date
            date = None

    events = await _fetch_provider(canon, market, date=date)
    if not events:
        events = _mock_events(canon, market, date=date)
    return JSONResponse(events, status_code=200)

if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
