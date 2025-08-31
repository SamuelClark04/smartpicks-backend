# main.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional, List, Dict, Any
from datetime import datetime

app = FastAPI(title="SmartPicks backend v1")

# --- helpers
def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _norm_league(league: Optional[str], sport: Optional[str]) -> str:
    x = (league or sport or "").strip().lower()
    aliases = {
        "nba": "nba", "basketball_nba": "nba",
        "nfl": "nfl", "americanfootball_nfl": "nfl",
        "ncaa_football": "ncaa_football", "ncaaf": "ncaa_football",
        "americanfootball_ncaaf": "ncaa_football",
        "mlb": "mlb", "baseball_mlb": "mlb",
        "nhl": "nhl", "icehockey_nhl": "nhl",
    }
    return aliases.get(x, x or "demo")

def _sample_events(league_code: str, market: str, date_str: Optional[str]) -> List[Dict[str, Any]]:
    # 3 mock events with FanDuel book + a couple markets
    # matches the iOS decoder shape exactly
    teams = {
        "nba": [("LAL","GSW"),("BOS","MIA"),("DEN","PHX")],
        "nfl": [("KC","BUF"),("DAL","PHI"),("CIN","BAL")],
        "mlb": [("NYY","BOS"),("LAD","SF"),("ATL","NYM")],
        "nhl": [("TOR","MTL"),("NYR","BOS"),("VGK","COL")],
        "ncaa_football": [("UGA","BAMA"),("OSU","MICH"),("TEX","OKLA")],
        "demo": [("Away A","Home A"),("Away B","Home B"),("Away C","Home C")],
    }.get(league_code, [("Away","Home")]*3)

    mk = market or "player_props"
    mk_points = {"nba":"player_points","nfl":"player_rush_yards","mlb":"player_total_bases","nhl":"player_shots_on_goal","ncaa_football":"player_rush_yards"}.get(league_code,"player_points")

    events = []
    for i,(away,home) in enumerate(teams, start=1):
        ev_id = f"{league_code}_{i}"
        events.append({
            "id": ev_id,
            "sport_key": league_code,
            "sport_title": league_code.upper(),
            "commence_time": _iso_now(),
            "home_team": home,
            "away_team": away,
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "last_update": _iso_now(),
                "markets": [
                    {
                        "key": mk_points,
                        "outcomes": [
                            {"name":"Over",  "price": -115, "point": 24.5, "description":"Player A"},
                            {"name":"Under", "price": -105, "point": 24.5, "description":"Player A"}
                        ]
                    },
                    {
                        "key": "player_assists" if league_code=="nba" else mk,
                        "outcomes": [
                            {"name":"Over",  "price": +120, "point": 6.5, "description":"Player B"}
                        ]
                    }
                ]
            }]
        })
    return events

@app.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok", status_code=200)

@app.get("/")
def root() -> PlainTextResponse:
    return PlainTextResponse("SmartPicks backend is alive. Try /api/upcoming?league=mlb&market=player_props", status_code=200)

@app.get("/api/upcoming")
def upcoming(
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    market: Optional[str] = Query("player_props"),
):
    league_code = _norm_league(league, sport)
    print(f"[UPCOMING] league={league_code!r} market={market!r}")
    events = _sample_events(league_code, market or "player_props", None)
    return JSONResponse(events, status_code=200)

@app.get("/api/odds")
def odds(
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    market: Optional[str] = Query("player_props"),
):
    league_code = _norm_league(league, sport)
    date_str = (date or "").strip()
    # be lenient about date, never 422
    if date_str:
        try: datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError: pass

    print(f"[ODDS] league={league_code!r} date={date_str!r} market={market!r}")
    events = _sample_events(league_code, market or "player_props", date_str or None)
    return JSONResponse(events, status_code=200)

# Render will run: uvicorn main:app --host 0.0.0.0 --port $PORT
