# main.py
import os
import time
import json
import math
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import httpx
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------
# Config & Environment
# -----------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
# If you have CollegeFootballData (cfbd) key later:
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")

# Odds API base
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# CORS for your iOS app / local dev
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

# Caching (in-memory TTL)
class TTLCache:
    def __init__(self, ttl_seconds: int = 600, max_items: int = 512):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.store: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str):
        item = self.store.get(key)
        if not item: return None
        ts, val = item
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: object):
        if len(self.store) >= self.max_items:
            # drop oldest
            oldest_key = min(self.store.keys(), key=lambda k: self.store[k][0])
            self.store.pop(oldest_key, None)
        self.store[key] = (time.time(), value)

cache = TTLCache(ttl_seconds=600)

# -----------------------------
# Models (shape compatible with your iOS app)
# -----------------------------
class APIOutcome(BaseModel):
    name: str
    price: float
    point: Optional[float] = None
    description: Optional[str] = None

class APIMarket(BaseModel):
    key: str
    outcomes: List[APIOutcome]

class APIBookmaker(BaseModel):
    key: str
    title: str
    last_update: Optional[str] = None
    markets: List[APIMarket]

class APIEvent(BaseModel):
    id: str
    sport_key: str
    sport_title: Optional[str] = None
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[APIBookmaker]

# Correlation Signals
class CorrelationSignal(BaseModel):
    id: str
    kind: str
    eventId: str
    players: List[str]
    teams: List[str]
    markets: List[str]
    boost: float
    reason: str

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="SmartPicks Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
# -----------------------------
SPORT_MAP = {
    # Frontend may send either of these
    "nba": "basketball_nba",
    "basketball_nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "americanfootball_nfl": "americanfootball_nfl",
    "ncaa_football": "americanfootball_ncaaf",
    "americanfootball_ncaaf": "americanfootball_ncaaf",
    "mlb": "baseball_mlb",
    "baseball_mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "icehockey_nhl": "icehockey_nhl",
}

DEFAULT_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars"]

def _ckey(*parts) -> str:
    return "|".join(str(p) for p in parts if p is not None)

def _auth_headers() -> Dict[str,str]:
    # The Odds API uses apiKey as query param, not Authorization, but we keep header path for future use.
    return {"Accept": "application/json"}

# Odds API fetcher
async def _fetch_events(
    http: httpx.AsyncClient,
    sport_key: str,
    date: Optional[str],
    market: str,
    odds_format: str = "american",
    regions: str = "us",
    books: Optional[List[str]] = None
) -> List[APIEvent]:
    if not ODDS_API_KEY:
        raise HTTPException(status_code=401, detail="ODDS_API_KEY missing on server")

    # Decide route
    # /sports/{sport_key}/odds vs /sports/{sport_key}/events ? The v4 “odds” endpoint returns pricing.
    # For upcoming without date we just call odds without date filter; the API doesn’t support date filter directly,
    # so the frontend narrows by commence_time.
    route = f"/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": market if market != "all" else "h2h,spreads,totals,player_points,player_rebounds,player_assists,player_threes,player_steals,player_blocks,player_turnovers,player_pass_yards,player_pass_attempts,player_pass_completions,player_pass_tds,player_interceptions,player_rush_yards,player_rush_attempts,player_rush_tds,player_receiving_yards,player_receptions,player_receiving_tds,player_longest_reception,player_longest_rush,player_hits,player_runs,player_rbis,player_home_runs,player_total_bases,player_walks,player_strikeouts,pitcher_outs,pitcher_hits_allowed,player_shots_on_goal,player_goals,player_points,goalie_saves",
        "oddsFormat": odds_format,
        "bookmakers": ",".join(books or DEFAULT_BOOKS),
    }

    url = f"{ODDS_API_BASE}{route}"
    r = await http.get(url, headers=_auth_headers(), params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    raw = r.json()

    # Normalize into our APIEvent model
    events: List[APIEvent] = []
    for e in raw:
        # Convert The Odds API format into our expected shape
        bookmakers: List[APIBookmaker] = []
        for b in e.get("bookmakers", []):
            markets: List[APIMarket] = []
            for m in b.get("markets", []):
                outs: List[APIOutcome] = []
                for o in m.get("outcomes", []):
                    # The Odds API player markets differ per book; map to our universal outcome shape
                    desc = o.get("description")
                    outs.append(APIOutcome(
                        name=o.get("name", ""),
                        price=float(o.get("price", 0)),
                        point=o.get("point"),
                        description=desc
                    ))
                markets.append(APIMarket(key=m.get("key",""), outcomes=outs))
            bookmakers.append(APIBookmaker(
                key=b.get("key",""),
                title=b.get("title","").strip() or b.get("key",""),
                last_update=b.get("last_update"),
                markets=markets
            ))
        events.append(APIEvent(
            id=str(e.get("id", "")),
            sport_key=sport_key,
            sport_title=e.get("sport_title"),
            commence_time=e.get("commence_time"),
            home_team=e.get("home_team",""),
            away_team=e.get("away_team",""),
            bookmakers=bookmakers
        ))
    return events

# -----------------------------
# Signal adapters (NBA H2H implemented; others safe stubs)
# -----------------------------

@dataclass
class EvLite:
    id: str
    sport_key: str
    home_team: str
    away_team: str
    bookmakers: List[Dict]

def _bld_team_id(name: str) -> Optional[int]:
    # Simplified: stable hash → pretend team id
    return int(hashlib.md5(name.encode()).hexdigest()[:6], 16)

def _bld_search_player(name: str) -> Optional[int]:
    # Simplified: stable hash as player id
    if not name: return None
    return int(hashlib.md5(name.encode()).hexdigest()[:6], 16)

def _bld_player_team_id(pid: int) -> Optional[int]:
    # Fake: derive team from pid for demo; your real impl should map player → current team
    return (pid % 2) and 1 or 2

def _bld_player_h2h(pid: int, opp_tid: int, n: int = 8) -> Dict[str,float]:
    # Placeholder: in real impl, query a stats provider (balldontlie, scraper, etc.)
    # Return recent averages vs opponent.
    # Keys: "pts","ast","reb","fg3m"
    # We return deterministic pseudo-values so UI has something consistent.
    seed = int(pid) ^ int(opp_tid) ^ n
    pts = 18 + (seed % 16) * 0.5         # 18..25.5
    ast = 3 + (seed % 8) * 0.4           # 3..5.8
    reb = 4 + (seed % 10) * 0.5          # 4..8.5
    th3 = 1 + (seed % 6) * 0.3           # 1..2.5
    return {"pts": pts, "ast": ast, "reb": reb, "fg3m": th3}

def adapter_nba_h2h(ev: APIEvent) -> List[CorrelationSignal]:
    """NBA head-to-head signals: if a player's recent H2H avg vs this opponent beats today's line, add a boost."""
    if ev.sport_key.lower() not in ("nba","basketball_nba"):
        return []

    # collect player lines for this event
    player_lines: Dict[str, Dict[str, float]] = {}
    for bk in ev.bookmakers:
        for m in bk.markets:
            if not m.key.startswith("player_"):
                continue
            for o in m.outcomes:
                if o.description and o.point is not None:
                    player_lines.setdefault(o.description, {})[m.key] = o.point
    if not player_lines:
        return []

    # resolve team ids for opponent detection
    home_tid = _bld_team_id(ev.home_team)
    away_tid = _bld_team_id(ev.away_team)
    if home_tid is None or away_tid is None:
        return []

    # market → (stat_key, base, slope, cap)
    mappings: List[Tuple[str, str, float, float, float]] = [
        ("player_points",   "pts",  0.035, 0.008, 0.060),
        ("player_assists",  "ast",  0.030, 0.010, 0.060),
        ("player_rebounds", "reb",  0.030, 0.009, 0.060),
        ("player_threes",   "fg3m", 0.028, 0.012, 0.055),
    ]

    sigs: List[CorrelationSignal] = []
    for name, lines in player_lines.items():
        pid = _bld_search_player(name)
        if not pid:
            continue
        p_team = _bld_player_team_id(pid)
        if not p_team:
            continue
        # identify opponent team id and name
        if p_team == home_tid:
            opp_tid, opp_name = away_tid, ev.away_team
        elif p_team == away_tid:
            opp_tid, opp_name = home_tid, ev.home_team
        else:
            continue

        h2h = _bld_player_h2h(pid, opp_tid, n=8)
        if not h2h:
            continue
        for mk, stat, base, slope, cap in mappings:
            line = lines.get(mk)
            if line is None:
                continue
            val = float(h2h.get(stat, 0.0))
            gap = val - float(line)
            threshold = 0.5 if stat == "fg3m" else 1.0
            if gap >= threshold:
                boost = min(cap, base + gap * slope)
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((ev.id+name+mk+"h2h").encode()).hexdigest(),
                    kind="head_to_head",
                    eventId=ev.id,
                    players=[name],
                    teams=[],
                    markets=[mk],
                    boost=round(boost, 3),
                    reason=f"H2H vs {opp_name}: avg {val:.1f} > line {float(line):.1f}"
                ))
    return sigs

# Stubs (safe no-ops until hooked to real data):
def adapter_nfl_recent_form(ev: APIEvent) -> List[CorrelationSignal]:
    return []

def adapter_mlb_light_h2h(ev: APIEvent) -> List[CorrelationSignal]:
    return []

def adapter_nhl_recent_form(ev: APIEvent) -> List[CorrelationSignal]:
    return []

def adapter_cfb_cfbd(ev: APIEvent) -> List[CorrelationSignal]:
    # fill later with CFBD once key is added
    return []

def build_signals_for_event(ev: APIEvent) -> List[CorrelationSignal]:
    sigs: List[CorrelationSignal] = []
    try:
        _n = len(sigs)
        sigs.extend(adapter_nba_h2h(ev))
        print(f"signals.adapter nba_h2h added={len(sigs)-_n}")
    except Exception as e:
        print(f"signals.adapter nba_h2h error: {e}")

    try:
        _n = len(sigs)
        sigs.extend(adapter_nfl_recent_form(ev))
        print(f"signals.adapter nfl_recent_form added={len(sigs)-_n}")
    except Exception as e:
        print(f"signals.adapter nfl_recent_form error: {e}")

    try:
        _n = len(sigs)
        sigs.extend(adapter_mlb_light_h2h(ev))
        print(f"signals.adapter mlb_light_h2h added={len(sigs)-_n}")
    except Exception as e:
        print(f"signals.adapter mlb_light_h2h error: {e}")

    try:
        _n = len(sigs)
        sigs.extend(adapter_nhl_recent_form(ev))
        print(f"signals.adapter nhl_recent_form added={len(sigs)-_n}")
    except Exception as e:
        print(f"signals.adapter nhl_recent_form error: {e}")

    try:
        _n = len(sigs)
        sigs.extend(adapter_cfb_cfbd(ev))
        print(f"signals.adapter cfb_cfbd added={len(sigs)-_n}")
    except Exception as e:
        print(f"signals.adapter cfb_cfbd error: {e}")

    return sigs

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.0"}

@app.get("/api/odds", response_model=List[APIEvent])
async def api_odds(
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    date: Optional[str] = Query(None, description="YYYY-MM-DD (frontend may filter locally)"),
    market: str = Query("all"),
    x_forwarded_for: Optional[str] = Header(None)
):
    code_in = (league or sport or "").lower()
    sport_key = SPORT_MAP.get(code_in, code_in)
    if not sport_key:
        raise HTTPException(status_code=422, detail="Missing league/sport")

    ck = _ckey("odds", sport_key, date or "upcoming", market)
    cached = cache.get(ck)
    if cached is not None:
        return cached

    async with httpx.AsyncClient() as http:
        events = await _fetch_events(http, sport_key, date, market)
        cache.set(ck, events)
        return events

@app.get("/api/upcoming", response_model=List[APIEvent])
async def api_upcoming(
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    market: str = Query("all")
):
    # identical to /api/odds without date constraint
    return await api_odds(league=league, sport=sport, date=None, market=market)

@app.get("/api/signals", response_model=List[CorrelationSignal])
async def api_signals(event_id: str = Query(..., alias="event_id"),
                      league: Optional[str] = Query(None), sport: Optional[str] = Query(None)):
    """
    Build signals for a single event ID. We refetch odds for the sport (cache will be used),
    then locate the event and generate signals.
    """
    code_in = (league or sport or "").lower()
    sport_key = SPORT_MAP.get(code_in, code_in) if code_in else None
    if not sport_key:
        # Best effort: scan across major sports in cache
        candidates = ["basketball_nba","americanfootball_nfl","americanfootball_ncaaf","baseball_mlb","icehockey_nhl"]
        sport_key = next((s for s in candidates if cache.get(_ckey("odds", s, "upcoming", "all"))), "basketball_nba")

    # Try cached events first
    ck = _ckey("odds", sport_key, "upcoming", "all")
    events: Optional[List[APIEvent]] = cache.get(ck)

    # If missing, fetch fresh
    if events is None:
        async with httpx.AsyncClient() as http:
            events = await _fetch_events(http, sport_key, None, "all")
            cache.set(ck, events)

    # Find the event by id
    ev = next((e for e in events if str(e.id) == str(event_id)), None)
    if not ev:
        # Last resort: return empty (frontend treats signals as optional)
        return []

    return build_signals_for_event(ev)
