# main.py — SmartPicks Backend (FastAPI)
#
# What this provides (exactly what your app expects):
# - GET /api/odds?league=...&date=YYYY-MM-DD&market=all|csv
# - GET /api/upcoming?league=...&market=...
#   (aliases /odds and /upcoming also provided, because the app tries both)
# - GET /api/signals?event_id=...&league=...
#
# Response models EXACTLY match your Swift structs:
# APIEvent { id, sport_key, sport_title, commence_time, home_team, away_team, bookmakers[] }
# APIBookmaker { key, title, last_update, markets[] }
# APIMarket { key, outcomes[] }
# APIOutcome { name, price, point?, description? }
#
# Notes:
# - Uses The Odds API v4. Set env var ODDS_API_KEY.
# - Adds CORS for your iOS app. Set ALLOW_ORIGINS="*"
# - In-memory TTL cache (10 min) keyed by sport/date/markets.
# - Forwards upstream 'x-requests-remaining' header to the client.
# - /api/signals builds NBA H2H boosts from the returned player lines.
#   (NFL/MLB/NHL/CFB adapters stubbed so UI stays happy; expand when ready.)

import os
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Query, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------------------------------------
# Environment / Config
# ------------------------------------------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "").strip()  # placeholder for later
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# ------------------------------------------------------------
# Caching layer (simple TTL in-memory)
# ------------------------------------------------------------
class TTLCache:
    def __init__(self, ttl_seconds: int = 600, max_items: int = 512):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.store: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str):
        item = self.store.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: object):
        if len(self.store) >= self.max_items:
            # Drop oldest
            oldest_key = min(self.store, key=lambda k: self.store[k][0]) if self.store else None
            if oldest_key:
                self.store.pop(oldest_key, None)
        self.store[key] = (time.time(), value)

cache = TTLCache(ttl_seconds=600)

# ------------------------------------------------------------
# Pydantic models that MATCH your Swift Decodables (non-optionals included)
# ------------------------------------------------------------
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
    last_update: str   # non-optional in Swift
    markets: List[APIMarket]

class APIEvent(BaseModel):
    id: str
    sport_key: str
    sport_title: str   # non-optional in Swift
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[APIBookmaker]

# Correlation signal model (for /api/signals) matching your Swift struct
class CorrelationSignal(BaseModel):
    id: str
    kind: str             # "head_to_head" | "matchup_trend" | ...
    eventId: str
    players: List[str]
    teams: List[str]
    markets: List[str]
    boost: float
    reason: str

# ------------------------------------------------------------
# FastAPI app + CORS
# ------------------------------------------------------------
app = FastAPI(title="SmartPicks Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Helpers & mappings
# ------------------------------------------------------------
# ---- Generic JSON fetch with caching (per-URL) ----
def _year_from_iso(ts: str) -> Optional[int]:
    # '2025-09-04T00:30:00Z' -> 2025
    try:
        return int((ts or "")[:4])
    except Exception:
        return None
def _http_get_json(url: str, params: Optional[Dict[str, str]] = None, headers: Optional[Dict[str, str]] = None, ttl: int = 600):
    """
    Small helper that caches JSON GET responses using the global TTL cache.
    Keyed by URL + sorted params.
    """
    key = _ckey("httpjson", url, json.dumps(params or {}, sort_keys=True))
    cached = cache.get(key)
    if cached is not None:
        return cached
    import anyio
    async def _fetch():
        async with httpx.AsyncClient() as http:
            r = await http.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
    data = anyio.run(_fetch)
    cache.set(key, data)
    return data
    # ----- MLB (statsapi.mlb.com) helpers -----
def _mlb_teams() -> Dict[str, dict]:
    data = _http_get_json(f"{MLB_API_BASE}/teams", params={"sportId": "1"})
    tmap = {}
    for t in data.get("teams", []):
        keys = {
            t.get("name",""), t.get("teamName",""), t.get("clubName",""),
            t.get("locationName",""), t.get("shortName",""), t.get("abbreviation","")
        }
        for k in {k.lower() for k in keys if k}:
            tmap[k] = t
    return tmap

def _mlb_team_id_by_name(name: str) -> Optional[int]:
    if not name:
        return None
    tmap = _mlb_teams()
    n = name.lower()
    if n in tmap:
        return tmap[n]["id"]
    for k, v in tmap.items():
        if n in k or k in n:
            return v["id"]
    return None

def _mlb_search_player_id(name: str) -> Optional[int]:
    data = _http_get_json(f"{MLB_API_BASE}/people/search", params={"names": name})
    people = data.get("people", [])
    if not people:
        return None
    # exact match first
    nlow = name.lower()
    for p in people:
        full = p.get("fullName","").lower()
        if full == nlow:
            return p.get("id")
    return people[0].get("id")

def _mlb_player_vs_opp_avg(person_id: int, opp_team_id: int, season: Optional[int] = None) -> Dict[str, float]:
    from datetime import datetime
    if season is None:
        season = datetime.utcnow().year
    params = {"stats": "gameLog", "season": str(season)}
    data = _http_get_json(f"{MLB_API_BASE}/people/{person_id}/stats", params=params)
    splits = (data.get("stats") or [{}])[0].get("splits", []) if data.get("stats") else []
    # aggregate vs opponent
    agg = {"tb":0.0,"h":0.0,"r":0.0,"rbi":0.0,"hr":0.0,"bb":0.0,"so":0.0,"outs":0.0}
    cnt = 0
    for s in splits:
        opp = ((s.get("opponent") or {}).get("id"))
        if not opp or int(opp) != int(opp_team_id):
            continue
        stat = s.get("stat", {})
        agg["tb"] += float(stat.get("totalBases") or 0)
        agg["h"]  += float(stat.get("hits") or 0)
        agg["r"]  += float(stat.get("runs") or 0)
        agg["rbi"]+= float(stat.get("rbi") or 0)
        agg["hr"] += float(stat.get("homeRuns") or 0)
        agg["bb"] += float(stat.get("baseOnBalls") or 0)
        agg["so"] += float(stat.get("strikeOuts") or 0)
        # pitcher outs: outs = IP * 3; gameLog has "inningsPitched" like "5.2"
        ip = stat.get("inningsPitched")
        if ip:
            try:
                whole, frac = ip.split(".")
                outs = int(whole)*3 + (2 if frac=="2" else (1 if frac=="1" else 0))
                agg["outs"] += outs
            except Exception:
                pass
        cnt += 1
    if cnt == 0:
        return {}
    return {k: v/cnt for k, v in agg.items()}
    # ----- NHL (statsapi.web.nhl.com) helpers -----
def _nhl_teams() -> Dict[str, dict]:
    data = _http_get_json(f"{NHL_API_BASE}/teams")
    tmap = {}
    for t in data.get("teams", []):
        keys = {
            t.get("name",""), t.get("teamName",""), t.get("abbreviation",""), t.get("shortName","")
        }
        for k in {k.lower() for k in keys if k}:
            tmap[k] = t
    return tmap

def _nhl_team_id_by_name(name: str) -> Optional[int]:
    if not name:
        return None
    tmap = _nhl_teams()
    n = name.lower()
    if n in tmap:
        return tmap[n]["id"]
    for k, v in tmap.items():
        if n in k or k in n:
            return v["id"]
    return None

def _nhl_search_player_id(name: str) -> Optional[int]:
    data = _http_get_json(f"{NHL_API_BASE}/people/", params={"search": name})
    people = data.get("people", [])
    if not people:
        return None
    nlow = name.lower()
    for p in people:
        if p.get("fullName","").lower() == nlow:
            return p.get("id")
    return people[0].get("id")

def _nhl_season_str() -> str:
    from datetime import datetime
    now = datetime.utcnow()
    start = now.year if now.month >= 8 else now.year - 1
    return f"{start}{start+1}"

def _nhl_player_vs_opp_avg(person_id: int, opp_team_id: int) -> Dict[str, float]:
    season = _nhl_season_str()
    data = _http_get_json(f"{NHL_API_BASE}/people/{person_id}/stats", params={"stats": "gameLog", "season": season})
    splits = (data.get("stats") or [{}])[0].get("splits", []) if data.get("stats") else []
    agg = {"shots":0.0,"goals":0.0,"assists":0.0,"points":0.0}
    cnt = 0
    for s in splits:
        opp = ((s.get("opponent") or {}).get("id"))
        if not opp or int(opp) != int(opp_team_id):
            continue
        stat = s.get("stat", {})
        agg["shots"]   += float(stat.get("shots") or 0)
        agg["goals"]   += float(stat.get("goals") or 0)
        agg["assists"] += float(stat.get("assists") or 0)
        agg["points"]  += float(stat.get("points") or 0)
        cnt += 1
    if cnt == 0:
        return {}
    return {k: v/cnt for k, v in agg.items()}
    # ----- CFB (CollegeFootballData) helpers -----
def _cfbd_headers() -> Dict[str, str]:
    if not CFBD_API_KEY:
        return {}
    return {"Authorization": f"Bearer {CFBD_API_KEY}"}

def _cfbd_player_h2h_averages(team_name: str, opp_name: str, year: int, limit_games: int = 12) -> Dict[str, Dict[str, float]]:
    """
    Returns { playerName: { 'passingYards': avg, 'passingTDs': avg, 'rushingYards': avg,
                            'rushingTDs': avg, 'receivingYards': avg, 'receptions': avg } }
    using CFBD /games/players for team vs opponent in the given season.
    If the structure isn’t as expected, returns {} safely.
    """
    headers = _cfbd_headers()
    if not headers:
        return {}
    try:
        params = {"year": year, "team": team_name, "opponent": opp_name}
        data = _http_get_json(f"{CFBD_API_BASE}/games/players", params=params, headers=headers)
        # data is a list of game dicts; each has "players": [{ "player": "...", "stat": "...", "statType": "..."}] (CFBD format varies by endpoint)
        # We’ll handle the common response where each game has an array of players with stat fields.
        agg: Dict[str, Dict[str, float]] = {}
        cnt: Dict[str, int] = {}
        games = data if isinstance(data, list) else []
        for g in games[:limit_games]:
            players = g.get("players") or []
            for p in players:
                name = p.get("player") or p.get("name")
                if not name:
                    continue
                stats = p.get("stat") or p  # sometimes values are flattened
                # Read commonly provided keys if present; skip silently if missing
                py = float(stats.get("passingYards", 0) or 0)
                ptd= float(stats.get("passingTDs", 0) or 0)
                ry = float(stats.get("rushingYards", 0) or 0)
                rtd= float(stats.get("rushingTDs", 0) or 0)
                recy=float(stats.get("receivingYards", 0) or 0)
                rec = float(stats.get("receptions", 0) or 0)

                row = agg.setdefault(name, {"passingYards":0,"passingTDs":0,"rushingYards":0,"rushingTDs":0,"receivingYards":0,"receptions":0})
                row["passingYards"]     += py
                row["passingTDs"]       += ptd
                row["rushingYards"]     += ry
                row["rushingTDs"]       += rtd
                row["receivingYards"]   += recy
                row["receptions"]       += rec
                cnt[name] = cnt.get(name, 0) + 1
        # average
        out: Dict[str, Dict[str, float]] = {}
        for name, row in agg.items():
            n = max(1, cnt.get(name, 1))
            out[name] = {k: v / n for k, v in row.items()}
        return out
    except Exception:
        return {}
SPORT_MAP = {
    # Your app sends either "league" short code or "sport" apiKey
    "nba": "basketball_nba",
    "basketball_nba": "basketball_nba",

    "nfl": "americanfootball_nfl",
    "americanfootball_nfl": "americanfootball_nfl",

    "ncaa_football": "americanfootball_ncaaf",      # important: your app uses "ncaa_football"
    "americanfootball_ncaaf": "americanfootball_ncaaf",

    "mlb": "baseball_mlb",
    "baseball_mlb": "baseball_mlb",

    "nhl": "icehockey_nhl",
    "icehockey_nhl": "icehockey_nhl",
}

HUMAN_TITLES = {
    "basketball_nba": "NBA",
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAA Football",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
}

DEFAULT_BOOKMAKERS = ["fanduel", "draftkings", "betmgm", "caesars"]
# Optional public stat APIs (used for real H2H signals)
BALLDONTLIE_API_BASE = "https://www.balldontlie.io/api/v1"
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "").strip()  # optional

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

NHL_API_BASE = "https://statsapi.web.nhl.com/api/v1"
# A big union so your Props browser & heatmap can discover a lot out of one call
CFBD_API_BASE = "https://api.collegefootballdata.com"
SPORTSDATA_NFL_BASE = "https://api.sportsdata.io/v3/nfl/stats/json"
SPORTSDATA_NFL_KEY = os.getenv("SPORTSDATA_NFL_KEY", "").strip()
ALL_MARKETS = ",".join([
    "h2h","spreads","totals",
    # NBA
    "player_points","player_rebounds","player_assists","player_threes","player_steals","player_blocks","player_turnovers",
    "player_points_rebounds_assists","player_points_rebounds","player_points_assists","player_rebounds_assists",
    # NFL/CFB
    "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_interceptions",
    "player_rush_yards","player_rush_attempts","player_rush_tds","player_receiving_yards","player_receptions",
    "player_receiving_tds","player_longest_reception","player_longest_rush",
    # MLB
    "player_hits","player_runs","player_rbis","player_home_runs","player_total_bases","player_walks",
    "player_strikeouts","pitcher_outs","pitcher_hits_allowed",
    # NHL
    "player_shots_on_goal","player_goals","player_points","goalie_saves"
])

# Per-sport market allowlists (to avoid 422 from The Odds API when requesting unsupported markets)
SPORT_MARKETS: Dict[str, List[str]] = {
    "basketball_nba": [
        "h2h","spreads","totals",
        "player_points","player_rebounds","player_assists","player_threes",
        "player_steals","player_blocks","player_turnovers",
        "player_points_rebounds_assists","player_points_rebounds",
        "player_points_assists","player_rebounds_assists",
    ],
    "americanfootball_nfl": [
        "h2h","spreads","totals",
        "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_interceptions",
        "player_rush_yards","player_rush_attempts","player_rush_tds",
        "player_receiving_yards","player_receptions","player_receiving_tds",
        "player_longest_reception","player_longest_rush",
    ],
    "americanfootball_ncaaf": [
        "h2h","spreads","totals",
        "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_interceptions",
        "player_rush_yards","player_rush_attempts","player_rush_tds",
        "player_receiving_yards","player_receptions","player_receiving_tds",
        "player_longest_reception","player_longest_rush",
    ],
    "baseball_mlb": [
        "h2h","spreads","totals",
        "player_hits","player_runs","player_rbis","player_home_runs","player_total_bases","player_walks",
        "player_strikeouts","pitcher_outs","pitcher_hits_allowed",
    ],
    "icehockey_nhl": [
        "h2h","spreads","totals",
        "player_shots_on_goal","player_goals","player_assists","player_points","goalie_saves",
    ],
}

def _filter_markets_for_sport(requested_csv: str, sport_key: str) -> str:
    """Intersect requested markets with sport-allowed list to prevent upstream 422s."""
    requested = [m.strip() for m in (requested_csv or "").split(",") if m.strip()]
    allowed = set(SPORT_MARKETS.get(sport_key, []))
    # default to generic team markets if sport not found
    if not allowed:
        allowed = {"h2h","spreads","totals"}
    filtered = [m for m in requested if m in allowed]
    # if nothing matched, fall back to generic team markets
    if not filtered:
        filtered = list({"h2h","spreads","totals"})
    return ",".join(filtered)

def _ckey(*parts) -> str:
    return "|".join(str(p) for p in parts if p is not None)

def normalize_market_param(market: Optional[str]) -> str:
    if not market or market.lower() == "all":
        return ALL_MARKETS
    return market

def map_league_or_sport(league: Optional[str], sport: Optional[str]) -> str:
    code_in = (league or sport or "").strip().lower()
    mapped = SPORT_MAP.get(code_in)
    if not mapped:
        raise HTTPException(status_code=422, detail="Missing or unknown league/sport")
    return mapped

# ------------------------------------------------------------
# Upstream fetch & normalization
# ------------------------------------------------------------
async def fetch_odds_events(
    http: httpx.AsyncClient,
    sport_key: str,
    markets_csv: str,
    bookmakers: Optional[List[str]] = None
) -> Tuple[List[APIEvent], Dict[str, str]]:
    """
    Calls The Odds API and normalizes results into your Swift JSON shape.
    Returns (events, upstream_headers) so we can forward x-requests-remaining.
    """
    if not ODDS_API_KEY:
        raise HTTPException(status_code=401, detail="Server missing ODDS_API_KEY")

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "oddsFormat": "american",
        "markets": _filter_markets_for_sport(markets_csv, sport_key),
        "bookmakers": ",".join(bookmakers or DEFAULT_BOOKMAKERS),
    }

    r = await http.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    raw = r.json()
    upstream_headers = {k.lower(): v for k, v in r.headers.items()}

    events: List[APIEvent] = []
    for e in raw:
        bks: List[APIBookmaker] = []
        for b in e.get("bookmakers", []) or []:
            mkts: List[APIMarket] = []
            for m in b.get("markets", []) or []:
                outs: List[APIOutcome] = []
                for o in m.get("outcomes", []) or []:
                    outs.append(APIOutcome(
                        name=str(o.get("name", "")),
                        price=float(o.get("price", 0) or 0),
                        point=o.get("point"),
                        description=o.get("description"),
                    ))
                mkts.append(APIMarket(key=str(m.get("key","")), outcomes=outs))
            bks.append(APIBookmaker(
                key=str(b.get("key","")),
                title=(b.get("title") or b.get("key") or "").strip(),
                last_update=str(b.get("last_update") or ""),   # non-optional for Swift
                markets=mkts
            ))
        events.append(APIEvent(
            id=str(e.get("id","")),
            sport_key=sport_key,
            sport_title=HUMAN_TITLES.get(sport_key, str(e.get("sport_title") or "")),
            commence_time=str(e.get("commence_time") or ""),
            home_team=str(e.get("home_team") or ""),
            away_team=str(e.get("away_team") or ""),
            bookmakers=bks
        ))
    return events, upstream_headers

def filter_by_date(events: List[APIEvent], date_yyyy_mm_dd: Optional[str]) -> List[APIEvent]:
    """Optional server-side filter (the app also filters locally)."""
    if not date_yyyy_mm_dd:
        return events
    want = date_yyyy_mm_dd.strip()
    out: List[APIEvent] = []
    for e in events:
        # commence_time is ISO8601 with "Z". We compare by date substring.
        # Example: 2025-03-02T01:00:00Z → '2025-03-02'
        date_prefix = (e.commence_time or "")[:10]
        if date_prefix == want:
            out.append(e)
    return out

# ------------------------------------------------------------
# Correlation Signals (NBA H2H implemented; others stubbed)
# ------------------------------------------------------------
def _stable_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def _fake_team_id(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:6], 16)

def _fake_player_id(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:6], 16)

def _fake_player_team(pid: int) -> int:
    # Deterministic pseudo team for demo purposes
    return 1 if (pid % 2) else 2

def _fake_h2h_avg(pid: int, opp_tid: int) -> Dict[str, float]:
    # Deterministic pseudo stats keyed by player & opponent to give consistent boosts
    seed = pid ^ opp_tid
    return {
        "pts": 18 + (seed % 16) * 0.5,     # 18..25.5
        "ast": 3 + (seed % 8) * 0.4,       # 3..5.8
        "reb": 4 + (seed % 10) * 0.5,      # 4..8.5
        "fg3m": 1 + (seed % 6) * 0.3       # 1..2.5
    }
# ----- NBA (balldontlie) helpers -----
def _nba_list_teams() -> Dict[str, dict]:
    data = _http_get_json(f"{BALLDONTLIE_API_BASE}/teams")
    teams = {}
    for t in data.get("data", []):
        keyset = {
            t.get("full_name","").lower(),
            t.get("name","").lower(),
            t.get("city","").lower(),
            t.get("abbreviation","").lower()
        }
        for k in keyset:
            if k:
                teams[k] = t
    return teams

def _nba_team_id_by_name(name: str) -> Optional[int]:
    if not name:
        return None
    tmap = _nba_list_teams()
    n = name.lower()
    if n in tmap:
        return tmap[n]["id"]
    # fuzzy contains
    for k, v in tmap.items():
        if n in k or k in n:
            return v["id"]
    return None

def _nba_search_player_id(name: str) -> Optional[int]:
    if not name:
        return None
    params = {"search": name, "per_page": 5}
    headers = {}
    if BALLDONTLIE_API_KEY:
        headers["Authorization"] = f"Bearer {BALLDONTLIE_API_KEY}"
    data = _http_get_json(f"{BALLDONTLIE_API_BASE}/players", params=params, headers=headers)
    best = None
    nlow = name.lower()
    for p in data.get("data", []):
        full = f"{p.get('first_name','')} {p.get('last_name','')}".strip().lower()
        if full == nlow:
            return p["id"]
        if nlow in full and best is None:
            best = p["id"]
    return best

def _nba_player_vs_opp_avg(player_id: int, opp_team_id: int, max_pages: int = 3) -> Dict[str, float]:
    """
    Average the player's last ~100 stats vs specific opponent by scanning game logs.
    """
    headers = {}
    if BALLDONTLIE_API_KEY:
        headers["Authorization"] = f"Bearer {BALLDONTLIE_API_KEY}"
    page = 1
    pts = ast = reb = fg3m = count = 0
    while page <= max_pages:
        params = {"player_ids[]": player_id, "per_page": 100, "page": page}
        data = _http_get_json(f"{BALLDONTLIE_API_BASE}/stats", params=params, headers=headers)
        arr = data.get("data", [])
        if not arr:
            break
        for s in arr:
            g = s.get("game", {}) or {}
            ht = (g.get("home_team_id") or 0)
            vt = (g.get("visitor_team_id") or 0)
            # opponent match: one of the two teams is the opponent and the other is player's team
            if ht == opp_team_id or vt == opp_team_id:
                pts += float(s.get("pts") or 0)
                ast += float(s.get("ast") or 0)
                reb += float(s.get("reb") or 0)
                fg3m += float(s.get("fg3m") or 0)
                count += 1
        if not data.get("meta", {}).get("next_page"):
            break
        page += 1
    if count == 0:
        return {}
    return {"pts": pts/count, "ast": ast/count, "reb": reb/count, "fg3m": fg3m/count}
def adapter_nba_h2h(ev: APIEvent) -> List[CorrelationSignal]:
    if ev.sport_key.lower() not in ("basketball_nba", "nba"):
        return []
    # collect prop lines per player
    player_lines: Dict[str, Dict[str, float]] = {}
    for bk in ev.bookmakers:
        for mk in bk.markets:
            if not mk.key.startswith("player_"):
                continue
            for o in mk.outcomes:
                player = o.description if (o.name.lower() in ("over", "under") and o.description) else o.name
                if not player or o.point is None:
                    continue
                player_lines.setdefault(player, {})[mk.key] = float(o.point)

    if not player_lines:
        return []

    home_tid = _nba_team_id_by_name(ev.home_team)
    away_tid = _nba_team_id_by_name(ev.away_team)
    if not home_tid or not away_tid:
        return []

    mappings = [
        ("player_points",   "pts",  0.035, 0.008, 0.060),
        ("player_assists",  "ast",  0.030, 0.010, 0.060),
        ("player_rebounds", "reb",  0.030, 0.009, 0.060),
        ("player_threes",   "fg3m", 0.028, 0.012, 0.055),
    ]

    sigs: List[CorrelationSignal] = []
    for player_name, lines in player_lines.items():
        pid = _nba_search_player_id(player_name)
        if not pid:
            continue
        # Determine opponent: naive heuristic—assume player's team is not known; compute both and pick stronger sample
        # (balldontlie doesn't give team for the search result directly in v1 response reliably)
        # Try vs away and vs home; pick whichever returns data.
        h2h_vs_away = _nba_player_vs_opp_avg(pid, away_tid)
        h2h_vs_home = _nba_player_vs_opp_avg(pid, home_tid)
        # choose the one that has any stat
        h2h = h2h_vs_away if h2h_vs_away else h2h_vs_home
        opp_name = ev.away_team if h2h is h2h_vs_away else ev.home_team

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
                    id=_stable_id(ev.id + player_name + mk + "h2h"),
                    kind="head_to_head",
                    eventId=ev.id,
                    players=[player_name],
                    teams=[],
                    markets=[mk],
                    boost=round(boost, 3),
                    reason=f"H2H vs {opp_name}: avg {val:.1f} > line {float(line):.1f}"
                ))
    return sigs
def adapter_nfl_recent_form(ev: APIEvent) -> List[CorrelationSignal]:
    """
    NFL recent-form adapter using SportsDataIO.
    For both teams in this event, we pull PlayerGameStatsByTeam for the most recent up-to-4 weeks
    up to the event's week, average per-player key stats, and compare them to the posted lines.
    Emits boosts when recent avg exceeds the line by a sensible margin.
    """
    if ev.sport_key.lower() not in ("americanfootball_nfl","nfl"):
        return []
    if not SPORTSDATA_NFL_KEY:
        return []

    # Collect posted lines per player across markets we support
    player_lines: Dict[str, Dict[str, float]] = {}
    wanted = {
        "player_pass_yards","player_pass_tds",
        "player_rush_yards","player_rush_tds",
        "player_receiving_yards","player_receptions"
    }
    for bk in ev.bookmakers:
        for mk in bk.markets:
            if mk.key not in wanted:
                continue
            for o in mk.outcomes:
                name = o.description if (o.name.lower() in ("over","under") and o.description) else o.name
                if not name or o.point is None:
                    continue
                player_lines.setdefault(name, {})[mk.key] = float(o.point)
    if not player_lines:
        return []

    # Derive season/week from kickoff time
    sw = _nfl_season_week_from_iso(ev.commence_time)
    if not sw:
        return []
    season, week = sw

    # Map teams to SportsDataIO abbreviations
    home_abbr = _nfl_team_abbrev(ev.home_team)
    away_abbr = _nfl_team_abbrev(ev.away_team)
    if not home_abbr or not away_abbr:
        return []

    # Pull up to the last 4 weeks of game stats for each team (clamped to >=1)
    def _team_recent(season: int, week: int, team_abbr: str, back_weeks: int = 4) -> List[dict]:
        rows: List[dict] = []
        for w in range(max(1, week - back_weeks), week):
            try:
                data = _sdio_get(f"PlayerGameStatsByTeam/{season}/{w}/{team_abbr}")
                if isinstance(data, list):
                    rows.extend(data)
            except Exception:
                # ignore missing weeks or API hiccups
                continue
        return rows

    home_rows = _team_recent(season, week, home_abbr)
    away_rows = _team_recent(season, week, away_abbr)
    if not home_rows and not away_rows:
        return []

    # Aggregate per player averages across pulled games
    def _avg_by_player(rows: List[dict]) -> Dict[str, Dict[str, float]]:
        agg: Dict[str, Dict[str, float]] = {}
        cnt: Dict[str, int] = {}
        for r in rows:
            name = (r.get("Name") or r.get("PlayerName") or "").strip()
            if not name:
                continue
            row = agg.setdefault(name, {"PassingYards":0.0,"PassingTouchdowns":0.0,"RushingYards":0.0,"RushingTouchdowns":0.0,"ReceivingYards":0.0,"Receptions":0.0})
            row["PassingYards"]      += float(r.get("PassingYards") or 0)
            row["PassingTouchdowns"] += float(r.get("PassingTouchdowns") or 0)
            row["RushingYards"]      += float(r.get("RushingYards") or 0)
            row["RushingTouchdowns"] += float(r.get("RushingTouchdowns") or 0)
            row["ReceivingYards"]    += float(r.get("ReceivingYards") or 0)
            row["Receptions"]        += float(r.get("Receptions") or 0)
            cnt[name] = cnt.get(name, 0) + 1
        for name, row in list(agg.items()):
            n = max(1, cnt.get(name, 1))
            agg[name] = {k: v / n for k, v in row.items()}
        return agg

    home_avg = _avg_by_player(home_rows)
    away_avg = _avg_by_player(away_rows)

    # Compare averages to lines and emit signals
    # market -> (statKey in SDIO, base, slope, cap, threshold)
    mappings = {
        "player_pass_yards":      ("PassingYards",       0.032, 0.008, 0.060, 12.0),
        "player_pass_tds":        ("PassingTouchdowns",  0.030, 0.030, 0.070, 0.6),
        "player_rush_yards":      ("RushingYards",       0.032, 0.010, 0.060, 8.0),
        "player_rush_tds":        ("RushingTouchdowns",  0.028, 0.030, 0.070, 0.6),
        "player_receiving_yards": ("ReceivingYards",     0.032, 0.010, 0.060, 8.0),
        "player_receptions":      ("Receptions",         0.030, 0.020, 0.065, 0.8),
    }

    sigs: List[CorrelationSignal] = []
    for player, lines in player_lines.items():
        row = home_avg.get(player) or away_avg.get(player)
        opp_name = ev.away_team if (player in home_avg) else (ev.home_team if (player in away_avg) else None)
        if not row or not opp_name:
            continue
        for mk, cfg in mappings.items():
            stat, base, slope, cap, thr = cfg
            line = lines.get(mk)
            if line is None:
                continue
            val = float(row.get(stat, 0.0))
            gap = val - float(line)
            if gap >= thr:
                boost = min(cap, base + gap * slope)
                sigs.append(CorrelationSignal(
                    id=_stable_id(ev.id + player + mk + "nflrf"),
                    kind="matchup_trend",
                    eventId=ev.id,
                    players=[player],
                    teams=[],
                    markets=[mk],
                    boost=round(boost, 3),
                    reason=f"Recent form vs last 4: avg {val:.1f} > line {float(line):.1f} (opp {opp_name})"
                ))
    return sigs

def adapter_mlb_light_h2h(ev: APIEvent) -> List[CorrelationSignal]:
    if ev.sport_key.lower() not in ("baseball_mlb", "mlb"):
        return []
    player_lines: Dict[str, Dict[str, float]] = {}
    for bk in ev.bookmakers:
        for mk in bk.markets:
            if not mk.key.startswith("player_") and mk.key not in ("pitcher_strikeouts","pitcher_outs"):
                continue
            for o in mk.outcomes:
                name = o.description if (o.name.lower() in ("over","under") and o.description) else o.name
                if not name or o.point is None:
                    continue
                player_lines.setdefault(name, {})[mk.key] = float(o.point)

    if not player_lines:
        return []

    home_tid = _mlb_team_id_by_name(ev.home_team)
    away_tid = _mlb_team_id_by_name(ev.away_team)
    if not home_tid or not away_tid:
        return []

    sigs: List[CorrelationSignal] = []
    from datetime import datetime
    season = datetime.utcnow().year

    for player, lines in player_lines.items():
        pid = _mlb_search_player_id(player)
        if not pid:
            continue
        # try both opponents; prefer the one that yields data
        vs_away = _mlb_player_vs_opp_avg(pid, away_tid, season)
        vs_home = _mlb_player_vs_opp_avg(pid, home_tid, season)
        h2h = vs_away if vs_away else vs_home
        opp_name = ev.away_team if h2h is vs_away else ev.home_team
        if not h2h:
            continue

        # Hitter markets
        mappings_hit = [
            ("player_total_bases", "tb",  0.030, 0.015, 0.065),
            ("player_hits",        "h",   0.030, 0.020, 0.070),
            ("player_runs",        "r",   0.025, 0.015, 0.055),
            ("player_rbis",        "rbi", 0.025, 0.015, 0.055),
            ("player_home_runs",   "hr",  0.020, 0.040, 0.080),
            ("player_walks",       "bb",  0.020, 0.015, 0.050),
        ]
        for mk, stat, base, slope, cap in mappings_hit:
            line = lines.get(mk)
            if line is None:
                continue
            val = float(h2h.get(stat, 0.0))
            gap = val - float(line)
            thr = 0.5 if mk in ("player_total_bases","player_hits") else 0.2
            if gap >= thr:
                boost = min(cap, base + gap * slope)
                sigs.append(CorrelationSignal(
                    id=_stable_id(ev.id + player + mk + "mlbh2h"),
                    kind="head_to_head",
                    eventId=ev.id,
                    players=[player],
                    teams=[],
                    markets=[mk],
                    boost=round(boost, 3),
                    reason=f"H2H vs {opp_name}: avg {val:.2f} > line {float(line):.2f}"
                ))
        # Pitcher markets
        mappings_pit = [
            ("pitcher_strikeouts", "so",    0.030, 0.010, 0.060),
            ("pitcher_outs",       "outs",  0.025, 0.003, 0.060),
        ]
        for mk, stat, base, slope, cap in mappings_pit:
            line = lines.get(mk)
            if line is None:
                continue
            val = float(h2h.get(stat, 0.0))
            gap = val - float(line)
            thr = 0.5 if mk == "pitcher_strikeouts" else 1.0
            if gap >= thr:
                boost = min(cap, base + gap * slope)
                sigs.append(CorrelationSignal(
                    id=_stable_id(ev.id + player + mk + "mlbpitch"),
                    kind="head_to_head",
                    eventId=ev.id,
                    players=[player],
                    teams=[],
                    markets=[mk],
                    boost=round(boost, 3),
                    reason=f"H2H vs {opp_name}: avg {val:.2f} > line {float(line):.2f}"
                ))
    return sigs

def adapter_nhl_recent_form(ev: APIEvent) -> List[CorrelationSignal]:
    if ev.sport_key.lower() not in ("icehockey_nhl","nhl"):
        return []
    player_lines: Dict[str, Dict[str, float]] = {}
    for bk in ev.bookmakers:
        for mk in bk.markets:
            if mk.key not in ("player_shots_on_goal","player_goals","player_assists","player_points"):
                continue
            for o in mk.outcomes:
                name = o.description if (o.name.lower() in ("over","under") and o.description) else o.name
                if not name or o.point is None:
                    continue
                player_lines.setdefault(name, {})[mk.key] = float(o.point)

    if not player_lines:
        return []

    home_tid = _nhl_team_id_by_name(ev.home_team)
    away_tid = _nhl_team_id_by_name(ev.away_team)
    if not home_tid or not away_tid:
        return []

    sigs: List[CorrelationSignal] = []
    for player, lines in player_lines.items():
        pid = _nhl_search_player_id(player)
        if not pid:
            continue
        vs_away = _nhl_player_vs_opp_avg(pid, away_tid)
        vs_home = _nhl_player_vs_opp_avg(pid, home_tid)
        h2h = vs_away if vs_away else vs_home
        opp_name = ev.away_team if h2h is vs_away else ev.home_team
        if not h2h:
            continue

        mappings = [
            ("player_shots_on_goal", "shots",   0.030, 0.020, 0.070),
            ("player_goals",         "goals",   0.025, 0.050, 0.080),
            ("player_assists",       "assists", 0.025, 0.040, 0.070),
            ("player_points",        "points",  0.025, 0.035, 0.070),
        ]
        for mk, stat, base, slope, cap in mappings:
            line = lines.get(mk)
            if line is None:
                continue
            val = float(h2h.get(stat, 0.0))
            gap = val - float(line)
            thr = 0.4 if mk in ("player_shots_on_goal","player_points") else 0.2
            if gap >= thr:
                boost = min(cap, base + gap * slope)
                sigs.append(CorrelationSignal(
                    id=_stable_id(ev.id + player + mk + "nhlh2h"),
                    kind="head_to_head",
                    eventId=ev.id,
                    players=[player],
                    teams=[],
                    markets=[mk],
                    boost=round(boost, 3),
                    reason=f"H2H vs {opp_name}: avg {val:.2f} > line {float(line):.2f}"
                ))
    return sigs

def adapter_cfb_cfbd(ev: APIEvent) -> List[CorrelationSignal]:
    if ev.sport_key.lower() not in ("americanfootball_ncaaf","ncaa_football","cfb"):
        return []
    if not CFBD_API_KEY:
        return []

    # collect CFB lines of interest (QB/RB/WR/TE)
    player_lines: Dict[str, Dict[str, float]] = {}
    for bk in ev.bookmakers:
        for mk in bk.markets:
            if mk.key not in (
                "player_pass_yards","player_pass_tds",
                "player_rush_yards","player_rush_tds",
                "player_receiving_yards","player_receptions"
            ):
                continue
            for o in mk.outcomes:
                # O/U style vs single-side
                name = o.description if (o.name.lower() in ("over","under") and o.description) else o.name
                if not name or o.point is None:
                    continue
                player_lines.setdefault(name, {})[mk.key] = float(o.point)

    if not player_lines:
        return []

    yr = _year_from_iso(ev.commence_time) or 0
    if not yr:
        return []

    # Build per-opponent averages for both directions and pick one with data
    h2h_home = _cfbd_player_h2h_averages(ev.home_team, ev.away_team, yr)
    h2h_away = _cfbd_player_h2h_averages(ev.away_team, ev.home_team, yr)

    sigs: List[CorrelationSignal] = []
    # mappings: market -> (statKey, base, slope, cap, threshold)
    mappings = {
        "player_pass_yards":      ("passingYards",   0.030, 0.008, 0.060, 10.0),
        "player_pass_tds":        ("passingTDs",     0.028, 0.020, 0.070, 0.7),
        "player_rush_yards":      ("rushingYards",   0.030, 0.010, 0.060, 8.0),
        "player_rush_tds":        ("rushingTDs",     0.028, 0.030, 0.070, 0.6),
        "player_receiving_yards": ("receivingYards", 0.030, 0.010, 0.060, 8.0),
        "player_receptions":      ("receptions",     0.030, 0.020, 0.065, 0.8),
    }

    for player, lines in player_lines.items():
        # try to find averages in either home or away dict
        row = h2h_home.get(player) or h2h_away.get(player)
        opp_name = ev.away_team if (player in h2h_home) else (ev.home_team if (player in h2h_away) else None)
        if not row or not opp_name:
            continue
        for mk, cfg in mappings.items():
            stat, base, slope, cap, thr = cfg
            line = lines.get(mk)
            if line is None:
                continue
            val = float(row.get(stat, 0.0))
            gap = val - float(line)
            if gap >= thr:
                boost = min(cap, base + gap * slope)
                sigs.append(CorrelationSignal(
                    id=_stable_id(ev.id + player + mk + "cfbd"),
                    kind="head_to_head",
                    eventId=ev.id,
                    players=[player],
                    teams=[],
                    markets=[mk],
                    boost=round(boost, 3),
                    reason=f"H2H vs {opp_name}: avg {val:.1f} > line {float(line):.1f}"
                ))
    return sigs

def build_signals_for_event(ev: APIEvent) -> List[CorrelationSignal]:
    sigs: List[CorrelationSignal] = []
    try:
        before = len(sigs)
        sigs.extend(adapter_nba_h2h(ev))
        print(f"signals.adapter nba_h2h added={len(sigs)-before}")
    except Exception as e:
        print(f"signals.adapter nba_h2h error: {e}")

    for name, fn in [
        ("nfl_recent_form", adapter_nfl_recent_form),
        ("mlb_light_h2h", adapter_mlb_light_h2h),
        ("nhl_recent_form", adapter_nhl_recent_form),
        ("cfb_cfbd", adapter_cfb_cfbd),
    ]:
        try:
            before = len(sigs)
            sigs.extend(fn(ev))
            print(f"signals.adapter {name} added={len(sigs)-before}")
        except Exception as e:
            print(f"signals.adapter {name} error: {e}")

    return sigs

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.0"}

def _odds_impl(
    league: Optional[str],
    sport: Optional[str],
    date: Optional[str],
    market: Optional[str],
    response: Response
):
    sport_key = map_league_or_sport(league, sport)
    markets_csv = normalize_market_param(market)
    ck = _ckey("odds", sport_key, date or "upcoming", markets_csv)

    cached = cache.get(ck)
    if cached is not None:
        # cached is (events, headers_dict)
        events, hdrs = cached
        # forward credits header if we cached it
        rem = hdrs.get("x-requests-remaining") if isinstance(hdrs, dict) else None
        if rem is not None:
            response.headers["x-requests-remaining"] = str(rem)
        # server-side date filter is optional; app also filters by local day
        return filter_by_date(events, date)

    async def _do_fetch():
        async with httpx.AsyncClient() as http:
            events, hdrs = await fetch_odds_events(http, sport_key, markets_csv)
            cache.set(ck, (events, hdrs))
            rem = hdrs.get("x-requests-remaining")
            if rem is not None:
                response.headers["x-requests-remaining"] = str(rem)
            return filter_by_date(events, date)

    # Run fetch (sync wrapper since FastAPI lets us call async from sync via anyio)
    import anyio
    return anyio.run(_do_fetch)

@app.get("/api/odds", response_model=List[APIEvent])
def api_odds(
    response: Response,
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    market: Optional[str] = Query("all")
):
    return _odds_impl(league, sport, date, market, response)

@app.get("/api/upcoming", response_model=List[APIEvent])
def api_upcoming(
    response: Response,
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    market: Optional[str] = Query("all")
):
    return _odds_impl(league, sport, None, market, response)

# Aliases (your app tries these fallbacks too)
@app.get("/odds", response_model=List[APIEvent])
def odds_alias(
    response: Response,
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    market: Optional[str] = Query("all")
):
    return _odds_impl(league, sport, date, market, response)

@app.get("/upcoming", response_model=List[APIEvent])
def upcoming_alias(
    response: Response,
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    market: Optional[str] = Query("all")
):
    return _odds_impl(league, sport, None, market, response)

@app.get("/api/signals", response_model=List[CorrelationSignal])
def api_signals(
    event_id: str = Query(..., alias="event_id"),
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None)
):
    """
    Build correlation signals for a single event.
    We re-use cached odds if available; otherwise we fetch.
    """
    sport_key = map_league_or_sport(league, sport) if (league or sport) else None

    # Try to find the event in cache first (across known sports if necessary)
    def _scan_cached_for_event(skey: Optional[str]) -> Optional[APIEvent]:
        keys_to_check = []
        if skey:
            keys_to_check.append(_ckey("odds", skey, "upcoming", ALL_MARKETS))
        else:
            for sk in ("basketball_nba","americanfootball_nfl","americanfootball_ncaaf","baseball_mlb","icehockey_nhl"):
                keys_to_check.append(_ckey("odds", sk, "upcoming", ALL_MARKETS))

        for ck in keys_to_check:
            cached = cache.get(ck)
            if not cached:
                continue
            events, _ = cached
            for ev in events:
                if str(ev.id) == str(event_id):
                    return ev
        return None

    ev = _scan_cached_for_event(sport_key)
    if ev:
        return build_signals_for_event(ev)

    # If not cached, fetch upcoming for a best-guess sport and try again
    import anyio
    async def _ensure_fetch_and_find() -> List[CorrelationSignal]:
        skey = sport_key or "basketball_nba"
        async with httpx.AsyncClient() as http:
            events, _ = await fetch_odds_events(http, skey, ALL_MARKETS)
            cache.set(_ckey("odds", skey, "upcoming", ALL_MARKETS), (events, {}))
        for e in events:
            if str(e.id) == str(event_id):
                return build_signals_for_event(e)
        return []  # not fatal; app treats signals as optional

    return anyio.run(_ensure_fetch_and_find)
