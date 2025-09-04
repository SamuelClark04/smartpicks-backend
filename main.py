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
from pydantic import BaseModel, ConfigDict
# Pydantic v2: allow fields starting with "model_" (e.g., model_prob)
class SPBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
import sqlite3
import gzip
from pathlib import Path
import urllib.parse
from contextlib import contextmanager

# ------------------------------------------------------------
# Environment / Config
# ------------------------------------------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "").strip()  # placeholder for later
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "us").strip()
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "smartpicks.sqlite"

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
IS_PG = DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")

# Lazy import for Postgres so local SQLite still works without extra deps
_psycopg = None
if IS_PG:
    try:
        import psycopg
        _psycopg = psycopg
    except Exception as e:
        # We will raise a clear error later on first DB access if psycopg is missing
        _psycopg = e  # store exception to report nicely

@contextmanager
def db_conn():
    """Context manager that yields a connection with autocommit for PG and row factory for sqlite."""
    if IS_PG:
        if isinstance(_psycopg, Exception):
            raise RuntimeError(
                "DATABASE_URL is set but psycopg is not installed. Add 'psycopg[binary]' to requirements.txt."
            )
        conn = _psycopg.connect(DATABASE_URL, autocommit=True)
        try:
            yield conn
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

# Helpers for SQL param style differences
def qmark(sql: str) -> str:
    """Convert sqlite style '?' placeholders to Postgres '%s' if needed."""
    if IS_PG:
        # naive but safe: replace '?' that are used as value placeholders
        parts = sql.split('?')
        return '%s'.join(parts)
    return sql

def _init_db():
    with db_conn() as conn:
        cur = conn.cursor()
        if IS_PG:
            # Enable extensions if desired (not required); keep DDL idempotent
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS odds_snapshots (
                  id SERIAL PRIMARY KEY,
                  sport TEXT NOT NULL,
                  ymd TEXT NOT NULL,
                  markets TEXT NOT NULL,
                  regions TEXT NOT NULL,
                  bookmakers TEXT NOT NULL,
                  fetched_at BIGINT NOT NULL,
                  body_gz BYTEA NOT NULL,
                  UNIQUE(sport, ymd, markets, regions, bookmakers)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS player_game_stats (
                  id SERIAL PRIMARY KEY,
                  league TEXT NOT NULL,
                  game_date TEXT NOT NULL,
                  player TEXT NOT NULL,
                  team TEXT,
                  opp TEXT,
                  minutes DOUBLE PRECISION,
                  pts DOUBLE PRECISION, ast DOUBLE PRECISION, reb DOUBLE PRECISION, fg3m DOUBLE PRECISION,
                  stl DOUBLE PRECISION, blk DOUBLE PRECISION, tov DOUBLE PRECISION,
                  rush_yd DOUBLE PRECISION, rec_yd DOUBLE PRECISION, pass_yd DOUBLE PRECISION, sog DOUBLE PRECISION,
                  UNIQUE(league, game_date, player)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pair_corr (
                  id SERIAL PRIMARY KEY,
                  league TEXT NOT NULL,
                  window TEXT NOT NULL,
                  player_a TEXT NOT NULL,
                  stat_a TEXT NOT NULL,
                  player_b TEXT NOT NULL,
                  stat_b TEXT NOT NULL,
                  r DOUBLE PRECISION NOT NULL,
                  n INTEGER NOT NULL,
                  updated_at BIGINT NOT NULL,
                  UNIQUE(league, window, player_a, stat_a, player_b, stat_b)
                );
                """
            )
        else:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS odds_snapshots (
                  id INTEGER PRIMARY KEY,
                  sport TEXT NOT NULL,
                  ymd TEXT NOT NULL,
                  markets TEXT NOT NULL,
                  regions TEXT NOT NULL,
                  bookmakers TEXT NOT NULL,
                  fetched_at INTEGER NOT NULL,
                  body_gz BLOB NOT NULL,
                  UNIQUE(sport, ymd, markets, regions, bookmakers)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS player_game_stats (
                  id INTEGER PRIMARY KEY,
                  league TEXT NOT NULL,
                  game_date TEXT NOT NULL,
                  player TEXT NOT NULL,
                  team TEXT,
                  opp TEXT,
                  minutes REAL,
                  pts REAL, ast REAL, reb REAL, fg3m REAL,
                  stl REAL, blk REAL, tov REAL,
                  rush_yd REAL, rec_yd REAL, pass_yd REAL, sog REAL,
                  UNIQUE(league, game_date, player)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pair_corr (
                  id INTEGER PRIMARY KEY,
                  league TEXT NOT NULL,
                  window TEXT NOT NULL,
                  player_a TEXT NOT NULL,
                  stat_a TEXT NOT NULL,
                  player_b TEXT NOT NULL,
                  stat_b TEXT NOT NULL,
                  r REAL NOT NULL,
                  n INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  UNIQUE(league, window, player_a, stat_a, player_b, stat_b)
                );
                """
            )

_init_db()


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
# === Odds disk-cache helpers (gzipped JSON in SQLite) ===
def _odds_cache_key(sport_key: str, ymd: str, markets_csv: str, regions: str, books: str) -> tuple:
    return (sport_key, ymd, markets_csv, regions, books)

def _odds_cache_get(sport_key: str, ymd: str, markets_csv: str, regions: str, books: str, max_age_sec: int = 600):
    ck = _odds_cache_key(sport_key, ymd, markets_csv, regions, books)
    with db_conn() as conn:
        cur = conn.cursor()
        sql = qmark("SELECT fetched_at, body_gz FROM odds_snapshots WHERE sport=? AND ymd=? AND markets=? AND regions=? AND bookmakers=?")
        cur.execute(sql, ck)
        row = cur.fetchone()
    if not row:
        return None
    fetched_at, body_gz = row
    if time.time() - fetched_at > max_age_sec:
        return None
    try:
        body = gzip.decompress(bytes(body_gz)) if IS_PG else gzip.decompress(body_gz)
        payload = json.loads(body.decode("utf-8"))
        return payload.get("events", [])
    except Exception:
        return None

def _odds_cache_put(sport_key: str, ymd: str, markets_csv: str, regions: str, books: str, events: list):
    ev_dicts = [e.dict() if hasattr(e, "dict") else e for e in events]
    blob = gzip.compress(json.dumps({"events": ev_dicts}, separators=(",", ":")).encode("utf-8"))
    ck = _odds_cache_key(sport_key, ymd, markets_csv, regions, books)
    with db_conn() as conn:
        cur = conn.cursor()
        if IS_PG:
            sql = qmark(
                """
                INSERT INTO odds_snapshots (sport, ymd, markets, regions, bookmakers, fetched_at, body_gz)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (sport, ymd, markets, regions, bookmakers)
                DO UPDATE SET fetched_at = EXCLUDED.fetched_at, body_gz = EXCLUDED.body_gz
                """
            )
            cur.execute(sql, (*ck, int(time.time()), blob))
        else:
            sql = qmark(
                """
                INSERT INTO odds_snapshots (sport, ymd, markets, regions, bookmakers, fetched_at, body_gz)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sport, ymd, markets, regions, bookmakers)
                DO UPDATE SET fetched_at=excluded.fetched_at, body_gz=excluded.body_gz
                """
            )
            cur.execute(sql, (*ck, int(time.time()), blob))


# ------------------------------------------------------------
# Pydantic models that MATCH your Swift Decodables (non-optionals included)
# ------------------------------------------------------------
class APIOutcome(SPBase):
    name: str
    price: float
    point: Optional[float] = None
    description: Optional[str] = None
    model_prob: Optional[float] = None  # de‑vig normalized probability within an outcome group

class APIMarket(SPBase):
    key: str
    outcomes: List[APIOutcome]

class APIBookmaker(SPBase):
    key: str
    title: str
    last_update: str   # non-optional in Swift
    markets: List[APIMarket]

class APIEvent(SPBase):
    id: str
    sport_key: str
    sport_title: str   # non-optional in Swift
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[APIBookmaker]

# Correlation signal model (for /api/signals) matching your Swift struct
class CorrelationSignal(SPBase):
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
app = FastAPI(title="SmartPicks Backend", version="1.0.1")

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

DEFAULT_BOOKMAKERS = [
    "fanduel","draftkings","betmgm","caesars",
    "pointsbetus","betrivers","barstool","bovada",
    "bet365","williamhill_us","betway","superbook",
    "prizepicks","underdog"
]
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
    "player_strikeouts","pitcher_strikeouts","pitcher_outs","pitcher_hits_allowed",
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
        "player_strikeouts","pitcher_strikeouts","pitcher_outs","pitcher_hits_allowed",
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

# ---- OddsAPI market chunk/merge helpers ----
def _split_markets(csv: str, chunk_size: int = 3) -> List[str]:
    items = [m.strip() for m in (csv or "").split(",") if m.strip()]
    return [",".join(items[i:i+chunk_size]) for i in range(0, len(items), chunk_size)]

def _merge_market_lists(a: List[APIMarket], b: List[APIMarket]) -> List[APIMarket]:
    # merge by market.key, de-duping outcomes by (name, point, description)
    by_key: Dict[str, APIMarket] = {m.key: APIMarket(key=m.key, outcomes=list(m.outcomes)) for m in a}
    for m in b:
        if m.key not in by_key:
            by_key[m.key] = APIMarket(key=m.key, outcomes=list(m.outcomes))
        else:
            seen = {(o.name, o.point, o.description) for o in by_key[m.key].outcomes}
            for o in m.outcomes:
                key = (o.name, o.point, o.description)
                if key not in seen:
                    by_key[m.key].outcomes.append(o)
                    seen.add(key)
    return list(by_key.values())

def _merge_bookmakers(a: List[APIBookmaker], b: List[APIBookmaker]) -> List[APIBookmaker]:
    by_key: Dict[str, APIBookmaker] = {bk.key: APIBookmaker(key=bk.key, title=bk.title, last_update=bk.last_update, markets=list(bk.markets)) for bk in a}
    for bk in b:
        if bk.key not in by_key:
            by_key[bk.key] = APIBookmaker(key=bk.key, title=bk.title, last_update=bk.last_update, markets=list(bk.markets))
        else:
            merged_markets = _merge_market_lists(by_key[bk.key].markets, bk.markets)
            by_key[bk.key] = APIBookmaker(
                key=by_key[bk.key].key,
                title=by_key[bk.key].title if by_key[bk.key].title else bk.title,
                last_update=bk.last_update or by_key[bk.key].last_update,
                markets=merged_markets
            )
    return list(by_key.values())

def _merge_events(a: List[APIEvent], b: List[APIEvent]) -> List[APIEvent]:
    by_id: Dict[str, APIEvent] = {e.id: APIEvent(**e.dict()) for e in a}
    for e in b:
        if e.id not in by_id:
            by_id[e.id] = APIEvent(**e.dict())
        else:
            merged_bk = _merge_bookmakers(by_id[e.id].bookmakers, e.bookmakers)
            by_id[e.id] = APIEvent(
                id=by_id[e.id].id,
                sport_key=by_id[e.id].sport_key or e.sport_key,
                sport_title=by_id[e.id].sport_title or e.sport_title,
                commence_time=by_id[e.id].commence_time or e.commence_time,
                home_team=by_id[e.id].home_team or e.home_team,
                away_team=by_id[e.id].away_team or e.away_team,
                bookmakers=merged_bk
            )
    return list(by_id.values())

def _ckey(*parts) -> str:
    return "|".join(str(p) for p in parts if p is not None)

# ---- American odds to implied probability helper ----
def _american_to_implied(price: float) -> float:
    try:
        p = float(str(price).strip())
        if p < 0:
            return (-p) / ((-p) + 100.0)
        return 100.0 / (p + 100.0)
    except Exception:
        return 0.0

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
    Returns (events, upstream_headers) so we can forward x-requests-remaining).

    Batches markets to avoid 422s and caches disk snapshots.
    """
    if not ODDS_API_KEY:
        raise HTTPException(status_code=401, detail="Server missing ODDS_API_KEY")

    # Filter by sport and then split into small chunks to stay under provider limits
    filtered_markets = _filter_markets_for_sport(markets_csv, sport_key)

    # unified string of books, reuse later for cache key & store
    books_str = ",".join(bookmakers or DEFAULT_BOOKMAKERS)

    # try disk snapshot cache first (stored under ymd='upcoming')
    cached_events = _odds_cache_get(
        sport_key,
        "upcoming",
        filtered_markets,
        ODDS_REGIONS,
        books_str
    )
    if cached_events:
        try:
            return [APIEvent(**e) for e in cached_events], {}
        except Exception:
            pass

    # chunk markets to avoid 422 limits upstream
    chunks = _split_markets(filtered_markets, chunk_size=3)

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    common = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "oddsFormat": "american",
        "bookmakers": books_str,
    }

    merged_events: List[APIEvent] = []
    last_headers: Dict[str, str] = {}

    async def _one_call(markets: str, no_bookmakers: bool = False) -> Tuple[List[APIEvent], Dict[str, str]]:
        params = {**common, "markets": markets}
        if no_bookmakers:
            params.pop("bookmakers", None)
        r = await http.get(url, params=params, timeout=30)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)

        raw = r.json()
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        evs: List[APIEvent] = []

        for e in raw:
            bks: List[APIBookmaker] = []
            for b in e.get("bookmakers", []) or []:
                mkts: List[APIMarket] = []
                for m in b.get("markets", []) or []:
                    outs: List[APIOutcome] = []
                    for o in m.get("outcomes", []) or []:
                        _name = str(o.get("name", "")).strip()
                        # try to recover a player/runner name consistently
                        _desc = (
                            o.get("description")
                            or o.get("player") or o.get("playerName")
                            or o.get("participant") or o.get("runner") or o.get("competitor")
                            or o.get("participant_name") or o.get("athlete") or o.get("entity")
                            or o.get("label") or o.get("selection")
                        )
                        if not _desc and _name.lower() in ("over", "under"):
                            d = o.get("outcome") or {}
                            _desc = d.get("description") or d.get("player") or _desc

                        try:
                            _price = float(o.get("price", 0) or 0)
                        except Exception:
                            _price = 0.0

                        outs.append(APIOutcome(
                            name=_name,
                            price=_price,
                            point=o.get("point"),
                            description=_desc or "",
                            model_prob=None,
                        ))

                    # de-vig within logical groups
                    groups: Dict[Tuple, List[int]] = {}
                    for idx, oc in enumerate(outs):
                        if oc.point is not None and oc.description:
                            key = (round(float(oc.point), 3), oc.description.strip().lower())
                        elif oc.point is not None:
                            key = (round(float(oc.point), 3),)
                        else:
                            key = ("all",)
                        groups.setdefault(key, []).append(idx)

                    for _, idxs in groups.items():
                        implieds = [max(1e-6, _american_to_implied(outs[i].price)) for i in idxs]
                        s = sum(implieds) if sum(implieds) > 1e-9 else 1.0
                        for i, imp in zip(idxs, implieds):
                            outs[i].model_prob = float(imp / s)

                    mkts.append(APIMarket(key=str(m.get("key","")), outcomes=outs))

                bks.append(APIBookmaker(
                    key=str(b.get("key","")),
                    title=(b.get("title") or b.get("key") or "").strip(),
                    last_update=str(b.get("last_update") or ""),
                    markets=mkts
                ))

            evs.append(APIEvent(
                id=_event_fallback_id(e),
                sport_key=sport_key,
                sport_title=HUMAN_TITLES.get(sport_key, str(e.get("sport_title") or "")),
                commence_time=str(e.get("commence_time") or ""),
                home_team=str(e.get("home_team") or ""),
                away_team=str(e.get("away_team") or ""),
                bookmakers=bks
            ))
        return evs, hdrs

    success_chunks = 0
    for mk_chunk in chunks:
        try:
            evs, hdrs = await _one_call(mk_chunk)
            merged_events = _merge_events(merged_events, evs)
            last_headers = hdrs
            success_chunks += 1
        except HTTPException as e:
            if e.status_code in (400, 404, 422):
                if e.status_code == 422:
                    try:
                        evs, hdrs = await _one_call(mk_chunk, no_bookmakers=True)
                        merged_events = _merge_events(merged_events, evs)
                        last_headers = hdrs
                        success_chunks += 1
                        print(f"[fetch_odds_events] 422 retry succeeded without bookmakers for chunk='{mk_chunk}'")
                        continue
                    except HTTPException as e2:
                        print(f"[fetch_odds_events] Skipping unsupported markets chunk='{mk_chunk}' status=422 (retry failed {e2.status_code})")
                        continue
                else:
                    print(f"[fetch_odds_events] Skipping unsupported markets chunk='{mk_chunk}' status={e.status_code}")
                    continue
            raise

    if success_chunks == 0:
        try:
            evs, hdrs = await _one_call("h2h,spreads,totals")
            merged_events = _merge_events(merged_events, evs)
            last_headers = hdrs
        except HTTPException:
            raise

    # If we still have no player props, try a lighter set without restricting books
    have_player = any(
        any(mk.key.startswith("player_") or mk.key.startswith("pitcher_")
            for bk in ev.bookmakers for mk in bk.markets)
        for ev in merged_events
    )
    if not have_player:
        light_by_sport = {
            "basketball_nba": ["player_points","player_rebounds","player_assists","player_threes"],
            "americanfootball_nfl": ["player_pass_yards","player_rush_yards","player_receiving_yards","player_receptions"],
            "americanfootball_ncaaf": ["player_pass_yards","player_rush_yards","player_receiving_yards","player_receptions"],
            "baseball_mlb": ["player_total_bases","player_hits","player_home_runs","pitcher_strikeouts","pitcher_outs"],
            "icehockey_nhl": ["player_shots_on_goal","player_points","player_goals","player_assists"],
        }
        light = light_by_sport.get(sport_key, [])
        if light:
            merged_events = []
            last_headers = {}
            for mk_chunk in _split_markets(",".join(light), chunk_size=4):
                try:
                    evs, hdrs = await _one_call(mk_chunk, no_bookmakers=True)
                    merged_events = _merge_events(merged_events, evs)
                    last_headers = hdrs
                except HTTPException as e:
                    if e.status_code in (400, 404, 422):
                        print(f"[fetch_odds_events] Light retry skipped chunk='{mk_chunk}' status={e.status_code}")
                        continue
                    raise

    try:
        _odds_cache_put(
            sport_key,
            "upcoming",
            filtered_markets,
            ODDS_REGIONS,
            books_str,
            merged_events
        )
    except Exception as _e:
        print(f"[odds_cache_put] non-fatal error: {_e}")

    return merged_events, last_headers

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
    
def _event_fallback_id(raw: dict) -> str:
    parts = [
        str(raw.get("id") or ""),
        str(raw.get("commence_time") or ""),
        str(raw.get("home_team") or ""),
        str(raw.get("away_team") or "")
    ]
    base = "||".join(parts).strip()
    return (raw.get("id") or hashlib.md5(base.encode()).hexdigest())

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
    ck = _ckey("odds", sport_key, date or "upcoming", markets_csv, ODDS_REGIONS)

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

    # Try the on-disk snapshot cache for 'upcoming' (we store by 'upcoming', app also filters by date)
    disk_evs = _odds_cache_get(sport_key, "upcoming", markets_csv, ODDS_REGIONS, ",".join(DEFAULT_BOOKMAKERS))
    if disk_evs:
        try:
            events = [APIEvent(**e) for e in disk_evs]
            cache.set(ck, (events, {}))
            return filter_by_date(events, date)
        except Exception:
            pass

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
    market: Optional[str] = Query("all", description="CSV of markets; 'all' requests team + player props")
):
    return _odds_impl(league, sport, date, market, response)

@app.get("/api/upcoming", response_model=List[APIEvent])
def api_upcoming(
    response: Response,
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    market: Optional[str] = Query("all", description="CSV of markets; 'all' requests team + player props")
):
    return _odds_impl(league, sport, None, market, response)

# Aliases (your app tries these fallbacks too)
@app.get("/odds", response_model=List[APIEvent])
def odds_alias(
    response: Response,
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    market: Optional[str] = Query("all", description="CSV of markets; 'all' requests team + player props")
):
    return _odds_impl(league, sport, date, market, response)

@app.get("/upcoming", response_model=List[APIEvent])
def upcoming_alias(
    response: Response,
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    market: Optional[str] = Query("all", description="CSV of markets; 'all' requests team + player props")
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


# ------------------------------------------------------------
# /api/corr: Lightweight on-demand correlation signals for slip
# ------------------------------------------------------------
@app.get("/api/corr", response_model=List[CorrelationSignal])
def api_corr(
    event_id: str = Query(..., description="Event id from /api/odds"),
    players: str = Query("", description="comma-separated player names in slip for this event"),
    markets: Optional[str] = Query(None, description="optional comma list of market keys to focus"),
    league: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
):
    """
    Lightweight, on-demand correlations that the app can call after a leg is added.
    - Reuses the same CorrelationSignal schema as /api/signals so the UI can merge them.
    - Conservative caps (±0.08) to avoid over-influence.
    """
    # helper: find event in cache (mirrors logic inside /api/signals)
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

    sport_key = map_league_or_sport(league, sport) if (league or sport) else None
    ev = _scan_cached_for_event(sport_key)

    if not ev:
        # Best-effort fetch of upcoming for the guessed sport, then try to find the event
        import anyio, httpx as _hx
        async def _ensure_fetch_and_find() -> Optional[APIEvent]:
            skey = sport_key or "basketball_nba"
            async with _hx.AsyncClient() as http:
                events, _ = await fetch_odds_events(http, skey, ALL_MARKETS)
                cache.set(_ckey("odds", skey, "upcoming", ALL_MARKETS), (events, {}))
            for e in events:
                if str(e.id) == str(event_id):
                    return e
            return None
        ev = anyio.run(_ensure_fetch_and_find)

    if not ev:
        return []

    # Start with the standard signals for this event (H2H/recent form adapters)
    sigs: List[CorrelationSignal] = []
    try:
        sigs.extend(build_signals_for_event(ev))
    except Exception as e:
        print(f"/api/corr base build error: {e}")

    # Augment with slip-aware, bounded correlations
    names = [p.strip() for p in (players or "").split(",") if p.strip()]
    wanted = set(m.strip() for m in (markets or "").split(",") if m) if markets else None

    # Rule A: same-player, multi-market → small role-stability boost
    for n in names:
        mk = list(wanted) if wanted else [
            "player_points","player_assists","player_rebounds","player_threes",
            "player_pass_yards","player_receiving_yards","player_rush_yards",
            "player_total_bases","player_hits","player_shots_on_goal"
        ]
        if mk:
            sigs.append(CorrelationSignal(
                id=_stable_id(f"same_player|{ev.id}|{n}"),
                kind="news",
                eventId=ev.id,
                players=[n],
                teams=[],
                markets=mk,
                boost=0.04,
                reason=f"{n}: consistent role across markets (recent form proxy)"
            ))

    # Rule B: scorer + facilitator pairing (NBA-ish) → mild bump
    if len(names) >= 2 and (ev.sport_key.lower() in ("basketball_nba","nba")):
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = names[i], names[j]
                if not wanted:
                    sigs.append(CorrelationSignal(
                        id=_stable_id(f"scorer_fac|{ev.id}|{a}|{b}"),
                        kind="team_style",
                        eventId=ev.id,
                        players=[a, b],
                        teams=[],
                        markets=["player_points","player_assists"],
                        boost=0.06,
                        reason="High-usage scorer + facilitator in same game"
                    ))

    # Rule C: tiny game-wide tailwind (kept very small)
    sigs.append(CorrelationSignal(
        id=_stable_id(f"game_tailwind|{ev.id}"),
        kind="pace_injury",
        eventId=ev.id,
        players=[],
        teams=[],
        markets=[],
        boost=0.03,
        reason="Game-wide pace/injury tailwind (bounded)"
    ))

    # Optional: fold in any precomputed correlations from DB (bounded window)
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            sql = qmark("SELECT player_a, stat_a, player_b, stat_b, r, n FROM pair_corr WHERE league=? AND window=? LIMIT 200")
            cur.execute(sql, (ev.sport_key, "recent"))
            for (pa, sa, pb, sb, r, n) in cur.fetchall() or []:
                if abs(float(r)) < 0.15 or int(n) < 8:
                    continue
                boost = max(-0.08, min(0.08, 0.20 * float(r)))
                sigs.append(CorrelationSignal(
                    id=_stable_id(f"dbcorr|{ev.id}|{pa}|{sa}|{pb}|{sb}"),
                    kind="db_corr",
                    eventId=ev.id,
                    players=[pa, pb],
                    teams=[],
                    markets=[sa, sb],
                    boost=float(boost),
                    reason=f"DB correlations r={float(r):.2f} over n={int(n)}"
                ))
    except Exception as _e:
        print(f"/api/corr db merge skipped: {_e}")

    # Cap boosts conservatively and de-duplicate by id
    capped: Dict[str, CorrelationSignal] = {}
    # (safety) cap boosts to ±0.08
    for s in sigs:
        s.boost = float(max(-0.08, min(0.08, s.boost)))
        capped[s.id] = s
    dedup = {}
    for s in sigs:
        s.boost = float(max(-0.08, min(0.08, s.boost)))
        dedup[s.id] = s
    return list(dedup.values())


@app.post("/admin/backfill")
def admin_backfill(league: str = "nba", days: int = 30):
    """
    Stub endpoint: integrate your stats feed here and call
    `upsert_player_stat_row(...)` per player per game to build local correlations.
    """
    return {"ok": True, "league": league, "window_days": days, "rows": 0}


# Admin: prune old odds snapshots to save storage
@app.post("/admin/prune")
def admin_prune(max_age_days: int = 14):
    """Delete old odds snapshots older than N days to keep storage small."""
    cutoff = int(time.time() - max_age_days * 86400)
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            if IS_PG:
                cur.execute(qmark("DELETE FROM odds_snapshots WHERE fetched_at < ?"), (cutoff,))
            else:
                cur.execute("DELETE FROM odds_snapshots WHERE fetched_at < ?", (cutoff,))
        return {"ok": True, "deleted_older_than_days": max_age_days}
    except Exception as e:
        return {"ok": False, "error": str(e)}
