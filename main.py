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

# ---- SmartPicks odds quota guards ----
TEAM_MARKETS = "h2h,spreads,totals"
ODDS_PULL_PLAYER_ON_ODDS = os.getenv("ODDS_PULL_PLAYER_ON_ODDS", "0").strip() in ("1","true","yes")
ODDS_COOLDOWN_SEC = int(os.getenv("ODDS_COOLDOWN_SEC", "120").strip() or 120)
ODDS_MAX_PROP_EVENTS = int(os.getenv("ODDS_MAX_PROP_EVENTS", "0").strip() or 0)

FREE_SANDBOX = os.getenv("FREE_SANDBOX", "0").strip() == "1"
ODDS_DISABLE = os.getenv("ODDS_DISABLE", "0").strip() == "1"




# --- BEGIN: market safety layer ---
# Minimal per-sport allowlist (provider-compatible names). We’ll intersect with ODDS_MARKETS/Query.
SPORT_ALLOWED_MARKETS = {
    "basketball_nba": [
        "h2h","spreads","totals",
        "player_points","player_rebounds","player_assists","player_threes",
        "player_steals","player_blocks","player_turnovers",
        "player_points_rebounds_assists","player_points_rebounds","player_points_assists","player_rebounds_assists",
    ],
    "americanfootball_nfl": [
        "h2h","spreads","totals",
        "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_pass_interceptions",
        "player_rush_yards","player_rush_attempts","player_rush_tds",
        "player_receiving_yards","player_receptions","player_reception_tds",
        "player_reception_longest","player_rush_longest",
    ],
    "baseball_mlb": [
        "h2h","spreads","totals",
        "batter_total_bases","batter_hits","batter_runs_scored","batter_rbis","batter_home_runs","batter_walks",
        "batter_strikeouts","pitcher_strikeouts","pitcher_outs","pitcher_hits_allowed",
    ],
    "icehockey_nhl": [
        "h2h","spreads","totals",
        "player_points","player_goals","player_assists","player_shots_on_goal","player_total_saves",
    ],
    "americanfootball_ncaaf": [
        "h2h","spreads","totals",
        "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_pass_interceptions",
        "player_rush_yards","player_rush_attempts","player_rush_tds",
        "player_receiving_yards","player_receptions","player_reception_tds",
        "player_reception_longest","player_rush_longest",
    ],
}

# Keep a process-local “learned unsupported” set so once a market 422s we stop asking for it.
LEARNED_UNSUPPORTED = {}  # dict[str sport_key] -> set[str market]
# --- END: market safety layer ---

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
import random
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

# Track last upstream fetch per sport to avoid burst refetches
_last_fetch_ts: Dict[str, float] = {}
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

# ---- FastAPI health and root endpoints ----
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"name": "SmartPicks Backend", "version": app.version}

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
    Tolerates ESPN endpoints that return JSON with content-type=text/plain.
    """
    key = _ckey("httpjson", url, json.dumps(params or {}, sort_keys=True), json.dumps(headers or {}, sort_keys=True))
    cached = cache.get(key)
    if cached is not None:
        return cached

    import anyio

    async def _fetch():
        async with httpx.AsyncClient() as http:
            r = await http.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
            # Try r.json first; if it fails or content-type is text/plain, fall back to json.loads(r.text)
            try:
                if ctype in ("application/json", "application/hal+json", "text/json"):
                    return r.json()
                # ESPN often returns text/plain with JSON body
                return json.loads(r.text)
            except Exception:
                # Last resort: try bytes decode
                try:
                    return json.loads(r.content.decode("utf-8", errors="ignore"))
                except Exception:
                    raise

    data = anyio.run(_fetch)
    cache.set(key, data)
    return data
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
    "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_pass_interceptions",
    "player_rush_yards","player_rush_attempts","player_rush_tds","player_receiving_yards","player_receptions",
    "player_reception_tds","player_reception_longest","player_rush_longest",
    # MLB
    "batter_hits","batter_runs_scored","batter_rbis","batter_home_runs","batter_total_bases","batter_walks",
    "batter_strikeouts","pitcher_strikeouts","pitcher_outs","pitcher_hits_allowed",
    # NHL
    "player_shots_on_goal","player_goals","player_points","player_total_saves"
])

# --- ESPN endpoints & mapping helper ---
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_SCOREBOARD = {
    "basketball_nba": f"{ESPN_BASE}/basketball/nba/scoreboard",
    "americanfootball_nfl": f"{ESPN_BASE}/football/nfl/scoreboard",
    "americanfootball_ncaaf": f"{ESPN_BASE}/football/college-football/scoreboard",
    "baseball_mlb": f"{ESPN_BASE}/baseball/mlb/scoreboard",
    "icehockey_nhl": f"{ESPN_BASE}/hockey/nhl/scoreboard",
}
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
    "ncaaf": "americanfootball_ncaaf",
    "cfb": "americanfootball_ncaaf",

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
    "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_pass_interceptions",
    "player_rush_yards","player_rush_attempts","player_rush_tds","player_receiving_yards","player_receptions",
    "player_reception_tds","player_reception_longest","player_rush_longest",
    # MLB
    "batter_hits","batter_runs_scored","batter_rbis","batter_home_runs","batter_total_bases","batter_walks",
    "batter_strikeouts","pitcher_strikeouts","pitcher_outs","pitcher_hits_allowed",
    # NHL
    "player_shots_on_goal","player_goals","player_points","player_total_saves"
])

# Per-sport market allowlists (to avoid 422 from The Odds API when requesting unsupported markets)
SPORT_MARKETS: Dict[str, List[str]] = {
    "basketball_nba": [
        "h2h","spreads","totals",
        "player_points","player_rebounds","player_assists","player_threes",
        "player_steals","player_blocks","player_turnovers",
        "player_points_rebounds_assists","player_points_rebounds","player_points_assists","player_rebounds_assists",
    ],
    "americanfootball_nfl": [
        "h2h","spreads","totals",
        "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_pass_interceptions",
        "player_rush_yards","player_rush_attempts","player_rush_tds",
        "player_receiving_yards","player_receptions","player_reception_tds",
        "player_reception_longest","player_rush_longest",
    ],
    "americanfootball_ncaaf": [
        "h2h","spreads","totals",
        "player_pass_yards","player_pass_attempts","player_pass_completions","player_pass_tds","player_pass_interceptions",
        "player_rush_yards","player_rush_attempts","player_rush_tds",
        "player_receiving_yards","player_receptions","player_reception_tds",
        "player_reception_longest","player_rush_longest",
    ],
    "baseball_mlb": [
        "h2h","spreads","totals",
        "batter_total_bases","batter_hits","batter_runs_scored","batter_rbis","batter_home_runs","batter_walks",
        "batter_strikeouts","pitcher_strikeouts","pitcher_outs","pitcher_hits_allowed",
    ],
    "icehockey_nhl": [
        "h2h","spreads","totals",
        "player_points","player_goals","player_assists","player_shots_on_goal","player_total_saves",
    ],
    "tennis_atp": [
        "h2h","spreads","totals",
        "player_aces","player_double_faults","player_total_games_won"
    ],
    "tennis_wta": [
        "h2h","spreads","totals",
        "player_aces","player_double_faults","player_total_games_won"
    ],
}

# ---------- SANDBOX (free fallback) ----------

# ---- FREE EVENTS/PROPS FROM PUBLIC SOURCES (no OddsAPI) ----
# If FREE_SANDBOX=1 or ODDS_DISABLE=1 or missing ODDS_API_KEY,
# we can still return *real* upcoming events (ESPN) and *model* player props
# built from free public stats APIs (balldontlie for NBA). Other leagues
# fallback to synthetic props if we can't compute them for free.

def _espn_upcoming_events(sport_key: str) -> list[APIEvent]:
    """Return upcoming events using ESPN scoreboard for a given sport_key.
    Falls back to an empty list on error; caller may synthesize instead.
    """
    url = ESPN_SCOREBOARD.get(sport_key)
    if not url:
        return []
    try:
        data = _http_get_json(url)
        events = []
        # ESPN shapes vary; handle common layout
        items = []
        if isinstance(data, dict):
            if isinstance(data.get("events"), list):
                items = data["events"]
            elif isinstance(data.get("leagues"), list):
                items = data.get("events", [])
        for e in items or []:
            comp = (e.get("competitions") or [{}])[0]
            teams = comp.get("competitors") or []
            if len(teams) < 2:
                continue
            # ESPN marks home via "homeAway"
            home = next((t for t in teams if (t.get("homeAway") or "").lower() == "home"), None)
            away = next((t for t in teams if (t.get("homeAway") or "").lower() == "away"), None)
            if not home or not away:
                # fallback by index order
                home = teams[0]
                away = teams[1] if len(teams) > 1 else teams[0]
            home_name = (home.get("team") or {}).get("displayName") or home.get("displayName") or "Home"
            away_name = (away.get("team") or {}).get("displayName") or away.get("displayName") or "Away"
            ct = (comp.get("date") or e.get("date") or "").replace("+00:00", "Z")
            raw_id = e.get("id") or _stable_id(f"{sport_key}|{home_name}|{away_name}|{ct}")
            events.append(APIEvent(
                id=str(raw_id),
                sport_key=sport_key,
                sport_title=HUMAN_TITLES.get(sport_key, sport_key),
                commence_time=ct,
                home_team=str(home_name),
                away_team=str(away_name),
                bookmakers=[],
            ))
        return events
    except Exception:
        return []

# ---- Free/model props builders ----

def _model_nba_props_for_event(ev: APIEvent, wanted: list[str] | None) -> list[APIMarket]:
    """Build NBA player props using free balldontlie season averages.
    Markets: player_points, player_rebounds, player_assists, player_threes.
    We select ~6 top-minute players across both teams and set lines to their
    season averages, rounding sensibly. Prices are set to -115 for both sides.
    """
    try:
        tmap = _nba_list_teams()
        # Map team names to balldontlie IDs using fuzzy contains
        def _tid(name: str) -> int | None:
            if not name:
                return None
            n = name.lower()
            if n in tmap:
                return tmap[n]["id"]
            for k, v in tmap.items():
                if n in k or k in n:
                    return v["id"]
            return None
        home_tid = _tid(ev.home_team)
        away_tid = _tid(ev.away_team)
        if not home_tid and not away_tid:
            return []
        def _team_players(team_id: int) -> list[dict]:
            if not team_id:
                return []
            # balldontlie v1: /players?team_ids[]=ID&per_page=100
            data = _http_get_json(f"{BALLDONTLIE_API_BASE}/players", params={"team_ids[]": team_id, "per_page": 100})
            return data.get("data", []) if isinstance(data, dict) else []
        def _season_avgs(pids: list[int]) -> dict[int, dict]:
            # split into chunks of up to 25 ids (API limit-friendly)
            out: dict[int, dict] = {}
            for i in range(0, len(pids), 25):
                ids = pids[i:i+25]
                params = [("player_ids[]", pid) for pid in ids]
                # call with manual param list to allow repeated keys
                params_dict = {"per_page": 100}
                # Build query string manually via _http_get_json by giving params as dict won't repeat keys;
                # So build URL ourselves
                qs = "&".join([f"player_ids[]={pid}" for pid in ids])
                url = f"{BALLDONTLIE_API_BASE}/season_averages?{qs}"
                data = _http_get_json(url)
                for row in data.get("data", []) if isinstance(data, dict) else []:
                    out[int(row.get("player_id"))] = row
            return out
        home_players = _team_players(home_tid) if home_tid else []
        away_players = _team_players(away_tid) if away_tid else []
        # Collect player IDs and names
        players = []
        for p in home_players + away_players:
            pid = p.get("id")
            if not pid:
                continue
            full = f"{p.get('first_name','').strip()} {p.get('last_name','').strip()}".strip()
            players.append((pid, full))
        if not players:
            return []
        avgs = _season_avgs([pid for pid, _ in players])
        # Sort by minutes to pick prominent players
        top = []
        for pid, name in players:
            row = avgs.get(pid)
            if not row:
                continue
            top.append((float(row.get("min") or 0), pid, name, row))
        top.sort(reverse=True)
        top = top[:6]  # limit to ~6 players to keep payload lean
        if not top:
            return []
        mkts: list[APIMarket] = []
        need = set(wanted or ["player_points","player_rebounds","player_assists","player_threes"])
        def _mk_line(stat_key: str, rnd: float = 0.5) -> float | None:
            val = float(row.get(stat_key) or 0)
            if val <= 0:
                return None
            # round to nearest step
            step = 0.5
            return round(val / step) * step
        # build markets per stat, aggregating outcomes for each player at his line
        def _ensure_market(key: str) -> APIMarket:
            m = next((m for m in mkts if m.key == key), None)
            if m is None:
                m = APIMarket(key=key, outcomes=[])
                mkts.append(m)
            return m
        for _min, pid, name, row in top:
            mapping = [
                ("player_points", "pts"),
                ("player_rebounds", "reb"),
                ("player_assists", "ast"),
                ("player_threes", "fg3m"),
            ]
            for key, stat in mapping:
                if key not in need:
                    continue
                line = _mk_line(stat)
                if line is None:
                    continue
                m = _ensure_market(key)
                m.outcomes.extend([
                    APIOutcome(name="Over", price=-115, point=float(line), description=name, model_prob=0.5),
                    APIOutcome(name="Under", price=-115, point=float(line), description=name, model_prob=0.5),
                ])
        return mkts
    except Exception:
        return []


# ----------------- MLB, NFL, CFB, TENNIS free/model prop builders -----------------

def _model_mlb_props_for_event(ev: APIEvent, wanted: list[str] | None) -> list[APIMarket]:
    """Build MLB props using free MLB StatsAPI roster names (no key). Lines are heuristic.
    Markets: batter_total_bases, batter_hits, pitcher_strikeouts, pitcher_outs.
    """
    try:
        need = set(wanted or [
            "batter_total_bases","batter_hits","pitcher_strikeouts","pitcher_outs"
        ])
        # Map team names -> MLB team id using helper (present elsewhere in file)
        home_tid = _mlb_team_id_by_name(ev.home_team)
        away_tid = _mlb_team_id_by_name(ev.away_team)
        rng = _sandbox_rng(ev.id + "mlb")
        names_hitters: list[str] = []
        names_pitchers: list[str] = []
        def _roster_names(tid: int) -> tuple[list[str], list[str]]:
            if not tid:
                return [], []
            data = _http_get_json(f"{MLB_API_BASE}/teams/{tid}/roster", params={"rosterType":"active"})
            hitters, pitchers = [], []
            for p in (data.get("roster") or []):
                full = ((p.get("person") or {}).get("fullName") or p.get("name") or "").strip()
                pos = ((p.get("position") or {}).get("abbreviation") or "").upper()
                if not full:
                    continue
                if pos in ("P", "SP", "RP"):
                    pitchers.append(full)
                else:
                    hitters.append(full)
            return hitters, pitchers
        try:
            h_hit, h_pit = _roster_names(home_tid)
            a_hit, a_pit = _roster_names(away_tid)
            names_hitters = (h_hit[:4] + a_hit[:4]) or []
            names_pitchers = (h_pit[:2] + a_pit[:2]) or []
        except Exception:
            # fallback names synthesized from team labels
            base_h = [f"{(ev.home_team or 'Home').split()[0]} Batter {i}" for i in range(1,5)]
            base_a = [f"{(ev.away_team or 'Away').split()[0]} Batter {i}" for i in range(1,5)]
            names_hitters = base_h + base_a
            names_pitchers = [f"{(ev.home_team or 'Home').split()[0]} SP", f"{(ev.away_team or 'Away').split()[0]} SP"]
        mkts: list[APIMarket] = []
        def _ensure(key: str) -> APIMarket:
            m = next((m for m in mkts if m.key == key), None)
            if m is None:
                m = APIMarket(key=key, outcomes=[])
                mkts.append(m)
            return m
        # Hitters
        for n in names_hitters:
            if "batter_total_bases" in need:
                line = round(rng.uniform(0.5, 2.5) / 0.5) * 0.5
                _ensure("batter_total_bases").outcomes.extend([
                    APIOutcome(name="Over", price=-115, point=line, description=n, model_prob=0.5),
                    APIOutcome(name="Under", price=-115, point=line, description=n, model_prob=0.5),
                ])
            if "batter_hits" in need:
                line = round(rng.uniform(0.5, 1.5) / 0.5) * 0.5
                _ensure("batter_hits").outcomes.extend([
                    APIOutcome(name="Over", price=-115, point=line, description=n, model_prob=0.5),
                    APIOutcome(name="Under", price=-115, point=line, description=n, model_prob=0.5),
                ])
        # Pitchers
        for n in names_pitchers:
            if "pitcher_strikeouts" in need:
                line = round(rng.uniform(3.5, 7.5) / 0.5) * 0.5
                _ensure("pitcher_strikeouts").outcomes.extend([
                    APIOutcome(name="Over", price=-115, point=line, description=n, model_prob=0.5),
                    APIOutcome(name="Under", price=-115, point=line, description=n, model_prob=0.5),
                ])
            if "pitcher_outs" in need:
                line = round(rng.uniform(15.5, 18.5) / 0.5) * 0.5
                _ensure("pitcher_outs").outcomes.extend([
                    APIOutcome(name="Over", price=-115, point=line, description=n, model_prob=0.5),
                    APIOutcome(name="Under", price=-115, point=line, description=n, model_prob=0.5),
                ])
        return mkts
    except Exception:
        return []


def _model_nfl_props_for_event(ev: APIEvent, wanted: list[str] | None) -> list[APIMarket]:
    """Build NFL props (free, heuristic) using role-based generators.
    Markets: player_pass_yds, player_rush_yds, player_receiving_yds, player_receptions.
    """
    rng = _sandbox_rng(ev.id + "nfl")
    need = set(wanted or ["player_pass_yards","player_rush_yards","player_receiving_yards","player_receptions"])
    # Simple role placeholders if we can't resolve real rosters for free reliably
    qb = [f"{(ev.home_team or 'Home').split()[0]} QB", f"{(ev.away_team or 'Away').split()[0]} QB"]
    rb = [f"{(ev.home_team or 'Home').split()[0]} RB1", f"{(ev.away_team or 'Away').split()[0]} RB1"]
    wr = [f"{(ev.home_team or 'Home').split()[0]} WR1", f"{(ev.away_team or 'Away').split()[0]} WR1"]
    mkts: list[APIMarket] = []
    def _ensure(key: str) -> APIMarket:
        m = next((m for m in mkts if m.key == key), None)
        if m is None:
            m = APIMarket(key=key, outcomes=[])
            mkts.append(m)
        return m
    for n in qb:
        if "player_pass_yards" in need:
            line = round(rng.uniform(210, 285) / 5) * 5
            _ensure("player_pass_yards").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
    for n in rb:
        if "player_rush_yards" in need:
            line = round(rng.uniform(45, 95) / 0.5) * 0.5
            _ensure("player_rush_yards").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
    for n in wr:
        if "player_receiving_yards" in need:
            line = round(rng.uniform(45, 95) / 0.5) * 0.5
            _ensure("player_receiving_yards").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
        if "player_receptions" in need:
            line = round(rng.uniform(3.5, 7.5) / 0.5) * 0.5
            _ensure("player_receptions").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
    return mkts


def _model_cfb_props_for_event(ev: APIEvent, wanted: list[str] | None) -> list[APIMarket]:
    """Build CFB props (free heuristic). Uses role-based names.
    Markets: player_pass_yds, player_rush_yds, player_receiving_yds, player_receptions.
    """
    rng = _sandbox_rng(ev.id + "cfb")
    need = set(wanted or ["player_pass_yards","player_rush_yards","player_receiving_yards","player_receptions"])
    qb = [f"{(ev.home_team or 'Home').split()[0]} QB", f"{(ev.away_team or 'Away').split()[0]} QB"]
    rb = [f"{(ev.home_team or 'Home').split()[0]} RB1", f"{(ev.away_team or 'Away').split()[0]} RB1"]
    wr = [f"{(ev.home_team or 'Home').split()[0]} WR1", f"{(ev.away_team or 'Away').split()[0]} WR1"]
    mkts: list[APIMarket] = []
    def _ensure(key: str) -> APIMarket:
        m = next((m for m in mkts if m.key == key), None)
        if m is None:
            m = APIMarket(key=key, outcomes=[])
            mkts.append(m)
        return m
    for n in qb:
        if "player_pass_yards" in need:
            line = round(rng.uniform(185, 275) / 5) * 5
            _ensure("player_pass_yards").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
    for n in rb:
        if "player_rush_yards" in need:
            line = round(rng.uniform(55, 125) / 0.5) * 0.5
            _ensure("player_rush_yards").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
    for n in wr:
        if "player_receiving_yards" in need:
            line = round(rng.uniform(55, 115) / 0.5) * 0.5
            _ensure("player_receiving_yards").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
        if "player_receptions" in need:
            line = round(rng.uniform(3.5, 8.5) / 0.5) * 0.5
            _ensure("player_receptions").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
    return mkts


def _model_tennis_props_for_event(ev: APIEvent, wanted: list[str] | None) -> list[APIMarket]:
    """Build Tennis props (free heuristic) for ATP/WTA-style events. Names come from event teams.
    Markets: player_aces, player_double_faults, player_total_games_won.
    """
    rng = _sandbox_rng(ev.id + "tennis")
    need = set(wanted or ["player_aces","player_double_faults","player_total_games_won"])
    names = [ev.home_team or "Player A", ev.away_team or "Player B"]
    mkts: list[APIMarket] = []
    def _ensure(key: str) -> APIMarket:
        m = next((m for m in mkts if m.key == key), None)
        if m is None:
            m = APIMarket(key=key, outcomes=[])
            mkts.append(m)
        return m
    for n in names:
        if "player_aces" in need:
            line = round(rng.uniform(3.5, 12.5) / 0.5) * 0.5
            _ensure("player_aces").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
        if "player_double_faults" in need:
            line = round(rng.uniform(1.5, 4.5) / 0.5) * 0.5
            _ensure("player_double_faults").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
        if "player_total_games_won" in need:
            line = round(rng.uniform(9.5, 16.5) / 0.5) * 0.5
            _ensure("player_total_games_won").outcomes.extend([
                APIOutcome(name="Over", price=-115, point=float(line), description=n, model_prob=0.5),
                APIOutcome(name="Under", price=-115, point=float(line), description=n, model_prob=0.5),
            ])
    return mkts


def _model_props_for_event(ev: APIEvent, sport_key: str, wanted: list[str] | None) -> list[APIMarket]:
    """Dispatch to per-league free/model props builders."""
    if sport_key == "basketball_nba":
        return _model_nba_props_for_event(ev, wanted)
    if sport_key == "baseball_mlb":
        return _model_mlb_props_for_event(ev, wanted)
    if sport_key in ("americanfootball_nfl",):
        return _model_nfl_props_for_event(ev, wanted)
    if sport_key in ("americanfootball_ncaaf",):
        return _model_cfb_props_for_event(ev, wanted)
    if sport_key.startswith("tennis_") or sport_key == "tennis":
        return _model_tennis_props_for_event(ev, wanted)
    return []
def _sandbox_rng(seed_text: str) -> random.Random:
    h = hashlib.md5(seed_text.encode()).hexdigest()
    return random.Random(int(h[:8], 16))

def _mk_ou_market(key: str, who: str, line: float, price: int = -115) -> APIMarket:
    return APIMarket(
        key=key,
        outcomes=[
            APIOutcome(name="Over", price=price, point=round(line, 1), description=who, model_prob=0.5),
            APIOutcome(name="Under", price=price, point=round(line, 1), description=who, model_prob=0.5),
        ],
    )

def _sandbox_player_markets_for_sport(ev: APIEvent, sport_key: str, wanted: list[str]) -> list[APIMarket]:
    rng = _sandbox_rng(ev.id + sport_key)
    mkts: list[APIMarket] = []
    home_names = [f"{(ev.home_team or 'Home').split()[0]} Star {i}" for i in range(1, 3)]
    away_names = [f"{(ev.away_team or 'Away').split()[0]} Star {i}" for i in range(1, 3)]
    names = home_names + away_names
    if sport_key == "basketball_nba":
        base = {
            "player_points": (18, 35),
            "player_rebounds": (4, 12),
            "player_assists": (3, 10),
            "player_threes": (1, 5),
        }
    elif sport_key in ("americanfootball_nfl", "americanfootball_ncaaf"):
        # Use OddsAPI-compliant keys
        base = {
            "player_pass_yards": (190, 320),
            "player_rush_yards": (35, 95),
            "player_receptions": (3, 8),
            "player_receiving_yards": (30, 95),
        }
    elif sport_key == "baseball_mlb":
        # Use batter_* and pitcher_* keys to match OddsAPI
        base = {
            "batter_total_bases": (0.5, 2.5),
            "batter_hits": (0.5, 1.5),
            "pitcher_strikeouts": (3.5, 7.5),
            "pitcher_outs": (15.5, 18.5),
        }
    elif sport_key == "icehockey_nhl":
        base = {
            "player_shots_on_goal": (1.5, 3.5),
            "player_points": (0.5, 1.5),
            "player_goals": (0.5, 1.5),
        }
    else:
        base = {}
    allowed = set(wanted) if wanted else set(base.keys())
    for mk, (lo, hi) in base.items():
        if mk not in allowed:
            continue
        for n in names:
            line = rng.uniform(lo, hi)
            mkts.append(_mk_ou_market(mk, n, line))
    return mkts

def _sandbox_team_markets(ev: APIEvent) -> list[APIMarket]:
    rng = _sandbox_rng(ev.id)
    price_home = -120 if rng.random() > 0.5 else 110
    price_away = 100 if price_home == -120 else -105
    h2h = APIMarket(
        key="h2h",
        outcomes=[
            APIOutcome(name=ev.home_team, price=price_home),
            APIOutcome(name=ev.away_team, price=price_away),
        ],
    )
    spread = round(rng.uniform(-6, 6), 1)
    sp = APIMarket(
        key="spreads",
        outcomes=[
            APIOutcome(name=ev.home_team, price=-110, point=spread),
            APIOutcome(name=ev.away_team, price=-110, point=-spread),
        ],
    )
    total = round(rng.uniform(190, 235), 1)
    tot = APIMarket(
        key="totals",
        outcomes=[
            APIOutcome(name="Over", price=-110, point=total),
            APIOutcome(name="Under", price=-110, point=total),
        ],
    )
    return [h2h, sp, tot]

def _sandbox_events_for_sport(sport_key: str) -> list[APIEvent]:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    pairs = {
        "basketball_nba": [("Lakers", "Warriors"), ("Celtics", "Heat")],
        "americanfootball_nfl": [("Chiefs", "Ravens"), ("Bills", "Jets")],
        "americanfootball_ncaaf": [("Georgia", "Alabama"), ("Ohio State", "Michigan")],
        "baseball_mlb": [("Yankees", "Red Sox"), ("Dodgers", "Giants")],
        "icehockey_nhl": [("Maple Leafs", "Canadiens"), ("Rangers", "Bruins")],
    }.get(sport_key, [("Home", "Away"), ("East", "West")])
    evs: list[APIEvent] = []
    for idx, (home, away) in enumerate(pairs):
        raw_id = f"{sport_key}|{home}|{away}|{now.date()}|{idx}"
        evs.append(APIEvent(
            id=_stable_id(raw_id),
            sport_key=sport_key,
            sport_title=HUMAN_TITLES.get(sport_key, sport_key),
            commence_time=(now + timedelta(hours=2 + idx)).isoformat().replace("+00:00","Z"),
            home_team=home, away_team=away, bookmakers=[]
        ))
    # Prefer real ESPN events if available for this sport (still free)
    espn_evs = _espn_upcoming_events(sport_key)
    if espn_evs:
        return espn_evs
    return evs



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
    if not market:
        return TEAM_MARKETS
    if market.lower() == "all":
        # Protect your quota by default: only team markets unless explicitly enabled
        return ALL_MARKETS if ODDS_PULL_PLAYER_ON_ODDS else TEAM_MARKETS
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
    bookmakers: Optional[List[str]] = None,
    allow_props: bool = False,
    max_prop_events: int = 0,
) -> Tuple[List[APIEvent], Dict[str, str]]:
    """
    Calls The Odds API and normalizes results into your Swift JSON shape.
    Returns (events, upstream_headers) so we can forward x-requests-remaining.

    NEW (fix): Player props must be fetched per-event using
    `/sports/{sport}/events/{eventId}/odds`. Team markets (h2h/spreads/totals)
    still come from `/sports/{sport}/odds`. We split the request and merge.
    """
    if not ODDS_API_KEY:
        raise HTTPException(status_code=401, detail="Server missing ODDS_API_KEY")

    # 1) Filter the requested markets for this sport, then split into team vs player buckets.
    filtered_markets = _filter_markets_for_sport(markets_csv, sport_key)
    requested = [m.strip() for m in (filtered_markets or "").split(",") if m.strip()]
    team_set = {"h2h", "spreads", "totals"}
    team_markets = [m for m in requested if m in team_set]
    player_markets = [m for m in requested if m not in team_set]

    # Always ensure we at least request team markets so we get the events list/ids
    if not team_markets:
        team_markets = ["h2h", "spreads", "totals"]

    # Shared params
    books_csv = ",".join(bookmakers or DEFAULT_BOOKMAKERS)
    common = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "oddsFormat": "american",
        "bookmakers": books_csv,
    }

    def _normalize_events(raw: list) -> List[APIEvent]:
        evs: List[APIEvent] = []
        for e in raw or []:
            bks: List[APIBookmaker] = []
            for b in e.get("bookmakers", []) or []:
                mkts: List[APIMarket] = []
                for m in b.get("markets", []) or []:
                    outs: List[APIOutcome] = []
                    for o in m.get("outcomes", []) or []:
                        _name = str(o.get("name", "")).strip()
                        _desc = (
                            o.get("description")
                            or o.get("player")
                            or o.get("playerName")
                        )
                        if not _desc and _name.lower() in ("over", "under"):
                            for k in (
                                "participant", "runner", "competitor", "team",
                                "participant_name", "athlete", "entity", "label", "selection",
                            ):
                                val = o.get(k)
                                if val:
                                    _desc = str(val).strip()
                                    break
                        if not _desc:
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
                    # group de‑vig (same logic as before)
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
                    mkts.append(APIMarket(key=str(m.get("key", "")), outcomes=outs))
                bks.append(APIBookmaker(
                    key=str(b.get("key", "")),
                    title=(b.get("title") or b.get("key") or "").strip(),
                    last_update=str(b.get("last_update") or ""),
                    markets=mkts,
                ))
            evs.append(APIEvent(
                id=_event_fallback_id(e),
                sport_key=sport_key,
                sport_title=HUMAN_TITLES.get(sport_key, str(e.get("sport_title") or "")),
                commence_time=str(e.get("commence_time") or ""),
                home_team=str(e.get("home_team") or ""),
                away_team=str(e.get("away_team") or ""),
                bookmakers=bks,
            ))
        return evs

    merged_events: List[APIEvent] = []
    last_headers: Dict[str, str] = {}

    # 2) Fetch TEAM markets from /sports/{sport}/odds (chunked)
    team_chunks = _split_markets(",".join(team_markets), chunk_size=3)
    base_url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"

    async def _one_team_chunk(markets: str, *, no_bookmakers: bool = False) -> Tuple[List[APIEvent], Dict[str, str]]:
        params = {**common, "markets": markets}
        if no_bookmakers:
            params.pop("bookmakers", None)
        r = await http.get(base_url, params=params, timeout=30)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return _normalize_events(r.json()), {k.lower(): v for k, v in r.headers.items()}

    success = 0
    for mk_chunk in team_chunks:
        try:
            evs, hdrs = await _one_team_chunk(mk_chunk)
            merged_events = _merge_events(merged_events, evs)
            last_headers = hdrs
            success += 1
        except HTTPException as e:
            if e.status_code in (400, 404, 422):
                try:
                    evs, hdrs = await _one_team_chunk(mk_chunk, no_bookmakers=True)
                    merged_events = _merge_events(merged_events, evs)
                    last_headers = hdrs
                    success += 1
                except HTTPException as e2:
                    print(f"[fetch_odds_events] Skipping unsupported team markets chunk='{mk_chunk}' status={e2.status_code}")
                continue
            raise

    if success == 0:
        # as a last resort, try plain team trio
        evs, hdrs = await _one_team_chunk("h2h,spreads,totals")
        merged_events = _merge_events(merged_events, evs)
        last_headers = hdrs

    # 3) If player markets requested, fetch per-event using /events/{id}/odds.

    # Sandbox synth: if props were requested but upstream pull is disabled, synthesize
    if FREE_SANDBOX and player_markets and merged_events and not allow_props:
        for ev in merged_events:
            synth = _sandbox_player_markets_for_sport(ev, sport_key, player_markets)
            if synth:
                if ev.bookmakers:
                    ev.bookmakers[0] = APIBookmaker(
                        key=ev.bookmakers[0].key or "sandbox",
                        title=ev.bookmakers[0].title or "Sandbox",
                        last_update=ev.bookmakers[0].last_update,
                        markets=_merge_market_lists(ev.bookmakers[0].markets, synth),
                    )
                else:
                    ev.bookmakers.append(
                        APIBookmaker(key="sandbox", title="Sandbox", last_update="", markets=synth)
                    )
    if allow_props and player_markets and merged_events:
        # Respect max_prop_events to avoid exploding request counts
        events_for_props = merged_events[:max(0, int(max_prop_events))] if max_prop_events > 0 else []
        per_event_url_tpl = f"{ODDS_API_BASE}/sports/{sport_key}/events/{{eid}}/odds"
        player_chunks = _split_markets(",".join(player_markets), chunk_size=3)

        async def _one_event_chunk(event_id: str, markets: str, *, no_bookmakers: bool = False) -> List[APIMarket]:
            params = {**common, "markets": markets}
            if no_bookmakers:
                params.pop("bookmakers", None)
            r = await http.get(per_event_url_tpl.format(eid=urllib.parse.quote(event_id)), params=params, timeout=30)
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.text)
            # Response shape for event odds is usually an array with a single event
            raw = r.json()
            raw_event = raw[0] if isinstance(raw, list) and raw else raw
            mkts: List[APIMarket] = []
            for b in (raw_event.get("bookmakers") or []):
                for m in (b.get("markets") or []):
                    outs: List[APIOutcome] = []
                    for o in (m.get("outcomes") or []):
                        _name = str(o.get("name", "")).strip()
                        _desc = o.get("description") or o.get("player") or o.get("playerName") or ""
                        try:
                            _price = float(o.get("price", 0) or 0)
                        except Exception:
                            _price = 0.0
                        outs.append(APIOutcome(name=_name, price=_price, point=o.get("point"), description=_desc, model_prob=None))
                    mkts.append(APIMarket(key=str(m.get("key", "")), outcomes=outs))
            return mkts

        # For each event, pull player markets and merge into that event's bookmakers
        for ev in list(events_for_props):
            player_merged_by_book: Dict[str, List[APIMarket]] = {}
            for mk_chunk in player_chunks:
                try:
                    mkts = await _one_event_chunk(ev.id, mk_chunk)
                except HTTPException as e:
                    if e.status_code in (400, 404, 422):
                        # try without limiting bookmakers
                        try:
                            mkts = await _one_event_chunk(ev.id, mk_chunk, no_bookmakers=True)
                        except HTTPException as e2:
                            print(f"[fetch_odds_events] Skipping unsupported player chunk='{mk_chunk}' for event='{ev.id}' status={e2.status_code}")
                            continue
                    else:
                        raise
                # Merge by bookmaker key. Build a synthetic bookmaker if none exists yet.
                # First, bucket new markets by bookmaker from upstream response structure
                # (we lost bookmaker key in the simplified parser above), so instead merge into
                # a placeholder bookmaker "aggregated" to ensure UI still sees player markets.
                player_merged_by_book.setdefault("aggregated", [])
                player_merged_by_book["aggregated"] = _merge_market_lists(player_merged_by_book["aggregated"], mkts)

            # Attach/merge into event's bookmakers
            if not player_merged_by_book and FREE_SANDBOX:
                synth = _sandbox_player_markets_for_sport(ev, sport_key, player_markets)
                if synth:
                    player_merged_by_book["aggregated"] = synth

            if player_merged_by_book:
                agg_markets = player_merged_by_book["aggregated"]
                # if there is already a bookmaker, append player markets to the first one; otherwise create one
                if ev.bookmakers:
                    ev.bookmakers[0] = APIBookmaker(
                        key=ev.bookmakers[0].key or "agg",
                        title=ev.bookmakers[0].title or "Aggregated",
                        last_update=ev.bookmakers[0].last_update,
                        markets=_merge_market_lists(ev.bookmakers[0].markets, agg_markets),
                    )
                else:
                    ev.bookmakers.append(
                        APIBookmaker(key="agg", title="Aggregated", last_update="", markets=agg_markets)
                    )

    # persist a compressed snapshot for reuse
    try:
        _odds_cache_put(sport_key, "upcoming", filtered_markets, ODDS_REGIONS, books_csv, merged_events)
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
    """Safe stub for NFL recent-form correlation. Returns no signals for now.
    This prevents indentation/import errors during deploy until the full adapter
    is implemented. The rest of the pipeline assumes adapters may return [].
    """
    if ev.sport_key.lower() not in ("americanfootball_nfl", "nfl"):
        return []
    return []
