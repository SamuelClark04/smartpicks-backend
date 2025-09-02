# main.py
import os, time, random, hashlib, requests
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

APP_NAME = "smartpicks-backend"
# ---------- Logging ----------
import logging
logger = logging.getLogger(APP_NAME if 'APP_NAME' in globals() else 'smartpicks')
if not logger.handlers:
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))

# ---------- Simple counters (/metrics) ----------
METRICS = {
    "provider_calls_total": 0,
    "provider_errors_total": 0,
    "cache_hits_total": 0,
    "cache_misses_total": 0,
    "cache_stale_total": 0,
    "signals_built_total": 0,
    "pairs_returned_total": 0,
}

from fastapi import Response

# ---------- HTTP util (tiny retry/backoff wrapper) ----------
from random import random as _rand

def http_get(url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 10, retries: int = 2):
    """Lightweight GET with ≤2 retries on network/5xx; jittered backoff.
    Returns `requests.Response` or raises the last exception.
    """
    attempt = 0
    last_exc = None
    while attempt <= retries:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code >= 500:
                raise RuntimeError(f"server {r.status_code}")
            return r
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            # jittered backoff: 0.35s, 0.7s, 1.4s ...
            sleep_s = (0.35 * (2 ** attempt)) * (1.0 + 0.25 * _rand())
            try:
                time.sleep(sleep_s)
            except Exception:
                pass
            attempt += 1
    # Exhausted retries
    if isinstance(last_exc, Exception):
        raise last_exc
    raise RuntimeError("http_get failed without explicit exception")

# Provider keys (no fake mode)
THEODDSAPI_KEY = os.getenv("THEODDSAPI_KEY", "")
SPORTSDATA_IO_KEY = os.getenv("SPORTSDATA_IO_KEY", "")  # optional (future props provider)
# Optional: allow your iOS app / local dev
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- /metrics endpoint (Prometheus-style plaintext) ----------
@app.get("/metrics")
def metrics():
    # Prometheus-style plaintext
    lines = [
        "# TYPE smartpicks_provider_calls_total counter",
        f"smartpicks_provider_calls_total {METRICS['provider_calls_total']}",
        "# TYPE smartpicks_provider_errors_total counter",
        f"smartpicks_provider_errors_total {METRICS['provider_errors_total']}",
        "# TYPE smartpicks_cache_hits_total counter",
        f"smartpicks_cache_hits_total {METRICS['cache_hits_total']}",
        "# TYPE smartpicks_cache_misses_total counter",
        f"smartpicks_cache_misses_total {METRICS['cache_misses_total']}",
        "# TYPE smartpicks_cache_stale_total counter",
        f"smartpicks_cache_stale_total {METRICS['cache_stale_total']}",
        "# TYPE smartpicks_signals_built_total counter",
        f"smartpicks_signals_built_total {METRICS['signals_built_total']}",
        "# TYPE smartpicks_pairs_returned_total counter",
        f"smartpicks_pairs_returned_total {METRICS['pairs_returned_total']}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")

# ---------- Models (match iOS) ----------
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
    last_update: str
    markets: List[APIMarket]

class APIEvent(BaseModel):
    id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[APIBookmaker]

# Signals expected by iOS

class CorrelationSignal(BaseModel):
    id: str
    kind: str            # matchup_trend | head_to_head | team_style | pace_injury | news
    eventId: str
    players: List[str]   # 0,1,2
    teams: List[str]
    markets: List[str]   # [] means applies to any
    boost: float         # 0..0.35 normally
    reason: str

# --- Pair models for /api/pairs ---
class PairLeg(BaseModel):
    player: Optional[str] = None
    team: Optional[str] = None
    market: str
    line: Optional[float] = None

class PairSuggestion(BaseModel):
    eventId: str
    score: float
    reason: str
    legs: List[PairLeg]

# ---------- Helpers ----------
FANDUEL_KEYS = {"fanduel", "fan-duel", "fan_duel"}

def fanduel_first(bookmakers: List[APIBookmaker]) -> List[APIBookmaker]:
    return sorted(bookmakers, key=lambda b: (b.key.lower() not in FANDUEL_KEYS, b.title))

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def merge_events(events_lists: List[List[APIEvent]]) -> List[APIEvent]:
    by_id: Dict[str, APIEvent] = {}
    for events in events_lists:
        for ev in events:
            if ev.id not in by_id:
                by_id[ev.id] = ev
            else:
                base = by_id[ev.id]
                # merge bookmakers by key
                merged: Dict[str, APIBookmaker] = {bk.key: bk for bk in base.bookmakers}
                for bk in ev.bookmakers:
                    if bk.key in merged:
                        # merge markets by key (prefer more outcomes)
                        mk_map = {m.key: m for m in merged[bk.key].markets}
                        for m in bk.markets:
                            if m.key in mk_map:
                                if len(m.outcomes) > len(mk_map[m.key].outcomes):
                                    mk_map[m.key] = m
                            else:
                                mk_map[m.key] = m
                        merged[bk.key] = APIBookmaker(
                            key=bk.key,
                            title=bk.title,
                            last_update=max(bk.last_update, merged[bk.key].last_update),
                            markets=list(mk_map.values())
                        )
                    else:
                        merged[bk.key] = bk
                by_id[ev.id] = APIEvent(
                    **{**base.dict(), "bookmakers": fanduel_first(list(merged.values()))}
                )
    # sort by commence, then title
    return sorted(by_id.values(), key=lambda e: (e.commence_time, e.sport_title))

# ---------- Provider fetch (real) ----------
THEODDS_BASE = "https://api.the-odds-api.com/v4/sports"

def theodds_markets_for(league: str, market: str) -> List[str]:
    """
    Expand logical market groups into provider markets.
    """
    m = market.lower()
    if m == "all":
        return ["h2h", "spreads", "totals"]  # props are handled by explicit 'player_*' lists per league
    if m == "player_props":
        if league == "nba":
            return ["player_points", "player_assists", "player_rebounds", "player_threes"]
        if league == "nfl":
            return ["player_pass_yards", "player_rush_yards", "player_receiving_yards",
                    "player_receptions", "player_pass_tds", "player_rush_tds", "player_receiving_tds"]
        if league == "mlb":
            return ["player_total_bases", "player_hits", "player_home_runs", "player_strikeouts"]
        if league == "nhl":
            return ["player_points", "player_goals", "player_shots_on_goal"]
        return []
    return [m]

def fetch_from_provider(league: str, markets: List[str], date_str: Optional[str]) -> List[APIEvent]:
    """
    Fetch odds from The Odds API and map to APIEvent.
    Requires THEODDSAPI_KEY. If not configured, raise HTTPException 503.
    """
    if not THEODDSAPI_KEY:
        raise HTTPException(status_code=503, detail="Backend not configured: set THEODDSAPI_KEY")

    sport_key = {
        "nba": "basketball_nba",
        "nfl": "americanfootball_nfl",
        "ncaa_football": "americanfootball_ncaaf",
        "mlb": "baseball_mlb",
        "nhl": "icehockey_nhl",
        "atp": "tennis_atp",
        "wta": "tennis_wta",
    }.get(league, league)

    events_parts: List[List[APIEvent]] = []

    # Build a deduped list of provider markets to query
    provider_markets: List[str] = []
    for m in markets:
        provider_markets.extend(theodds_markets_for(league, m))
    provider_markets = sorted(set(provider_markets)) if provider_markets else markets

    # If nothing expanded, use the original list (for h2h/spreads/totals/etc.)
    if not provider_markets:
        provider_markets = markets

    regions = os.getenv("THEODDSAPI_REGIONS", "us")
    odds_format = os.getenv("THEODDSAPI_FORMAT", "american")
    date_param = f"&dateFormat=iso" + (f"&commenceTimeFrom={date_str}T00:00:00Z" if date_str else "")

    for chunk_start in range(0, len(provider_markets), 3):
        # Call in small chunks to respect URL length/rate limits
        chunk = provider_markets[chunk_start:chunk_start+3]
        markets_param = ",".join(chunk)

        url = f"{THEODDS_BASE}/{sport_key}/odds?regions={regions}&markets={markets_param}&oddsFormat={odds_format}&apiKey={THEODDSAPI_KEY}&dateFormat=iso"
        t0 = _now()
        try:
            resp = http_get(url, timeout=12)
        except Exception as e:
            try:
                METRICS["provider_errors_total"] += 1
            except Exception:
                pass
            logger.warning(f"odds.fetch error league={league} mkts={markets_param}: {e}")
            raise HTTPException(status_code=502, detail=f"Provider error: {e}")
        dt = int((__import__('time').time() - t0) * 1000)
        if resp.status_code != 200:
            try:
                METRICS["provider_errors_total"] += 1
            except Exception:
                pass
            logger.warning(f"odds.fetch bad_status league={league} mkts={markets_param} status={resp.status_code}")
            raise HTTPException(status_code=resp.status_code, detail=f"Provider response {resp.status_code}: {resp.text[:240]}")
        try:
            METRICS["provider_calls_total"] += 1
        except Exception:
            pass
        data = resp.json()
        # Map provider JSON to our pydantic models directly (shapes match closely)
        try:
            batch = [APIEvent(**ev) for ev in data]
        except Exception as e:
            logger.error(f"odds.map error league={league} mkts={markets_param}: {e}")
            raise HTTPException(status_code=500, detail=f"Mapping error: {e}")
        logger.info(f"odds.fetch ok league={league} mkts={markets_param} ms={dt} events={len(batch)}")
        events_parts.append(batch)

    return merge_events(events_parts)

# ---------- Correlation rules (domain pairs) ----------
CORRELATION_RULES: Dict[str, List[Tuple[str, float, str]]] = {
    # (same dictionary you provided; trimmed for brevity in this snippet)
    "player_pass_yards": [
        ("player_receiving_yards", 0.18, "QB ⇄ WR/TE passing-to-receiving yardage"),
        ("player_receptions",      0.16, "QB volume ↔ receiver catches"),
        ("player_pass_completions",0.12, "More completions → more yards"),
        ("player_pass_attempts",   0.08, "Attempts ↔ passing volume"),
        ("player_pass_tds",        0.06, "Yardage can lead to TDs")
    ],
    # --- NBA same-player correlations ---
    "player_points": [
        ("player_assists", 0.14, "Scoring nights often come with playmaking"),
        ("player_threes",  0.12, "High-usage scorers generate 3PA/3PM"),
        ("player_rebounds",0.08, "Second-chance points boost totals"),
    ],
    "player_assists": [
        ("player_points", 0.16, "Playmaker ↔ scorer synergy"),
        ("player_threes", 0.10, "Drive-and-kick → 3PM"),
    ],
    "player_rebounds": [
        ("player_points", 0.07, "Putbacks/second chances"),
        ("player_assists",0.05, "Rebound → outlet/kick-ahead assists"),
    ],
    "player_threes": [
        ("player_assists", 0.12, "Catch-and-shoot fed by creators"),
        ("player_points",  0.09, "3PM contribute to scoring props"),
    ],
    # --- NFL same-player correlations ---
    "player_receiving_yards": [
        ("player_receptions",    0.22, "Targets drive receiving yards"),
        ("player_receiving_tds", 0.12, "Air yards translate to TDs"),
    ],
    "player_receptions": [
        ("player_receiving_yards", 0.20, "Catch volume ↔ receiving yards"),
        ("player_receiving_tds",   0.08, "Red-zone involvement follows volume"),
    ],
    "player_rush_yards": [
        ("player_rush_tds", 0.10, "Ground volume → TD chances"),
    ],
    "player_rush_tds": [
        ("player_rush_yards", 0.10, "Rushing success → TD conversion"),
    ],
    "player_receiving_tds": [
        ("player_receiving_yards", 0.12, "Downfield usage → TD equity"),
        ("player_receptions",      0.10, "High targets → scoring ops"),
    ],
    "player_pass_tds": [
        ("player_pass_yards",      0.08, "Yardage drives TD potential"),
        ("player_receiving_tds",   0.06, "QB TDs pair with WR/TE TDs"),
    ],
    # --- MLB same-player correlations ---
    "player_total_bases": [
        ("player_hits",       0.18, "Hits drive total bases"),
        ("player_home_runs",  0.10, "Home runs are 4 TB by definition"),
    ],
    "player_hits": [
        ("player_total_bases", 0.16, "More hits → more total bases"),
    ],
    "player_home_runs": [
        ("player_total_bases", 0.12, "HRs inflate total bases"),
    ],
    # --- NHL same-player correlations ---
    "player_goals": [
        ("player_points",        0.14, "Goal scoring ↔ point totals"),
        ("player_shots_on_goal", 0.12, "More shots → more goals probability"),
    ],
    "player_shots_on_goal": [
        ("player_goals",  0.12, "Shot volume → goal likelihood"),
        ("player_points", 0.09,  "Shot volume → points via goals/assists"),
    ],
}

# Lightweight cross-player pairing rules (synergies across different players in same event)
CROSS_PLAYER_RULES: Dict[str, List[Tuple[str, float, str]]] = {
    # NBA
    "player_assists": [
        ("player_points", 0.08, "Playmaker → scorer synergy"),
        ("player_threes", 0.07, "Drive-and-kick → 3PM synergy")
    ],
    "player_points": [
        ("player_assists", 0.05, "Scorer usage ↔ teammate assists"),
        ("player_threes", 0.05, "High-usage scorers create 3PA")
    ],
    "player_rebounds": [
        ("player_points", 0.04, "Second-chance points from rebounds")
    ],
    # NFL
    "player_pass_yards": [("player_receiving_yards", 0.12, "QB → WR receiving yards linkage")],
    "player_rush_yards": [("player_rush_attempts", 0.08, "Ground game volume linkage")],
    "player_receptions": [("player_receiving_yards", 0.10, "Targets → yards")],
    # MLB (keep modest; cross-team noise is high)
    "player_total_bases": [("player_hits", 0.06, "Lineup rallies correlate hitter production")],
    # We avoid negative pairings here to keep scores within 0..1 for suggestions
}

def build_signals_for_event(ev: APIEvent) -> List[CorrelationSignal]:
    """
    Heuristic signals: domain pair hints + same-game synergies + light pace/injury proxy.
    Replace/extend with real news/injuries/head-to-head integrations later.
    """
    sigs: List[CorrelationSignal] = []
    eid = ev.id

    # 1) If many prop markets available -> slight pace/injury tailwind
    market_keys = {m.key for bk in ev.bookmakers for m in bk.markets}
    if len([k for k in market_keys if k.startswith("player_")]) >= 8:
        sigs.append(CorrelationSignal(
            id=hashlib.md5((eid+"-pace").encode()).hexdigest(),
            kind="pace_injury", eventId=eid, players=[], teams=[],
            markets=[], boost=0.04, reason="Game-wide tailwind from pace/injuries"
        ))

        # 1b) High Total tailwind (NBA-style)
        total_points = None
        for bk in ev.bookmakers:
            for m in bk.markets:
                if m.key == "totals":
                    # take the first Over/Under point found
                    for o in m.outcomes:
                        if o.point is not None:
                            total_points = o.point
                            break
            if total_points is not None:
                break
        if total_points is not None and total_points >= 228:
            sigs.append(CorrelationSignal(
                id=hashlib.md5((eid+"-hi-total").encode()).hexdigest(),
                kind="pace_injury", eventId=eid, players=[], teams=[],
                markets=["player_points","player_assists","player_threes"], boost=0.04,
                reason=f"High total ({int(total_points)}) → pace/efficiency tailwind"
            ))

        # 1c) Favorite Moneyline → star usage bump (points)
        fav_team = None
        for bk in ev.bookmakers:
            for m in bk.markets:
                if m.key == "h2h":
                    # pick strongest favorite (most negative price)
                    fav = sorted(m.outcomes, key=lambda o: o.price)[0] if m.outcomes else None
                    if fav and fav.price <= -150:
                        fav_team = fav.name
                        break
            if fav_team: break
        if fav_team:
            sigs.append(CorrelationSignal(
                id=hashlib.md5((eid+"-fav-ml").encode()).hexdigest(),
                kind="matchup_trend", eventId=eid, players=[], teams=[fav_team],
                markets=["player_points"], boost=0.03,
                reason=f"Favorite ({fav_team}) → star scoring usage bump"
            ))

        # 1d) Recent form proxy: multi-market strength for same player in this event
        # (Using available markets to infer role stability)
        counts: Dict[str, int] = {}
        for bk in ev.bookmakers:
            for m in bk.markets:
                if m.key.startswith("player_"):
                    for o in m.outcomes:
                        if o.description:
                            counts[o.description] = counts.get(o.description, 0) + 1
        for player, n in counts.items():
            if n >= 2:
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((eid+player+"-form").encode()).hexdigest(),
                    kind="news", eventId=eid, players=[player], teams=[],
                    markets=[], boost=0.04,
                    reason="Recent form / role stability across markets"
                ))

    # 2) NBA scorer + facilitator synergy (fake roster from outcomes)
    players = set()
    for bk in ev.bookmakers:
        for m in bk.markets:
            for o in m.outcomes:
                if o.description: players.add(o.description)
    players = list(players)
    if players:
        # create a few scorer/assist matchups
        for p in random.sample(players, k=min(2,len(players))):
            sigs.append(CorrelationSignal(
                id=hashlib.md5((eid+p+"-pf").encode()).hexdigest(),
                kind="team_style", eventId=eid, players=[p], teams=[],
                markets=["player_points","player_assists"], boost=0.06,
                reason="High-usage scorer + facilitator in same game"
            ))
    # 3) External adapters (free sources)
    try:
        _n = len(sigs)
        sigs.extend(adapter_nfl_injuries(ev))
        logger.info(f"signals.adapter nfl_injuries added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter nfl_injuries error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_mlb_lineups(ev))
        logger.info(f"signals.adapter mlb_lineups added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter mlb_lineups error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_nhl_lineups(ev))
        logger.info(f"signals.adapter nhl_lineups added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter nhl_lineups error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_nba_injuries(ev))
        logger.info(f"signals.adapter nba_injuries added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter nba_injuries error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_cfb_rules(ev))
        logger.info(f"signals.adapter cfb_rules added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter cfb_rules error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_tennis_rules(ev))
        logger.info(f"signals.adapter tennis_rules added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter tennis_rules error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_nba_recent_form(ev))
        logger.info(f"signals.adapter nba_recent_form added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter nba_recent_form error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_nba_h2h(ev))
        logger.info(f"signals.adapter nba_h2h added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter nba_h2h error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_mlb_recent_form(ev))
        logger.info(f"signals.adapter mlb_recent_form added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter mlb_recent_form error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_mlb_h2h(ev))
        logger.info(f"signals.adapter mlb_h2h added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter mlb_h2h error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_nhl_recent_form(ev))
        logger.info(f"signals.adapter nhl_recent_form added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter nhl_recent_form error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_nhl_h2h(ev))
        logger.info(f"signals.adapter nhl_h2h added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter nhl_h2h error")
        pass
    try:
        _n = len(sigs)
        sigs.extend(adapter_nfl_recent_form(ev))
        logger.info(f"signals.adapter nfl_recent_form added={len(sigs)-_n}")
    except Exception:
        logger.info("signals.adapter nfl_recent_form error")
        pass
    return sigs

# ---------- Simple cache for events by id (supports /api/signals) ----------

EVENT_CACHE: Dict[str, APIEvent] = {}

# ---------- Lightweight TTL cache for odds (limits API usage) ----------
from time import time as _now

ODDS_CACHE: Dict[Tuple[str, Tuple[str, ...], Optional[str]], Tuple[float, List[APIEvent]]] = {}
CACHE_TTL_DEFAULT = int(os.getenv("ODDS_CACHE_TTL", "120"))  # seconds

def _cache_key(league: str, markets: List[str], date: Optional[str]) -> Tuple[str, Tuple[str, ...], Optional[str]]:
    # Use a stable tuple of sorted markets as part of the key
    return (league, tuple(sorted(markets)), date)

def get_cached_odds(league: str, markets: List[str], date: Optional[str], max_age: Optional[int] = None) -> Optional[List[APIEvent]]:
    ttl = CACHE_TTL_DEFAULT if max_age is None else max_age
    key = _cache_key(league, markets, date)
    hit = ODDS_CACHE.get(key)
    if not hit:
        try:
            METRICS["cache_misses_total"] += 1
            logger.info(f"odds.cache miss league={league} markets={sorted(markets)} date={date}")
        except Exception:
            pass
        return None
    ts, events = hit
    age = _now() - ts
    if age > ttl:
        try:
            METRICS["cache_stale_total"] += 1
            logger.info(f"odds.cache stale league={league} markets={sorted(markets)} date={date} age_s={int(age)} ttl_s={ttl}")
        except Exception:
            pass
        return None
    try:
        METRICS["cache_hits_total"] += 1
        logger.info(f"odds.cache hit league={league} markets={sorted(markets)} date={date} size={len(events)} age_s={int(age)}")
    except Exception:
        pass
    return events

def set_cached_odds(league: str, markets: List[str], date: Optional[str], events: List[APIEvent]):
    key = _cache_key(league, markets, date)
    ODDS_CACHE[key] = (_now(), events)
    try:
        logger.info(f"odds.cache write league={league} markets={sorted(markets)} date={date} size={len(events)}")
    except Exception:
        pass

# ---------- Signal Adapters (scaffold config & caches) ----------
# Free/low-cost adapters are toggleable by env var to protect quotas.


ENABLE_NBA_INJ = os.getenv("ENABLE_NBA_INJ", "false").lower() == "true"  # SportsData.io (optional)
INJ_TTL = int(os.getenv("INJ_TTL", "21600"))  # 6 hours
INJ_CACHE: Dict[str, Tuple[float, List[CorrelationSignal]]] = {}  # eventId -> (ts, signals)
ENABLE_NBA_FORM = os.getenv("ENABLE_NBA_FORM", "true").lower() == "true"  # balldontlie (free)

# Free adapters toggles + small caches
ENABLE_NFL_INJ = os.getenv("ENABLE_NFL_INJ", "false").lower() == "true"  # Sleeper (free)
NFL_INJ_TTL = int(os.getenv("NFL_INJ_TTL", "21600"))
NFL_INJ_CACHE: Dict[str, Tuple[float, List[CorrelationSignal]]] = {}

ENABLE_MLB_LINEUPS = os.getenv("ENABLE_MLB_LINEUPS", "false").lower() == "true"  # MLB StatsAPI (free)
MLB_LINEUPS_TTL = int(os.getenv("MLB_LINEUPS_TTL", "21600"))
MLB_LINEUPS_CACHE: Dict[str, Tuple[float, List[CorrelationSignal]]] = {}

ENABLE_NHL_LINEUPS = os.getenv("ENABLE_NHL_LINEUPS", "false").lower() == "true"  # NHL StatsAPI (free)
NHL_LINEUPS_TTL = int(os.getenv("NHL_LINEUPS_TTL", "21600"))
NHL_LINEUPS_CACHE: Dict[str, Tuple[float, List[CorrelationSignal]]] = {}

# MLB/NHL recent-form toggles
ENABLE_MLB_FORM = os.getenv("ENABLE_MLB_FORM", "true").lower() == "true"
ENABLE_NHL_FORM = os.getenv("ENABLE_NHL_FORM", "true").lower() == "true"

# Tiny caches for player id lookups
MLB_PLAYER_ID_CACHE: Dict[str, Optional[int]] = {}
NHL_PLAYER_ID_CACHE: Dict[str, Optional[int]] = {}


# Extra rule toggles
ENABLE_CFB_RULES = os.getenv("ENABLE_CFB_RULES", "true").lower() == "true"
ENABLE_TENNIS_RULES = os.getenv("ENABLE_TENNIS_RULES", "true").lower() == "true"

# --- CFBD (CollegeFootballData) optional free integration ---
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "").strip()
ENABLE_CFB_CFBD = os.getenv("ENABLE_CFB_CFBD", "true").lower() == "true"  # enable if a key is present
CFBD_TTL = int(os.getenv("CFBD_TTL", "21600"))  # 6 hours
CFBD_TEAM_CACHE: Dict[str, Tuple[float, Optional[str]]] = {}

def _cfb_season() -> int:
    now = datetime.utcnow()
    # CFB season crosses years; start in August
    return now.year if now.month >= 8 else now.year - 1

def _cfbd_school_name(raw_team: str) -> Optional[str]:
    """Resolve a team display name from odds to CFBD school name via /teams (cached)."""
    key = raw_team.strip()
    hit = CFBD_TEAM_CACHE.get(key)
    if hit and _now() - hit[0] < CFBD_TTL:
        return hit[1]
    if not CFBD_API_KEY:
        CFBD_TEAM_CACHE[key] = (_now(), None)
        return None
    try:
        r = http_get(
            "https://api.collegefootballdata.com/teams",
            headers={"Authorization": f"Bearer {CFBD_API_KEY}"},
            params={"search": raw_team},
            timeout=10,
        )
        if r.status_code != 200:
            CFBD_TEAM_CACHE[key] = (_now(), None)
            return None
        rows = r.json() or []
        name = None
        if rows:
            # prefer exact school field; fallback to first match
            name = rows[0].get("school") or rows[0].get("team")
        CFBD_TEAM_CACHE[key] = (_now(), name)
        return name
    except Exception:
        CFBD_TEAM_CACHE[key] = (_now(), None)
        return None

# CFBD: team season stats (tempo, efficiency) — cached
CFBD_STATS_CACHE: Dict[Tuple[str, int], Tuple[float, Optional[Dict[str, float]]]] = {}


def _cfbd_team_season_stats(raw_team: str) -> Optional[Dict[str, float]]:
    """Return a small set of season metrics for a team using CFBD /stats/season.
    Metrics returned (when available): plays_pg, yards_per_play, pass_rate.
    """
    if not CFBD_API_KEY or not ENABLE_CFB_CFBD:
        return None
    school = _cfbd_school_name(raw_team) or raw_team
    year = _cfb_season()
    key = (school, year)
    hit = CFBD_STATS_CACHE.get(key)
    if hit and _now() - hit[0] < CFBD_TTL:
        return hit[1]
    try:
        r = http_get(
            "https://api.collegefootballdata.com/stats/season",
            headers={"Authorization": f"Bearer {CFBD_API_KEY}"},
            params={"year": year, "team": school},
            timeout=10,
        )
        if r.status_code != 200:
            CFBD_STATS_CACHE[key] = (_now(), None)
            return None
        rows = r.json() or []
        if not rows:
            CFBD_STATS_CACHE[key] = (_now(), None)
            return None
        row = rows[0]
        # CFBD keys vary slightly; guard each
        plays_pg = float(row.get("playsPerGame") or row.get("plays_pg") or 0.0)
        ypp = float(row.get("yardsPerPlay") or row.get("yardsPerPlayOffense") or 0.0)
        # pass attempts per game / total plays per game → crude pass rate
        pass_att_pg = float(row.get("passAttemptsPerGame") or row.get("passAtt_pg") or 0.0)
        pass_rate = (pass_att_pg / plays_pg) if plays_pg > 0 else 0.0
        out = {"plays_pg": plays_pg, "yards_per_play": ypp, "pass_rate": pass_rate}
        CFBD_STATS_CACHE[key] = (_now(), out)
        return out
    except Exception:
        CFBD_STATS_CACHE[key] = (_now(), None)
        return None

# In-memory caches for adapters
PLAYER_ID_CACHE: Dict[str, Optional[int]] = {}
FORM_CACHE: Dict[int, Tuple[float, Dict[str, float]]] = {}  # pid -> (ts, stats dict)
FORM_TTL = int(os.getenv("FORM_TTL", "21600"))  # 6 hours by default

# balldontlie team-id cache
TEAM_ID_CACHE: Dict[str, Optional[int]] = {}

def _bld_team_id(team_name: str) -> Optional[int]:
    """Find NBA team id via balldontlie /teams search; cached by name."""
    if team_name in TEAM_ID_CACHE:
        return TEAM_ID_CACHE[team_name]
    try:
        r = http_get("https://www.balldontlie.io/api/v1/teams", params={"search": team_name}, timeout=8)
        if r.status_code != 200:
            TEAM_ID_CACHE[team_name] = None
            return None
        data = r.json().get("data", [])
        if not data:
            TEAM_ID_CACHE[team_name] = None
            return None
        tid = int(data[0]["id"])  # best match
        TEAM_ID_CACHE[team_name] = tid
        return tid
    except Exception:
        TEAM_ID_CACHE[team_name] = None
        return None

def _cache_get_form(pid: int) -> Optional[Dict[str, float]]:
    hit = FORM_CACHE.get(pid)
    if not hit: return None
    ts, stats = hit
    if _now() - ts > FORM_TTL: return None
    return stats

# Set form stats in cache
def _cache_set_form(pid: int, stats: Dict[str, float]):
    FORM_CACHE[pid] = (_now(), stats)

# Helper: Find NBA player id via balldontlie, cached by name
def _bld_search_player(full_name: str) -> Optional[int]:
    """Find NBA player id via balldontlie. Cached by name. Returns None if not found."""
    if full_name in PLAYER_ID_CACHE:
        return PLAYER_ID_CACHE[full_name]
    try:
        r = http_get(
            "https://www.balldontlie.io/api/v1/players",
            params={"search": full_name}, timeout=8
        )
        if r.status_code != 200:
            PLAYER_ID_CACHE[full_name] = None
            return None
        data = r.json().get("data", [])
        if not data:
            PLAYER_ID_CACHE[full_name] = None
            return None
        pid = int(data[0]["id"])  # take best match
        PLAYER_ID_CACHE[full_name] = pid
        return pid
    except Exception:
        PLAYER_ID_CACHE[full_name] = None
        return None

# Cache for player -> current team id (balldontlie)
PLAYER_TEAM_CACHE: Dict[int, Optional[int]] = {}

def _bld_player_team_id(pid: int) -> Optional[int]:
    """Return current team id for player pid via balldontlie /players/{id}. Cached."""
    if pid in PLAYER_TEAM_CACHE:
        return PLAYER_TEAM_CACHE[pid]
    try:
        r = http_get(f"https://www.balldontlie.io/api/v1/players/{pid}", timeout=8)
        if r.status_code != 200:
            PLAYER_TEAM_CACHE[pid] = None
            return None
        team = (r.json() or {}).get("team") or {}
        tid = team.get("id")
        PLAYER_TEAM_CACHE[pid] = int(tid) if tid is not None else None
        return PLAYER_TEAM_CACHE[pid]
    except Exception:
        PLAYER_TEAM_CACHE[pid] = None
        return None

# Season averages helper (recent form; cached)
def _bld_recent_form(pid: int) -> Optional[Dict[str, float]]:
    c = _cache_get_form(pid)
    if c is not None:
        return c
    try:
        r = http_get(
            "https://www.balldontlie.io/api/v1/season_averages",
            params={"player_ids[]": pid}, timeout=8
        )
        if r.status_code != 200:
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        row = data[0]
        stats = {
            "pts": float(row.get("pts", 0)),
            "ast": float(row.get("ast", 0)),
            "reb": float(row.get("reb", 0)),
            "stl": float(row.get("stl", 0)),
            "blk": float(row.get("blk", 0)),
            "fg3m": float(row.get("fg3m", 0)),
        }
        _cache_set_form(pid, stats)
        return stats
    except Exception:
        return None

# Last-5 games helper (cached): pts/ast/reb/fg3m means
LAST5_CACHE: Dict[int, Tuple[float, Dict[str, float]]] = {}
LAST5_TTL = int(os.getenv("LAST5_TTL", "21600"))  # 6 hours

def _bld_last5(pid: int) -> Optional[Dict[str, float]]:
    hit = LAST5_CACHE.get(pid)
    if hit and _now() - hit[0] < LAST5_TTL:
        return hit[1]
    try:
        r = http_get(
            "https://www.balldontlie.io/api/v1/stats",
            params={"player_ids[]": pid, "per_page": 5, "postseason": "false"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        n = len(data)
        s_pts = sum(float(d.get("pts", 0)) for d in data)
        s_ast = sum(float(d.get("ast", 0)) for d in data)
        s_reb = sum(float(d.get("reb", 0)) for d in data)
        s_3m  = sum(float(d.get("fg3m", 0)) for d in data)
        out = {"pts": s_pts/n, "ast": s_ast/n, "reb": s_reb/n, "fg3m": s_3m/n}
        LAST5_CACHE[pid] = (_now(), out)
        return out
    except Exception:
        return None

# --- NBA Head-to-Head (H2H) helper: cache + fetch recent games vs specific opponent ---
H2H_CACHE: Dict[Tuple[int, int], Tuple[float, Dict[str, float]]] = {}
H2H_TTL = int(os.getenv("H2H_TTL", "21600"))  # 6 hours

def _bld_player_h2h(pid: int, opp_team_id: int, n: int = 8) -> Optional[Dict[str, float]]:
    """Return averages (pts/ast/reb/fg3m) for recent games vs opp_team_id. Cached by (pid, opp)."""
    if opp_team_id is None:
        return None
    key = (pid, opp_team_id)
    hit = H2H_CACHE.get(key)
    if hit and _now() - hit[0] < H2H_TTL:
        return hit[1]
    try:
        # Pull a page of recent games and filter locally to opponent
        r = http_get(
            "https://www.balldontlie.io/api/v1/stats",
            params={"player_ids[]": pid, "per_page": 30, "postseason": "false"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        rows = []
        for d in data:
            g = d.get("game") or {}
            h_id = g.get("home_team_id")
            v_id = g.get("visitor_team_id")
            if h_id == opp_team_id or v_id == opp_team_id:
                rows.append(d)
        if not rows:
            return None
        rows = rows[:max(1, n)]  # limit to last n vs this opp
        n_used = len(rows)
        s_pts = sum(float(x.get("pts", 0)) for x in rows)
        s_ast = sum(float(x.get("ast", 0)) for x in rows)
        s_reb = sum(float(x.get("reb", 0)) for x in rows)
        s_3m  = sum(float(x.get("fg3m", 0)) for x in rows)
        out = {"pts": s_pts/n_used, "ast": s_ast/n_used, "reb": s_reb/n_used, "fg3m": s_3m/n_used}
        H2H_CACHE[key] = (_now(), out)
        return out
    except Exception:
        return None

# NBA recent form adapter (season averages above line for key stat markets)
def adapter_nba_recent_form(ev: APIEvent) -> List[CorrelationSignal]:
    if not ENABLE_NBA_FORM or ev.sport_key.lower() != "nba":
        return []
    sigs: List[CorrelationSignal] = []
    player_lines: Dict[str, Dict[str, float]] = {}
    for bk in ev.bookmakers:
        for m in bk.markets:
            if m.key.startswith("player_"):
                for o in m.outcomes:
                    if o.description and o.point is not None:
                        player_lines.setdefault(o.description, {})[m.key] = o.point
    if not player_lines:
        return []
    mappings = [
        ("player_points",   "pts",  0.05, "Season scoring above line"),
        ("player_assists",  "ast",  0.05, "Season assists above line"),
        ("player_rebounds", "reb",  0.05, "Season rebounds above line"),
        ("player_threes",   "fg3m", 0.04, "Season 3PM above line"),
    ]
    for name, lines in player_lines.items():
        pid = _bld_search_player(name)
        if not pid:
            continue
        form = _bld_recent_form(pid)
        if not form:
            continue
        for mk, stat, base_boost, why in mappings:
            line = lines.get(mk)
            if line is None:
                continue
            val = form.get(stat, 0.0)
            gap = val - float(line)
            if gap >= 0.7:
                boost = min(base_boost + gap * 0.01, base_boost + 0.04)
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((ev.id+name+mk+"form").encode()).hexdigest(),
                    kind="news", eventId=ev.id, players=[name], teams=[], markets=[mk],
                    boost=round(boost, 3), reason=f"{why} (avg {val:.1f} > line {line:.1f})"
                ))
        # Last-5 games nudges (smaller than season avg; additive if both apply)
        l5 = _bld_last5(pid)
        if l5:
            mappings_l5 = [
                ("player_points",   "pts",  0.03, "Last-5 scoring above line"),
                ("player_assists",  "ast",  0.03, "Last-5 assists above line"),
                ("player_rebounds", "reb",  0.03, "Last-5 rebounds above line"),
                ("player_threes",   "fg3m", 0.025, "Last-5 3PM above line"),
            ]
            for mk2, stat2, base2, why2 in mappings_l5:
                line2 = lines.get(mk2)
                if line2 is None:
                    continue
                val2 = l5.get(stat2, 0.0)
                gap2 = val2 - float(line2)
                if gap2 >= 0.6:
                    boost2 = min(base2 + gap2 * 0.008, base2 + 0.03)
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+name+mk2+"l5").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[name], teams=[], markets=[mk2],
                        boost=round(boost2, 3), reason=f"{why2} (L5 {val2:.1f} > line {line2:.1f})"
                    ))
    return sigs
def adapter_nba_h2h(ev: APIEvent) -> List[CorrelationSignal]:
    """NBA head-to-head signals: if a player's recent H2H average vs this opponent beats today's line, add a boost."""
    if ev.sport_key.lower() != "nba":
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
            # require a meaningful edge before boosting
            threshold = 0.5 if stat == "fg3m" else 1.0
            if gap >= threshold:
                boost = min(cap, base + gap * slope)
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((ev.id+name+mk+"h2h").encode()).hexdigest(),
                    kind="head_to_head", eventId=ev.id, players=[name], teams=[], markets=[mk],
                    boost=round(boost, 3), reason=f"H2H vs {opp_name}: avg {val:.1f} > line {float(line):.1f}"
                ))
    return sigs
def adapter_nfl_injuries(ev: APIEvent) -> List[CorrelationSignal]:
    if not ENABLE_NFL_INJ:
        return []
    # Per-event cache
    hit = NFL_INJ_CACHE.get(ev.id)
    if hit and _now() - hit[0] < NFL_INJ_TTL:
        return hit[1]
    data = _sleeper_players()
    if not data:
        NFL_INJ_CACHE[ev.id] = (_now(), [])
        return []
    teams = set(_nfl_event_abbrs(ev))
    sigs: List[CorrelationSignal] = []
    # Simple mapping of status strings to OUT/QUESTIONABLE buckets
    out_markers = {"out", "o", "ir", "injured_reserve", "pup", "doubtful"}
    q_markers = {"questionable", "q"}
    # Scan players for the two teams
    for pid, p in data.items():
        try:
            tm = str(p.get("team") or "").upper()
            if tm not in teams:
                continue
            pos = (p.get("position") or "").upper()
            # Build a display name
            name = p.get("full_name") or ("{} {}".format(p.get("first_name","").strip(), p.get("last_name","").strip()).strip())
            # Sleeper status fields vary; check both
            st = str(p.get("injury_status") or p.get("status") or "").lower()
            if not st:
                continue
            if st in out_markers:
                # Team-level positive bumps when key positions are OUT
                if pos == "QB":
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+tm+pos+"qbout").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[], teams=[tm],
                        markets=["player_rush_yards","player_rush_attempts"],
                        boost=0.06, reason=f"{name} ({tm}) OUT at QB → rush volume up"
                    ))
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+tm+pos+"qbpassdown").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[], teams=[tm],
                        markets=["player_pass_yards"],
                        boost=-0.05, reason=f"{name} ({tm}) OUT at QB → passing yards down"
                    ))
                elif pos in {"WR","TE"}:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+tm+pos+"recup").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[], teams=[tm],
                        markets=["player_receptions","player_receiving_yards"],
                        boost=0.04, reason=f"{name} ({tm}) OUT → target redistribution ↑"
                    ))
                elif pos == "RB":
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+tm+pos+"rbout").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[], teams=[tm],
                        markets=["player_receptions","player_receiving_yards"],
                        boost=0.03, reason=f"{name} ({tm}) OUT at RB → receiving backs usage ↑"
                    ))
            elif st in q_markers:
                # Smaller nudges for Questionable
                if pos == "QB":
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+tm+pos+"qbq").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[], teams=[tm],
                        markets=["player_rush_yards","player_rush_attempts"],
                        boost=0.03, reason=f"{name} ({tm}) Questionable at QB → contingency rush lean"
                    ))
                elif pos in {"WR","TE"}:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+tm+pos+"recq").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[], teams=[tm],
                        markets=["player_receptions","player_receiving_yards"],
                        boost=0.02, reason=f"{name} ({tm}) Questionable → target redistribution risk"
                    ))
        except Exception:
            continue
    NFL_INJ_CACHE[ev.id] = (_now(), sigs)
    return sigs

def adapter_mlb_lineups(ev: APIEvent) -> List[CorrelationSignal]:
    if not ENABLE_MLB_LINEUPS:
        return []
    # Per-event cache
    hit = MLB_LINEUPS_CACHE.get(ev.id)
    if hit and _now() - hit[0] < MLB_LINEUPS_TTL:
        return hit[1]

    gamepk = _mlb_find_gamepk(ev)
    if not gamepk:
        MLB_LINEUPS_CACHE[ev.id] = (_now(), [])
        return []
    try:
        r = http_get(f"https://statsapi.mlb.com/api/v1/game/{gamepk}/boxscore", timeout=10)
        if r.status_code != 200:
            MLB_LINEUPS_CACHE[ev.id] = (_now(), [])
            return []
        box = r.json().get("teams", {})
    except Exception:
        MLB_LINEUPS_CACHE[ev.id] = (_now(), [])
        return []

    sigs: List[CorrelationSignal] = []
    for side in ("home","away"):
        team = box.get(side) or {}
        players = team.get("players") or {}
        # 1) Starting pitcher confirmation → small K bump
        for pid, pdata in players.items():
            try:
                pos = (pdata.get("position") or {}).get("abbreviation")
                bo = pdata.get("battingOrder")  # e.g., "100", "200" for 1st, 2nd
                person = pdata.get("person") or {}
                name = person.get("fullName") or person.get("boxscoreName")
                if pos == "P" and pdata.get("isSubstitute") is False:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+name+"sp_k").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[name] if name else [], teams=[],
                        markets=["pitcher_strikeouts"], boost=0.03,
                        reason="Starting pitcher confirmed"
                    ))
                # 2) Hitters batting 2–4 → small TB bump
                if bo and len(str(bo)) >= 3:
                    try:
                        spot = int(str(bo)[:1])
                        if 2 <= spot <= 4 and name:
                            sigs.append(CorrelationSignal(
                                id=hashlib.md5((ev.id+name+"tb_boost").encode()).hexdigest(),
                                kind="news", eventId=ev.id, players=[name], teams=[],
                                markets=["player_total_bases"], boost=0.02,
                                reason=f"Batting order {spot} confirmed"
                            ))
                    except Exception:
                        pass
            except Exception:
                continue

    MLB_LINEUPS_CACHE[ev.id] = (_now(), sigs)
    return sigs


# --- MLB recent form adapter ---
def adapter_mlb_recent_form(ev: APIEvent) -> List[CorrelationSignal]:
    if not ENABLE_MLB_FORM or ev.sport_key.lower() != "mlb":
        return []
    # collect MLB prop lines by player
    player_lines: Dict[str, Dict[str, float]] = {}
    for bk in ev.bookmakers:
        for m in bk.markets:
            if m.key.startswith("player_"):
                for o in m.outcomes:
                    if o.description and o.point is not None:
                        player_lines.setdefault(o.description, {})[m.key] = o.point
    if not player_lines:
        return []
    sigs: List[CorrelationSignal] = []
    for name, lines in player_lines.items():
        pid = _mlb_search_player(name)
        if not pid:
            continue
        form = _mlb_lastN_hitting(pid, n=7)
        if form:
            # TB boost
            line = lines.get("player_total_bases")
            if line is not None:
                gap = form.get("tb", 0.0) - float(line)
                if gap >= 0.5:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+name+"mlb_tb_form").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[name], teams=[], markets=["player_total_bases"],
                        boost=0.03, reason=f"Last-7 TB avg {form['tb']:.1f} > line {line:.1f}"
                    ))
            # Hits boost
            line = lines.get("player_hits")
            if line is not None:
                gap = form.get("h", 0.0) - float(line)
                if gap >= 0.4:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+name+"mlb_hits_form").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[name], teams=[], markets=["player_hits"],
                        boost=0.025, reason=f"Last-7 H avg {form['h']:.1f} > line {line:.1f}"
                    ))
            # HR boost
            line = lines.get("player_home_runs")
            if line is not None:
                gap = form.get("hr", 0.0) - float(line)
                if gap >= 0.2:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+name+"mlb_hr_form").encode()).hexdigest(),
                        kind="news", eventId=ev.id, players=[name], teams=[], markets=["player_home_runs"],
                        boost=0.03, reason=f"Last-7 HR avg {form['hr']:.2f} > line {line:.1f}"
                    ))
        # Pitcher K boost
        if "player_strikeouts" in lines:
            formp = _mlb_lastN_pitching(pid, n=3)
            if formp:
                line = lines.get("player_strikeouts")
                if line is not None:
                    gap = formp.get("k", 0.0) - float(line)
                    if gap >= 0.5:
                        sigs.append(CorrelationSignal(
                            id=hashlib.md5((ev.id+name+"mlb_k_form").encode()).hexdigest(),
                            kind="news", eventId=ev.id, players=[name], teams=[], markets=["player_strikeouts"],
                            boost=0.03, reason=f"Last-3 K avg {formp['k']:.1f} > line {line:.1f}"
                        ))
    return sigs


# --- MLB helper: player's current team abbreviation (from people/{id}) ---
MLB_PLAYER_TEAM_ABBR_CACHE: Dict[int, Optional[str]] = {}

def _mlb_player_team_abbr(pid: int) -> Optional[str]:
    if pid in MLB_PLAYER_TEAM_ABBR_CACHE:
        return MLB_PLAYER_TEAM_ABBR_CACHE[pid]
    try:
        r = http_get(f"https://statsapi.mlb.com/api/v1/people/{pid}", timeout=8)
        if r.status_code != 200:
            MLB_PLAYER_TEAM_ABBR_CACHE[pid] = None
            return None
        people = (r.json() or {}).get("people", [])
        team = (people[0] or {}).get("currentTeam", {}) if people else {}
        abbr = team.get("abbreviation") or _mlb_guess_abbr(team.get("name", ""))
        MLB_PLAYER_TEAM_ABBR_CACHE[pid] = abbr
        return abbr
    except Exception:
        MLB_PLAYER_TEAM_ABBR_CACHE[pid] = None
        return None

# --- MLB head-to-head adapter (light) ---
def adapter_mlb_h2h(ev: APIEvent) -> List[CorrelationSignal]:
    if ev.sport_key.lower() != "mlb":
        return []
    # collect MLB prop lines by player
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

    home_abbr, away_abbr = _mlb_event_abbrs(ev)
    sigs: List[CorrelationSignal] = []
    for name, lines in player_lines.items():
        pid = _mlb_search_player(name)
        if not pid:
            continue
        team_abbr = _mlb_player_team_abbr(pid)
        if not team_abbr:
            continue
        team_abbr = team_abbr.upper()
        if team_abbr == home_abbr.upper():
            opp = away_abbr
        elif team_abbr == away_abbr.upper():
            opp = home_abbr
        else:
            # fallback: try both opponents and take the first that yields data
            opp = None
        # Hitters H2H
        if opp:
            h2h_hit = _mlb_h2h_lastN(pid, opp, group="hitting", n=7)
        else:
            h2h_hit = _mlb_h2h_lastN(pid, home_abbr, group="hitting", n=7) or _mlb_h2h_lastN(pid, away_abbr, group="hitting", n=7)
        if h2h_hit:
            # TB
            ln = lines.get("player_total_bases")
            if ln is not None:
                gap = float(h2h_hit.get("tb", 0.0)) - float(ln)
                if gap >= 0.4:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+name+"mlb_h2h_tb").encode()).hexdigest(),
                        kind="head_to_head", eventId=ev.id, players=[name], teams=[], markets=["player_total_bases"],
                        boost=0.022, reason=f"H2H vs {opp or 'opp'}: avg TB {h2h_hit.get('tb',0):.2f} > line {float(ln):.2f}"
                    ))
            # Hits
            ln = lines.get("player_hits")
            if ln is not None:
                gap = float(h2h_hit.get("h", 0.0)) - float(ln)
                if gap >= 0.3:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+name+"mlb_h2h_h").encode()).hexdigest(),
                        kind="head_to_head", eventId=ev.id, players=[name], teams=[], markets=["player_hits"],
                        boost=0.018, reason=f"H2H vs {opp or 'opp'}: avg H {h2h_hit.get('h',0):.2f} > line {float(ln):.2f}"
                    ))
            # Home Runs
            ln = lines.get("player_home_runs")
            if ln is not None:
                gap = float(h2h_hit.get("hr", 0.0)) - float(ln)
                if gap >= 0.12:
                    sigs.append(CorrelationSignal(
                        id=hashlib.md5((ev.id+name+"mlb_h2h_hr").encode()).hexdigest(),
                        kind="head_to_head", eventId=ev.id, players=[name], teams=[], markets=["player_home_runs"],
                        boost=0.02, reason=f"H2H vs {opp or 'opp'}: avg HR {h2h_hit.get('hr',0):.2f} > line {float(ln):.2f}"
                    ))
        # Pitcher K H2H (if a line exists for this player)
        if "player_strikeouts" in lines:
            if opp:
                h2h_p = _mlb_h2h_lastN(pid, opp, group="pitching", n=5)
            else:
                h2h_p = _mlb_h2h_lastN(pid, home_abbr, group="pitching", n=5) or _mlb_h2h_lastN(pid, away_abbr, group="pitching", n=5)
            if h2h_p:
                ln = lines.get("player_strikeouts")
                if ln is not None:
                    gap = float(h2h_p.get("k", 0.0)) - float(ln)
                    if gap >= 0.4:
                        sigs.append(CorrelationSignal(
                            id=hashlib.md5((ev.id+name+"mlb_h2h_k").encode()).hexdigest(),
                            kind="head_to_head", eventId=ev.id, players=[name], teams=[], markets=["player_strikeouts"],
                            boost=0.02, reason=f"H2H vs {opp or 'opp'}: avg K {h2h_p.get('k',0):.2f} > line {float(ln):.2f}"
                        ))
    return sigs


def adapter_nhl_lineups(ev: APIEvent) -> List[CorrelationSignal]:
    if not ENABLE_NHL_LINEUPS:
        return []
    hit = NHL_LINEUPS_CACHE.get(ev.id)
    if hit and _now() - hit[0] < NHL_LINEUPS_TTL:
        return hit[1]
    sigs: List[CorrelationSignal] = []
    gamepk = _nhl_find_gamepk(ev)
    if not gamepk:
        NHL_LINEUPS_CACHE[ev.id] = (_now(), sigs)
        return sigs
    try:
        r = http_get(f"https://statsapi.web.nhl.com/api/v1/game/{gamepk}/boxscore", timeout=10)
        if r.status_code != 200:
            NHL_LINEUPS_CACHE[ev.id] = (_now(), sigs)
            return sigs
        box = r.json().get("teams", {})
    except Exception:
        NHL_LINEUPS_CACHE[ev.id] = (_now(), sigs)
        return sigs
    for side in ("home", "away"):
        team = box.get(side) or {}
        players = team.get("players") or {}
        scratches = team.get("scratches") or []
        if scratches:
            sigs.append(CorrelationSignal(
                id=hashlib.md5((ev.id+side+"nhl_scratch").encode()).hexdigest(),
                kind="news", eventId=ev.id, players=[], teams=[], markets=["player_points"],
                boost=0.02, reason="Scratches reported → lineup volatility"
            ))
        goalie_found = any(((pdata.get("position") or {}).get("abbreviation") == "G") for pdata in players.values())
        if goalie_found:
            sigs.append(CorrelationSignal(
                id=hashlib.md5((ev.id+side+"nhl_goalie").encode()).hexdigest(),
                kind="team_style", eventId=ev.id, players=[], teams=[], markets=["totals"],
                boost=0.02, reason="Goalie listed → totals confidence tweak"
            ))
    NHL_LINEUPS_CACHE[ev.id] = (_now(), sigs)
    return sigs


# --- NHL recent-form adapter ---
def adapter_nhl_recent_form(ev: APIEvent) -> List[CorrelationSignal]:
    if not ENABLE_NHL_FORM or ev.sport_key.lower() != "nhl":
        return []
    # collect NHL prop lines by player
    player_lines: Dict[str, Dict[str, float]] = {}
    for bk in ev.bookmakers:
        for m in bk.markets:
            if m.key.startswith("player_"):
                for o in m.outcomes:
                    if o.description and o.point is not None:
                        player_lines.setdefault(o.description, {})[m.key] = o.point
    if not player_lines:
        return []
    sigs: List[CorrelationSignal] = []
    for name, lines in player_lines.items():
        pid = _nhl_search_player(name)
        if not pid:
            continue
        l5 = _nhl_lastN(pid, n=5)
        if not l5:
            continue
        # Goals boost
        line = lines.get("player_goals")
        if line is not None:
            gap = l5.get("g", 0.0) - float(line)
            if gap >= 0.2:
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((ev.id+name+"nhl_g_l5").encode()).hexdigest(),
                    kind="news", eventId=ev.id, players=[name], teams=[], markets=["player_goals"],
                    boost=0.025, reason=f"Last-5 avg G {l5['g']:.2f} > line {float(line):.2f}"
                ))
        # Shots on goal boost
        line = lines.get("player_shots_on_goal")
        if line is not None:
            gap = l5.get("sog", 0.0) - float(line)
            if gap >= 0.6:
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((ev.id+name+"nhl_sog_l5").encode()).hexdigest(),
                    kind="news", eventId=ev.id, players=[name], teams=[], markets=["player_shots_on_goal"],
                    boost=0.02, reason=f"Last-5 avg SOG {l5['sog']:.2f} > line {float(line):.2f}"
                ))
    return sigs


# --- NHL head-to-head adapter ---
def adapter_nhl_h2h(ev: APIEvent) -> List[CorrelationSignal]:
    """NHL head-to-head: if a player's recent averages vs today's opponent beat the line, add a small boost."""
    if ev.sport_key.lower() != "nhl":
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

    opp_home = ev.away_team  # for home-team players
    opp_away = ev.home_team  # for away-team players

    sigs: List[CorrelationSignal] = []
    for name, lines in player_lines.items():
        pid = _nhl_search_player(name)
        if not pid:
            continue
        # determine which opponent name to use by checking player's current team
        team = _nhl_player_team_name(pid)
        if not team:
            continue
        opp_name = opp_away if team.lower() == ev.away_team.lower() else opp_home
        h2h = _nhl_h2h_lastN(pid, opp_name, n=6)
        if not h2h:
            continue
        # Goals
        if (ln := lines.get("player_goals")) is not None:
            gap = float(h2h.get("g", 0.0)) - float(ln)
            if gap >= 0.15:
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((ev.id+name+"nhl_h2h_g").encode()).hexdigest(),
                    kind="head_to_head", eventId=ev.id, players=[name], teams=[], markets=["player_goals"],
                    boost=0.02, reason=f"H2H vs {opp_name}: avg G {h2h['g']:.2f} > line {float(ln):.2f}"
                ))
        # Shots on goal
        if (ln := lines.get("player_shots_on_goal")) is not None:
            gap = float(h2h.get("sog", 0.0)) - float(ln)
            if gap >= 0.4:
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((ev.id+name+"nhl_h2h_sog").encode()).hexdigest(),
                    kind="head_to_head", eventId=ev.id, players=[name], teams=[], markets=["player_shots_on_goal"],
                    boost=0.018, reason=f"H2H vs {opp_name}: avg SOG {h2h['sog']:.2f} > line {float(ln):.2f}"
                ))
    return sigs


# --- CFB odds-only adapter: compact rules for college football ---
def adapter_cfb_rules(ev: APIEvent) -> List[CorrelationSignal]:
    """Odds-only nudges for college football; no external API calls."""
    if not ENABLE_CFB_RULES or ev.sport_key.lower() not in {"ncaa_football", "ncaaf", "cfb"}:
        return []
    sigs: List[CorrelationSignal] = []
    total_points = None
    best_fav = None  # (team_name, price)
    for bk in ev.bookmakers:
        for m in bk.markets:
            if m.key == "totals":
                for o in m.outcomes:
                    if o.point is not None:
                        total_points = o.point
                        break
            if m.key == "h2h" and m.outcomes:
                fav = sorted(m.outcomes, key=lambda o: o.price)[0]
                if best_fav is None or fav.price < best_fav[1]:
                    best_fav = (fav.name, fav.price)
    if total_points is not None and total_points >= 62:
        sigs.append(CorrelationSignal(
            id=hashlib.md5((ev.id+"cfb-hi-total").encode()).hexdigest(),
            kind="pace_injury", eventId=ev.id, players=[], teams=[],
            markets=["player_receptions","player_receiving_yards","player_rush_yards"],
            boost=0.04, reason=f"High total ({int(total_points)}) → pace/efficiency tailwind"
        ))
    if best_fav and best_fav[1] <= -300:
        sigs.append(CorrelationSignal(
            id=hashlib.md5((ev.id+best_fav[0]+"cfb-fav").encode()).hexdigest(),
            kind="matchup_trend", eventId=ev.id, players=[], teams=[best_fav[0]],
            markets=["player_rush_yards","player_receiving_yards"],
            boost=0.03, reason=f"Heavy favorite ({best_fav[0]}) → volume skew to primaries"
        ))
    return sigs

# --- Tennis odds-only adapter: compact rules for ATP/WTA ---
def adapter_tennis_rules(ev: APIEvent) -> List[CorrelationSignal]:
    """Odds-only nudges for tennis; no external API calls."""
    if not ENABLE_TENNIS_RULES or ev.sport_key.lower() not in {"atp", "wta"}:
        return []
    sigs: List[CorrelationSignal] = []
    best_fav = None  # (name, price)
    total_games = None
    for bk in ev.bookmakers:
        for m in bk.markets:
            if m.key == "h2h" and m.outcomes:
                fav = sorted(m.outcomes, key=lambda o: o.price)[0]
                if best_fav is None or fav.price < best_fav[1]:
                    best_fav = (fav.name, fav.price)
            if m.key == "totals":
                for o in m.outcomes:
                    if o.point is not None:
                        total_games = o.point
                        break
    if best_fav and best_fav[1] <= -250:
        sigs.append(CorrelationSignal(
            id=hashlib.md5((ev.id+best_fav[0]+"ten-fav").encode()).hexdigest(),
            kind="matchup_trend", eventId=ev.id, players=[], teams=[best_fav[0]],
            markets=["h2h","spreads"], boost=0.04,
            reason=f"Strong favorite ({best_fav[0]}) → ML/handicap lean"
        ))
    if total_games is not None and total_games <= 21.5:
        sigs.append(CorrelationSignal(
            id=hashlib.md5((ev.id+"ten-low-total").encode()).hexdigest(),
            kind="team_style", eventId=ev.id, players=[], teams=[],
            markets=["h2h","spreads"], boost=0.03,
            reason="Low total → straight-sets/short match lean"
        ))
    elif total_games is not None and total_games >= 24.5:
        sigs.append(CorrelationSignal(
            id=hashlib.md5((ev.id+"ten-high-total").encode()).hexdigest(),
            kind="team_style", eventId=ev.id, players=[], teams=[],
            markets=["totals"], boost=0.03,
            reason="High total → long match tiebreak risk"
        ))
    return sigs

# --- NBA helper: map common team names to SportsData-style abbreviations
_NBA_ABBR = {
    "Lakers":"LAL","Los Angeles Lakers":"LAL",
    "Clippers":"LAC","Los Angeles Clippers":"LAC",
    "Warriors":"GSW","Golden State Warriors":"GSW",
    "Suns":"PHX","Phoenix Suns":"PHX",
    "Kings":"SAC","Sacramento Kings":"SAC",
    "Trail Blazers":"POR","Portland Trail Blazers":"POR",
    "Jazz":"UTA","Utah Jazz":"UTA",
    "Nuggets":"DEN","Denver Nuggets":"DEN",
    "Thunder":"OKC","Oklahoma City Thunder":"OKC",
    "Mavericks":"DAL","Dallas Mavericks":"DAL",
    "Spurs":"SAS","San Antonio Spurs":"SAS",
    "Rockets":"HOU","Houston Rockets":"HOU",
    "Grizzlies":"MEM","Memphis Grizzlies":"MEM",
    "Pelicans":"NOP","New Orleans Pelicans":"NOP",
    "Timberwolves":"MIN","Minnesota Timberwolves":"MIN",
    "Bucks":"MIL","Milwaukee Bucks":"MIL",
    "Bulls":"CHI","Chicago Bulls":"CHI",
    "Cavaliers":"CLE","Cleveland Cavaliers":"CLE",
    "Pistons":"DET","Detroit Pistons":"DET",
    "Pacers":"IND","Indiana Pacers":"IND",
    "Hawks":"ATL","Atlanta Hawks":"ATL",
    "Heat":"MIA","Miami Heat":"MIA",
    "Magic":"ORL","Orlando Magic":"ORL",
    "Hornets":"CHA","Charlotte Hornets":"CHA",
    "Wizards":"WAS","Washington Wizards":"WAS",
    "Knicks":"NYK","New York Knicks":"NYK",
    "Nets":"BKN","Brooklyn Nets":"BKN",
    "Celtics":"BOS","Boston Celtics":"BOS",
    "76ers":"PHI","Philadelphia 76ers":"PHI",
    "Raptors":"TOR","Toronto Raptors":"TOR"
}
def _nba_event_abbrs(ev: APIEvent) -> List[str]:
    names = [ev.home_team, ev.away_team]
    out = []
    for n in names:
        ab = _NBA_ABBR.get(n) or _NBA_ABBR.get(n.strip()) or _NBA_ABBR.get(n.split()[-1], None)
        if not ab:
            parts = n.replace("Los Angeles","LA").replace("Golden State","GS").split()
            initials = "".join(w[0].upper() for w in parts if w and w[0].isalpha())
            ab = initials if 2 <= len(initials) <= 3 else n[:3].upper()
        out.append(ab)
    return out

# --- NBA injuries adapter (SportsData.io; compact, guarded, cached)
def adapter_nba_injuries(ev: APIEvent) -> List[CorrelationSignal]:
    # Optional: only if enabled and key present
    if not ENABLE_NBA_INJ or not SPORTSDATA_IO_KEY:
        return []
    # Use shared INJ_CACHE keyed by event
    hit = INJ_CACHE.get(ev.id)
    if hit and _now() - hit[0] < INJ_TTL:
        return hit[1]
    sigs: List[CorrelationSignal] = []
    try:
        # SportsData.io: pull league-wide injuries and filter by this event's two teams
        url = "https://api.sportsdata.io/v3/nba/injuries/json/Injuries"
        r = http_get(url, headers={"Ocp-Apim-Subscription-Key": SPORTSDATA_IO_KEY}, timeout=10)
        if r.status_code != 200:
            INJ_CACHE[ev.id] = (_now(), sigs)
            return sigs
        data = r.json() if isinstance(r.json(), list) else []
        teams = set(_nba_event_abbrs(ev))
        for row in data:
            try:
                tm = str(row.get("Team") or "").upper()
                if tm not in teams:
                    continue
                status = str(row.get("InjuryStatus") or row.get("Status") or "").lower()
                name = row.get("Name") or row.get("PlayerName") or ""
                pos = (row.get("Position") or "").upper()
                if not status or not name:
                    continue
                # map to OUT / Q buckets
                if status in {"out","doubtful","inactive","injured reserve","ir"}:
                    # star out → teammates usage bumps
                    if pos in {"G","F","C"}:
                        sigs.append(CorrelationSignal(
                            id=hashlib.md5((ev.id+tm+name+"nb_out_pts").encode()).hexdigest(),
                            kind="news", eventId=ev.id, players=[], teams=[tm],
                            markets=["player_points","player_assists"], boost=0.04,
                            reason=f"{name} ({tm}) OUT → usage redistribution"
                        ))
                elif status in {"questionable","q"}:
                    if pos in {"G","F","C"}:
                        sigs.append(CorrelationSignal(
                            id=hashlib.md5((ev.id+tm+name+"nb_q").encode()).hexdigest(),
                            kind="news", eventId=ev.id, players=[], teams=[tm],
                            markets=["player_points"], boost=0.02,
                            reason=f"{name} ({tm}) Questionable → volatility in usage"
                        ))
            except Exception:
                continue
    except Exception:
        pass
    INJ_CACHE[ev.id] = (_now(), sigs)
    return sigs

# --- NFL helper: map team names to abbreviations (for free APIs like Sleeper)
_NFL_ABBR = {
    "Chiefs":"KC","Kansas City Chiefs":"KC",
    "Bengals":"CIN","Cincinnati Bengals":"CIN",
    "Bills":"BUF","Buffalo Bills":"BUF",
    "Jets":"NYJ","New York Jets":"NYJ",
    "Cowboys":"DAL","Dallas Cowboys":"DAL",
    "Eagles":"PHI","Philadelphia Eagles":"PHI",
    "Giants":"NYG","New York Giants":"NYG",
    "Patriots":"NE","New England Patriots":"NE",
    "Dolphins":"MIA","Miami Dolphins":"MIA",
    "Ravens":"BAL","Baltimore Ravens":"BAL",
    "Steelers":"PIT","Pittsburgh Steelers":"PIT",
    "Browns":"CLE","Cleveland Browns":"CLE",
    "Jaguars":"JAX","Jacksonville Jaguars":"JAX",
    "Titans":"TEN","Tennessee Titans":"TEN",
    "Colts":"IND","Indianapolis Colts":"IND",
    "Texans":"HOU","Houston Texans":"HOU",
    "Broncos":"DEN","Denver Broncos":"DEN",
    "Raiders":"LV","Las Vegas Raiders":"LV",
    "Chargers":"LAC","Los Angeles Chargers":"LAC",
    "Chiefs":"KC","Kansas City":"KC",
    "Packers":"GB","Green Bay Packers":"GB",
    "Bears":"CHI","Chicago Bears":"CHI",
    "Vikings":"MIN","Minnesota Vikings":"MIN",
    "Lions":"DET","Detroit Lions":"DET",
    "Saints":"NO","New Orleans Saints":"NO",
    "Buccaneers":"TB","Tampa Bay Buccaneers":"TB",
    "Panthers":"CAR","Carolina Panthers":"CAR",
    "Falcons":"ATL","Atlanta Falcons":"ATL",
    "Seahawks":"SEA","Seattle Seahawks":"SEA",
    "49ers":"SF","San Francisco 49ers":"SF",
    "Rams":"LAR","Los Angeles Rams":"LAR",
    "Cardinals":"ARI","Arizona Cardinals":"ARI",
    "Commanders":"WAS","Washington Commanders":"WAS"
}
def _nfl_event_abbrs(ev: APIEvent) -> List[str]:
    names = [ev.home_team, ev.away_team]
    abbrs = []
    for n in names:
        ab = _NFL_ABBR.get(n) or _NFL_ABBR.get(n.strip()) or _NFL_ABBR.get(n.split()[-1], None)
        if not ab:
            # fallback: take uppercase letters of words (e.g., 'Los Angeles Chargers' -> 'LAC')
            parts = n.replace("Los Angeles","LA").split()
            initials = "".join(w[0].upper() for w in parts if w and w[0].isalpha())
            ab = initials if 2 <= len(initials) <= 3 else n[:3].upper()
        abbrs.append(ab)
    return abbrs

# --- MLB helpers: guess team abbreviations and get both teams from event
_MLB_ABBR_OVERRIDES = {
    "White Sox": "CWS", "Chicago White Sox": "CWS",
    "Red Sox": "BOS",  "Boston Red Sox": "BOS",
    "Cubs": "CHC",     "Chicago Cubs": "CHC",
    "Yankees": "NYY",  "New York Yankees": "NYY",
    "Mets": "NYM",     "New York Mets": "NYM",
    "Dodgers": "LAD",  "Los Angeles Dodgers": "LAD",
    "Angels": "LAA",   "Los Angeles Angels": "LAA",
    "Giants": "SF",    "San Francisco Giants": "SF",
    "A's": "OAK",      "Athletics": "OAK", "Oakland Athletics": "OAK",
    "Cardinals": "STL", "St. Louis Cardinals": "STL",
    "Diamondbacks": "ARI", "Arizona Diamondbacks": "ARI",
    "Blue Jays": "TOR", "Toronto Blue Jays": "TOR",
    "Rays": "TB",      "Tampa Bay Rays": "TB",
    "Guardians": "CLE", "Cleveland Guardians": "CLE",
    "Braves": "ATL",    "Atlanta Braves": "ATL",
    "Brewers": "MIL",   "Milwaukee Brewers": "MIL",
    "Twins": "MIN",     "Minnesota Twins": "MIN",
    "Tigers": "DET",    "Detroit Tigers": "DET",
    "Reds": "CIN",      "Cincinnati Reds": "CIN",
    "Pirates": "PIT",   "Pittsburgh Pirates": "PIT",
    "Mariners": "SEA",  "Seattle Mariners": "SEA",
    "Rockies": "COL",   "Colorado Rockies": "COL",
    "Padres": "SD",     "San Diego Padres": "SD",
    "Nationals": "WSH", "Washington Nationals": "WSH",
    "Orioles": "BAL",   "Baltimore Orioles": "BAL",
    "Phillies": "PHI",  "Philadelphia Phillies": "PHI",
    "Royals": "KC",     "Kansas City Royals": "KC",
    "Rangers": "TEX",   "Texas Rangers": "TEX",
    "Marlins": "MIA",   "Miami Marlins": "MIA"
}
def _mlb_guess_abbr(name: str) -> str:
    n = name.strip()
    if n in _MLB_ABBR_OVERRIDES:
        return _MLB_ABBR_OVERRIDES[n]
    parts = n.replace("Los Angeles", "LA").replace("St. Louis", "STL").split()
    initials = "".join(w[0].upper() for w in parts if w and w[0].isalpha())
    if 2 <= len(initials) <= 3:
        return initials
    return n[:3].upper()
def _mlb_event_abbrs(ev: APIEvent) -> List[str]:
    return [_mlb_guess_abbr(ev.home_team), _mlb_guess_abbr(ev.away_team)]

# --- Date helper for events (Y-m-d from commence_time) ---

def _event_date_ymd(ev: APIEvent) -> str:
    try:
        dt = datetime.fromisoformat(ev.commence_time.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")

# --- MLB: Find gamePk for event's teams and date ---
def _mlb_find_gamepk(ev: APIEvent) -> Optional[int]:
    # Look up MLB schedule for the event date and match by team abbreviations
    date = _event_date_ymd(ev)
    ab_home, ab_away = _mlb_event_abbrs(ev)
    try:
        r = http_get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": date}, timeout=10
        )
        if r.status_code != 200:
            return None
        sched = r.json()
        for d in sched.get("dates", []):
            for g in d.get("games", []):
                try:
                    t_home = g.get("teams", {}).get("home", {}).get("team", {})
                    t_away = g.get("teams", {}).get("away", {}).get("team", {})
                    gh = (t_home.get("abbreviation") or _mlb_guess_abbr(t_home.get("name", ""))).upper()
                    ga = (t_away.get("abbreviation") or _mlb_guess_abbr(t_away.get("name", ""))).upper()
                    if {gh, ga} == {ab_home.upper(), ab_away.upper()}:
                        return int(g.get("gamePk"))
                except Exception:
                    continue
    except Exception:
        return None
    return None

# --- MLB recent-form helpers (StatsAPI) ---
MLB_FORM_HIT_CACHE: Dict[Tuple[int, int], Tuple[float, Dict[str, float]]] = {}
MLB_FORM_PITCH_CACHE: Dict[Tuple[int, int], Tuple[float, Dict[str, float]]] = {}
MLB_FORM_TTL = int(os.getenv("MLB_FORM_TTL", "21600"))  # 6 hours

def _mlb_search_player(full_name: str) -> Optional[int]:
    if full_name in MLB_PLAYER_ID_CACHE:
        return MLB_PLAYER_ID_CACHE[full_name]
    try:
        r = http_get("https://statsapi.mlb.com/api/v1/people", params={"search": full_name}, timeout=10)
        if r.status_code != 200:
            MLB_PLAYER_ID_CACHE[full_name] = None
            return None
        people = (r.json() or {}).get("people", [])
        pid = int(people[0]["id"]) if people else None
        MLB_PLAYER_ID_CACHE[full_name] = pid
        return pid
    except Exception:
        MLB_PLAYER_ID_CACHE[full_name] = None
        return None

def _mlb_lastN_hitting(pid: int, n: int = 7) -> Optional[Dict[str, float]]:
    key = (pid, n)
    hit = MLB_FORM_HIT_CACHE.get(key)
    if hit and _now() - hit[0] < MLB_FORM_TTL:
        return hit[1]
    try:
        season = datetime.utcnow().year
        r = http_get(
            f"https://statsapi.mlb.com/api/v1/people/{pid}/stats",
            params={"stats": "gameLog", "group": "hitting", "season": season},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        splits = ((r.json() or {}).get("stats") or [{}])[0].get("splits", [])
        rows = splits[:max(1, n)]
        if not rows:
            return None
        s_tb = s_h = s_hr = 0.0
        for row in rows:
            st = row.get("stat", {})
            s_tb += float(st.get("totalBases", 0))
            s_h  += float(st.get("hits", 0))
            s_hr += float(st.get("homeRuns", 0))
        k = len(rows)
        out = {"tb": s_tb/k, "h": s_h/k, "hr": s_hr/k}
        MLB_FORM_HIT_CACHE[key] = (_now(), out)
        return out
    except Exception:
        return None

def _mlb_lastN_pitching(pid: int, n: int = 3) -> Optional[Dict[str, float]]:
    key = (pid, n)
    hit = MLB_FORM_PITCH_CACHE.get(key)
    if hit and _now() - hit[0] < MLB_FORM_TTL:
        return hit[1]
    try:
        season = datetime.utcnow().year
        r = http_get(
            f"https://statsapi.mlb.com/api/v1/people/{pid}/stats",
            params={"stats": "gameLog", "group": "pitching", "season": season},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        splits = ((r.json() or {}).get("stats") or [{}])[0].get("splits", [])
        rows = splits[:max(1, n)]
        if not rows:
            return None
        s_k = 0.0
        for row in rows:
            st = row.get("stat", {})
            s_k += float(st.get("strikeOuts", 0))
        k = len(rows)
        out = {"k": s_k/k}
        MLB_FORM_PITCH_CACHE[key] = (_now(), out)
        return out
    except Exception:
        return None

# MLB: last-N averages vs a specific opponent (by team abbreviation)
MLB_H2H_CACHE: Dict[Tuple[int, str, str, int], Tuple[float, Dict[str, float]]] = {}
MLB_H2H_TTL = int(os.getenv("MLB_H2H_TTL", "21600"))

def _mlb_h2h_lastN(pid: int, opp_abbr: str, group: str = "hitting", n: int = 7) -> Optional[Dict[str, float]]:
    if not opp_abbr:
        return None
    key = (pid, opp_abbr.upper(), group, n)
    hit = MLB_H2H_CACHE.get(key)
    if hit and _now() - hit[0] < MLB_H2H_TTL:
        return hit[1]
    try:
        season = datetime.utcnow().year
        r = http_get(
            f"https://statsapi.mlb.com/api/v1/people/{pid}/stats",
            params={"stats": "gameLog", "group": group, "season": season},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        splits = ((r.json() or {}).get("stats") or [{}])[0].get("splits", [])
        if not splits:
            return None
        rows = []
        target = opp_abbr.upper()
        for row in splits:
            opp = (row.get("opponent") or {}).get("abbreviation") or (row.get("opponent") or {}).get("name")
            if not opp:
                continue
            cand = str(opp).upper() if len(str(opp)) <= 4 else _mlb_guess_abbr(str(opp))
            if cand.upper() == target:
                rows.append(row)
        if not rows:
            return None
        rows = rows[:max(1, n)]
        k = len(rows)
        if group == "hitting":
            s_tb = s_h = s_hr = 0.0
            for row in rows:
                st = row.get("stat", {})
                s_tb += float(st.get("totalBases", 0))
                s_h  += float(st.get("hits", 0))
                s_hr += float(st.get("homeRuns", 0))
            out = {"tb": s_tb/k, "h": s_h/k, "hr": s_hr/k}
        else:  # pitching
            s_k = 0.0
            for row in rows:
                st = row.get("stat", {})
                s_k += float(st.get("strikeOuts", 0))
            out = {"k": s_k/k}
        MLB_H2H_CACHE[key] = (_now(), out)
        return out
    except Exception:
        return None

# --- NHL: Find gamePk for event's teams and date (StatsAPI) ---
def _nhl_find_gamepk(ev: APIEvent) -> Optional[int]:
    date = _event_date_ymd(ev)
    home, away = ev.home_team.lower(), ev.away_team.lower()
    try:
        r = http_get(
            "https://statsapi.web.nhl.com/api/v1/schedule",
            params={"date": date}, timeout=10
        )
        if r.status_code != 200:
            return None
        sched = r.json()
        for d in sched.get("dates", []):
            for g in d.get("games", []):
                try:
                    gh = (g.get("teams", {}).get("home", {}).get("team", {}).get("name", "")).lower()
                    ga = (g.get("teams", {}).get("away", {}).get("team", {}).get("name", "")).lower()
                    if {gh, ga} == {home, away}:
                        return int(g.get("gamePk"))
                except Exception:
                    continue
    except Exception:
        return None
    return None

# --- NHL recent-form helpers (Suggest + StatsAPI) ---
NHL_PLAYER_ID_CACHE: Dict[str, Optional[int]] = {}
NHL_FORM_CACHE: Dict[Tuple[int, int], Tuple[float, Dict[str, float]]] = {}
NHL_FORM_TTL = int(os.getenv("NHL_FORM_TTL", "21600"))  # 6 hours


def _nhl_search_player(full_name: str) -> Optional[int]:
    if full_name in NHL_PLAYER_ID_CACHE:
        return NHL_PLAYER_ID_CACHE[full_name]
    try:
        # NHL suggest service: "id|First Last|..."
        q = full_name.strip().lower().replace(" ", "%20")
        r = http_get(f"https://suggest.svc.nhl.com/svc/suggest/v1/minplayers/{q}/10", timeout=8)
        if r.status_code != 200:
            NHL_PLAYER_ID_CACHE[full_name] = None
            return None
        items = (r.json() or {}).get("suggestions", [])
        if not items:
            NHL_PLAYER_ID_CACHE[full_name] = None
            return None
        pid = int(str(items[0]).split("|", 1)[0])
        NHL_PLAYER_ID_CACHE[full_name] = pid
        return pid
    except Exception:
        NHL_PLAYER_ID_CACHE[full_name] = None
        return None


def _nhl_lastN(pid: int, n: int = 5) -> Optional[Dict[str, float]]:
    key = (pid, n)
    hit = NHL_FORM_CACHE.get(key)
    if hit and _now() - hit[0] < NHL_FORM_TTL:
        return hit[1]
    try:
        season = datetime.utcnow().year
        season_str = f"{season-1}{season}" if datetime.utcnow().month < 9 else f"{season}{season+1}"
        r = http_get(
            f"https://statsapi.web.nhl.com/api/v1/people/{pid}/stats",
            params={"stats": "gameLog", "season": season_str}, timeout=10
        )
        if r.status_code != 200:
            return None
        splits = ((r.json() or {}).get("stats") or [{}])[0].get("splits", [])
        rows = splits[:max(1, n)]
        if not rows:
            return None
        s_g = s_sog = 0.0
        for row in rows:
            st = row.get("stat", {})
            s_g   += float(st.get("goals", 0))
            s_sog += float(st.get("shots", 0))
        k = len(rows)
        out = {"g": s_g/k, "sog": s_sog/k}
        NHL_FORM_CACHE[key] = (_now(), out)
        return out
    except Exception:
        return None

# NHL: resolve player's current team name (cached via StatsAPI people/{id})
NHL_PLAYER_TEAM_CACHE: Dict[int, Optional[str]] = {}

def _nhl_player_team_name(pid: int) -> Optional[str]:
    if pid in NHL_PLAYER_TEAM_CACHE:
        return NHL_PLAYER_TEAM_CACHE[pid]
    try:
        r = http_get(f"https://statsapi.web.nhl.com/api/v1/people/{pid}", timeout=8)
        if r.status_code != 200:
            NHL_PLAYER_TEAM_CACHE[pid] = None
            return None
        data = (r.json() or {}).get("people", [])
        team = (data[0] or {}).get("currentTeam", {}) if data else {}
        name = team.get("name")
        NHL_PLAYER_TEAM_CACHE[pid] = name
        return name
    except Exception:
        NHL_PLAYER_TEAM_CACHE[pid] = None
        return None

# NHL: last-N averages vs a specific opponent (by team name)
NHL_H2H_CACHE: Dict[Tuple[int, str, int], Tuple[float, Dict[str, float]]] = {}
NHL_H2H_TTL = int(os.getenv("NHL_H2H_TTL", "21600"))

def _nhl_h2h_lastN(pid: int, opp_team_name: str, n: int = 6) -> Optional[Dict[str, float]]:
    if not opp_team_name:
        return None
    key = (pid, opp_team_name.lower(), n)
    hit = NHL_H2H_CACHE.get(key)
    if hit and _now() - hit[0] < NHL_H2H_TTL:
        return hit[1]
    try:
        season = datetime.utcnow().year
        season_str = f"{season-1}{season}" if datetime.utcnow().month < 9 else f"{season}{season+1}"
        r = http_get(
            f"https://statsapi.web.nhl.com/api/v1/people/{pid}/stats",
            params={"stats": "gameLog", "season": season_str}, timeout=10
        )
        if r.status_code != 200:
            return None
        splits = ((r.json() or {}).get("stats") or [{}])[0].get("splits", [])
        if not splits:
            return None
        rows = []
        opp_l = opp_team_name.lower()
        for row in splits:
            try:
                game = row.get("team", {})
                # StatsAPI puts opponent in `opponent` inside the stat row for game logs
                opp = (row.get("opponent") or {}).get("name")
                if opp and opp.lower() == opp_l:
                    rows.append(row)
            except Exception:
                continue
        if not rows:
            return None
        rows = rows[:max(1, n)]
        k = len(rows)
        s_g = s_sog = 0.0
        for row in rows:
            st = row.get("stat", {})
            s_g += float(st.get("goals", 0))
            s_sog += float(st.get("shots", 0))
        out = {"g": s_g/k, "sog": s_sog/k}
        NHL_H2H_CACHE[key] = (_now(), out)
        return out
    except Exception:
        return None

# --- NFL (Sleeper) free players snapshot (cached)
SLEEPER_PLAYERS_TTL = int(os.getenv("SLEEPER_PLAYERS_TTL", "86400"))  # 24h
_SLEEPER_CACHE: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)

def _sleeper_players() -> Optional[Dict[str, Any]]:
    ts, data = _SLEEPER_CACHE
    if _now() - ts < SLEEPER_PLAYERS_TTL and data is not None:
        return data
    try:
        r = http_get("https://api.sleeper.app/v1/players/nfl", timeout=15)
        if r.status_code != 200:
            return data  # keep stale if present
        new = r.json()
        # store
        globals()["_SLEEPER_CACHE"] = (_now(), new)
        return new
    except Exception:
        return data

# --- NFL recent form via Sleeper weekly stats (free) ---
SLEEPER_STATE_TTL = int(os.getenv("SLEEPER_STATE_TTL", "900"))  # 15 min
_SLEEPER_STATE_CACHE: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)

def _sleeper_state() -> Optional[Dict[str, Any]]:
    ts, data = _SLEEPER_STATE_CACHE
    if _now() - ts < SLEEPER_STATE_TTL and data is not None:
        return data
    try:
        r = http_get("https://api.sleeper.app/v1/state/nfl", timeout=8)
        if r.status_code != 200:
            return data
        new = r.json()
        globals()["_SLEEPER_STATE_CACHE"] = (_now(), new)
        return new
    except Exception:
        return data

# Cache for mapping name -> Sleeper player_id for quick lookups
NFL_NAME_TO_ID: Dict[str, Optional[str]] = {}

def _nfl_lookup_player_id(full_name: str) -> Optional[str]:
    if full_name in NFL_NAME_TO_ID:
        return NFL_NAME_TO_ID[full_name]
    data = _sleeper_players()
    if not data:
        NFL_NAME_TO_ID[full_name] = None
        return None
    # Exact match first (case-insensitive), then loose contains
    lname = full_name.strip().lower()
    best = None
    for pid, row in data.items():
        nm = str(row.get("full_name") or "").strip()
        if nm.lower() == lname:
            best = pid
            break
        # fallback: loose match on last name token
        if not best and lname and nm and lname.split()[-1] in nm.lower():
            best = pid
    NFL_NAME_TO_ID[full_name] = best
    return best

# TTL cache for recent-form aggregates by sleeper_id
NFL_FORM_TTL = int(os.getenv("NFL_FORM_TTL", "21600"))  # 6 hours
NFL_FORM_CACHE: Dict[Tuple[str, int], Tuple[float, Dict[str, float]]] = {}

def _nfl_recent_form(full_name: str, weeks: int = 3) -> Optional[Dict[str, float]]:
    """Return last-`weeks` averages for key stats using Sleeper weekly stats.
    Stats keys: pass_yd, pass_td, rush_yd, rush_td, rec_yd, rec_td, rec (receptions).
    """
    pid = _nfl_lookup_player_id(full_name)
    if not pid:
        return None
    key = (pid, weeks)
    hit = NFL_FORM_CACHE.get(key)
    if hit and _now() - hit[0] < NFL_FORM_TTL:
        return hit[1]
    st = _sleeper_state() or {}
    season = int(st.get("season") or datetime.utcnow().year)
    week = int(st.get("week") or 1)
    agg = {"pass_yd":0.0,"pass_td":0.0,"rush_yd":0.0,"rush_td":0.0,"rec_yd":0.0,"rec_td":0.0,"rec":0.0}
    count = 0
    try:
        for w in range(max(1, week - weeks + 1), week + 1):
            r = http_get(f"https://api.sleeper.app/v1/stats/nfl/regular/{season}/{w}", timeout=8)
            if r.status_code != 200:
                continue
            rows = r.json() or []
            # rows is a list of dicts, each may include a `player_id` and stat fields
            found = None
            for row in rows:
                if str(row.get("player_id")) == str(pid):
                    found = row
                    break
            if not found:
                continue
            for k in agg.keys():
                agg[k] += float(found.get(k, 0) or 0)
            count += 1
        if count == 0:
            return None
        out = {k: (v / count) for k, v in agg.items()}
        NFL_FORM_CACHE[key] = (_now(), out)
        return out
    except Exception:
        return None

# --- NFL recent-form adapter (Sleeper last-3 weeks vs current lines) ---
def adapter_nfl_recent_form(ev: APIEvent) -> List[CorrelationSignal]:
    if ev.sport_key.lower() != "nfl":
        return []
    # Build player → {market: line}
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
    # Map markets → (form_key, base, slope, cap, min_gap)
    mapdefs: List[Tuple[str, str, float, float, float, float]] = [
        ("player_pass_yards",      "pass_yd", 0.030, 0.0005, 0.055, 15.0),
        ("player_pass_tds",        "pass_td", 0.020, 0.0100, 0.045, 0.4),
        ("player_rush_yards",      "rush_yd", 0.028, 0.0008, 0.055, 10.0),
        ("player_rush_tds",        "rush_td", 0.020, 0.0100, 0.045, 0.3),
        ("player_receiving_yards", "rec_yd",  0.030, 0.0008, 0.055, 10.0),
        ("player_receptions",      "rec",     0.028, 0.0150, 0.055, 0.6),
        ("player_receiving_tds",   "rec_td",  0.020, 0.0100, 0.045, 0.3),
    ]
    sigs: List[CorrelationSignal] = []
    for name, lines in player_lines.items():
        form = _nfl_recent_form(name, weeks=3)
        if not form:
            continue
        for mk, fkey, base, slope, cap, min_gap in mapdefs:
            ln = lines.get(mk)
            if ln is None:
                continue
            val = float(form.get(fkey, 0.0))
            gap = val - float(ln)
            if gap >= min_gap:
                boost = min(cap, base + gap * slope)
                sigs.append(CorrelationSignal(
                    id=hashlib.md5((ev.id+name+mk+"nfl_form").encode()).hexdigest(),
                    kind="news", eventId=ev.id, players=[name], teams=[], markets=[mk],
                    boost=round(boost, 3), reason=f"Last-3 avg {val:.1f} > line {float(ln):.1f}"
                ))
    return sigs

# ---------- Endpoints ----------
def resolve_league(league: Optional[str], sport: Optional[str]) -> str:
    if league: return league.lower()
    if sport:
        # map the apiKey-ish values to league codes used in iOS
        return {
            "basketball_nba": "nba",
            "americanfootball_nfl": "nfl",
            "americanfootball_ncaaf": "ncaa_football",
            "baseball_mlb": "mlb",
            "icehockey_nhl": "nhl",
            "tennis_atp": "atp",
            "tennis_wta": "wta",
        }.get(sport.lower(), sport.lower())
    raise HTTPException(status_code=422, detail="Missing 'league' or 'sport'")

def parse_markets(market: Optional[str]) -> List[str]:
    # Default now pulls props + team markets so the app has full context
    if not market or market.strip() == "":
        return ["player_props","h2h","spreads","totals"]
    m = [x.strip() for x in market.split(",") if x.strip()]
    # Explicit override for props-only, if needed by callers
    if "player_props_only" in m:
        return ["player_props"]
    if "all" in m:
        return ["player_props","h2h","spreads","totals"]
    return m

@app.get("/health")
def healthz():
    return {"ok": True, "ts": now_iso(), "provider": "theoddsapi", "configured": bool(THEODDSAPI_KEY)}

@app.get("/healthz")
def healthz_alias():
    return {"ok": True, "ts": now_iso(), "provider": "theoddsapi", "configured": bool(THEODDSAPI_KEY), "alias": "healthz"}
    
@app.get("/api/odds", response_model=List[APIEvent])
def get_odds(
    league: Optional[str] = None,
    sport: Optional[str] = None,
    date: Optional[str] = None,
    market: Optional[str] = "player_props"
):
    lg = resolve_league(league, sport)
    markets = parse_markets(market)
    parts = []
    # --- TTL cache check to reduce provider calls ---
    cached = get_cached_odds(lg, markets, date)
    if cached is not None:
        # refresh EVENT_CACHE for downstream /api/signals usage
        for ev in cached:
            EVENT_CACHE[ev.id] = ev
        return cached
    # You can parallelize these calls if you wire an async client
    for m in markets:
        parts.append(fetch_from_provider(lg, [m], date))
    merged = merge_events(parts)
    # store in TTL cache
    set_cached_odds(lg, markets, date, merged)
    for ev in merged:
        EVENT_CACHE[ev.id] = ev
    return merged

@app.get("/api/upcoming", response_model=List[APIEvent])
def get_upcoming(
    league: Optional[str] = None,
    sport: Optional[str] = None,
    market: Optional[str] = "player_props"
):
    return get_odds(league=league, sport=sport, date=None, market=market)

@app.get("/api/signals", response_model=List[CorrelationSignal])
def get_signals(event_id: str):
    ev = EVENT_CACHE.get(event_id)
    if not ev:
        try:
            logger.info(f"signals.miss event={event_id}")
        except Exception:
            pass
        return []
    try:
        sigs = build_signals_for_event(ev)
        try:
            METRICS["signals_built_total"] += len(sigs)
        except Exception:
            pass
        logger.info(f"signals.ok event={event_id} sport={ev.sport_key} sigs={len(sigs)}")
        return sigs
    except Exception as e:
        try:
            logger.warning(f"signals.err event={event_id} sport={ev.sport_key} err={e}")
        except Exception:
            pass
        return []

 # --- Pair scoring helper: cap signal contribution and clamp to 0..1 ---
def _pair_score(base_w: float, mk_a: str, mk_b: str, bonus_map: Dict[str, float]) -> float:
    sig_a = bonus_map.get(mk_a, 0.0)
    sig_b = bonus_map.get(mk_b, 0.0)
    # limit how much signals can add in total
    sig_part = min(0.20, 0.5*sig_a + 0.5*sig_b)
    return max(0.0, min(1.0, base_w + sig_part))

@app.get("/api/pairs", response_model=List[PairSuggestion])
def get_pairs(event_id: str, top_k: int = 8):
    ev = EVENT_CACHE.get(event_id)
    if not ev:
        return []
    # Build player → markets map with lines
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
    # Pull signals once and compute a market bonus map
    sigs = build_signals_for_event(ev)
    market_bonus: Dict[str, float] = {}
    for s in sigs:
        for mk in (s.markets or []):
            market_bonus[mk] = market_bonus.get(mk, 0.0) + s.boost
    suggestions: List[PairSuggestion] = []
    # Same-player pairs
    for player, mks in player_lines.items():
        keys = list(mks.keys())
        for a in keys:
            rules = CORRELATION_RULES.get(a, [])
            if not rules:
                continue
            for b, w, why in rules:
                if b not in mks:
                    continue
                score = _pair_score(w, a, b, market_bonus)
                suggestions.append(PairSuggestion(
                    eventId=ev.id,
                    score=round(score, 3),
                    reason=f"{why} (+signals)",
                    legs=[
                        PairLeg(player=player, market=a, line=mks.get(a)),
                        PairLeg(player=player, market=b, line=mks.get(b)),
                    ],
                ))
    # Cross-player pairs: limited neighbors to stay cheap
    players_list = list(player_lines.items())
    for i in range(len(players_list)):
        p1, mks1 = players_list[i]
        for j in range(i+1, min(i+6, len(players_list))):
            p2, mks2 = players_list[j]
            for a, rules in CROSS_PLAYER_RULES.items():
                if a not in mks1:
                    continue
                for b, w, why in rules:
                    if b not in mks2:
                        continue
                    score = _pair_score(w, a, b, market_bonus)
                    suggestions.append(PairSuggestion(
                        eventId=ev.id,
                        score=round(score, 3),
                        reason=f"{why} (+signals)",
                        legs=[
                            PairLeg(player=p1, market=a, line=mks1.get(a)),
                            PairLeg(player=p2, market=b, line=mks2.get(b)),
                        ],
                    ))
    suggestions.sort(key=lambda x: x.score, reverse=True)
    out = suggestions[:max(1, min(20, top_k))]
    try:
        METRICS["pairs_returned_total"] += len(out)
    except Exception:
        pass
    return out

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
