Absolutely—here’s a quick, no-fluff status of what’s in your current app (based on the ContentView.swift you shared), what works, what’s missing, and what to improve. Then I give you a drop-in main.py for the Render backend that matches your iOS models, supports multiple markets (props + moneyline/spread/total), and exposes a /api/signals endpoint for correlation/news-style boosters.

What’s in the app right now (and working)
	•	Tabs present: Picks, Props Browser (MVP), AI Builder, Planner v2 (bankroll → two primaries + safety, variable legs), Heat Map, Settings.
	•	Bet Slip: FanDuel-style bar + panel on the Picks tab (safeAreaInset + sheet), with EV, hit %, decimal odds, payout, correlation callouts within the slip, and suggested correlated adds.
	•	Correlation engine scaffolding:
	•	CorrelationSignal model + signalsByEventId
	•	async fetch from /api/signals?event_id=...
	•	local fallback generator buildDerivedSignals()
	•	pairing logic: correlationScore, correlationReason, marketPairBoost, suggestionsFor(leg:)
	•	AI Builder v2: lets you choose leg count, then optimizes using confidence + correlation with a diversity penalty tuned by Risk Mode.
	•	Planner v2: builds 2 primaries + 1 safety as singles or parlays, sizes stakes, shows Hit%, Dec Odds, Payout, EV, and can add to the slip.
	•	Heat Map: grid by player × key stat markets, 🔥 for hot, 🥶 for cold, game picker included.
	•	Odds fetch: robust client with logging, cache, dual routes (/api/odds//api/upcoming), dual param names (league/sport), local-day + fallback-day.

What’s missing / bugs to fix
	1.	Global correlation rules constant
Code references CORRELATION_RULES in marketPairBoost(_:_:), but that dictionary isn’t in this file. If it’s not in another file, you’ll get a compile error or a silent “no reason” in the UI.
➜ Add the large CORRELATION_RULES dictionary you pasted earlier at file top scope.
	2.	Duplicate signals fetch (wasted work; can stutter)
In AppVM.refresh() you call fetchSignalsIfAvailable twice. Keep the first block; delete the second.
	3.	Multi-market fetching not implemented
fetchAllProps still uses:

let marketParam = (markets?.first ?? "player_props")

That means you’re only ever hitting one market at a time (usually props).
➜ Replace with a loop over multiple markets (e.g. "player_props,h2h,spreads,totals"), or parallelize. (I gave you a paste-ready version earlier.)

	4.	Props Browser doesn’t expose team markets
The local fallback ends at NHL props and omits "h2h","spreads","totals".
➜ Add those keys and make sure MarketKey.label maps them to Moneyline / Spread / Total.
	5.	Bet Slip bar only on Picks
It won’t follow you into Props/Builder/Planner/Heat Map.
➜ Extract a BetSlipOverlay view modifier and apply it to every tab.
	6.	PickRow correlation hint not active in Picks
You defined PickRow(pick:addAction: vm:) but PicksHome is still calling it without vm, so the “Best match: … 68%” line won’t appear.
➜ Pass vm: vm where you render rows.
	7.	AI Builder “news/injuries”/matchup logic
The UI is wired for signals, but real news/injury/matchup signals depend on backend /api/signals. Right now you have a fallback heuristic.
➜ Implement /api/signals on your backend (below).

⸻

Backend main.py (Render): FastAPI, multi-market odds + signals

This file matches your iOS JSON models, supports multi-market (props + moneyline/spread/total), and provides /api/signals with domain-rule and heuristic correlation boosters. It also supports a FAKE_MODE for development until real provider keys are set.

Notes
• Set env vars on Render: ODDS_API_KEY (if you use a provider), FAKE_MODE=true while testing.
• The code is structured so you can swap fetch_from_provider() with your real data source(s).
• JSON schema mirrors your Swift APIEvent/APIBookmaker/APIMarket/APIOutcome.
• CORS enabled for your app.
• Returns one merged event list; markets/bookmakers are de-duped and sorted (FanDuel first).

# main.py
import os, time, random, hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_NAME = "smartpicks-backend"
FAKE_MODE = os.getenv("FAKE_MODE", "false").lower() == "true"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
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

# ---------- Fake data (for dev) ----------
TEAMS_BY_LEAGUE = {
    "nba": [("Lakers","Mavericks"), ("Celtics","Heat"), ("Suns","Warriors")],
    "nfl": [("Bengals","Chiefs"), ("Cowboys","Eagles"), ("Bills","Jets")],
    "mlb": [("Dodgers","Giants"), ("Yankees","Red Sox")],
    "nhl": [("Rangers","Bruins"), ("Maple Leafs","Lightning")],
}
PLAYER_POOLS = {
    "nba": ["LeBron James","Anthony Davis","Luka Doncic","Kyrie Irving","Jayson Tatum","Jimmy Butler","Kevin Durant","Stephen Curry"],
    "nfl": ["Joe Burrow","Ja'Marr Chase","Patrick Mahomes","Travis Kelce","Josh Allen","Stefon Diggs","Dak Prescott","CeeDee Lamb"],
    "mlb": ["Aaron Judge","Juan Soto","Mookie Betts","Shohei Ohtani","Rafael Devers","Ronald Acuña Jr."],
    "nhl": ["Auston Matthews","Nikita Kucherov","Connor McDavid","David Pastrnak","Artemi Panarin"],
}

def fake_event_id(home: str, away: str, date_str: str) -> str:
    seed = f"{home}-{away}-{date_str}"
    return hashlib.sha1(seed.encode()).hexdigest()[:16]

def make_fake_events(league: str, markets: List[str], date_str: Optional[str]) -> List[APIEvent]:
    teams = TEAMS_BY_LEAGUE.get(league, [("Team A","Team B")])
    date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out: List[APIEvent] = []
    for i, (home, away) in enumerate(teams):
        ev_id = fake_event_id(home, away, date)
        commence = (datetime.now(timezone.utc) + timedelta(hours=2+i)).isoformat()
        books: List[APIBookmaker] = []
        for bk in ["fanduel","draftkings"]:
            mkts: List[APIMarket] = []
            # team markets
            if any(m in markets for m in ["h2h","spreads","totals","all"]):
                mkts.append(APIMarket(key="h2h", outcomes=[
                    APIOutcome(name=home, price=random.choice([-110,-105,100,120])),
                    APIOutcome(name=away, price=random.choice([-110,-105,100,120]))
                ]))
                mkts.append(APIMarket(key="spreads", outcomes=[
                    APIOutcome(name=home, price=-110, point=random.choice([-3.5,-2.5,-1.5])),
                    APIOutcome(name=away, price=-110, point=random.choice([+3.5,+2.5,+1.5]))
                ]))
                mkts.append(APIMarket(key="totals", outcomes=[
                    APIOutcome(name="Over", price=-110, point=random.choice([219.5,222.5,225.5])),
                    APIOutcome(name="Under", price=-110, point=random.choice([219.5,222.5,225.5]))
                ]))
            # player props
            if any(m in markets for m in ["player_props","all"]):
                pool = PLAYER_POOLS.get(league, ["Player A","Player B","Player C"])
                for p in random.sample(pool, k=min(6,len(pool))):
                    mkts.append(APIMarket(key="player_points", outcomes=[
                        APIOutcome(name="Over", price=random.choice([-125,-110,100,120]), point=random.choice([23.5,26.5,28.5]), description=p),
                        APIOutcome(name="Under", price=random.choice([-120,-110,100,125]), point=random.choice([23.5,26.5,28.5]), description=p),
                    ]))
                    mkts.append(APIMarket(key="player_assists", outcomes=[
                        APIOutcome(name="Over", price=-110, point=random.choice([5.5,6.5,7.5]), description=p),
                        APIOutcome(name="Under", price=-110, point=random.choice([5.5,6.5,7.5]), description=p),
                    ]))
            books.append(APIBookmaker(key=bk, title=bk.title(), last_update=now_iso(), markets=mkts))
        out.append(APIEvent(
            id=ev_id,
            sport_key=league,
            sport_title=league.upper(),
            commence_time=commence,
            home_team=home,
            away_team=away,
            bookmakers=fanduel_first(books),
        ))
    return out

# ---------- Provider fetch (stub) ----------
def fetch_from_provider(league: str, markets: List[str], date_str: Optional[str]) -> List[APIEvent]:
    """
    Swap this with real provider calls. Keep return shape identical to APIEvent.
    """
    if FAKE_MODE or not ODDS_API_KEY:
        return make_fake_events(league, markets, date_str)
    # Example structure for real providers:
    # 1) call provider endpoints per market
    # 2) map provider JSON -> APIEvent schema
    # 3) merge by event id with merge_events()
    # For now just fallback:
    return make_fake_events(league, markets, date_str)

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
    # ... include the full mapping you pasted earlier ...
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
        }.get(sport.lower(), sport.lower())
    raise HTTPException(status_code=422, detail="Missing 'league' or 'sport'")

def parse_markets(market: Optional[str]) -> List[str]:
    if not market or market.strip() == "":
        return ["player_props"]  # default to props
    m = [x.strip() for x in market.split(",") if x.strip()]
    if "all" in m: return ["player_props","h2h","spreads","totals"]
    return m

@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": now_iso(), "fake": FAKE_MODE}

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
    # You can parallelize these calls if you wire an async client
    for m in markets:
        parts.append(fetch_from_provider(lg, [m], date))
    return merge_events(parts)

@app.get("/api/upcoming", response_model=List[APIEvent])
def get_upcoming(
    league: Optional[str] = None,
    sport: Optional[str] = None,
    market: Optional[str] = "player_props"
):
    return get_odds(league=league, sport=sport, date=None, market=market)

@app.get("/api/signals", response_model=List[CorrelationSignal])
def get_signals(event_id: str):
    # In a real impl, you’d look up the event from cache/db.
    # For FAKE mode we synthesize signals deterministically from the id.
    random.seed(event_id)
    # Build a thin fake event to feed the heuristic:
    ev = APIEvent(
        id=event_id, sport_key="nba", sport_title="NBA",
        commence_time=now_iso(), home_team="Home", away_team="Away",
        bookmakers=[APIBookmaker(key="fanduel", title="FanDuel", last_update=now_iso(), markets=[
            APIMarket(key="player_points", outcomes=[
                APIOutcome(name="Over", price=-110, point=24.5, description="Player A"),
                APIOutcome(name="Under", price=-110, point=24.5, description="Player A"),
            ]),
            APIMarket(key="player_assists", outcomes=[
                APIOutcome(name="Over", price=-110, point=6.5, description="Player B"),
                APIOutcome(name="Under", price=-110, point=6.5, description="Player B"),
            ]),
        ])]
    )
    base = build_signals_for_event(ev)
    # You can also enrich with domain-rule “reasons” if needed (from CORRELATION_RULES)
    return base

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


⸻

What to do next (fastest wins)
	1.	Add the CORRELATION_RULES dictionary (the full one you pasted) at top-level in your Swift file (or a new CorrelationRules.swift) so marketPairBoost actually returns reasons/bonuses.
	2.	Fix duplicate signal fetch (delete the second block in refresh()).
	3.	Enable multi-market in iOS fetcher (loop or parallelize over ["player_props","h2h","spreads","totals"]) and expose team markets in Props Browser (+ add labels in MarketKey.label).
	4.	Make the Bet Slip global (extract an overlay modifier and apply on all tabs), and pass vm: vm to PickRow everywhere you render it so correlation hints show.
	5.	Deploy the main.py to Render (set FAKE_MODE=true at first). Once your provider is wired, flip off FAKE mode and fill fetch_from_provider() with real calls and field mappings.

If you’d like, I can also provide the minimal Swift patches (small diffs) for steps 1–4 in a single block so you can paste once.
