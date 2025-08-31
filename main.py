import os, time, json
from typing import Dict, Tuple
from fastapi import FastAPI, Response, Request, HTTPException
import httpx
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.environ["ODDS_API_KEY"]
UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "https://api.the-odds-api.com")
UPSTREAM_VERSION = os.environ.get("UPSTREAM_VERSION", "v4")
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "900"))
PREFERRED_BOOK = os.environ.get("PREFERRED_BOOK", "fanduel").lower()

app = FastAPI(title="SmartPicks Backend", version="1.0.0")

# Simple in-memory cache: url -> (expires_at, body_bytes, headers_subset)
_cache: Dict[str, Tuple[float, bytes, Dict[str, str]]] = {}

async def fetch_upstream(path: str, query: Dict[str, str]) -> Tuple[bytes, Dict[str, str]]:
    # always add apiKey server-side
    q = dict(query)
    q["apiKey"] = ODDS_API_KEY
    url = f"{UPSTREAM_BASE}/{UPSTREAM_VERSION}{path}"

    # build canonical cache key
    items = sorted(q.items())
    key = url + "?" + "&".join([f"{k}={v}" for k,v in items])

    # cache hit?
    now = time.time()
    hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1], hit[2]

    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get(url, params=q, headers={"Accept":"application/json"})
    # bubble errors with upstream body for clarity
    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    # keep useful headers
    hdrs = {}
    for h in ["x-requests-remaining", "x-requests-used", "cache-control"]:
        if h in r.headers:
            hdrs[h] = r.headers[h]

    body = r.content
    _cache[key] = (now + CACHE_TTL, body, hdrs)
    return body, hdrs

@app.get("/health")
async def health():
    return {"ok": True, "cache_items": len(_cache)}

@app.get("/status")
async def status():
    # best-effort status by pinging sports list (cheap)
    body, hdrs = await fetch_upstream("/sports", {})
    return {
        "sports_count": len(json.loads(body)),
        "requests_remaining": hdrs.get("x-requests-remaining"),
        "requests_used": hdrs.get("x-requests-used")
    }

# ---- Pass-through Odds API endpoints (same schema as upstream) ----

@app.get("/v4/sports")
async def sports():
    body, hdrs = await fetch_upstream("/sports", {})
    return Response(content=body, media_type="application/json", headers=hdrs)

@app.get("/v4/sports/{sport}/odds-markets")
async def markets(sport: str):
    # markets endpoint name varies; support both "markets" and "odds-markets"
    try:
        body, hdrs = await fetch_upstream(f"/sports/{sport}/odds-markets", {})
    except HTTPException:
        body, hdrs = await fetch_upstream(f"/sports/{sport}/markets", {})
    return Response(content=body, media_type="application/json", headers=hdrs)

@app.get("/v4/sports/{sport}/odds")
async def odds(sport: str, request: Request):
    # forward allowed query params (we add apiKey internally)
    allowed = {"regions","markets","oddsFormat","dateFormat","commenceTimeFrom","commenceTimeTo"}
    q = {k: v for k, v in request.query_params.items() if k in allowed}
    # fetch upstream (cached)
    body, hdrs = await fetch_upstream(f"/sports/{sport}/odds", q)
    return Response(content=body, media_type="application/json", headers=hdrs)

# ---- Lightweight AI / correlation endpoint (uses same odds feed) ----

@app.get("/ai/picks")
async def ai_picks(sport: str, limit: int = 50):
    """
    Returns top N 'picks' with a simple confidence model and preferred book ordering.
    Schema is *not* the upstream one; this is for your app’s AI tab if you want.
    """
    body, _ = await fetch_upstream(f"/sports/{sport}/odds",
                                   {"regions":"us", "oddsFormat":"american"})
    events = json.loads(body)

    def american_to_decimal(odds: float) -> float:
        return 1.0 + (odds/100.0 if odds > 0 else 100.0/(-odds))

    def implied(odds: float) -> float:
        return 100/(odds+100) if odds > 0 else (-odds)/((-odds)+100)

    picks = []
    for ev in events:
        # sort books with preferred first
        books = sorted(ev.get("bookmakers", []),
                       key=lambda b: 0 if b.get("key","").lower()==PREFERRED_BOOK else 1)
        seen = set()
        for bk in books:
            for m in bk.get("markets", []):
                for out in m.get("outcomes", []):
                    name = out.get("name","")
                    desc = out.get("description")
                    player = desc if name.lower() in ("over","under") else name
                    line = out.get("point")
                    price = out.get("price")
                    if price is None or player is None: continue
                    # de-dupe event|market|player|line
                    dkey = f"{ev['id']}|{m['key']}|{player}|{line}"
                    if dkey in seen: continue
                    seen.add(dkey)
                    imp = implied(price)
                    conf = max(0.35, min(0.9,
                        0.5 + (0.18 if abs(price) >= 140 else 0.0) + (0.07 if imp > 0.55 else -0.03)))
                    picks.append({
                        "event_id": ev["id"],
                        "event_label": f"{ev['away_team']} @ {ev['home_team']}",
                        "commence_time": ev["commence_time"],
                        "bookmaker": bk.get("title"),
                        "player": player,
                        "market_key": m["key"],
                        "line": line,
                        "over_odds": price,
                        "confidence": conf
                    })
    # sort by time then confidence
    picks.sort(key=lambda p: (p["commence_time"], -p["confidence"]))
    return {"picks": picks[:limit]}
