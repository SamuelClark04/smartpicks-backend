    # Filter by sport and then split into small chunks to stay under provider limits
    filtered_markets = _filter_markets_for_sport(markets_csv, sport_key)

    # unified string of books, reuse later for cache key & store
    books_str = ",".join(bookmakers or DEFAULT_BOOKMAKERS)

    # try disk snapshot cache first
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
    chunks = _split_markets(filtered_markets, chunk_size=3)\n