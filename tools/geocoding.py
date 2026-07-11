"""
Place Search MCP tool (OpenStreetMap).

Exposes a single tool, ``find_nearby_places(category, near=None, latitude=None,
longitude=None, radius_m=None, limit=None)``, which finds nearby points of
interest (restaurants, cafes, pharmacies, ATMs, …) via **Overpass**. ``near``
accepts a place name that is geocoded internally via **Nominatim**, a coordinate
string, a map URL with coordinates, an OpenStreetMap object URL, or the caller
can pass explicit ``latitude`` / ``longitude``.

Passing ``place_details=True`` switches the tool into a place-lookup mode: rather
than searching for POIs, it returns rich structured info *about* the place named
in ``near`` or at the supplied coordinates (coordinates, bounding box, address
breakdown, population, wikidata/wikipedia links, website) — the "where/what is
X" question.

By default it uses the public OpenStreetMap APIs. Set ``GEO_NOMINATIM_URL`` /
``GEO_OVERPASS_URL`` to self-host. Nominatim's usage policy (a descriptive
User-Agent and a 1 req/sec cap on the public API) is honored — see config.py.
"""

import logging
import math
import re
import time
import asyncio
from typing import Annotated
from urllib.parse import parse_qs, unquote, urlparse

import anyio
import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from config import geocoding_settings as cfg
from .cache import TTLCache
from .serialize import to_json, log_call, log_result, redact_secrets

log = logging.getLogger(__name__)

# Reuse async clients per event loop to keep API calls from paying repeated
# connection setup. The client class id keeps tests that monkeypatch
# httpx.AsyncClient isolated from clients created under earlier patches.
_http_clients: dict[tuple[int, int], httpx.AsyncClient] = {}


def _backend_error(exc: Exception, endpoint: str) -> str:
    """Format an HTTP error without exposing basic-auth URL credentials."""
    try:
        password = urlparse(endpoint).password or ""
    except ValueError:
        password = ""
    return redact_secrets(exc, password)


def _http_client() -> httpx.AsyncClient:
    key = (id(asyncio.get_running_loop()), id(httpx.AsyncClient))
    client = _http_clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient()
        _http_clients[key] = client
    return client

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake a failure for a
# real place. A valid-but-empty result (a search that found nothing) is NOT a
# failure — it is returned as normal output. See the README "Error handling".

# Place data changes slowly and agent loops re-ask the same lookups, so the
# finished responses are cached. See the README "Caching" section.
_cache = TTLCache(cfg.cache_ttl_seconds, cfg.cache_max_entries)

# Both OpenStreetMap backends (Nominatim and Overpass) rate-limit aggressive
# callers (Nominatim caps at ~1 req/sec; Overpass returns 429/504 when its few
# query slots are saturated). A model can fire several find_nearby_places calls at
# once, so a single shared limiter serializes every request to either backend
# through a lock and spaces them by the configured interval. anyio.Lock is
# FIFO-fair, so concurrent callers are effectively queued and dispatched in
# arrival order — a burst waits its turn instead of stampeding the APIs. anyio
# primitives keep this on the event loop without blocking it. The limiter is
# module-global so the spacing holds across every concurrent tool invocation.
class _RateLimiter:
    """Serializes calls and spaces consecutive ones by `interval` seconds."""

    def __init__(self) -> None:
        self._lock = anyio.Lock()
        self._last_call = 0.0

    async def acquire(self, interval: float) -> None:
        """Block until at least `interval` seconds have passed since the previous
        acquire (no-op when interval <= 0, e.g. when self-hosting)."""
        if interval <= 0:
            return
        async with self._lock:
            wait = self._last_call + interval - time.monotonic()
            if wait > 0:
                await anyio.sleep(wait)
            self._last_call = time.monotonic()


_osm_limiter = _RateLimiter()


# --------------------------- category mapping ---------------------------

# Natural-language category -> Overpass tag filters (OR'd together). Keys are
# matched as substrings of the (diet-stripped) category, longest first, so a
# small model can pass everyday words instead of OSM tag syntax. Anything not
# matched here falls back to a name search (see _build_filters).
_CATEGORY_FILTERS: dict[str, list[str]] = {
    "fast food": ['["amenity"="fast_food"]'],
    "restaurant": ['["amenity"="restaurant"]'],
    "cafe": ['["amenity"="cafe"]'],
    "coffee": ['["amenity"="cafe"]'],
    "bakery": ['["shop"="bakery"]'],
    "pizza": ['["amenity"~"restaurant|fast_food"]["cuisine"~"pizza"]'],
    "bar": ['["amenity"="bar"]', '["amenity"="pub"]'],
    "pub": ['["amenity"="pub"]'],
    "nightclub": ['["amenity"="nightclub"]'],
    "supermarket": ['["shop"="supermarket"]'],
    "grocery": ['["shop"~"supermarket|convenience|greengrocer"]'],
    "convenience store": ['["shop"="convenience"]'],
    "mall": ['["shop"="mall"]'],
    "shopping": ['["shop"="mall"]', '["shop"="department_store"]'],
    "clothes": ['["shop"="clothes"]'],
    "clothing": ['["shop"="clothes"]'],
    "bookstore": ['["shop"="books"]'],
    "book shop": ['["shop"="books"]'],
    "hardware": ['["shop"="hardware"]', '["shop"="doityourself"]'],
    "hairdresser": ['["shop"="hairdresser"]'],
    "barber": ['["shop"="hairdresser"]'],
    "salon": ['["shop"="hairdresser"]', '["shop"="beauty"]'],
    "laundry": ['["shop"="laundry"]', '["amenity"="laundry"]'],
    "pharmacy": ['["amenity"="pharmacy"]'],
    "drugstore": ['["amenity"="pharmacy"]'],
    "hospital": ['["amenity"="hospital"]'],
    "clinic": ['["amenity"~"clinic|doctors"]'],
    "doctor": ['["amenity"="doctors"]'],
    "dentist": ['["amenity"="dentist"]'],
    "veterinary": ['["amenity"="veterinary"]'],
    "vet": ['["amenity"="veterinary"]'],
    "atm": ['["amenity"="atm"]'],
    "bank": ['["amenity"="bank"]'],
    "fuel": ['["amenity"="fuel"]'],
    "gas station": ['["amenity"="fuel"]'],
    "petrol": ['["amenity"="fuel"]'],
    "charging station": ['["amenity"="charging_station"]'],
    "ev charging": ['["amenity"="charging_station"]'],
    "parking": ['["amenity"="parking"]'],
    "toilet": ['["amenity"="toilets"]'],
    "restroom": ['["amenity"="toilets"]'],
    "bathroom": ['["amenity"="toilets"]'],
    "park": ['["leisure"="park"]'],
    "playground": ['["leisure"="playground"]'],
    "gym": ['["leisure"="fitness_centre"]', '["amenity"="gym"]'],
    "fitness": ['["leisure"="fitness_centre"]'],
    "swimming pool": ['["leisure"="swimming_pool"]', '["leisure"="water_park"]'],
    "beach": ['["natural"="beach"]'],
    "hotel": ['["tourism"~"hotel|guest_house|motel"]'],
    "lodging": ['["tourism"~"hotel|guest_house|motel|hostel"]'],
    "hostel": ['["tourism"="hostel"]'],
    "museum": ['["tourism"="museum"]'],
    "attraction": ['["tourism"="attraction"]'],
    "viewpoint": ['["tourism"="viewpoint"]'],
    "library": ['["amenity"="library"]'],
    "school": ['["amenity"="school"]'],
    "university": ['["amenity"="university"]'],
    "police": ['["amenity"="police"]'],
    "post office": ['["amenity"="post_office"]'],
    "cinema": ['["amenity"="cinema"]'],
    "movie theater": ['["amenity"="cinema"]'],
    "theatre": ['["amenity"="theatre"]'],
    "place of worship": ['["amenity"="place_of_worship"]'],
    "church": ['["amenity"="place_of_worship"]["religion"="christian"]'],
    "mosque": ['["amenity"="place_of_worship"]["religion"="muslim"]'],
    "synagogue": ['["amenity"="place_of_worship"]["religion"="jewish"]'],
    "temple": ['["amenity"="place_of_worship"]'],
    "bus stop": ['["highway"="bus_stop"]', '["amenity"="bus_station"]'],
    "train station": ['["railway"="station"]'],
    "subway": ['["station"="subway"]', '["railway"="station"]["subway"="yes"]'],
    "metro": ['["station"="subway"]', '["railway"="station"]["subway"="yes"]'],
    "airport": ['["aeroway"="aerodrome"]'],
}

# Categories sorted longest-key-first so multi-word keys ("fast food") win over
# the shorter substrings they contain ("food").
_CATEGORY_ITEMS = sorted(_CATEGORY_FILTERS.items(), key=lambda kv: -len(kv[0]))

# Food-place defaults used when only a diet is given (e.g. "vegan").
_FOOD_FILTERS = ['["amenity"~"restaurant|cafe|fast_food"]']

# POI tag keys a name/brand search (e.g. "Starbucks") is constrained to. A bare
# `["name"~"…",i]` regex over node+way+relation has no indexed key for Overpass
# to narrow on, so it must regex-match the name of EVERY element in the radius —
# in a dense area that scans hundreds of thousands of elements and blows past the
# query's own [timeout:N], returning nothing. Pairing the name regex with an
# indexed key existence check (`["name"~"…",i]["amenity"]`) lets Overpass select
# the small set of POIs first and regex only those, turning a 76s timeout into a
# ~3s answer. Kept to the four keys that cover essentially every consumer-facing
# named POI a brand search targets — each one adds a sub-query, and the extra
# cost of rarer keys (office/craft) pushed the query back past the timeout.
_NAME_SEARCH_KEYS = ("amenity", "shop", "tourism", "leisure")

# Diet modifiers -> the OSM diet:* tag they imply.
_DIET_TAGS: dict[str, str] = {
    "vegan": '["diet:vegan"~"yes|only"]',
    "vegetarian": '["diet:vegetarian"~"yes|only"]',
    "halal": '["diet:halal"~"yes|only"]',
    "kosher": '["diet:kosher"~"yes|only"]',
    "gluten free": '["diet:gluten_free"~"yes|only"]',
    "gluten-free": '["diet:gluten_free"~"yes|only"]',
}

# Tag keys, in priority order, used to label a result's primary category.
_PRIMARY_KEYS = (
    "amenity", "shop", "tourism", "leisure", "healthcare",
    "craft", "office", "natural", "railway", "highway", "aeroway",
)


def _build_filters(category: str) -> list[str]:
    """Translate a natural-language category into Overpass tag filters.

    Recognizes a diet modifier (vegan/vegetarian/halal/kosher/gluten-free) and a
    base category from _CATEGORY_FILTERS; falls back to a name search for
    anything unknown (e.g. a brand like "Starbucks").
    """
    text = (category or "").lower().strip()

    diet_tag = None
    for word, tag in _DIET_TAGS.items():
        if word in text:
            diet_tag = tag
            text = text.replace(word, " ").strip()
            break

    base_filters = None
    for key, filters in _CATEGORY_ITEMS:
        if key in text:
            base_filters = filters
            break

    if base_filters is None:
        if diet_tag:
            base_filters = _FOOD_FILTERS  # a bare diet implies a place to eat
        else:
            # Unknown category: search POI names instead. Strip characters that
            # could break out of the Overpass quoted string (prevents injection).
            term = re.sub(r'["\\]', "", category or "").strip()
            if not term:
                return []
            # Constrain the name regex to elements carrying an indexed POI key so
            # Overpass narrows by that key before regex-matching names, instead of
            # scanning every element in the radius (which times out). See
            # _NAME_SEARCH_KEYS.
            return [f'["name"~"{term}",i]["{key}"]' for key in _NAME_SEARCH_KEYS]

    if diet_tag:
        return [bf + diet_tag for bf in base_filters]
    return base_filters


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in meters."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _clamp(value, default: int, maximum: int) -> int:
    """Clamp a model-supplied count into [1, maximum]; None uses the default."""
    if value is None:
        return min(default, maximum)
    try:
        v = int(value)
    except (TypeError, ValueError):
        return min(default, maximum)
    return max(1, min(v, maximum))


def _compose_address(tags: dict) -> str | None:
    """Build a short street address from OSM addr:* tags, if any are present."""
    house = tags.get("addr:housenumber")
    street = tags.get("addr:street")
    city = tags.get("addr:city")
    line = " ".join(p for p in [house, street] if p).strip()
    parts = [p for p in [line or None, city] if p]
    return ", ".join(parts) if parts else None


def _primary_category(tags: dict) -> str | None:
    for key in _PRIMARY_KEYS:
        if key in tags:
            return tags[key]
    return None


# OSM place values treated as a "town" worth re-centering a follow-up search on.
# Suburbs/hamlets are excluded: a suburb isn't a separate municipality, and a
# hamlet is usually too small to host the kind of POI a model is hunting for.
_TOWN_PLACE_TYPES = "city|town|village"


def _parse_population(value) -> int | None:
    """Coerce an OSM population tag (a string, sometimes comma-grouped) to an int.
    Returns None for missing/unparseable values so the field is simply omitted."""
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# ------------------------------- Nominatim -------------------------------

# Relative location references a model may pass as `near`, echoing a user's
# "near me" / "near here". This server has no user location, so these can't be
# geocoded — Nominatim would just fail to find a place called "me". We catch them
# up front and return an actionable error instead of that opaque failure.
_RELATIVE_LOCATION_TERMS = frozenset({
    "me", "here", "near me", "near here", "nearby", "near by", "around me",
    "around here", "close to me", "close by", "my location",
    "my current location", "current location", "this location", "my position",
    "current position", "where i am", "where i m", "my area", "my city",
    "you", "your location", "your area",
})


def _is_relative_location(near: str) -> bool:
    """True if `near` is a relative reference (me/here/nearby/…) rather than an
    actual place. Strips punctuation and collapses whitespace before matching so
    quoting and spacing don't slip a variant past the check."""
    cleaned = " ".join(re.sub(r"[^\w\s]", " ", near.lower()).split())
    return cleaned in _RELATIVE_LOCATION_TERMS


def _parse_bbox(box) -> dict | None:
    """Nominatim's `boundingbox` is [south, north, west, east] as strings.
    Coerce to a labelled dict of floats; return None if it's missing/malformed."""
    if not box or len(box) != 4:
        return None
    try:
        south, north, west, east = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return {"south": south, "north": north, "west": west, "east": east}


def _parse_float(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _valid_lat_lon(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _parse_coordinates(text: str | None) -> tuple[float, float] | None:
    """Extract latitude/longitude from plain text or common map URLs.

    Supports values a model is likely to lift from map search results:
    ``45.515118,-122.679485``, ``45.515118 -122.679485``, Apple Maps ``ll=``,
    and OpenStreetMap ``#map=zoom/lat/lon`` fragments.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    query = parse_qs(parsed.query)
    for key in ("ll", "sll"):
        values = query.get(key)
        if values:
            coords = _parse_coordinates(values[0])
            if coords:
                return coords

    fragment = unquote(parsed.fragment or "")
    match = re.search(
        r"(?:^|/)map=\d+(?:\.\d+)?/([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)",
        fragment,
    )
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if _valid_lat_lon(lat, lon):
            return lat, lon

    match = re.search(
        r"(?<![-+\d.])([-+]?\d+(?:\.\d+)?)\s*[,/ ]\s*([-+]?\d+(?:\.\d+)?)(?![\d.])",
        raw,
    )
    if not match:
        return None
    lat, lon = float(match.group(1)), float(match.group(2))
    if _valid_lat_lon(lat, lon):
        return lat, lon
    return None


def _parse_osm_object_url(text: str | None) -> tuple[str, str] | None:
    """Extract a Nominatim lookup id from openstreetmap.org node/way/relation URLs."""
    raw = (text or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host not in {"openstreetmap.org", "www.openstreetmap.org"}:
        return None
    match = re.search(r"/(node|way|relation)/(\d+)(?:/|$)", parsed.path)
    if not match:
        return None
    prefix = {"node": "N", "way": "W", "relation": "R"}[match.group(1)]
    return prefix, match.group(2)


def _format_place(entry: dict) -> dict:
    """Build the rich place-details payload from a detailed `_geocode` entry.

    Folds the most useful `extratags`/`namedetails` Nominatim returns (population,
    wikidata/wikipedia, website, opening hours, elevation, official name) up into
    flat fields, dropping anything absent so the response stays compact.
    """
    extra = entry.get("extratags") or {}
    names = entry.get("namedetails") or {}
    place = {
        "name": entry.get("name"),
        "latitude": entry.get("latitude"),
        "longitude": entry.get("longitude"),
        "class": entry.get("category"),
        "type": entry.get("type"),
        "address": entry.get("address"),
        "bounding_box": entry.get("bounding_box"),
        "population": _parse_population(extra.get("population")),
        "wikidata": extra.get("wikidata"),
        "wikipedia": extra.get("wikipedia"),
        "website": extra.get("website") or extra.get("url") or extra.get("contact:website"),
        "phone": extra.get("phone") or extra.get("contact:phone"),
        "opening_hours": extra.get("opening_hours"),
        "elevation_m": _parse_float(extra.get("ele")),
        "official_name": names.get("official_name") or names.get("name:en"),
        "importance": entry.get("importance"),
        "osm_type": entry.get("osm_type"),
        "osm_id": entry.get("osm_id"),
    }
    return {k: v for k, v in place.items() if v not in (None, {}, "")}


async def _geocode(query: str, limit: int, detailed: bool = False) -> list[dict]:
    """Geocode a free-text query via Nominatim. Returns a (possibly empty) list
    of matches. Raises ToolError only on a real failure (network / bad status).

    When ``detailed`` is set, requests Nominatim's ``extratags``/``namedetails``
    and surfaces the bounding box, importance, and OSM id so a place-details
    lookup can report population, wikidata/wikipedia links, etc. The basic
    (non-detailed) shape is unchanged — the nearby-search path relies on it."""
    q = (query or "").strip()
    if not q:
        raise ToolError("Empty query. Provide a place name or address to geocode.")

    cache_key = "geocode\x00" + "\x00".join(
        (q.lower(), str(limit), cfg.language, "d" if detailed else "")
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "q": q,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
    }
    if detailed:
        params["extratags"] = 1
        params["namedetails"] = 1
    if cfg.nominatim_email.strip():
        params["email"] = cfg.nominatim_email.strip()

    url = cfg.nominatim_url.rstrip("/") + "/search"
    headers = {"User-Agent": cfg.user_agent, "Accept-Language": cfg.language}

    await _osm_limiter.acquire(cfg.min_request_interval_seconds)
    try:
        client = _http_client()
        resp = await client.get(
            url,
            params=params,
            headers=headers,
            timeout=cfg.http_timeout_seconds,
        )
    except httpx.TimeoutException:
        raise ToolError(f"Geocoding request timed out after {cfg.http_timeout_seconds}s.")
    except httpx.HTTPError as exc:
        raise ToolError(
            f"Network error contacting the geocoder: {_backend_error(exc, url)}"
        )

    if resp.status_code == 429:
        raise ToolError(
            "The geocoder rate-limited the request (HTTP 429). The public "
            "Nominatim API allows ~1 request/second; slow down or self-host."
        )
    if resp.status_code >= 400:
        raise ToolError(
            f"Geocoder error (HTTP {resp.status_code}): {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError:
        raise ToolError("Geocoder returned a non-JSON response.")
    if not isinstance(data, list):
        raise ToolError("Geocoder returned an unexpected JSON response shape.")

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        entry = {
            "name": item.get("display_name"),
            "latitude": lat,
            "longitude": lon,
            "category": item.get("category") or item.get("class"),
            "type": item.get("type"),
            "address": item.get("address"),
        }
        if detailed:
            entry["bounding_box"] = _parse_bbox(item.get("boundingbox"))
            entry["extratags"] = item.get("extratags") or {}
            entry["namedetails"] = item.get("namedetails") or {}
            entry["importance"] = item.get("importance")
            entry["osm_type"] = item.get("osm_type")
            entry["osm_id"] = item.get("osm_id")
        results.append(entry)

    _cache.set(cache_key, results)
    return results


async def _reverse_geocode(lat: float, lon: float, detailed: bool = False) -> dict:
    """Reverse-geocode coordinates via Nominatim."""
    if not _valid_lat_lon(lat, lon):
        raise ToolError(
            f"Invalid coordinates: latitude {lat!r}, longitude {lon!r}. "
            "Latitude must be -90..90 and longitude -180..180."
        )

    cache_key = "reverse_geocode\x00" + "\x00".join(
        (f"{lat:.7f}", f"{lon:.7f}", cfg.language, "d" if detailed else "")
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "lat": f"{lat:.7f}",
        "lon": f"{lon:.7f}",
        "format": "jsonv2",
        "addressdetails": 1,
    }
    if detailed:
        params["extratags"] = 1
        params["namedetails"] = 1
    if cfg.nominatim_email.strip():
        params["email"] = cfg.nominatim_email.strip()

    url = cfg.nominatim_url.rstrip("/") + "/reverse"
    headers = {"User-Agent": cfg.user_agent, "Accept-Language": cfg.language}

    await _osm_limiter.acquire(cfg.min_request_interval_seconds)
    try:
        client = _http_client()
        resp = await client.get(
            url,
            params=params,
            headers=headers,
            timeout=cfg.http_timeout_seconds,
        )
    except httpx.TimeoutException:
        raise ToolError(
            f"Reverse geocoding request timed out after {cfg.http_timeout_seconds}s."
        )
    except httpx.HTTPError as exc:
        raise ToolError(
            f"Network error contacting the geocoder: {_backend_error(exc, url)}"
        )

    if resp.status_code == 429:
        raise ToolError(
            "The geocoder rate-limited the request (HTTP 429). The public "
            "Nominatim API allows ~1 request/second; slow down or self-host."
        )
    if resp.status_code >= 400:
        raise ToolError(
            f"Geocoder error (HTTP {resp.status_code}): {resp.text[:200]}"
        )

    try:
        item = resp.json()
    except ValueError:
        raise ToolError("Geocoder returned a non-JSON response.")
    if not isinstance(item, dict):
        raise ToolError("Geocoder returned an unexpected JSON response shape.")

    if item.get("error"):
        raise ToolError(f"Could not reverse-geocode {lat:.7f}, {lon:.7f}: {item['error']}")

    entry = {
        "name": item.get("display_name"),
        "latitude": _parse_float(item.get("lat")) or lat,
        "longitude": _parse_float(item.get("lon")) or lon,
        "category": item.get("category") or item.get("class"),
        "type": item.get("type"),
        "address": item.get("address"),
    }
    if detailed:
        entry["bounding_box"] = _parse_bbox(item.get("boundingbox"))
        entry["extratags"] = item.get("extratags") or {}
        entry["namedetails"] = item.get("namedetails") or {}
        entry["importance"] = item.get("importance")
        entry["osm_type"] = item.get("osm_type")
        entry["osm_id"] = item.get("osm_id")

    _cache.set(cache_key, entry)
    return entry


async def _lookup_osm_object(osm_type: str, osm_id: str, detailed: bool = False) -> dict:
    """Look up one OSM node/way/relation through Nominatim's lookup endpoint."""
    osm_ref = f"{osm_type}{osm_id}"
    cache_key = "lookup_osm_object\x00" + "\x00".join(
        (osm_ref, cfg.language, "d" if detailed else "")
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "osm_ids": osm_ref,
        "format": "jsonv2",
        "addressdetails": 1,
    }
    if detailed:
        params["extratags"] = 1
        params["namedetails"] = 1
    if cfg.nominatim_email.strip():
        params["email"] = cfg.nominatim_email.strip()

    url = cfg.nominatim_url.rstrip("/") + "/lookup"
    headers = {"User-Agent": cfg.user_agent, "Accept-Language": cfg.language}

    await _osm_limiter.acquire(cfg.min_request_interval_seconds)
    try:
        client = _http_client()
        resp = await client.get(
            url,
            params=params,
            headers=headers,
            timeout=cfg.http_timeout_seconds,
        )
    except httpx.TimeoutException:
        raise ToolError(f"OSM object lookup timed out after {cfg.http_timeout_seconds}s.")
    except httpx.HTTPError as exc:
        raise ToolError(
            f"Network error contacting the geocoder: {_backend_error(exc, url)}"
        )

    if resp.status_code == 429:
        raise ToolError(
            "The geocoder rate-limited the request (HTTP 429). The public "
            "Nominatim API allows ~1 request/second; slow down or self-host."
        )
    if resp.status_code >= 400:
        raise ToolError(
            f"Geocoder error (HTTP {resp.status_code}): {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError:
        raise ToolError("Geocoder returned a non-JSON response.")

    if not isinstance(data, list):
        raise ToolError("Geocoder returned an unexpected JSON response shape.")
    if not data:
        raise ToolError(f"Could not find OpenStreetMap object {osm_ref}.")
    item = data[0]
    if not isinstance(item, dict):
        raise ToolError(f"OpenStreetMap object {osm_ref} was malformed.")
    try:
        lat = float(item["lat"])
        lon = float(item["lon"])
    except (KeyError, TypeError, ValueError):
        raise ToolError(f"OpenStreetMap object {osm_ref} did not include coordinates.")

    entry = {
        "name": item.get("display_name"),
        "latitude": lat,
        "longitude": lon,
        "category": item.get("category") or item.get("class"),
        "type": item.get("type"),
        "address": item.get("address"),
    }
    if detailed:
        entry["bounding_box"] = _parse_bbox(item.get("boundingbox"))
        entry["extratags"] = item.get("extratags") or {}
        entry["namedetails"] = item.get("namedetails") or {}
        entry["importance"] = item.get("importance")
        entry["osm_type"] = item.get("osm_type")
        entry["osm_id"] = item.get("osm_id")

    _cache.set(cache_key, entry)
    return entry


async def _place_lookup(query: str) -> dict:
    """Resolve a place name/address to rich structured info (the top match) plus
    a few lighter alternatives. Raises ToolError when nothing matches."""
    matches = await _geocode(query, cfg.max_place_matches, detailed=True)
    if not matches:
        raise ToolError(
            f"Could not find a place matching '{query}'. Try a more specific "
            "name or a full address."
        )
    payload: dict = {"query": query, "place": _format_place(matches[0])}
    alternatives = [
        {
            "name": m.get("name"),
            "latitude": m.get("latitude"),
            "longitude": m.get("longitude"),
            "type": m.get("type"),
        }
        for m in matches[1:]
    ]
    if alternatives:
        payload["alternatives"] = alternatives
    return payload


async def _place_lookup_coords(lat: float, lon: float) -> dict:
    entry = await _reverse_geocode(lat, lon, detailed=True)
    return {"query": f"{lat:.7f},{lon:.7f}", "place": _format_place(entry)}


async def _place_lookup_osm_object(osm_type: str, osm_id: str) -> dict:
    entry = await _lookup_osm_object(osm_type, osm_id, detailed=True)
    return {"query": f"{osm_type}{osm_id}", "place": _format_place(entry)}


# -------------------------------- Overpass -------------------------------

async def _overpass(query_ql: str) -> list[dict]:
    """Run an Overpass QL query and return its `elements`. Raises ToolError on
    failure; an empty element list is returned as-is (valid-but-empty)."""
    cache_key = "overpass\x00" + query_ql
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    # Space requests so a burst of concurrent tool calls is queued rather than
    # stampeding Overpass (which answers a flood with 429/504). Shares the single
    # OpenStreetMap limiter with Nominatim. Done after the cache check so a cache
    # hit doesn't needlessly consume the rate budget.
    await _osm_limiter.acquire(cfg.min_request_interval_seconds)

    headers = {"User-Agent": cfg.user_agent}
    try:
        client = _http_client()
        resp = await client.post(
            cfg.overpass_url,
            data={"data": query_ql},
            headers=headers,
            timeout=cfg.overpass_timeout_seconds,
        )
    except httpx.TimeoutException:
        raise ToolError(
            f"Overpass request timed out after {cfg.overpass_timeout_seconds}s. "
            "Try a smaller radius or a more specific category."
        )
    except httpx.HTTPError as exc:
        raise ToolError(
            "Network error contacting Overpass: "
            f"{_backend_error(exc, cfg.overpass_url)}"
        )

    if resp.status_code == 429:
        raise ToolError(
            "Overpass rate-limited the request (HTTP 429). Wait a moment and "
            "retry, or point GEO_OVERPASS_URL at your own instance."
        )
    if resp.status_code in (504, 502, 503):
        raise ToolError(
            f"Overpass is overloaded (HTTP {resp.status_code}). Retry shortly or "
            "narrow the query (smaller radius / more specific category)."
        )
    if resp.status_code >= 400:
        raise ToolError(f"Overpass error (HTTP {resp.status_code}): {resp.text[:200]}")

    try:
        data = resp.json()
    except ValueError:
        # Overpass emits plain-text error pages for malformed queries.
        raise ToolError(f"Overpass returned a non-JSON response: {resp.text[:200]}")
    if not isinstance(data, dict):
        raise ToolError("Overpass returned an unexpected JSON response shape.")

    remark = data.get("remark", "")
    if remark and "error" in remark.lower():
        raise ToolError(f"Overpass reported an error: {remark.strip()}")

    elements = data.get("elements", [])
    if not isinstance(elements, list):
        raise ToolError("Overpass returned a non-list `elements` value.")
    _cache.set(cache_key, elements)
    return elements


async def _nearby_towns(
    lat: float, lon: float, n: int, exclude: str | None
) -> list[dict]:
    """Find populated places (city/town/village) within nearby_towns_radius_m of
    the center, nearest first, so a model can launch follow-up searches centered on
    neighboring towns. Each carries its distance from the original center so the
    model can judge relevance. `exclude` (the center's display name) drops the
    origin place itself, which would otherwise appear at ~0 m as its own neighbor.
    """
    radius = cfg.nearby_towns_radius_m
    query_ql = (
        f"[out:json][timeout:{int(cfg.overpass_timeout_seconds)}];\n"
        f'node["place"~"^({_TOWN_PLACE_TYPES})$"]'
        f"(around:{radius},{lat:.7f},{lon:.7f});\n"
        f"out center tags {cfg.overpass_max_elements};"
    )
    elements = await _overpass(query_ql)

    # `town in display_name` only excludes names that are substrings of the center
    # label (e.g. "Sacramento" out of "Sacramento, …"), so a distinct neighbor like
    # "West Sacramento" — not a substring — is correctly kept.
    exclude_l = (exclude or "").lower()
    towns = []
    for el in elements:
        if el.get("type") == "node":
            e_lat, e_lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center") or {}
            e_lat, e_lon = center.get("lat"), center.get("lon")
        if e_lat is None or e_lon is None:
            continue

        tags = el.get("tags") or {}
        name = tags.get("name")
        if not name:
            continue
        if exclude_l and name.lower() in exclude_l:
            continue

        dist = _haversine_m(lat, lon, e_lat, e_lon)
        town = {
            "name": name,
            "latitude": e_lat,
            "longitude": e_lon,
            "distance_m": int(round(dist)),
            "place_type": tags.get("place"),
            "population": _parse_population(tags.get("population")),
        }
        towns.append({k: v for k, v in town.items() if v is not None})

    towns.sort(key=lambda t: t["distance_m"])
    return towns[:n]


# The tool's model-facing description and the per-arg cap descriptions are built
# at registration time (below) from `cfg`, so the configured caps/defaults appear
# as concrete numbers the model can target. The schemas list the meaningful keys
# only (echoed inputs like query_category are omitted); plain (non-f) strings keep
# the JSON braces from needing escaping.
_RETURN_SCHEMA = (
    "Search returns {center:{latitude,longitude,name?},count,results:[{name,"
    "latitude,longitude,distance_m,category,cuisine?,address?,opening_hours?,"
    "phone?,website?}],nearby_towns?:[{name,latitude,longitude,distance_m,"
    "place_type?,population?}]}"
)

_PLACE_RETURN_SCHEMA = (
    "place_details returns {place:{name,latitude,longitude,class,type,address,"
    "bounding_box?,population?,wikidata?,wikipedia?,website?,phone?,opening_hours?,"
    "elevation_m?},alternatives?:[{name,latitude,longitude,type}]}"
)


def register(mcp: FastMCP) -> None:
    description = (
        "Find places via OpenStreetMap. Default: search POIs of `category` near a "
        'location — near="city/address", near="lat,lon", map URL with coords, '
        "OpenStreetMap node/way/relation URL, OR latitude+longitude (no "
        '"near me"; ask the user). category is plain language (restaurant, coffee, '
        "pharmacy, atm, "
        "hotel, park, gym, …); prefix food with a diet (vegan/vegetarian/halal/"
        'kosher/gluten-free); unknown words match names ("Starbucks"). '
        f"include_nearby_towns=true also lists towns within {cfg.nearby_towns_radius_m} m "
        "to recenter a follow-up search.\n"
        "place_details=true: ignore category/radius/limit and return rich info "
        "ABOUT the place in `near` or at latitude+longitude (coords, bounding "
        "box, address, population, wikidata/wikipedia, website) — answers "
        "'where/what is X'.\n"
        f"{_RETURN_SCHEMA}\n{_PLACE_RETURN_SCHEMA}"
    )

    @mcp.tool(description=description)
    async def find_nearby_places(
        category: str = "",
        near: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_m: Annotated[
            int | None,
            Field(
                description=f"Search radius in meters. Default "
                f"{cfg.default_radius_m}, max {cfg.max_radius_m} (larger is clamped)."
            ),
        ] = None,
        limit: Annotated[
            int | None,
            Field(
                description=f"Max results, nearest-first. Default "
                f"{cfg.default_nearby_results}, max {cfg.max_nearby_results} "
                "(larger is clamped)."
            ),
        ] = None,
        include_nearby_towns: bool = False,
        nearby_towns_limit: Annotated[
            int | None,
            Field(
                description="Max nearby towns, nearest-first. Default & max "
                f"{cfg.max_nearby_towns} (larger is clamped). Needs "
                "include_nearby_towns=true."
            ),
        ] = None,
        place_details: bool = False,
    ) -> str:
        """Find nearby places via OpenStreetMap. The model-facing guidance lives in
        the @mcp.tool(description=...) above (built with the live caps from cfg).

        :param category: What to find (plain language). Ignored when place_details=true.
        :param near: Place/address (geocoded), "lat,lon", or map URL with coords.
        :param latitude: Coords (with longitude).
        :param longitude: Coords (with latitude).
        :param include_nearby_towns: Also return surrounding towns to recenter on.
        :param place_details: Look up rich info about the place in `near` instead
            of searching for nearby POIs.
        """
        log_call(
            log,
            "find_nearby_places",
            category=category,
            near=near,
            latitude=latitude,
            longitude=longitude,
            radius_m=radius_m,
            limit=limit,
            include_nearby_towns=include_nearby_towns,
            nearby_towns_limit=nearby_towns_limit,
            place_details=place_details,
        )

        if (latitude is None) != (longitude is None):
            raise ToolError("Provide both `latitude` and `longitude`, or neither.")

        # place_details mode: look up rich info ABOUT a place rather than POIs
        # around it. Accepts a name/address or coordinates, including coordinates
        # embedded in common map URLs returned by map search engines.
        if place_details:
            coords = None
            osm_object = None
            if latitude is not None and longitude is not None:
                coords = (float(latitude), float(longitude))
            elif near and near.strip():
                coords = _parse_coordinates(near)
                osm_object = _parse_osm_object_url(near) if not coords else None

            if coords:
                payload = await _place_lookup_coords(*coords)
                return log_result(log, "find_nearby_places", to_json(payload))
            if osm_object:
                payload = await _place_lookup_osm_object(*osm_object)
                return log_result(log, "find_nearby_places", to_json(payload))

            if not (near and near.strip()):
                raise ToolError(
                    "place_details lookup needs a place name/address in `near`, "
                    "a coordinate string like '45.515,-122.679', or "
                    "`latitude` and `longitude`."
                )
            if _is_relative_location(near):
                raise ToolError(
                    f"'{near.strip()}' is a relative location, and this server has "
                    "no access to the user's location. Pass an actual place name "
                    "or address as `near`."
                )
            payload = await _place_lookup(near.strip())
            return log_result(log, "find_nearby_places", to_json(payload))

        if not (category or "").strip():
            raise ToolError("Empty category. Say what to look for, e.g. 'pharmacy'.")

        # Resolve the search center.
        center_name = None
        if latitude is not None and longitude is not None:
            lat, lon = float(latitude), float(longitude)
            if not _valid_lat_lon(lat, lon):
                raise ToolError(
                    f"Invalid coordinates: latitude {lat!r}, longitude {lon!r}. "
                    "Latitude must be -90..90 and longitude -180..180."
                )
        elif near and near.strip():
            coords = _parse_coordinates(near)
            if coords:
                lat, lon = coords
            elif osm_object := _parse_osm_object_url(near):
                match = await _lookup_osm_object(*osm_object)
                lat = match["latitude"]
                lon = match["longitude"]
                center_name = match["name"]
            elif _is_relative_location(near):
                raise ToolError(
                    f"'{near.strip()}' is a relative location, and this server "
                    "has no access to the user's location. Ask the user which "
                    "place to search and pass it as `near` (e.g. 'Portland, OR'), "
                    "or pass `latitude` and `longitude` directly."
                )
            else:
                matches = await _geocode(near, 1)
                if not matches:
                    raise ToolError(
                        f"Could not find a location for '{near.strip()}'. Try a more "
                        "specific place name, or pass latitude/longitude directly."
                    )
                lat = matches[0]["latitude"]
                lon = matches[0]["longitude"]
                center_name = matches[0]["name"]
        else:
            raise ToolError(
                "No location given. Provide `near` (a place name) or both "
                "`latitude` and `longitude`."
            )

        radius = _clamp(radius_m, cfg.default_radius_m, cfg.max_radius_m)
        n = _clamp(limit, cfg.default_nearby_results, cfg.max_nearby_results)

        filters = _build_filters(category)
        if not filters:
            raise ToolError(f"Could not interpret the category '{category}'.")

        # Build the Overpass query: every filter against nodes, ways, and
        # relations within the radius, via the `nwr` shorthand. `out center tags`
        # yields a center point for ways/relations so each result has coordinates.
        # `nwr` (one statement per filter) is used instead of three separate
        # node/way/relation statements deliberately: they're logically identical,
        # but for a name-regex filter the split form makes Overpass re-resolve the
        # `around` spatial set per element type and is dramatically slower — a
        # name search measured ~58s (past the timeout, returning nothing) split
        # vs ~3s as nwr. Exact-match category filters are fast either way, so nwr
        # is a safe win for both.
        lines = [
            f"  nwr{f}(around:{radius},{lat:.7f},{lon:.7f});" for f in filters
        ]
        query_ql = (
            f"[out:json][timeout:{int(cfg.overpass_timeout_seconds)}];\n"
            "(\n" + "\n".join(lines) + "\n);\n"
            f"out center tags {cfg.overpass_max_elements};"
        )

        elements = await _overpass(query_ql)

        results = []
        for el in elements:
            if el.get("type") == "node":
                e_lat, e_lon = el.get("lat"), el.get("lon")
            else:
                center = el.get("center") or {}
                e_lat, e_lon = center.get("lat"), center.get("lon")
            if e_lat is None or e_lon is None:
                continue

            tags = el.get("tags") or {}
            dist = _haversine_m(lat, lon, e_lat, e_lon)
            place = {
                "name": tags.get("name"),
                "latitude": e_lat,
                "longitude": e_lon,
                "distance_m": int(round(dist)),
                "category": _primary_category(tags),
                "cuisine": tags.get("cuisine"),
                "address": _compose_address(tags),
                "opening_hours": tags.get("opening_hours"),
                "phone": tags.get("phone") or tags.get("contact:phone"),
                "website": tags.get("website") or tags.get("contact:website"),
            }
            results.append({k: v for k, v in place.items() if v is not None})

        results.sort(key=lambda p: p["distance_m"])
        results = results[:n]

        payload = {
            "query_category": category.strip(),
            "center": {
                "latitude": lat,
                "longitude": lon,
                "name": center_name,
            },
            "radius_m": radius,
            "count": len(results),
            "results": results,
        }

        # Optional companion list of surrounding towns to seed follow-up searches.
        # Only added when asked, so the common case stays lean. max_nearby_towns
        # doubles as the default (omitting the count returns up to the cap).
        if include_nearby_towns:
            towns_n = _clamp(
                nearby_towns_limit, cfg.max_nearby_towns, cfg.max_nearby_towns
            )
            payload["nearby_towns_radius_m"] = cfg.nearby_towns_radius_m
            payload["nearby_towns"] = await _nearby_towns(
                lat, lon, towns_n, center_name
            )

        return log_result(log, "find_nearby_places", to_json(payload))
