"""
Place Search MCP tool (OpenStreetMap).

Exposes a single tool, ``find_nearby_places(category, near=None, latitude=None,
longitude=None, radius_m=None, limit=None)``, which finds nearby points of
interest (restaurants, cafes, pharmacies, ATMs, …) via **Overpass**. ``near``
accepts a place name that is geocoded internally via **Nominatim**, so a query
like "vegan restaurants in Portland" needs only one tool call; alternatively the
caller passes explicit ``latitude``/``longitude``.

By default it uses the public OpenStreetMap APIs. Set ``GEO_NOMINATIM_URL`` /
``GEO_OVERPASS_URL`` to self-host. Nominatim's usage policy (a descriptive
User-Agent and a 1 req/sec cap on the public API) is honored — see config.py.
"""

import logging
import math
import re
import time

import anyio
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import geocoding_settings as cfg
from .cache import TTLCache
from .serialize import to_json, log_call, log_result

log = logging.getLogger(__name__)

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake a failure for a
# real place. A valid-but-empty result (a search that found nothing) is NOT a
# failure — it is returned as normal output. See the README "Error handling".

# Place data changes slowly and agent loops re-ask the same lookups, so the
# finished responses are cached. See the README "Caching" section.
_cache = TTLCache(cfg.cache_ttl_seconds, cfg.cache_max_entries)

# Nominatim's public API permits at most one request per second. We serialize
# calls through this lock and space them by the configured interval so a burst of
# tool calls can't trip the rate limit. anyio primitives keep this on the event
# loop without blocking it.
_nominatim_lock = anyio.Lock()
_last_nominatim_call = 0.0


async def _throttle_nominatim() -> None:
    """Block until at least min_request_interval_seconds has passed since the
    last Nominatim call (no-op when the interval is 0, e.g. self-hosting)."""
    global _last_nominatim_call
    interval = cfg.min_request_interval_seconds
    if interval <= 0:
        return
    async with _nominatim_lock:
        wait = _last_nominatim_call + interval - time.monotonic()
        if wait > 0:
            await anyio.sleep(wait)
        _last_nominatim_call = time.monotonic()


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
            return [f'["name"~"{term}",i]']

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


async def _geocode(query: str, limit: int) -> list[dict]:
    """Geocode a free-text query via Nominatim. Returns a (possibly empty) list
    of matches. Raises ToolError only on a real failure (network / bad status)."""
    q = (query or "").strip()
    if not q:
        raise ToolError("Empty query. Provide a place name or address to geocode.")

    cache_key = "geocode\x00" + "\x00".join((q.lower(), str(limit), cfg.language))
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "q": q,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
    }
    if cfg.nominatim_email.strip():
        params["email"] = cfg.nominatim_email.strip()

    url = cfg.nominatim_url.rstrip("/") + "/search"
    headers = {"User-Agent": cfg.user_agent, "Accept-Language": cfg.language}

    await _throttle_nominatim()
    try:
        async with httpx.AsyncClient(timeout=cfg.http_timeout_seconds) as client:
            resp = await client.get(url, params=params, headers=headers)
    except httpx.TimeoutException:
        raise ToolError(f"Geocoding request timed out after {cfg.http_timeout_seconds}s.")
    except httpx.HTTPError as exc:
        raise ToolError(f"Network error contacting the geocoder: {exc}")

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

    results = []
    for item in data:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        results.append(
            {
                "name": item.get("display_name"),
                "latitude": lat,
                "longitude": lon,
                "category": item.get("category") or item.get("class"),
                "type": item.get("type"),
                "address": item.get("address"),
            }
        )

    _cache.set(cache_key, results)
    return results


# -------------------------------- Overpass -------------------------------

async def _overpass(query_ql: str) -> list[dict]:
    """Run an Overpass QL query and return its `elements`. Raises ToolError on
    failure; an empty element list is returned as-is (valid-but-empty)."""
    cache_key = "overpass\x00" + query_ql
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    headers = {"User-Agent": cfg.user_agent}
    try:
        async with httpx.AsyncClient(timeout=cfg.overpass_timeout_seconds) as client:
            resp = await client.post(
                cfg.overpass_url, data={"data": query_ql}, headers=headers
            )
    except httpx.TimeoutException:
        raise ToolError(
            f"Overpass request timed out after {cfg.overpass_timeout_seconds}s. "
            "Try a smaller radius or a more specific category."
        )
    except httpx.HTTPError as exc:
        raise ToolError(f"Network error contacting Overpass: {exc}")

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

    remark = data.get("remark", "")
    if remark and "error" in remark.lower():
        raise ToolError(f"Overpass reported an error: {remark.strip()}")

    elements = data.get("elements", [])
    _cache.set(cache_key, elements)
    return elements


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def find_nearby_places(
        category: str,
        near: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_m: int | None = None,
        limit: int | None = None,
    ) -> str:
        """
        Find points of interest near a location via OpenStreetMap — e.g. "vegan
        restaurants in Portland", "pharmacies in Berlin", "museums near the Louvre".

        Give the location ONE of two ways:
        - `near`: an explicit place name or address (geocoded for you), or
        - `latitude` + `longitude`: explicit coordinates (used if both given).

        This server has no access to the user's location, if the user
        says "near me", ask them where before calling, or pass coordinates.

        `category` is plain language, not OSM tags: restaurant, coffee, bar,
        supermarket, pharmacy, hospital, atm, bank, gas station, hotel, museum,
        park, gym, etc. Prefix food with a diet (vegan, vegetarian, halal, kosher,
        gluten free). Unknown categories match place names, so brands like
        "Starbucks" work too.

        :param category: What to look for, in plain language.
        :param near: Explicit place name/address to search around (geocoded for
            you).
        :param latitude: Search-center latitude (use with longitude).
        :param longitude: Search-center longitude (use with latitude).
        :param radius_m: Search radius in meters (capped); omit for default.
        :param limit: Max places to return (capped); sorted nearest-first.
        :return: JSON with resolved `center`, `radius_m`, and a nearest-first
            `results` list (name, coordinates, distance_m, category, and
            cuisine/address/phone/website/opening_hours when available).
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
        )
        if not (category or "").strip():
            raise ToolError("Empty category. Say what to look for, e.g. 'pharmacy'.")

        # Resolve the search center.
        center_name = None
        if latitude is not None and longitude is not None:
            lat, lon = float(latitude), float(longitude)
        elif near and near.strip():
            if _is_relative_location(near):
                raise ToolError(
                    f"'{near.strip()}' is a relative location, and this server "
                    "has no access to the user's location. Ask the user which "
                    "place to search and pass it as `near` (e.g. 'Portland, OR'), "
                    "or pass `latitude` and `longitude` directly."
                )
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
        # relations within the radius. `out center tags` yields a center point
        # for ways/relations so each result has coordinates.
        lines = []
        for f in filters:
            for elem in ("node", "way", "relation"):
                lines.append(f"  {elem}{f}(around:{radius},{lat:.7f},{lon:.7f});")
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

        return log_result(
            log,
            "find_nearby_places",
            to_json(
                {
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
            ),
        )
