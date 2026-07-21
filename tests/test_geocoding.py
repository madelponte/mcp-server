"""Tests for tools/geocoding.py — filters, math, clamping, and async I/O."""

import httpx
import pytest
from fastmcp.exceptions import ToolError

import tools.geocoding as geo
from tools.geocoding import (
    _build_filters,
    _NAME_SEARCH_KEYS,
    _haversine_m,
    _clamp,
    _compose_address,
    _primary_category,
    _parse_population,
    _parse_bbox,
    _parse_float,
    _parse_coordinates,
    _parse_osm_object_url,
    _format_place,
    _is_relative_location,
    _geocode,
    _place_lookup,
    _overpass,
)
from conftest import run


# --------------------------- _build_filters ---------------------------

def test_build_filters_known_category():
    assert _build_filters("restaurant") == ['["amenity"="restaurant"]']


def test_build_filters_longest_key_wins():
    # "fast food" must beat the shorter "food"-ish matches.
    assert _build_filters("fast food") == ['["amenity"="fast_food"]']


def test_build_filters_diet_plus_category():
    out = _build_filters("vegan restaurant")
    assert out == ['["amenity"="restaurant"]["diet:vegan"~"yes|only"]']


def test_build_filters_bare_diet_implies_food():
    out = _build_filters("vegan")
    assert out == ['["amenity"~"restaurant|cafe|fast_food"]["diet:vegan"~"yes|only"]']


def test_build_filters_unknown_falls_back_to_name_search():
    # A name/brand search pairs the name regex with each indexed POI key so
    # Overpass can narrow by that key first (a bare name regex over every element
    # in the radius times out). See _NAME_SEARCH_KEYS.
    out = _build_filters("Starbucks")
    assert out == [f'["name"~"Starbucks",i]["{key}"]' for key in _NAME_SEARCH_KEYS]
    # Every filter constrains on the name AND an indexed key, never name alone.
    assert all(f.startswith('["name"~"Starbucks",i]["') for f in out)


def test_build_filters_strips_injection_chars():
    out = _build_filters('Star"bucks\\')
    assert out == [f'["name"~"Starbucks",i]["{key}"]' for key in _NAME_SEARCH_KEYS]


def test_build_filters_empty_returns_empty():
    assert _build_filters("") == []
    assert _build_filters('"\\') == []


# --------------------------- _haversine_m ---------------------------

def test_haversine_zero_distance():
    assert _haversine_m(40.0, -73.0, 40.0, -73.0) == 0.0


def test_haversine_known_distance():
    # NYC to LA is ~3,936 km; allow generous tolerance.
    d = _haversine_m(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3_900_000 < d < 4_000_000


# --------------------------- _clamp ---------------------------

def test_clamp_none_uses_default():
    assert _clamp(None, default=8, maximum=20) == 8


def test_clamp_none_default_capped_to_max():
    assert _clamp(None, default=50, maximum=20) == 20


def test_clamp_value_within_range():
    assert _clamp(5, default=8, maximum=20) == 5


def test_clamp_value_above_max():
    assert _clamp(100, default=8, maximum=20) == 20


def test_clamp_value_below_one():
    assert _clamp(0, default=8, maximum=20) == 1
    assert _clamp(-3, default=8, maximum=20) == 1


def test_clamp_invalid_uses_default():
    assert _clamp("abc", default=8, maximum=20) == 8


# --------------------------- _compose_address ---------------------------

def test_compose_address_full():
    tags = {"addr:housenumber": "10", "addr:street": "Main St", "addr:city": "Townsville"}
    assert _compose_address(tags) == "10 Main St, Townsville"


def test_compose_address_partial():
    assert _compose_address({"addr:city": "Townsville"}) == "Townsville"
    assert _compose_address({"addr:street": "Main St"}) == "Main St"


def test_compose_address_none():
    assert _compose_address({}) is None


# --------------------------- _primary_category ---------------------------

def test_primary_category_priority():
    # amenity outranks shop.
    assert _primary_category({"shop": "books", "amenity": "cafe"}) == "cafe"


def test_primary_category_none():
    assert _primary_category({"foo": "bar"}) is None


# --------------------------- _parse_population ---------------------------

def test_parse_population_int():
    assert _parse_population(50000) == 50000


def test_parse_population_comma_string():
    assert _parse_population("1,234,567") == 1234567


def test_parse_population_none_and_invalid():
    assert _parse_population(None) is None
    assert _parse_population("abc") is None


# --------------------------- _parse_bbox / _parse_float ---------------------------

def test_parse_bbox_valid():
    # Nominatim order: [south, north, west, east]
    out = _parse_bbox(["48.81", "48.90", "2.22", "2.46"])
    assert out == {"south": 48.81, "north": 48.90, "west": 2.22, "east": 2.46}


def test_parse_bbox_invalid():
    assert _parse_bbox(None) is None
    assert _parse_bbox(["1", "2", "3"]) is None  # wrong length
    assert _parse_bbox(["a", "b", "c", "d"]) is None  # non-numeric


def test_parse_float():
    assert _parse_float("35") == 35.0
    assert _parse_float(" 12.5 ") == 12.5
    assert _parse_float(None) is None
    assert _parse_float("abc") is None


# --------------------------- _format_place ---------------------------

def test_format_place_folds_extratags_and_drops_empty():
    entry = {
        "name": "Paris, France",
        "latitude": 48.8566,
        "longitude": 2.3522,
        "category": "place",
        "type": "city",
        "address": {"country": "France"},
        "bounding_box": {"south": 48.8, "north": 48.9, "west": 2.2, "east": 2.5},
        "extratags": {"population": "2,100,000", "wikidata": "Q90",
                      "wikipedia": "en:Paris", "ele": "35"},
        "namedetails": {"official_name": "Ville de Paris"},
        "importance": 0.9,
        "osm_type": "relation",
        "osm_id": 7444,
    }
    out = _format_place(entry)
    assert out["class"] == "place"
    assert out["population"] == 2100000
    assert out["wikidata"] == "Q90"
    assert out["elevation_m"] == 35.0
    assert out["official_name"] == "Ville de Paris"
    # Empty/absent fields are omitted (no phone/website/opening_hours here).
    assert "phone" not in out
    assert "website" not in out


def test_format_place_minimal_entry():
    out = _format_place({"name": "X", "latitude": 1.0, "longitude": 2.0})
    assert out == {"name": "X", "latitude": 1.0, "longitude": 2.0}


# --------------------------- _parse_coordinates ---------------------------

def test_parse_coordinates_plain_text():
    assert _parse_coordinates("45.515118,-122.679485") == (45.515118, -122.679485)
    assert _parse_coordinates("45.515118 -122.679485") == (45.515118, -122.679485)


def test_parse_coordinates_map_urls():
    apple = "https://maps.apple.com/place?ll=45.515118%2C-122.679485"
    osm = "https://openstreetmap.org/#map=18/45.5044526/-122.5494963"
    assert _parse_coordinates(apple) == (45.515118, -122.679485)
    assert _parse_coordinates(osm) == (45.5044526, -122.5494963)


def test_parse_coordinates_rejects_invalid_ranges():
    assert _parse_coordinates("123.0,-122.0") is None
    assert _parse_coordinates("45.0,-222.0") is None
    assert _parse_coordinates("Portland, OR") is None


def test_parse_osm_object_url():
    assert _parse_osm_object_url("https://openstreetmap.org/node/13252567900") == (
        "N", "13252567900"
    )
    assert _parse_osm_object_url("https://www.openstreetmap.org/way/5013364") == (
        "W", "5013364"
    )
    assert _parse_osm_object_url("https://openstreetmap.org/relation/7444") == (
        "R", "7444"
    )
    assert _parse_osm_object_url("https://example.com/node/13252567900") is None


# --------------------------- _is_relative_location ---------------------------

@pytest.mark.parametrize("near", ["me", "here", "near me", "around here", "my location", "MY LOCATION"])
def test_is_relative_location_true(near):
    assert _is_relative_location(near) is True


def test_is_relative_location_punctuation_variants():
    assert _is_relative_location("near me!") is True
    assert _is_relative_location('"here"') is True


def test_is_relative_location_real_place_false():
    assert _is_relative_location("Portland, OR") is False
    assert _is_relative_location("Paris") is False


# --------------------------- _geocode (async, mocked) ---------------------------

def _no_throttle(monkeypatch):
    monkeypatch.setattr(geo.cfg, "min_request_interval_seconds", 0)


def _fresh_cache(monkeypatch):
    from tools.cache import TTLCache
    monkeypatch.setattr(geo, "_cache", TTLCache(0))  # disable caching for determinism


def test_geocode_success(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)

    def handler(request):
        return httpx.Response(200, json=[
            {"display_name": "Paris, France", "lat": "48.8566", "lon": "2.3522",
             "class": "place", "type": "city", "address": {"country": "France"}},
        ])

    patch_httpx(handler)
    out = run(_geocode("Paris", 1))
    assert len(out) == 1
    assert out[0]["name"] == "Paris, France"
    assert out[0]["latitude"] == 48.8566
    assert out[0]["longitude"] == 2.3522


def test_geocode_empty_results(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, json=[]))
    assert run(_geocode("Nowhereville XYZ", 1)) == []


def test_geocode_empty_query_raises(monkeypatch):
    _no_throttle(monkeypatch)
    with pytest.raises(ToolError):
        run(_geocode("   ", 1))


def test_geocode_rate_limited_raises(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)
    patch_httpx(lambda req: httpx.Response(429, text="slow down"))
    with pytest.raises(ToolError) as exc:
        run(_geocode("Paris", 1))
    assert "429" in str(exc.value)


def test_geocode_non_json_raises(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(ToolError):
        run(_geocode("Paris", 1))


# --------------------------- _geocode detailed / _place_lookup ---------------------------

def test_geocode_detailed_surfaces_extratags(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)

    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[
            {"display_name": "Paris, France", "lat": "48.8566", "lon": "2.3522",
             "class": "place", "type": "city",
             "boundingbox": ["48.81", "48.90", "2.22", "2.46"],
             "extratags": {"population": "2100000", "wikidata": "Q90"},
             "namedetails": {"official_name": "Ville de Paris"},
             "importance": 0.9, "osm_type": "relation", "osm_id": 7444,
             "address": {"country": "France"}},
        ])

    patch_httpx(handler)
    out = run(_geocode("Paris", 5, detailed=True))
    # extratags/namedetails were requested and surfaced.
    assert "extratags=1" in captured["url"]
    assert "namedetails=1" in captured["url"]
    assert out[0]["bounding_box"]["north"] == 48.90
    assert out[0]["extratags"]["wikidata"] == "Q90"


def test_place_lookup_builds_payload(monkeypatch):
    async def fake_geocode(query, limit, detailed=False):
        assert detailed is True
        return [
            {"name": "Paris, France", "latitude": 48.8566, "longitude": 2.3522,
             "category": "place", "type": "city",
             "extratags": {"population": "2100000", "wikidata": "Q90"},
             "namedetails": {}, "bounding_box": None},
            {"name": "Paris, TX", "latitude": 33.66, "longitude": -95.55,
             "type": "city", "extratags": {}, "namedetails": {}},
        ]

    monkeypatch.setattr(geo, "_geocode", fake_geocode)
    out = run(_place_lookup("Paris"))
    assert out["query"] == "Paris"
    assert out["place"]["population"] == 2100000
    assert out["place"]["wikidata"] == "Q90"
    assert out["alternatives"][0]["name"] == "Paris, TX"


def test_place_lookup_no_match_raises(monkeypatch):
    async def fake_geocode(query, limit, detailed=False):
        return []

    monkeypatch.setattr(geo, "_geocode", fake_geocode)
    with pytest.raises(ToolError):
        run(_place_lookup("Nowhereville XYZ"))


# --------------------------- place_details tool mode ---------------------------

def test_find_nearby_places_place_details_happy_path(monkeypatch, tool_fns):
    import json as _json

    async def fake_geocode(query, limit, detailed=False):
        return [{"name": "Portland, OR", "latitude": 45.52, "longitude": -122.68,
                 "category": "place", "type": "city",
                 "extratags": {"population": "650000"}, "namedetails": {}}]

    monkeypatch.setattr(geo, "_geocode", fake_geocode)
    fn = tool_fns["find_nearby_places"]
    out = _json.loads(run(fn(near="Portland, OR", place_details=True)))
    assert out["place"]["name"] == "Portland, OR"
    assert out["place"]["population"] == 650000


def test_find_nearby_places_place_details_accepts_coordinates(monkeypatch, tool_fns):
    import json as _json

    async def fake_reverse_geocode(lat, lon, detailed=False):
        return {"name": "Downtown Portland", "latitude": lat, "longitude": lon,
                "category": "place", "type": "neighbourhood",
                "address": {"city": "Portland"}, "extratags": {}, "namedetails": {}}

    monkeypatch.setattr(geo, "_reverse_geocode", fake_reverse_geocode)
    fn = tool_fns["find_nearby_places"]
    out = _json.loads(run(fn(near="45.515118,-122.679485", place_details=True)))
    assert out["query"] == "45.5151180,-122.6794850"
    assert out["place"]["name"] == "Downtown Portland"


def test_find_nearby_places_place_details_accepts_osm_url(monkeypatch, tool_fns):
    import json as _json

    async def fake_lookup_osm_object(osm_type, osm_id, detailed=False):
        assert (osm_type, osm_id, detailed) == ("W", "5013364", True)
        return {"name": "Eiffel Tower", "latitude": 48.8582637, "longitude": 2.2942401,
                "category": "tourism", "type": "attraction",
                "address": {"city": "Paris"}, "extratags": {}, "namedetails": {}}

    monkeypatch.setattr(geo, "_lookup_osm_object", fake_lookup_osm_object)
    fn = tool_fns["find_nearby_places"]
    out = _json.loads(run(fn(near="https://openstreetmap.org/way/5013364", place_details=True)))
    assert out["query"] == "W5013364"
    assert out["place"]["name"] == "Eiffel Tower"


def test_find_nearby_places_place_details_requires_near(tool_fns):
    fn = tool_fns["find_nearby_places"]
    with pytest.raises(ToolError) as exc:
        run(fn(category="", place_details=True))
    assert "near" in str(exc.value)


def test_find_nearby_places_place_details_rejects_relative(tool_fns):
    fn = tool_fns["find_nearby_places"]
    with pytest.raises(ToolError) as exc:
        run(fn(near="near me", place_details=True))
    assert "relative location" in str(exc.value)


# --------------------------- _overpass (async, mocked) ---------------------------

def test_overpass_success(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)
    elements = [{"type": "node", "lat": 1.0, "lon": 2.0, "tags": {"name": "Cafe"}}]
    patch_httpx(lambda req: httpx.Response(200, json={"elements": elements}))
    assert run(_overpass("[out:json];node;out;")) == elements


def test_overpass_remark_error_raises(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, json={"remark": "runtime error: timeout"}))
    with pytest.raises(ToolError) as exc:
        run(_overpass("[out:json];node;out;"))
    assert "error" in str(exc.value).lower()


def test_overpass_rate_limited_raises(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)
    patch_httpx(lambda req: httpx.Response(429, text="too many"))
    with pytest.raises(ToolError):
        run(_overpass("[out:json];node;out;"))


def test_overpass_non_json_raises(monkeypatch, patch_httpx):
    _no_throttle(monkeypatch)
    _fresh_cache(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, text="Error: bad query"))
    with pytest.raises(ToolError):
        run(_overpass("bad query"))


def test_rate_limiter_serializes_and_spaces_concurrent_calls(monkeypatch):
    """Concurrent acquires are queued and dispatched one per interval, in order,
    rather than all firing at once (the burst that overloads Overpass)."""
    import anyio as _anyio

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        # Advance the limiter's clock so the next waiter computes its own delay
        # off the time this one "finished" — mimics real elapsed time.
        clock[0] += seconds

    clock = [1000.0]
    monkeypatch.setattr(geo.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(geo.anyio, "sleep", fake_sleep)

    limiter = geo._RateLimiter()

    async def main():
        # Five callers arrive together; the lock serializes them.
        async with _anyio.create_task_group() as tg:
            for _ in range(5):
                tg.start_soon(limiter.acquire, 1.0)

    run(main())

    # First caller doesn't wait; the next four are each spaced by ~1s.
    assert len(sleeps) == 4
    assert all(abs(s - 1.0) < 1e-9 for s in sleeps)


def test_rate_limiter_disabled_when_interval_zero(monkeypatch):
    slept = []
    monkeypatch.setattr(geo.anyio, "sleep", lambda s: slept.append(s))
    limiter = geo._RateLimiter()
    run(limiter.acquire(0))
    assert slept == []


# --------------------------- find_nearby_places tool (validation) ---------------------------

async def _empty_nearby_towns(lat, lon, n, exclude):
    return []


def test_find_nearby_places_empty_category_raises(tool_fns):
    fn = tool_fns["find_nearby_places"]
    with pytest.raises(ToolError):
        run(fn(category="  "))


def test_find_nearby_places_no_location_raises(tool_fns):
    fn = tool_fns["find_nearby_places"]
    with pytest.raises(ToolError):
        run(fn(category="cafe"))


def test_find_nearby_places_relative_location_raises(tool_fns):
    fn = tool_fns["find_nearby_places"]
    with pytest.raises(ToolError) as exc:
        run(fn(category="cafe", near="near me"))
    assert "relative location" in str(exc.value)


def test_find_nearby_places_happy_path(monkeypatch, tool_fns):
    """End-to-end with geocode + overpass monkeypatched."""
    import json as _json

    async def fake_geocode(query, limit):
        return [{"name": "Portland, OR", "latitude": 45.52, "longitude": -122.68}]

    async def fake_overpass(query_ql):
        return [
            {"type": "node", "lat": 45.521, "lon": -122.681,
             "tags": {"name": "Near Cafe", "amenity": "cafe", "cuisine": "coffee_shop"}},
            {"type": "node", "lat": 45.6, "lon": -122.7,
             "tags": {"name": "Far Cafe", "amenity": "cafe"}},
        ]

    nearby_call = {}

    async def fake_nearby_towns(lat, lon, n, exclude):
        nearby_call.update(lat=lat, lon=lon, n=n, exclude=exclude)
        return [{"name": "Beaverton", "latitude": 45.49, "longitude": -122.80,
                 "distance_m": 12000, "place_type": "city"}]

    monkeypatch.setattr(geo, "_geocode", fake_geocode)
    monkeypatch.setattr(geo, "_overpass", fake_overpass)
    monkeypatch.setattr(geo, "_nearby_towns", fake_nearby_towns)

    fn = tool_fns["find_nearby_places"]
    out = _json.loads(run(fn(category="cafe", near="Portland", radius_m=2000, limit=5)))
    assert out["query_category"] == "cafe"
    assert out["center"]["name"] == "Portland, OR"
    assert out["count"] == 2
    # Results are sorted nearest-first.
    assert out["results"][0]["name"] == "Near Cafe"
    assert out["results"][0]["distance_m"] <= out["results"][1]["distance_m"]
    assert out["nearby_towns"][0]["name"] == "Beaverton"
    assert out["nearby_towns_radius_m"] == geo.cfg.nearby_towns_radius_m
    assert nearby_call == {
        "lat": 45.52,
        "lon": -122.68,
        "n": geo.cfg.max_nearby_towns,
        "exclude": "Portland, OR",
    }


def test_find_nearby_places_accepts_coordinates_in_near(monkeypatch, tool_fns):
    import json as _json

    captured = {}

    async def fake_overpass(query_ql):
        captured["ql"] = query_ql
        return [
            {"type": "node", "lat": 45.5150268, "lon": -122.6799045,
             "tags": {"name": "Starbucks", "amenity": "cafe"}},
        ]

    monkeypatch.setattr(geo, "_overpass", fake_overpass)
    monkeypatch.setattr(geo, "_nearby_towns", _empty_nearby_towns)
    fn = tool_fns["find_nearby_places"]
    out = _json.loads(run(fn(
        category="coffee",
        near="https://maps.apple.com/place?ll=45.515118%2C-122.679485",
        radius_m=1000,
    )))
    assert out["center"]["latitude"] == 45.515118
    assert out["center"]["longitude"] == -122.679485
    assert "(around:1000,45.5151180,-122.6794850)" in captured["ql"]


def test_find_nearby_places_accepts_osm_url_as_center(monkeypatch, tool_fns):
    import json as _json

    async def fake_lookup_osm_object(osm_type, osm_id, detailed=False):
        assert (osm_type, osm_id, detailed) == ("N", "13252567900", False)
        return {"name": "Portland Lux Coffee", "latitude": 45.5044526,
                "longitude": -122.5494963}

    captured = {}

    async def fake_overpass(query_ql):
        captured["ql"] = query_ql
        return []

    monkeypatch.setattr(geo, "_lookup_osm_object", fake_lookup_osm_object)
    monkeypatch.setattr(geo, "_overpass", fake_overpass)
    monkeypatch.setattr(geo, "_nearby_towns", _empty_nearby_towns)
    fn = tool_fns["find_nearby_places"]
    out = _json.loads(run(fn(
        category="coffee",
        near="https://openstreetmap.org/node/13252567900",
        radius_m=1000,
    )))
    assert out["center"]["name"] == "Portland Lux Coffee"
    assert "(around:1000,45.5044526,-122.5494963)" in captured["ql"]


def test_find_nearby_places_clamps_radius(monkeypatch, tool_fns):
    """radius_m above the configured max is clamped."""
    import json as _json

    captured = {}

    async def fake_overpass(query_ql):
        captured["ql"] = query_ql
        return []

    monkeypatch.setattr(geo, "_overpass", fake_overpass)
    monkeypatch.setattr(geo, "_nearby_towns", _empty_nearby_towns)
    fn = tool_fns["find_nearby_places"]
    out = _json.loads(run(fn(
        category="cafe", latitude=45.0, longitude=-122.0,
        radius_m=10_000_000,
    )))
    assert out["radius_m"] == geo.cfg.max_radius_m
    # The query uses the `nwr` shorthand (one statement per filter), not three
    # separate node/way/relation statements — the split form is pathologically
    # slow for a name-regex filter (see geocoding.py).
    assert "nwr" in captured["ql"]
    assert "\n  node" not in captured["ql"]


def test_find_nearby_places_name_search_query_is_key_constrained(monkeypatch, tool_fns):
    """A brand/name search must never emit a bare `["name"~...]` filter: without an
    indexed key to narrow on, Overpass scans every element in the radius and times
    out. Each name filter is paired with a POI key (see _NAME_SEARCH_KEYS)."""
    import json as _json

    captured = {}

    async def fake_overpass(query_ql):
        captured["ql"] = query_ql
        return []

    monkeypatch.setattr(geo, "_overpass", fake_overpass)
    monkeypatch.setattr(geo, "_nearby_towns", _empty_nearby_towns)
    fn = tool_fns["find_nearby_places"]
    run(fn(category="Starbucks", latitude=45.0, longitude=-122.0))
    ql = captured["ql"]
    # The name regex appears, always immediately followed by an indexed key.
    assert '["name"~"Starbucks",i]' in ql
    assert '["name"~"Starbucks",i](' not in ql  # never name-only, then `(around:`
    for key in geo._NAME_SEARCH_KEYS:
        assert f'["name"~"Starbucks",i]["{key}"]' in ql
