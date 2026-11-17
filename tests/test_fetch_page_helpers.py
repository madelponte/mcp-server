"""Tests for the pure helpers in tools/fetch_page.py."""

import pytest

import tools.fetch_page as fp
import tools.serialize as serialize
from tools.fetch_page import (
    _join_note,
    _set_content,
    _resolve_query_control,
    _compile_query,
    _segment_text,
    _extract_matches,
    QueryMatchTimeoutError,
    _format_match_windows,
    _normalize_reddit_url,
    _reddit_oauth_url,
    _reddit_rss_url,
    _reddit_oembed_url,
    _reddit_post_id,
    _reddit_search_query,
    _is_reddit_url,
    _compact_reddit_json,
    _compact_reddit_rss,
    _provenance,
    _is_contentless,
)


# --------------------------- _join_note ---------------------------

def test_join_note_appends():
    assert _join_note("a", "b") == "a b"


def test_join_note_alone():
    assert _join_note(None, "b") == "b"
    assert _join_note("", "b") == "b"


# --------------------------- _set_content ---------------------------

def test_set_content_no_truncation():
    payload = {}
    _set_content(payload, "short content")
    assert payload["content"] == "short content"
    assert "truncated" not in payload
    assert "next_offset" not in payload


def test_set_content_truncates_and_sets_next_offset(monkeypatch):
    monkeypatch.setattr(fp.cfg, "max_page_chars", 10)
    payload = {}
    _set_content(payload, "a" * 50)
    assert payload["truncated"] is True
    assert payload["next_offset"] == 10
    # Full content size is reported so the model can size what remains.
    assert payload["content_length"] == 50
    assert "offset=" in payload["note"]
    # Narrowing hint is added by default.
    assert "query=" in payload["note"]


def test_set_content_no_content_length_when_not_truncated():
    payload = {}
    _set_content(payload, "short content")
    assert "content_length" not in payload


def test_set_content_content_length_is_full_size_with_offset(monkeypatch):
    # content_length reports the whole content's size, not the post-offset slice.
    monkeypatch.setattr(fp.cfg, "max_page_chars", 10)
    payload = {}
    _set_content(payload, "a" * 50, offset=5)
    assert payload["truncated"] is True
    assert payload["content_length"] == 50
    assert payload["next_offset"] == 15


def test_set_content_no_narrow_hint_when_hint_false(monkeypatch):
    monkeypatch.setattr(fp.cfg, "max_page_chars", 10)
    payload = {}
    _set_content(payload, "a" * 50, hint=False)
    assert payload["truncated"] is True
    assert "query=" not in payload["note"]
    assert "offset=" in payload["note"]


def test_set_content_with_offset(monkeypatch):
    monkeypatch.setattr(fp.cfg, "max_page_chars", 1000)
    payload = {}
    _set_content(payload, "0123456789", offset=5)
    assert payload["content"] == "56789"
    assert payload["offset"] == 5


def test_set_content_offset_reports_nearest_heading_anchor():
    content = "# Intro {#intro}\n\nfirst\n\n## Details {#cite-details}\n\nsecond"
    offset = content.index("second")
    payload = {}
    _set_content(payload, content, offset=offset)
    assert payload["content"] == "second"
    assert payload["continuation_anchor"] == "cite-details"


def test_set_content_offset_past_end():
    payload = {}
    _set_content(payload, "0123456789", offset=100)
    assert payload["content"] == ""
    assert "at or past the end" in payload["note"]


# --------------------------- query controls ---------------------------

def test_resolve_query_control_uses_default_and_clamps():
    assert _resolve_query_control(None, default=2, maximum=8, minimum=0) == 2
    assert _resolve_query_control(100, default=2, maximum=8, minimum=0) == 8
    assert _resolve_query_control(-5, default=2, maximum=8, minimum=0) == 0
    assert _resolve_query_control(0, default=10, maximum=10, minimum=1) == 1


# --------------------------- _compile_query ---------------------------

def test_compile_query_valid_regex():
    pat = _compile_query(r"foo\d+")
    assert pat.search("foo123")
    assert not pat.search("bar")


def test_compile_query_invalid_regex_falls_back_to_literal():
    # Unbalanced bracket is not valid regex -> matched literally.
    pat = _compile_query("cost (usd")
    assert pat.search("the cost (usd here")


def test_compile_query_nested_quantifier_falls_back_to_literal():
    pat = _compile_query("(a+)+")
    # Treated as the literal string, not a catastrophic-backtracking regex.
    assert pat.search("value (a+)+ here")
    assert not pat.search("aaaa")


def test_compile_query_overlong_falls_back_to_literal():
    long_q = "x" * 300
    pat = _compile_query(long_q)
    assert pat.search("prefix " + long_q)


def test_compile_query_is_case_insensitive():
    assert _compile_query("Hello").search("hello world")


# --------------------------- _segment_text ---------------------------

def test_segment_text_splits_and_drops_blanks():
    assert _segment_text("a\n\n  b  \n\nc\n") == ["a", "b", "c"]


# --------------------------- _extract_matches ---------------------------

def _text(*lines):
    return "\n".join(lines)


def test_extract_matches_with_context():
    text = _text("l0", "l1", "match here", "l3", "l4")
    windows, total_matches, total_windows = _extract_matches(
        text, "match", context=1, max_windows=10
    )
    assert total_matches == 1
    assert total_windows == 1
    assert windows[0]["content"] == "l1\nmatch here\nl3"
    assert windows[0]["line_start"] == 2
    assert windows[0]["line_end"] == 4
    assert windows[0]["match_lines"] == [3]
    assert windows[0]["match_quality"] == "literal_substring"


def test_extract_matches_prepends_nearest_heading_citation_anchor():
    text = _text(
        "# First {#first}",
        "unrelated paragraph",
        "## Details {#cite-details}",
        "intro",
        "target passage",
    )
    windows, _, _ = _extract_matches(text, "target", context=0, max_windows=10)
    assert windows[0]["content"] == "## Details {#cite-details}\ntarget passage"
    assert windows[0]["heading"] == {
        "text": "Details",
        "anchor": "cite-details",
        "line": 3,
    }


def test_extract_matches_stays_within_preceding_heading_section():
    text = _text(
        "## Prior {#prior}",
        "filler",
        "target passage",
        "## Next {#next}",
    )
    windows, _, _ = _extract_matches(text, "target", context=1, max_windows=10)
    assert windows[0]["content"].startswith("## Prior {#prior}\n")
    assert "## Next {#next}" not in windows[0]["content"]


def test_extract_matches_reports_exact_line_and_regex_quality():
    exact, _, _ = _extract_matches("needle", "needle", context=0, max_windows=1)
    regex, _, _ = _extract_matches("item 42", r"item \d+", context=0, max_windows=1)
    assert exact[0]["match_quality"] == "exact_line"
    assert regex[0]["match_quality"] == "regex_pattern"


def test_extract_matches_does_not_merge_across_heading_boundaries():
    text = _text(
        "## Alpha {#alpha}",
        "target alpha",
        "## Beta {#beta}",
        "target beta",
    )
    windows, _, total_windows = _extract_matches(
        text, "target", context=2, max_windows=10
    )
    assert total_windows == 2
    assert [window["heading"]["anchor"] for window in windows] == ["alpha", "beta"]
    assert [window["match_lines"] for window in windows] == [[2], [4]]


def test_extract_matches_merges_adjacent():
    text = _text("match a", "x", "match b")
    windows, total_matches, total_windows = _extract_matches(
        text, "match", context=1, max_windows=10
    )
    # Two matches within one segment of each other merge into a single window.
    assert total_matches == 2
    assert total_windows == 1
    assert len(windows) == 1


def test_extract_matches_respects_max_windows():
    text = _text("match", "a", "b", "c", "match", "d", "e", "f", "match")
    windows, total_matches, total_windows = _extract_matches(
        text, "match", context=0, max_windows=2
    )
    assert total_matches == 3
    assert total_windows == 3
    assert len(windows) == 2  # capped


def test_extract_matches_no_match():
    windows, total_matches, total_windows = _extract_matches(
        "nothing here", "absent", context=2, max_windows=10
    )
    assert windows == []
    assert total_matches == 0


def test_extract_matches_empty_text():
    assert _extract_matches("", "x", context=1, max_windows=5) == ([], 0, 0)


def test_extract_matches_interrupts_catastrophic_backtracking(monkeypatch):
    # This ambiguous alternation evades the nested-quantifier heuristic but is
    # still exponential in a traditional backtracking engine on a failed match.
    monkeypatch.setattr(fp, "_QUERY_MATCH_BUDGET_SECONDS", 0.01)
    with pytest.raises(QueryMatchTimeoutError):
        _extract_matches(
            "a" * 10_000 + "!",
            r"(a|aa)+$",
            context=0,
            max_windows=1,
        )


# --------------------------- _format_match_windows ---------------------------

def test_format_match_windows_single():
    assert _format_match_windows(["only"]) == "only"


def test_format_match_windows_multiple_labels():
    out = _format_match_windows(["one", "two"])
    assert "match 1 of 2" in out
    assert "match 2 of 2" in out
    assert "one" in out and "two" in out


# --------------------------- _normalize_reddit_url ---------------------------

def test_normalize_reddit_url_appends_json():
    out = _normalize_reddit_url("https://www.reddit.com/r/python/comments/abc/title")
    assert out.endswith(".json")
    assert "www.reddit.com" in out


def test_normalize_reddit_url_subdomain():
    out = _normalize_reddit_url("https://old.reddit.com/r/x/comments/abc/title/")
    assert out.endswith(".json")


def test_normalize_reddit_url_already_json():
    url = "https://www.reddit.com/r/x/comments/abc/title.json"
    assert _normalize_reddit_url(url).endswith(".json")
    assert _normalize_reddit_url(url).count(".json") == 1


def test_reddit_representation_urls():
    url = "https://www.reddit.com/r/python/comments/abc/title/?context=3"
    oauth = _reddit_oauth_url(url)
    assert oauth.startswith("https://oauth.reddit.com/")
    assert "raw_json=1" in oauth
    assert "limit=500" in oauth
    assert _reddit_rss_url(url).endswith("/title/.rss")
    assert "context=" not in _reddit_rss_url(url)


def test_reddit_listing_representation_urls():
    url = "https://old.reddit.com/r/python/search?q=asyncio&restrict_sr=1"
    assert _reddit_rss_url(url) == (
        "https://www.reddit.com/r/python/search.rss?q=asyncio&restrict_sr=1"
    )


def test_normalize_reddit_url_non_reddit_unchanged():
    url = "https://example.com/r/python/comments/abc"
    assert _normalize_reddit_url(url) == url


def test_normalize_reddit_url_lookalike_unchanged():
    url = "https://notreddit.com/r/x/comments/abc"
    assert _normalize_reddit_url(url) == url


def test_reddit_oembed_url_uses_post_url_not_json_endpoint():
    out = _reddit_oembed_url(
        "https://old.reddit.com/r/x/comments/abc/title.json?context=3"
    )
    assert "/oembed?" in out
    assert "title.json" not in out
    assert "www.reddit.com" in out
    assert "context%3D3" not in out


@pytest.mark.parametrize("url", [
    "https://reddit.com/r/x/comments/abc/t",
    "https://www.reddit.com/r/x/",
    "https://old.reddit.com/r/x/comments/abc/t",
    "https://np.reddit.com/r/x/comments/abc/t",
    "https://WWW.Reddit.com:443/r/x/",
])
def test_is_reddit_url_content_hosts(url):
    assert _is_reddit_url(url)


@pytest.mark.parametrize("url", [
    "https://developers.reddit.com/docs/guides/migrate/public-api",
    "https://support.reddithelp.com/hc/en-us/articles/1",
    "https://redditinc.com/policies",
    "https://notreddit.com/r/x/comments/abc",
    "https://evil.com/?u=https://www.reddit.com/r/x",
])
def test_is_reddit_url_rejects_non_content_hosts(url):
    assert not _is_reddit_url(url)


def test_reddit_post_id():
    assert _reddit_post_id("https://www.reddit.com/r/x/comments/AbC12/title/") == "abc12"
    assert _reddit_post_id("https://www.reddit.com/comments/abc/") == "abc"
    assert _reddit_post_id("https://www.reddit.com/r/x/comments/abc/t/def/") == "abc"
    assert _reddit_post_id("https://www.reddit.com/r/x/comments/abc/t.json") == "abc"
    assert _reddit_post_id("https://www.reddit.com/user/alice/comments/abc/t/") == "abc"
    assert _reddit_post_id("https://www.reddit.com/user/alice/comments/") is None
    assert _reddit_post_id("https://www.reddit.com/r/x/") is None
    assert _reddit_post_id("https://example.com/r/x/comments/abc/t") is None


def test_reddit_search_query_prefers_title_then_slug():
    url = "https://www.reddit.com/r/redditdev/comments/abc/moving_data_api_apps/"
    assert _reddit_search_query(url, "Moving Data API Apps: dates") == (
        "site:reddit.com r/redditdev Moving Data API Apps: dates"
    )
    assert _reddit_search_query(url, None) == (
        "site:reddit.com r/redditdev moving data api apps"
    )
    assert _reddit_search_query("https://www.reddit.com/comments/abc/", None) is None


def test_reddit_search_query_stays_within_brave_limits():
    url = "https://www.reddit.com/r/x/comments/abc/t/"
    query = _reddit_search_query(url, " ".join(["word"] * 200))
    assert len(query.split()) <= 75
    assert len(query) <= 600


def test_reddit_rss_retired_latch_opens_and_expires(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(fp.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fp, "_reddit_rss_retired_until", 0.0)
    assert fp._reddit_rss_available()
    fp._mark_reddit_rss_retired()
    assert not fp._reddit_rss_available()
    clock[0] += fp._REDDIT_RSS_RETIRED_TTL_SECONDS
    assert fp._reddit_rss_available()


@pytest.mark.parametrize("fetched, retired", [
    ({"status": 429, "text": ""}, False),
    ({"status": 503, "text": "<html>down</html>"}, False),
    # Reddit answers unknown subreddits/posts and search feeds with Atom errors.
    ({"status": 404, "text": "<?xml version='1.0'?><feed xmlns='x'></feed>"}, False),
    ({"status": 403, "text": "<?xml version='1.0'?><feed xmlns='x'></feed>"}, False),
    ({"status": 200, "text": "<feed><entry/></feed>"}, False),
    ({"status": 410, "text": "<feed></feed>"}, True),
    ({"status": 403, "text": "<html>blocked</html>"}, True),
    ({"status": 403, "text": ""}, True),
    ({"status": 200, "text": "<!doctype html><html></html>"}, True),
])
def test_rss_response_means_retired(fetched, retired):
    assert fp._rss_response_means_retired(fetched) is retired


@pytest.mark.parametrize("url, expected", [
    ("https://www.reddit.com/r/python/search?q=x", True),
    ("https://www.reddit.com/search/?q=x", True),
    ("https://www.reddit.com/r/python/", False),
    ("https://www.reddit.com/r/python/comments/abc/search/", True),
])
def test_is_reddit_search_url(url, expected):
    assert fp._is_reddit_search_url(url) is expected


# --------------------------- archive listing plan ---------------------------

@pytest.fixture
def listing_cap(monkeypatch):
    monkeypatch.setattr(fp.cfg, "reddit_archive_max_listing", 7)


@pytest.mark.parametrize("url", [
    "https://www.reddit.com/r/Python/",
    "https://www.reddit.com/r/Python/top/?t=week",
    "https://old.reddit.com/r/Python/new.rss",
])
def test_archive_plan_subreddit(listing_cap, url):
    meta, requests = fp._reddit_archive_listing_plan(url)
    assert meta == {"listing": "r/Python"}
    path, params = requests["posts"]
    assert path == "/posts/search"
    assert params == {
        "subreddit": "Python", "sort": "desc", "limit": 7,
        "fields": fp._ARCHIVE_POST_LISTING_FIELDS,
    }


def test_archive_plan_subreddit_search_uses_title(listing_cap):
    meta, requests = fp._reddit_archive_listing_plan(
        "https://www.reddit.com/r/Python/search/?q=async+io&restrict_sr=1"
    )
    assert meta == {"listing": "r/Python search", "query": "async io", "match": "title"}
    assert requests["posts"][1]["title"] == "async io"
    assert "query" not in requests["posts"][1]


@pytest.mark.parametrize("url, keys", [
    ("https://www.reddit.com/user/spez/", {"posts", "comments"}),
    ("https://www.reddit.com/u/spez", {"posts", "comments"}),
    ("https://www.reddit.com/user/spez/submitted/", {"posts"}),
    ("https://www.reddit.com/user/spez/comments/", {"comments"}),
])
def test_archive_plan_user(listing_cap, url, keys):
    meta, requests = fp._reddit_archive_listing_plan(url)
    assert meta == {"listing": "u/spez"}
    assert set(requests) == keys
    for path, params in requests.values():
        assert params["author"] == "spez"
        assert params["limit"] == 7
    if "comments" in requests:
        assert requests["comments"][1]["fields"] == fp._ARCHIVE_COMMENT_LISTING_FIELDS


@pytest.mark.parametrize("url", [
    "https://www.reddit.com/",
    "https://www.reddit.com/search/?q=x",
    "https://www.reddit.com/r/all/",
    "https://www.reddit.com/r/python+rust/",
    "https://www.reddit.com/r/python/wiki/index",
    "https://www.reddit.com/r/python/search/",
    "https://www.reddit.com/user/spez/saved/",
])
def test_archive_plan_rejects_uncovered_pages(listing_cap, url):
    with pytest.raises(RuntimeError):
        fp._reddit_archive_listing_plan(url)


def test_archive_post_item_builds_permalink_and_drops_self_url():
    item = fp._archive_post_item({
        "id": "abc", "subreddit": "Python", "title": "T", "author": "a",
        "score": 3, "num_comments": 2, "created_utc": 0,
        "url": "https://www.reddit.com/r/Python/comments/abc/t/",
        "selftext": "x" * 1000, "link_flair_text": "Discussion", "over_18": False,
    })
    assert item["permalink"] == "https://www.reddit.com/r/Python/comments/abc/"
    assert "url" not in item
    assert item["created"] == "1970-01-01T00:00:00Z"
    assert item["flair"] == "Discussion"
    assert "nsfw" not in item
    assert len(item["text"]) == fp._ARCHIVE_LISTING_EXCERPT_CHARS
    link = fp._archive_post_item({"id": "abc", "subreddit": "s", "url": "https://example.com/x"})
    assert link["url"] == "https://example.com/x"


def test_archive_items_mark_placeholder_text():
    post = fp._archive_post_item({"id": "a", "subreddit": "s", "selftext": " [removed] "})
    assert "text" not in post and post["text_removed"] is True
    empty = fp._archive_post_item({"id": "a", "subreddit": "s", "selftext": ""})
    assert "text" not in empty and "text_removed" not in empty
    comment = fp._archive_comment_item({"id": "c", "subreddit": "s", "body": "[deleted]"})
    assert "text" not in comment and comment["text_removed"] is True


def test_reddit_query_text_has_one_heading_per_item():
    text, lookup = fp._reddit_query_text({
        "post": {"title": "T {#x}", "author": "op", "selftext": "Body\nline 2",
                 "permalink": "https://www.reddit.com/r/s/comments/a/"},
        "comments": [{"author": "c", "score": 1, "depth": 1, "body": "Reply"}],
    })
    lines = text.split("\n")
    assert lines[0] == "# T { #x} {#post}"
    assert "## Comment 1: u/c, score 1, reply depth 1 {#comment-1}" in lines
    assert lookup == {"post": {"citation_url": "https://www.reddit.com/r/s/comments/a/"}}
    oembed_text, _ = fp._reddit_query_text({"title": "Oembed"})
    assert "Oembed" in oembed_text


def test_archive_comment_item_permalink():
    item = fp._archive_comment_item({
        "id": "c1", "subreddit": "s", "link_id": "t3_abc", "body": "hi", "author": "a",
    })
    assert item["permalink"] == "https://www.reddit.com/r/s/comments/abc/_/c1/"
    assert item["type"] == "comment"


# --------------------------- _compact_reddit_json ---------------------------

def test_compact_reddit_json_extracts_post_and_comments():
    data = [
        {"data": {"children": [{"data": {
            "title": "T", "author": "u", "subreddit": "py",
            "score": 10, "num_comments": 1, "selftext": "body",
        }}]}},
        {"data": {"children": [
            {"kind": "t1", "data": {
                "author": "c1", "score": 5, "body": "top comment",
                "replies": {"data": {"children": [
                    {"kind": "t1", "data": {"author": "c2", "score": 2, "body": "reply"}}
                ]}},
            }},
        ]}},
    ]
    out = _compact_reddit_json(data)
    assert out["post"]["title"] == "T"
    assert out["comments"][0]["body"] == "top comment"
    assert out["comments"][0]["depth"] == 0
    # Nested reply collected with incremented depth.
    assert out["comments"][1]["body"] == "reply"
    assert out["comments"][1]["depth"] == 1


def test_compact_reddit_json_passthrough_for_unexpected_shape():
    data = {"unexpected": True}
    assert _compact_reddit_json(data) == data


def test_compact_reddit_rss_extracts_post_and_comments(monkeypatch):
    monkeypatch.setattr(fp.cfg, "markdown", False)
    xml = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><author><name>/u/poster</name></author><content type="html">&lt;div class="md"&gt;&lt;p&gt;post body&lt;/p&gt;&lt;/div&gt;</content><id>t3_abc</id><link href="https://reddit.com/comments/abc"/><updated>2026-01-01T00:00:00Z</updated><title>Post title</title></entry>
      <entry><author><name>/u/commenter</name></author><content type="html">&lt;div class="md"&gt;&lt;p&gt;comment body&lt;/p&gt;&lt;/div&gt;</content><id>t1_def</id><link href="https://reddit.com/comments/abc/def"/><updated>2026-01-01T01:00:00Z</updated><title>Comment</title></entry>
    </feed>"""
    out = _compact_reddit_rss(xml, "https://www.reddit.com/comments/abc")
    assert out["post"]["title"] == "Post title"
    assert out["post"]["selftext"] == "post body"
    assert out["comments"][0]["author"] == "commenter"
    assert out["comments"][0]["body"] == "comment body"
    assert out["comments_returned"] == 1
    assert out["comments_incomplete"] is True


# --------------------------- _provenance ---------------------------

def test_provenance_happy_path_is_empty(monkeypatch):
    monkeypatch.setattr(serialize.server_settings, "debug", False)
    out = _provenance("https://e.com", "https://e.com", 200, "text/html", "direct")
    assert out == {}


def test_provenance_rewritten_url_adds_original():
    out = _provenance("https://e.com", "https://e.com/x.json", 200, "application/json", "direct")
    assert out["original_url"] == "https://e.com"


def test_provenance_non_200_adds_status(monkeypatch):
    monkeypatch.setattr(serialize.server_settings, "debug", False)
    out = _provenance("https://e.com", "https://e.com", 404, "text/html", "direct")
    assert out["status"] == 404
    assert out["content_type"] == "text/html"


def test_provenance_non_direct_via(monkeypatch):
    monkeypatch.setattr(serialize.server_settings, "debug", False)
    out = _provenance("https://e.com", "https://e.com", 200, "text/html", "flaresolverr")
    assert out["via"] == "flaresolverr"


def test_provenance_debug_forces_all(monkeypatch):
    monkeypatch.setattr(serialize.server_settings, "debug", True)
    out = _provenance("https://e.com", "https://e.com", 200, "text/html", "direct")
    assert out["original_url"] == "https://e.com"
    assert out["status"] == 200
    assert out["via"] == "direct"


# --------------------------- _is_contentless ---------------------------

def test_is_contentless_empty():
    assert _is_contentless("") is True


def test_is_contentless_punctuation_only():
    assert _is_contentless("; ; -- !!") is True


def test_is_contentless_has_word():
    assert _is_contentless("hi") is False
    assert _is_contentless("a real sentence") is False
