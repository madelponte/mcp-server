"""Tests for the pure helpers in tools/fetch_page.py."""

import pytest

import tools.fetch_page as fp
import tools.serialize as serialize
from tools.fetch_page import (
    _join_note,
    _set_content,
    _compile_query,
    _segment_text,
    _extract_matches,
    QueryMatchTimeoutError,
    _format_match_windows,
    _normalize_reddit_url,
    _reddit_oauth_url,
    _reddit_rss_url,
    _reddit_old_url,
    _reddit_oembed_url,
    _compact_reddit_json,
    _compact_reddit_rss,
    _compact_reddit_html,
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


def test_set_content_offset_past_end():
    payload = {}
    _set_content(payload, "0123456789", offset=100)
    assert payload["content"] == ""
    assert "at or past the end" in payload["note"]


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
    assert windows == ["l1\nmatch here\nl3"]


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
    assert _reddit_rss_url(url).endswith("/title/.rss?context=3")
    assert _reddit_old_url(url).startswith("https://old.reddit.com/")


def test_reddit_listing_representation_urls():
    url = "https://old.reddit.com/r/python/search?q=asyncio&restrict_sr=1"
    assert _reddit_rss_url(url) == (
        "https://www.reddit.com/r/python/search.rss?q=asyncio&restrict_sr=1"
    )
    assert _reddit_old_url(url) == url


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


def test_compact_old_reddit_html_extracts_targeted_content(monkeypatch):
    monkeypatch.setattr(fp.cfg, "markdown", False)
    html = """<html><body>
      <div class="thing link" data-fullname="t3_abc" data-author="poster"
           data-subreddit="python" data-score="12" data-comments-count="2"
           data-permalink="/r/python/comments/abc/title/" data-url="/r/python/comments/abc/title/">
        <a class="title">Post title</a><div class="entry"><div class="usertext-body"><div class="md"><p>post body</p></div></div></div>
      </div>
      <div class="thing comment" data-fullname="t1_def" data-type="comment" data-author="commenter">
        <div class="entry"><a class="author">commenter</a><span class="score unvoted">5 points</span><div class="usertext-body"><div class="md"><p>comment body</p></div></div></div>
      </div><div class="morecomments">more</div>
    </body></html>"""
    out = _compact_reddit_html(
        html, "https://old.reddit.com/r/python/comments/abc/title/"
    )
    assert out["post"]["title"] == "Post title"
    assert out["post"]["selftext"] == "post body"
    assert out["comments"][0]["body"] == "comment body"
    assert out["comments"][0]["depth"] == 0
    assert out["comments_incomplete"] is True
    assert out["more_comment_placeholders"] == 1


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
