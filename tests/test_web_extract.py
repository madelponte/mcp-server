"""Tests for tools/web_extract.py — pure HTML -> content helpers."""

from bs4 import BeautifulSoup

from tools.web_extract import (
    _trim_flagged,
    _trim,
    _extract_jsonld,
    _headings_outline,
    _toc_from_jsonld,
    _page_title,
    _page_description,
    _structured_from_html,
    _plain_text_from_html,
    _markdown_from_html,
    _norm_heading,
    _find_section,
)


def _soup(html):
    return BeautifulSoup(html, "lxml")


# --------------------------- trimming ---------------------------

def test_trim_flagged_under_limit():
    text, truncated = _trim_flagged("hello", 100)
    assert text == "hello"
    assert truncated is False


def test_trim_flagged_over_limit():
    text, truncated = _trim_flagged("abcdefghij", 5)
    assert truncated is True
    assert text.startswith("abcde")
    assert "truncated at 5 chars" in text


def test_trim_flagged_limit_zero_disables():
    text, truncated = _trim_flagged("abcdef", 0)
    assert text == "abcdef"
    assert truncated is False


def test_trim_wrapper_returns_text_only():
    assert _trim("abcdef", 3).startswith("abc")
    assert _trim("ab", 100) == "ab"


# --------------------------- JSON-LD ---------------------------

def test_extract_jsonld_single_object():
    html = '<script type="application/ld+json">{"@type":"Article"}</script>'
    out = _extract_jsonld(_soup(html))
    assert out == [{"@type": "Article"}]


def test_extract_jsonld_list_is_flattened():
    html = '<script type="application/ld+json">[{"a":1},{"b":2}]</script>'
    out = _extract_jsonld(_soup(html))
    assert out == [{"a": 1}, {"b": 2}]


def test_extract_jsonld_skips_invalid_json():
    html = '<script type="application/ld+json">{not valid}</script>'
    assert _extract_jsonld(_soup(html)) == []


def test_extract_jsonld_skips_empty():
    html = '<script type="application/ld+json">   </script>'
    assert _extract_jsonld(_soup(html)) == []


# --------------------------- headings outline ---------------------------

def test_headings_outline_levels_and_text():
    html = "<h1>Title</h1><h2>Sub</h2><h3>Deep</h3>"
    out = _headings_outline(_soup(html))
    assert out == [
        {"level": 1, "text": "Title"},
        {"level": 2, "text": "Sub"},
        {"level": 3, "text": "Deep"},
    ]


def test_headings_outline_skips_empty_headings():
    html = "<h1></h1><h2>Real</h2>"
    out = _headings_outline(_soup(html))
    assert out == [{"level": 2, "text": "Real"}]


def test_headings_outline_respects_max_items():
    html = "".join(f"<h2>H{i}</h2>" for i in range(10))
    out = _headings_outline(_soup(html), max_items=3)
    assert len(out) == 3


# --------------------------- TOC from JSON-LD ---------------------------

def test_toc_from_jsonld_recipe_steps():
    jsonld = [{"@type": "Recipe", "recipeInstructions": ["Mix", "Bake"]}]
    assert _toc_from_jsonld(jsonld) == ["Mix", "Bake"]


def test_toc_from_jsonld_recipe_step_dicts():
    jsonld = [{"@type": "HowTo", "step": [{"name": "Step one"}, {"text": "Step two"}]}]
    assert _toc_from_jsonld(jsonld) == ["Step one", "Step two"]


def test_toc_from_jsonld_article_headline():
    jsonld = [{"@type": "NewsArticle", "headline": "Big News"}]
    assert _toc_from_jsonld(jsonld) == ["Big News"]


def test_toc_from_jsonld_empty_returns_none():
    assert _toc_from_jsonld([{"@type": "Person", "name": "Bob"}]) is None


# --------------------------- title / description ---------------------------

def test_page_title_from_title_tag():
    assert _page_title(_soup("<title> Hello </title>")) == "Hello"


def test_page_title_falls_back_to_og_title():
    html = '<meta property="og:title" content="OG Title">'
    assert _page_title(_soup(html)) == "OG Title"


def test_page_title_none_when_absent():
    assert _page_title(_soup("<p>no title</p>")) is None


def test_page_description_meta_name():
    html = '<meta name="description" content="A page.">'
    assert _page_description(_soup(html)) == "A page."


def test_page_description_og_and_twitter_fallback():
    assert _page_description(_soup('<meta property="og:description" content="OG">')) == "OG"
    assert _page_description(_soup('<meta name="twitter:description" content="TW">')) == "TW"


def test_page_description_none_when_absent():
    assert _page_description(_soup("<p>x</p>")) is None


# --------------------------- structured_from_html ---------------------------

def test_structured_from_html_combines_fields():
    html = """
    <html><head><title>My Page</title>
    <meta name="description" content="Desc">
    <script type="application/ld+json">{"@type":"Article","headline":"H"}</script>
    </head><body><h1>Heading</h1></body></html>
    """
    out = _structured_from_html(html, "https://example.com")
    assert out["url"] == "https://example.com"
    assert out["title"] == "My Page"
    assert out["description"] == "Desc"
    assert {"level": 1, "text": "Heading"} in out["headings"]
    assert out["jsonld"] == [{"@type": "Article", "headline": "H"}]
    assert out["toc"] == ["H"]


def test_structured_from_html_no_jsonld_is_none():
    out = _structured_from_html("<title>T</title>", "https://e.com")
    assert out["jsonld"] is None
    assert out["toc"] is None


# --------------------------- plain text ---------------------------

def test_plain_text_strips_scripts_and_styles():
    html = "<body><script>var x=1;</script><style>.a{}</style><p>Visible</p></body>"
    out = _plain_text_from_html(html)
    assert "Visible" in out
    assert "var x" not in out
    assert ".a{" not in out


def test_plain_text_prefers_article():
    html = "<body><nav>Menu</nav><article><p>Body text</p></article></body>"
    out = _plain_text_from_html(html)
    assert "Body text" in out


def test_plain_text_collapses_blank_lines():
    html = "<body><p>a</p>\n\n\n\n<p>b</p></body>"
    out = _plain_text_from_html(html)
    assert "\n\n\n" not in out


# --------------------------- markdown ---------------------------

def test_markdown_keeps_headings_and_resolves_links():
    html = (
        '<body><article><h1>Title</h1>'
        '<p>See <a href="/page">link</a>.</p></article></body>'
    )
    out = _markdown_from_html(html, "https://example.com/base/")
    assert "# Title" in out
    assert "https://example.com/page" in out


def test_markdown_drops_unfollowable_links_keeping_text():
    html = '<body><article><p><a href="#frag">anchor</a> text</p></article></body>'
    out = _markdown_from_html(html, "https://example.com")
    assert "anchor" in out
    assert "#frag" not in out


def test_markdown_strips_nav_and_images():
    html = (
        '<body><nav>NavMenu</nav><article><p>Content</p>'
        '<img src="x.png" alt="img"></article></body>'
    )
    out = _markdown_from_html(html, "https://example.com")
    assert "Content" in out
    assert "NavMenu" not in out
    assert "x.png" not in out


# --------------------------- heading normalization ---------------------------

def test_norm_heading_lowercases_and_strips_punctuation():
    assert _norm_heading("  ## Hello, World ¶ ") == "hello, world"
    assert _norm_heading("Foo Bar") == "foo bar"


def test_norm_heading_collapses_whitespace():
    assert _norm_heading("a    b\tc") == "a b c"


# --------------------------- find_section ---------------------------

SECTION_HTML = """
<body>
  <h2>Introduction</h2>
  <p>Intro paragraph.</p>
  <h2>Methods</h2>
  <p>First method.</p>
  <p>Second method.</p>
  <h3>Subsection</h3>
  <p>Sub detail.</p>
  <h2>Results</h2>
  <p>Results paragraph.</p>
</body>
"""


def test_find_section_exact_match_collects_until_next_equal_heading():
    out = _find_section(_soup(SECTION_HTML), "Methods")
    assert out is not None
    assert out["matched_heading"] == "Methods"
    assert out["level"] == 2
    assert "First method." in out["text"]
    assert "Second method." in out["text"]
    assert "Sub detail." in out["text"]  # nested h3 is included
    assert "Results paragraph." not in out["text"]  # next h2 stops it
    assert out["next_heading"] == "Results"


def test_find_section_fuzzy_match():
    out = _find_section(_soup(SECTION_HTML), "method")
    assert out is not None
    assert out["matched_heading"] == "Methods"


def test_find_section_not_found_returns_none():
    assert _find_section(_soup(SECTION_HTML), "Nonexistent Heading") is None


def test_find_section_empty_query_returns_none():
    assert _find_section(_soup(SECTION_HTML), "") is None


def test_find_section_no_headings_returns_none():
    assert _find_section(_soup("<p>no headings here</p>"), "anything") is None
