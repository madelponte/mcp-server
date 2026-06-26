"""
HTML → content extraction helpers, shared by the web-search and fetch-page tools.

Pure, side-effect-free functions that turn a fetched HTML document into the
shapes the tools return: a markdown/plain-text rendering, a structured metadata
summary (title, description, heading outline, JSON-LD), or a single named
section. `search_web` uses `_structured_from_html` to enrich its results;
`fetch_page` uses the whole set. Kept free of any config or network dependency so
both tools can import it without coupling.
"""

import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter


# ---------------------------------------------------------------------------
# Text truncation
#
# Both tools cap how much text they return to protect the model's context
# window — `search_web` on each result snippet, `fetch_page` on page content
# (where `_trim_flagged`'s `truncated` flag also drives the offset-paging hint).
# ---------------------------------------------------------------------------

def _trim_flagged(text: str, limit: int) -> tuple[str, bool]:
    """Trim `text` to `limit` chars, also reporting whether anything was dropped.

    Returns ``(text, truncated)``. ``limit <= 0`` disables trimming. The marker
    appended on truncation keeps the old visible cue in the text itself; the
    boolean lets callers also flag it structurally on the payload.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + f"\n\n[... truncated at {limit} chars ...]", True


def _trim(text: str, limit: int) -> str:
    return _trim_flagged(text, limit)[0]


# ---------------------------------------------------------------------------
# Structured metadata extraction
# ---------------------------------------------------------------------------

def _extract_jsonld(soup: BeautifulSoup) -> list:
    """Pull JSON-LD structured data blocks from <script type=application/ld+json>."""
    out = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            out.extend(parsed)
        else:
            out.append(parsed)
    return out


def _headings_outline(soup: BeautifulSoup, max_items: int = 40) -> list[dict]:
    """Build a lightweight 'table of contents' from heading tags."""
    outline = []
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = " ".join(h.get_text(" ", strip=True).split())
        if not text:
            continue
        outline.append({"level": int(h.name[1]), "text": text})
        if len(outline) >= max_items:
            break
    return outline


def _toc_from_jsonld(jsonld: list) -> list[str] | None:
    """Step lists from Recipe/HowTo JSON-LD — content that reads as a table of
    contents but isn't expressed with heading tags.

    Deliberately does *not* treat an Article-type ``headline`` as a TOC entry: a
    headline is a one-line description, not a list of sections, and surfacing it
    here produced a bogus single-item "toc" (see ``_description_from_jsonld``,
    which now consumes the headline instead). The real table of contents for a
    normal page comes from its heading outline (see ``_table_of_contents``)."""
    toc: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            t = obj.get("@type")
            if t in ("Recipe", "HowTo") or (isinstance(t, list) and ("Recipe" in t or "HowTo" in t)):
                steps = obj.get("recipeInstructions") or obj.get("step") or []
                if isinstance(steps, list):
                    for s in steps:
                        if isinstance(s, str):
                            toc.append(s.strip())
                        elif isinstance(s, dict):
                            name = s.get("name") or s.get("text") or ""
                            if name:
                                toc.append(str(name).strip())
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(jsonld)
    return toc or None


def _description_from_jsonld(jsonld: list, title: str | None = None) -> str | None:
    """Pull a page description out of JSON-LD when the meta tags lack one.

    Many pages (Wikipedia among them) carry no ``<meta name="description">`` but
    do expose the summary in JSON-LD: a ``description``/``abstract`` field, or —
    as Wikipedia does — an Article ``headline`` holding the short description.
    Prefers an explicit ``description``/``abstract``; falls back to ``headline``
    only when it actually differs from the page title (otherwise it's just the
    title restated, not a description)."""
    descriptions: list[str] = []
    headlines: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            for key in ("description", "abstract"):
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    descriptions.append(v.strip())
            hl = obj.get("headline")
            if isinstance(hl, str) and hl.strip():
                headlines.append(hl.strip())
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(jsonld)
    if descriptions:
        return descriptions[0]
    norm_title = _norm_heading(title) if title else ""
    for hl in headlines:
        if _norm_heading(hl) != norm_title:
            return hl
    return None


def _table_of_contents(outline: list[dict], jsonld: list) -> list[str] | None:
    """The page's table of contents as a flat list of section titles.

    A real TOC is the heading outline, so use that when the page has headings;
    fall back to Recipe/HowTo step lists for step-based pages that carry no
    headings. Returns ``None`` when neither is available."""
    if outline:
        return [h["text"] for h in outline]
    return _toc_from_jsonld(jsonld) or None


def _page_title(soup: BeautifulSoup) -> str | None:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()
    return None


def _page_description(
    soup: BeautifulSoup, jsonld: list | None = None, title: str | None = None
) -> str | None:
    for sel in [
        ("meta", {"name": "description"}),
        ("meta", {"property": "og:description"}),
        ("meta", {"name": "twitter:description"}),
    ]:
        tag = soup.find(*sel)
        if tag and tag.get("content") and tag["content"].strip():
            return tag["content"].strip()
    # No usable meta description — fall back to JSON-LD (e.g. Wikipedia, which
    # exposes its short description only there, not as a meta tag).
    if jsonld:
        return _description_from_jsonld(jsonld, title)
    return None


def _structured_from_html(html: str, url: str) -> dict:
    """Return a structured representation of the whole page."""
    soup = BeautifulSoup(html, "lxml")
    jsonld = _extract_jsonld(soup)
    title = _page_title(soup)
    outline = _headings_outline(soup)
    return {
        "url": url,
        "title": title,
        "description": _page_description(soup, jsonld, title),
        "headings": outline,
        "jsonld": jsonld if jsonld else None,
        "toc": _table_of_contents(outline, jsonld),
    }


def _structured_section_from_html(
    html: str, url: str, section: str
) -> tuple[dict | None, list[str]]:
    """Structured metadata scoped to a single heading's subtree.

    Returns ``(payload, [])`` when the heading is found — ``headings``/``toc``
    then cover only that section and its sub-headings (down to the next heading
    at the same or higher level), not the whole page. Returns ``(None,
    available)`` when the heading isn't found, where ``available`` lists the
    page's headings for the caller's error message — mirroring ``_find_section``
    / ``_parse_section`` in the text path."""
    soup = BeautifulSoup(html, "lxml")
    jsonld = _extract_jsonld(soup)
    title = _page_title(soup)
    matched = _locate_heading(soup, section)
    if matched is None:
        available = [
            " ".join(h.get_text(" ", strip=True).split())
            for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        ]
        return None, [a for a in available if a]
    outline = _section_outline(matched)
    return {
        "url": url,
        "title": title,
        "section": outline[0]["text"],
        "description": _page_description(soup, jsonld, title),
        "headings": outline,
        "jsonld": jsonld if jsonld else None,
        "toc": [h["text"] for h in outline] or None,
    }, []


# ---------------------------------------------------------------------------
# Readable-text rendering
# ---------------------------------------------------------------------------

def _plain_text_from_soup(soup: BeautifulSoup) -> str:
    """Strip scripts/styles/nav from an already-parsed soup and return readable
    text. Mutates ``soup`` (decompose), so the caller must not reuse it after."""
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text)


def _plain_text_from_html(html: str) -> str:
    """Strip scripts/styles/nav and return readable text."""
    return _plain_text_from_soup(BeautifulSoup(html, "lxml"))


# ---------------------------------------------------------------------------
# Reference / citation links
#
# A [12]-style citation marker is an in-page anchor (`<a href="#cite_note-12">`)
# whose target footnote holds the real external source URL. We resolve those so a
# model can follow a page's references for deeper research, instead of dropping
# them as dead in-page links. This is the general footnote pattern (MediaWiki,
# many docs/academic pages), not a Wikipedia special case.
# ---------------------------------------------------------------------------

def _external_href(href: str, base_url: str) -> str | None:
    """Absolute http(s) URL for `href`, or None if it isn't independently
    followable (an in-page anchor, javascript:/data:/mailto:, or empty)."""
    href = (href or "").strip()
    if not href or href.startswith(("#", "javascript:", "data:", "mailto:")):
        return None
    return urljoin(base_url, href)


def _resolve_citation_href(id_map: dict, href: str, base_url: str) -> str | None:
    """For an in-page citation marker (e.g. [12] → ``#cite_note-12``), return the
    external source URL its footnote points to, so the marker becomes followable.
    None when the target is missing or is a plain explanatory note with no link.
    `id_map` maps element id → element (built once by the caller)."""
    frag = href[1:] if href.startswith("#") else href
    target = id_map.get(frag) if frag else None
    if target is None:
        return None
    for ext in target.find_all("a", href=True):
        url = _external_href(ext["href"], base_url)
        if url:
            return url
    return None


def _inline_link_urls(soup: BeautifulSoup, base_url: str) -> None:
    """Rewrite each link's text to ``text (url)`` so the URL survives
    ``get_text()`` (which drops href attributes). In-page citation markers are
    resolved to their source URL first. Used by the plain-text *section* path,
    where links would otherwise lose their targets — so a fetched section's
    references/citations stay followable. Mutates `soup`."""
    id_map = None
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"):
            if id_map is None:
                id_map = {t["id"]: t for t in soup.find_all(id=True)}
            url = _resolve_citation_href(id_map, href, base_url)
        else:
            url = _external_href(href, base_url)
        if not url:
            continue
        text = a.get_text(" ", strip=True)
        a.clear()
        a.append(f"{text} ({url})" if text else url)


def _markdown_from_soup(soup: BeautifulSoup, base_url: str) -> str:
    """Convert an already-parsed soup to markdown (see ``_markdown_from_html``).

    Mutates ``soup`` (decompose/unwrap), so the caller must extract anything else
    it needs (e.g. the title) before calling this and must not reuse the soup
    afterward. Split out so a caller can parse the HTML once and reuse the soup."""
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg",
                   "nav", "aside", "form", "button"]):
        t.decompose()
    # Site-chrome headers/footers are noise, but a header *inside* the article
    # carries its headline — drop only the ones outside any article/main.
    for t in soup(["header", "footer"]):
        if t.find_parent(["article", "main"]) is None:
            t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup

    id_map = None
    for a in root.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"):
            # A citation marker → point it at its source so it's followable;
            # a plain in-page anchor with no source is dropped (text kept).
            if id_map is None:
                id_map = {t["id"]: t for t in soup.find_all(id=True)}
            url = _resolve_citation_href(id_map, href, base_url)
            if url:
                a["href"] = url
            else:
                a.unwrap()
        elif not href or href.startswith(("javascript:", "data:")):
            a.unwrap()  # keep the text, drop the unfollowable link
        else:
            a["href"] = urljoin(base_url, href)

    md = MarkdownConverter(
        heading_style="atx",
        bullets="-",
        strip=["img"],
        escape_asterisks=False,
        escape_underscores=False,
    ).convert_soup(root)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def _markdown_from_html(html: str, base_url: str) -> str:
    """Convert page HTML to markdown, keeping the structure plain text loses.

    Headings, lists, tables, and hyperlinks survive the conversion, so the model
    sees the page the way a reader does — and can follow a link it found in the
    content with another fetch_page call. Link hrefs are resolved against
    `base_url` to absolute URLs for exactly that reason. Non-content elements
    (scripts, nav, site header/footer chrome, forms) and images are dropped;
    `escape_*` are off because the output feeds a model, not a markdown renderer.
    """
    return _markdown_from_soup(BeautifulSoup(html, "lxml"), base_url)


# ---------------------------------------------------------------------------
# Single-section extraction
# ---------------------------------------------------------------------------

def _norm_heading(s: str) -> str:
    """Normalize a heading for fuzzy comparison."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("¶#§*•·.:;-—–_ ")
    return s


def _locate_heading(soup: BeautifulSoup, section: str):
    """Find the heading tag matching `section`, or None.

    Prefers an exact (normalized) match, then a fuzzy substring match (either
    direction) of at least three chars. Shared by `_find_section` (text mode) and
    `_structured_section_from_html` (structured mode) so both scope to the same
    heading. Does not mutate the soup, so a caller can read JSON-LD/metadata from
    it afterwards."""
    target = _norm_heading(section)
    if not target:
        return None
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if not headings:
        return None
    for h in headings:
        if _norm_heading(h.get_text(" ", strip=True)) == target:
            return h
    for h in headings:
        ht = _norm_heading(h.get_text(" ", strip=True))
        if (target in ht or ht in target) and ht and len(ht) >= 3:
            return h
    return None


def _section_outline(matched, max_items: int = 40) -> list[dict]:
    """Heading outline for one section: the matched heading itself followed by
    its sub-headings, stopping at the next heading of the same or higher level."""
    matched_level = int(matched.name[1])
    outline = [{"level": matched_level, "text": " ".join(matched.get_text(" ", strip=True).split())}]
    for el in matched.find_all_next():
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            try:
                lvl = int(el.name[1])
            except ValueError:
                continue
            if lvl <= matched_level:
                break
            text = " ".join(el.get_text(" ", strip=True).split())
            if text:
                outline.append({"level": lvl, "text": text})
                if len(outline) >= max_items:
                    break
    return outline


def _find_section(soup: BeautifulSoup, section: str, base_url: str = "") -> dict | None:
    """Locate a heading matching `section` and return text up to the next equal/higher heading.

    `base_url` lets reference/citation links inside the section be resolved to
    followable absolute URLs (inlined into the text, since the section is rendered
    as plain text) — so a model can chase a fetched section's sources."""
    if not section:
        return None

    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()

    matched = _locate_heading(soup, section)
    if matched is None:
        return None

    # Make links in the section followable before the text is flattened.
    _inline_link_urls(soup, base_url)

    matched_level = int(matched.name[1])
    matched_text = " ".join(matched.get_text(" ", strip=True).split())

    pieces: list[str] = []
    next_heading_text: str | None = None

    for el in matched.find_all_next():
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            try:
                lvl = int(el.name[1])
            except ValueError:
                lvl = 99
            if lvl <= matched_level:
                next_heading_text = " ".join(el.get_text(" ", strip=True).split())
                break
            sub = " ".join(el.get_text(" ", strip=True).split())
            if sub:
                pieces.append(f"\n## {sub}\n")
            continue
        if el.name in ("p", "li", "pre", "code", "blockquote", "td", "th", "dd", "dt", "figcaption"):
            txt = el.get_text(" ", strip=True)
            if txt:
                pieces.append(txt)

    if not pieces:
        collected: list[str] = []
        for el in matched.find_all_next(string=False):
            if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                try:
                    lvl = int(el.name[1])
                except ValueError:
                    lvl = 99
                if lvl <= matched_level:
                    if next_heading_text is None:
                        next_heading_text = " ".join(el.get_text(" ", strip=True).split())
                    break
            txt = el.get_text(" ", strip=True) if hasattr(el, "get_text") else ""
            if txt and txt not in collected:
                collected.append(txt)
        body_text = "\n\n".join(collected)
    else:
        body_text = "\n\n".join(pieces)

    body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()

    return {
        "matched_heading": matched_text,
        "level": matched_level,
        "text": body_text,
        "next_heading": next_heading_text,
    }
