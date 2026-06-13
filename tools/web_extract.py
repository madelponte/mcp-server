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
    """Extract useful 'table of contents'-like info from JSON-LD when possible."""
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
            if isinstance(t, str) and t.endswith("Article"):
                hl = obj.get("headline")
                if hl and hl not in toc:
                    toc.append(str(hl).strip())
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(jsonld)
    return toc or None


def _page_title(soup: BeautifulSoup) -> str | None:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()
    return None


def _page_description(soup: BeautifulSoup) -> str | None:
    for sel in [
        ("meta", {"name": "description"}),
        ("meta", {"property": "og:description"}),
        ("meta", {"name": "twitter:description"}),
    ]:
        tag = soup.find(*sel)
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def _structured_from_html(html: str, url: str) -> dict:
    """Return a structured representation of the page."""
    soup = BeautifulSoup(html, "lxml")
    jsonld = _extract_jsonld(soup)
    return {
        "url": url,
        "title": _page_title(soup),
        "description": _page_description(soup),
        "headings": _headings_outline(soup),
        "jsonld": jsonld if jsonld else None,
        "toc": _toc_from_jsonld(jsonld),
    }


# ---------------------------------------------------------------------------
# Readable-text rendering
# ---------------------------------------------------------------------------

def _plain_text_from_html(html: str) -> str:
    """Strip scripts/styles/nav and return readable text."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _markdown_from_html(html: str, base_url: str) -> str:
    """Convert page HTML to markdown, keeping the structure plain text loses.

    Headings, lists, tables, and hyperlinks survive the conversion, so the model
    sees the page the way a reader does — and can follow a link it found in the
    content with another fetch_page call. Link hrefs are resolved against
    `base_url` to absolute URLs for exactly that reason. Non-content elements
    (scripts, nav, site header/footer chrome, forms) and images are dropped;
    `escape_*` are off because the output feeds a model, not a markdown renderer.
    """
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg",
                   "nav", "aside", "form", "button"]):
        t.decompose()
    # Site-chrome headers/footers are noise, but a header *inside* the article
    # carries its headline — drop only the ones outside any article/main.
    for t in soup(["header", "footer"]):
        if t.find_parent(["article", "main"]) is None:
            t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup

    for a in root.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "data:")):
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


# ---------------------------------------------------------------------------
# Single-section extraction
# ---------------------------------------------------------------------------

def _norm_heading(s: str) -> str:
    """Normalize a heading for fuzzy comparison."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("¶#§*•·.:;-—–_ ")
    return s


def _find_section(soup: BeautifulSoup, section: str) -> dict | None:
    """Locate a heading matching `section` and return text up to the next equal/higher heading."""
    if not section:
        return None

    target = _norm_heading(section)
    if not target:
        return None

    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()

    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if not headings:
        return None

    matched = None
    for h in headings:
        if _norm_heading(h.get_text(" ", strip=True)) == target:
            matched = h
            break
    if matched is None:
        for h in headings:
            ht = _norm_heading(h.get_text(" ", strip=True))
            if target in ht or ht in target:
                if ht and len(ht) >= 3:
                    matched = h
                    break
    if matched is None:
        return None

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
