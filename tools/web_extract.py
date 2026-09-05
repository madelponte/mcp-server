"""
HTML → content extraction helpers for fetch_page.

Pure, side-effect-free functions that turn a fetched HTML document into the
shapes the tools return: a markdown/plain-text rendering, a structured metadata
summary (title, description, heading outline, JSON-LD), or a single named
section. Kept free of any config or network dependency.
"""

import json
import re
import unicodedata
from urllib.parse import quote, unquote, urldefrag, urljoin, urlsplit

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter


# ---------------------------------------------------------------------------
# Text truncation
#
# fetch_page caps returned content to protect the model's context window.
# `_trim_flagged`'s `truncated` flag also drives the offset-paging hint.
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


_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_SOURCE_ANCHOR_RE = re.compile(r"^[^\s{}#]+$")


def _anchor_slug(value: str) -> str:
    """Deterministic, readable local anchor for heading text without an id."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", "-", value, flags=re.UNICODE).strip("-_")
    return value[:80] or "section"


def _is_heading_permalink(link) -> bool:
    href = str(link.get("href") or "").strip()
    if not href.startswith("#") or len(href) == 1:
        return False
    classes = {str(value).casefold() for value in link.get("class", [])}
    label = " ".join(
        str(link.get(field) or "") for field in ("title", "aria-label")
    ).casefold()
    return bool(
        classes.intersection({"headerlink", "anchor", "heading-anchor", "permalink"})
        or "permalink" in label
        or "link to this heading" in label
    )


def _is_decorative_heading_permalink(link) -> bool:
    if not _is_heading_permalink(link):
        return False
    classes = {str(value).casefold() for value in link.get("class", [])}
    text = " ".join(link.get_text(" ", strip=True).split())
    return bool(
        classes.intersection({"headerlink", "heading-anchor", "permalink"})
        or str(link.get("aria-hidden") or "").casefold() == "true"
        or text in {"", "#", "¶", "§", "🔗"}
    )


def _heading_text(heading) -> str:
    """Visible heading text without decorative permalink glyphs such as ``¶``."""
    parts = []
    for node in heading.find_all(string=True):
        link = (
            node.parent
            if getattr(node.parent, "name", None) == "a"
            else node.parent.find_parent("a")
        )
        if (
            link is not None
            and heading in link.parents
            and _is_decorative_heading_permalink(link)
        ):
            continue
        value = str(node).strip()
        if value:
            parts.append(value)
    return " ".join(" ".join(parts).split())


def _source_heading_fragment(heading) -> str:
    raw_fragment = heading.get("id")
    if raw_fragment is not None and str(raw_fragment).strip():
        return str(raw_fragment).strip()
    for link in heading.find_all("a", href=True):
        if _is_heading_permalink(link):
            return unquote(str(link["href"])[1:]).strip()
    parent = heading.parent
    if getattr(parent, "name", None) == "section" and parent.get("id"):
        return str(parent["id"]).strip()
    return ""


def _heading_anchor_records(soup: BeautifulSoup, base_url: str = "") -> dict[int, dict]:
    """Map heading tag identity to stable local/source citation metadata.

    A safe source fragment from the heading ID, permalink, or parent section is
    preserved exactly, making ``url#id`` directly citeable. Headings without one
    receive a deterministic
    ``cite-<slug>`` anchor that is stable within repeated extractions. Duplicate
    local anchors get document-order suffixes.
    """
    records: dict[int, dict] = {}
    used: dict[str, int] = {}
    source_url = urldefrag(base_url)[0]
    for heading in soup.find_all(_HEADING_TAGS):
        text = _heading_text(heading)
        if not text:
            continue
        fragment = _source_heading_fragment(heading)
        source_anchor = bool(fragment and _SOURCE_ANCHOR_RE.fullmatch(fragment))
        base_anchor = fragment if source_anchor else f"cite-{_anchor_slug(text)}"
        count = used.get(base_anchor, 0) + 1
        used[base_anchor] = count
        anchor = base_anchor if count == 1 else f"{base_anchor}-{count}"
        record = {
            "level": int(heading.name[1]),
            "text": text,
            "anchor": anchor,
        }
        if fragment:
            record["source_fragment"] = fragment
            if source_url:
                encoded = quote(fragment, safe="!$&'()*+,;=:@/?-._~")
                record["citation_url"] = f"{source_url}#{encoded}"
        records[id(heading)] = record
    return records


def _annotate_heading_anchors(soup: BeautifulSoup, root, base_url: str) -> None:
    """Append visible ``{#anchor}`` citation markers to content headings."""
    records = _heading_anchor_records(soup, base_url)
    for heading in root.find_all(_HEADING_TAGS):
        record = records.get(id(heading))
        if record is not None:
            for link in heading.find_all("a", href=True):
                if _is_decorative_heading_permalink(link):
                    link.decompose()
            heading.append(f" {{#{record['anchor']}}}")


def _headings_outline(
    soup: BeautifulSoup,
    max_items: int = 40,
    base_url: str | None = None,
) -> list[dict]:
    """Build a lightweight table of contents, with citation anchors when asked."""
    records = _heading_anchor_records(soup, base_url or "") if base_url is not None else None
    outline = []
    for h in soup.find_all(("h1", "h2", "h3", "h4")):
        text = _heading_text(h)
        if not text:
            continue
        item = {"level": int(h.name[1]), "text": text}
        if records is not None:
            item.update(records[id(h)])
        outline.append(item)
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


def _structured_from_html(html: str, url: str, max_images: int = 10) -> dict:
    """Return a structured representation of the whole page."""
    soup = BeautifulSoup(html, "lxml")
    jsonld = _extract_jsonld(soup)
    title = _page_title(soup)
    outline = _headings_outline(soup, base_url=url)
    root = soup.find("article") or soup.find("main") or soup.body or soup
    images = _prominent_image_records(soup, root, url, max_images=max_images)
    out = {
        "url": url,
        "title": title,
        "description": _page_description(soup, jsonld, title),
        "headings": outline,
        "jsonld": jsonld if jsonld else None,
        "toc": _table_of_contents(outline, jsonld),
    }
    if images:
        out["images"] = [item[2] for item in images]
        out["image_note"] = _IMAGE_NOTE
    return out


def _structured_section_from_html(
    html: str, url: str, section: str, max_images: int = 10
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
            _heading_text(h)
            for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        ]
        return None, [a for a in available if a]
    anchor_records = _heading_anchor_records(soup, url)
    outline = _section_outline(matched, anchor_records=anchor_records)
    root = soup.find("article") or soup.find("main") or soup.body or soup
    candidates = _section_image_candidates(matched)
    images = _prominent_image_records(
        soup, root, url, max_images=max_images, candidates=candidates
    )
    out = {
        "url": url,
        "title": title,
        "section": outline[0]["text"],
        "description": _page_description(soup, jsonld, title),
        "headings": outline,
        "jsonld": jsonld if jsonld else None,
        "toc": [h["text"] for h in outline] or None,
    }
    if images:
        out["images"] = [item[2] for item in images]
        out["image_note"] = _IMAGE_NOTE
    return out, []


# ---------------------------------------------------------------------------
# Prominent image descriptions
# ---------------------------------------------------------------------------

_IMAGE_DESCRIPTION_LIMIT = 300
_IMAGE_PLACEHOLDER_ATTR = "data-mcp-image-placeholder"
_IMAGE_NOTE = (
    "Each description replaces a prominent image at its page location and comes "
    "from page-provided alt text, captions, or metadata, not visual analysis."
)
_IMAGE_NOISE_RE = re.compile(
    r"(?:^|[-_\s])(icon|logo|avatar|emoji|sprite|badge|tracking|pixel|spacer)(?:$|[-_\s])",
    re.I,
)
_IMAGE_PROMINENT_RE = re.compile(
    r"(?:^|[-_\s])(hero|featured|lead|cover|main[-_]?image|article[-_]?image|wp[-_]?post[-_]?image)(?:$|[-_\s])",
    re.I,
)


def _clean_image_description(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    if not text:
        return None
    return text[:_IMAGE_DESCRIPTION_LIMIT].rstrip()


def _standalone_image_description(
    text: str | None, body: bytes | None
) -> tuple[str | None, str | None]:
    """Read an SVG's embedded title/description without inspecting its pixels."""
    source = text
    if source is None and body:
        source = body.decode("utf-8", errors="replace")
    if not source or "<svg" not in source[:2000].lower():
        return None, None
    soup = BeautifulSoup(source, "xml")
    title_tag = soup.find("title")
    desc_tag = soup.find("desc")
    title = _clean_image_description(
        title_tag.get_text(" ", strip=True) if title_tag else None
    )
    description = _clean_image_description(
        desc_tag.get_text(" ", strip=True) if desc_tag else None
    )
    if title and description and title.casefold() != description.casefold():
        return f"{title} — {description}", "embedded SVG title+description"
    if description:
        return description, "embedded SVG description"
    if title:
        return title, "embedded SVG title"
    return None, None


def _image_src(img, base_url: str) -> str | None:
    raw = (
        img.get("src")
        or img.get("data-src")
        or img.get("data-lazy-src")
        or ""
    ).strip()
    if not raw:
        srcset = (img.get("srcset") or img.get("data-srcset") or "").strip()
        raw = srcset.split(",", 1)[0].strip().split(" ", 1)[0] if srcset else ""
    if not raw or raw.startswith(("data:", "blob:", "javascript:")):
        return None
    resolved = urljoin(base_url, raw)
    return resolved[:500] if resolved.startswith(("http://", "https://")) else None


def _metadata_image_descriptions(
    soup: BeautifulSoup, base_url: str
) -> tuple[dict[str, str], set[str]]:
    """Map OpenGraph/Twitter/JSON-LD image URLs to their supplied descriptions."""
    descriptions: dict[str, str] = {}
    prominent_urls: set[str] = set()
    current: dict[str, str] = {}

    for meta in soup.find_all("meta"):
        key = (meta.get("property") or meta.get("name") or "").strip().lower()
        raw = meta.get("content")
        raw = " ".join(raw.split()).strip() if isinstance(raw, str) else ""
        if not raw:
            continue
        value = _clean_image_description(raw)
        if key in ("og:image", "og:image:url", "og:image:secure_url"):
            current["og"] = urljoin(base_url, raw)
            prominent_urls.add(current["og"])
        elif key == "og:image:alt" and current.get("og"):
            descriptions[current["og"]] = value
        elif key in ("twitter:image", "twitter:image:src"):
            current["twitter"] = urljoin(base_url, raw)
            prominent_urls.add(current["twitter"])
        elif key == "twitter:image:alt" and current.get("twitter"):
            descriptions[current["twitter"]] = value

    def walk(obj):
        if isinstance(obj, dict):
            kind = obj.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if "ImageObject" in kinds:
                raw_url = obj.get("contentUrl") or obj.get("url")
                if isinstance(raw_url, str) and raw_url.strip():
                    image_url = urljoin(base_url, raw_url.strip())
                    prominent_urls.add(image_url)
                    desc = None
                    for field in ("caption", "description", "name"):
                        desc = _clean_image_description(obj.get(field))
                        if desc:
                            break
                    if desc:
                        descriptions[image_url] = desc
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(_extract_jsonld(soup))
    return descriptions, prominent_urls


def _image_dimension(value) -> int | None:
    match = re.match(r"\s*(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def _section_image_candidates(matched) -> list:
    matched_level = int(matched.name[1])
    images = []
    for element in matched.find_all_next():
        if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            try:
                if int(element.name[1]) <= matched_level:
                    break
            except ValueError:
                pass
        if element.name == "img":
            images.append(element)
    return images


def _prominent_image_records(
    soup: BeautifulSoup,
    root,
    base_url: str,
    *,
    max_images: int = 10,
    candidates: list | None = None,
) -> list[tuple]:
    """Return ``(img, consumed_caption, public_record, marker)`` tuples."""
    if max_images <= 0:
        return []
    metadata, metadata_urls = _metadata_image_descriptions(soup, base_url)
    images = list(candidates) if candidates is not None else list(root.find_all("img"))
    records = []
    consumed_figures: set[int] = set()

    for img in images:
        if img.find_parent(["nav", "aside", "form", "button"]):
            continue
        chrome = img.find_parent(["header", "footer"])
        if chrome is not None and img.find_parent(["article", "main"]) is None:
            continue
        image_url = _image_src(img, base_url)
        figure = img.find_parent("figure")
        figure_key = id(figure) if figure is not None else None
        caption_tag = figure.find("figcaption") if figure is not None else None
        caption = (
            _clean_image_description(caption_tag.get_text(" ", strip=True))
            if caption_tag is not None and figure_key not in consumed_figures
            else None
        )
        alt = _clean_image_description(img.get("alt"))
        short = (
            _clean_image_description(img.get("aria-label"))
            or _clean_image_description(img.get("title"))
            or (metadata.get(image_url) if image_url else None)
        )
        descriptions = []
        sources = []
        for value, source in ((alt, "alt"), (caption, "caption"), (short, "metadata")):
            if value and value.casefold() not in {item.casefold() for item in descriptions}:
                descriptions.append(value)
                sources.append(source)
        description = descriptions[0] if descriptions else None
        if description and len(descriptions) > 1:
            label = "Caption" if sources[1] == "caption" else "Description"
            description += f" — {label}: {descriptions[1]}"
        description = _clean_image_description(description)

        parent = img.parent if getattr(img.parent, "attrs", None) is not None else None
        attrs = " ".join(
            str(value)
            for value in (
                img.get("id", ""),
                " ".join(img.get("class") or []),
                img.get("role", ""),
                parent.get("id", "") if parent else "",
                " ".join(parent.get("class") or []) if parent else "",
            )
        )
        hero = bool(_IMAGE_PROMINENT_RE.search(attrs)) or img.get("fetchpriority") == "high"
        metadata_image = bool(image_url and image_url in metadata_urls)
        width = _image_dimension(img.get("width"))
        height = _image_dimension(img.get("height"))
        known_small = bool(
            (width is not None and height is not None and width <= 96 and height <= 96)
            or (width is not None and height is None and width <= 96)
            or (height is not None and width is None and height <= 96)
        )
        large = bool(
            (width is not None and width >= 480)
            or (height is not None and height >= 300)
            or (
                width is not None
                and height is not None
                and width >= 300
                and height >= 180
            )
        )
        in_content = img.find_parent(["article", "main"]) is not None
        noisy = bool(_IMAGE_NOISE_RE.search(attrs)) or img.get("aria-hidden") == "true"
        if (known_small or noisy) and not (figure is not None or hero or metadata_image):
            continue
        if not (
            figure is not None
            or hero
            or metadata_image
            or large
            or in_content
            or bool(description)
        ):
            continue

        marker = (
            f"[Image at this location: {description}]"
            if description
            else "[Image at this location: no textual description was provided.]"
        )
        public = {"replaces_image": True}
        if description:
            public["description"] = description
            public["description_source"] = "+".join(sources[:2])
        if image_url:
            public["url"] = image_url
        records.append((img, caption_tag if caption else None, public, marker))
        if caption and figure_key is not None:
            consumed_figures.add(figure_key)
        if len(records) >= max_images:
            break
    return records


def _replace_prominent_images(
    soup: BeautifulSoup,
    root,
    base_url: str,
    max_images: int,
    candidates: list | None = None,
) -> list[dict]:
    records = _prominent_image_records(
        soup, root, base_url, max_images=max_images, candidates=candidates
    )
    replaced = []
    for img, caption_tag, public, marker in records:
        # Replacing an earlier figure image can remove its caption, including
        # any nested image that was also selected as a candidate. BeautifulSoup
        # raises if ``replace_with`` is called on that now-detached image.
        if img.parent is None:
            continue
        placeholder = soup.new_tag("span")
        placeholder[_IMAGE_PLACEHOLDER_ATTR] = "true"
        placeholder.string = marker
        img.replace_with(placeholder)
        replaced.append(public)
        if caption_tag is not None and caption_tag.parent is not None:
            caption_tag.decompose()
    return replaced


# ---------------------------------------------------------------------------
# Readable-text rendering
# ---------------------------------------------------------------------------

def _plain_text_from_soup(
    soup: BeautifulSoup, base_url: str = "", max_images: int = 10
) -> str:
    """Strip scripts/styles/nav from an already-parsed soup and return readable
    text. Mutates ``soup`` (decompose), so the caller must not reuse it after."""
    root = soup.find("article") or soup.find("main") or soup.body or soup
    _replace_prominent_images(soup, root, base_url, max_images)
    _annotate_heading_anchors(soup, root, base_url)
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()
    text = root.get_text("\n", strip=True)
    # ``get_text`` separates the marker node appended to a heading; fold it back
    # onto that heading's line so plain-text and markdown modes expose the same
    # citation syntax.
    text = re.sub(r"\n(?=\{#[^{}\s]+\}(?:\n|$))", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _plain_text_from_html(
    html: str, base_url: str = "", max_images: int = 10
) -> str:
    """Strip scripts/styles/nav and return readable text with image markers."""
    return _plain_text_from_soup(
        BeautifulSoup(html, "lxml"), base_url, max_images
    )


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


def _is_citation_marker(link) -> bool:
    """Whether an in-page link has common footnote/reference semantics."""
    href = str(link.get("href") or "").strip()
    if not href.startswith("#"):
        return False
    fragment = unquote(href[1:]).casefold()
    if re.match(r"^(?:cite|citation|ref|reference|fn|footnote|note)[_:-]", fragment):
        return True
    role = str(link.get("role") or "").casefold()
    if role in {"doc-noteref", "doc-biblioref"}:
        return True
    for tag in (link, *list(link.parents)[:2]):
        classes = {str(value).casefold() for value in tag.get("class", [])}
        if classes.intersection(
            {"citation", "reference", "references", "footnote", "footnote-ref"}
        ):
            return True
    return False


def _in_page_url(href: str, base_url: str, id_map: dict) -> str | None:
    fragment = unquote(href[1:]) if href.startswith("#") else ""
    source_url = urldefrag(base_url)[0]
    if not fragment or fragment not in id_map or not source_url:
        return None
    return f"{source_url}{href}"


def _resolve_citation_href(id_map: dict, href: str, base_url: str) -> str | None:
    """For an in-page citation marker (e.g. [12] → ``#cite_note-12``), return the
    external source URL its footnote points to, so the marker becomes followable.
    None when the target is missing or is a plain explanatory note with no link.
    `id_map` maps element id → element (built once by the caller)."""
    frag = unquote(href[1:] if href.startswith("#") else href)
    target = id_map.get(frag) if frag else None
    if target is None:
        return None
    base_host = urlsplit(base_url).netloc.casefold()
    for ext in target.find_all("a", href=True):
        url = _external_href(ext["href"], base_url)
        if not url:
            continue
        classes = {str(value).casefold() for value in ext.get("class", [])}
        rels = {str(value).casefold() for value in ext.get("rel", [])}
        link_host = urlsplit(url).netloc.casefold()
        explicitly_external = bool(
            "external" in classes
            or "mw:extlink" in rels
            or (link_host and base_host and link_host != base_host)
        )
        if explicitly_external:
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
            if _is_citation_marker(a):
                url = _resolve_citation_href(id_map, href, base_url)
            else:
                url = _in_page_url(href, base_url, id_map)
        else:
            url = _external_href(href, base_url)
        if not url:
            continue
        text = a.get_text(" ", strip=True)
        a.clear()
        a.append(f"{text} ({url})" if text else url)


def _markdown_from_soup(
    soup: BeautifulSoup, base_url: str, max_images: int = 10
) -> str:
    """Convert an already-parsed soup to markdown (see ``_markdown_from_html``).

    Mutates ``soup`` (decompose/unwrap), so the caller must extract anything else
    it needs (e.g. the title) before calling this and must not reuse the soup
    afterward. Split out so a caller can parse the HTML once and reuse the soup."""
    root = soup.find("article") or soup.find("main") or soup.body or soup
    _replace_prominent_images(soup, root, base_url, max_images)
    _annotate_heading_anchors(soup, root, base_url)
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg",
                   "nav", "aside", "form", "button"]):
        t.decompose()
    # Site-chrome headers/footers are noise, but a header *inside* the article
    # carries its headline — drop only the ones outside any article/main.
    for t in soup(["header", "footer"]):
        if t.find_parent(["article", "main"]) is None:
            t.decompose()
    id_map = None
    for a in root.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"):
            if id_map is None:
                id_map = {t["id"]: t for t in soup.find_all(id=True)}
            if _is_citation_marker(a):
                # Citation marker → point it at the footnote's external source.
                url = _resolve_citation_href(id_map, href, base_url)
            else:
                # Preserve real TOC/permalink anchors as source-page URLs;
                # never redirect them to an unrelated link inside the section.
                url = _in_page_url(href, base_url, id_map)
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


def _markdown_from_html(
    html: str, base_url: str, max_images: int = 10
) -> str:
    """Convert page HTML to markdown, keeping the structure plain text loses.

    Headings, lists, tables, and hyperlinks survive the conversion, so the model
    sees the page the way a reader does — and can follow a link it found in the
    content with another fetch_page call. Link hrefs are resolved against
    `base_url` to absolute URLs for exactly that reason. Prominent images become
    explicit in-place text markers using page-provided alt/caption/metadata;
    decorative images and non-content elements are dropped. `escape_*` is off
    because the output feeds a model, not a markdown renderer.
    """
    return _markdown_from_soup(
        BeautifulSoup(html, "lxml"), base_url, max_images
    )


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
        if _norm_heading(_heading_text(h)) == target:
            return h
    for h in headings:
        ht = _norm_heading(_heading_text(h))
        if (target in ht or ht in target) and ht and len(ht) >= 3:
            return h
    return None


def _section_outline(
    matched,
    max_items: int = 40,
    anchor_records: dict[int, dict] | None = None,
) -> list[dict]:
    """Heading outline for one section: the matched heading and its children."""
    matched_level = int(matched.name[1])
    matched_text = _heading_text(matched)
    first = {"level": matched_level, "text": matched_text}
    if anchor_records is not None:
        first.update(anchor_records[id(matched)])
    outline = [first]
    for el in matched.find_all_next():
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            try:
                lvl = int(el.name[1])
            except ValueError:
                continue
            if lvl <= matched_level:
                break
            text = _heading_text(el)
            if text:
                item = {"level": lvl, "text": text}
                if anchor_records is not None:
                    item.update(anchor_records[id(el)])
                outline.append(item)
                if len(outline) >= max_items:
                    break
    return outline


def _find_section(
    soup: BeautifulSoup,
    section: str,
    base_url: str = "",
    max_images: int = 10,
) -> dict | None:
    """Locate a heading matching `section` and return text up to the next equal/higher heading.

    `base_url` lets reference/citation links inside the section be resolved to
    followable absolute URLs (inlined into the text, since the section is rendered
    as plain text) — so a model can chase a fetched section's sources."""
    if not section:
        return None

    matched = _locate_heading(soup, section)
    if matched is None:
        return None

    anchor_records = _heading_anchor_records(soup, base_url)
    matched_record = anchor_records[id(matched)]
    root = soup.find("article") or soup.find("main") or soup.body or soup
    _replace_prominent_images(
        soup,
        root,
        base_url,
        max_images,
        candidates=_section_image_candidates(matched),
    )
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()

    # Make links in the section followable before the text is flattened.
    _inline_link_urls(soup, base_url)

    matched_level = int(matched.name[1])
    matched_text = matched_record["text"]

    pieces: list[str] = []
    text_block_tags = [
        "p",
        "li",
        "pre",
        "code",
        "blockquote",
        "td",
        "th",
        "dd",
        "dt",
        "figcaption",
    ]
    next_heading_text: str | None = None
    next_heading_anchor: str | None = None

    for el in matched.find_all_next():
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            try:
                lvl = int(el.name[1])
            except ValueError:
                lvl = 99
            record = anchor_records.get(id(el))
            if lvl <= matched_level:
                next_heading_text = record["text"] if record else _heading_text(el)
                next_heading_anchor = record["anchor"] if record else None
                break
            sub = record["text"] if record else _heading_text(el)
            if sub:
                anchor = f" {{#{record['anchor']}}}" if record else ""
                pieces.append(f"\n## {sub}{anchor}\n")
            continue
        if el.get(_IMAGE_PLACEHOLDER_ATTR) == "true":
            # A placeholder nested in a paragraph/list item is already included
            # when that parent is flattened; a standalone figure marker is not.
            if el.find_parent(["p", "li", "blockquote", "td", "th", "dd", "dt"]) is None:
                pieces.append(el.get_text(" ", strip=True))
            continue
        if el.name in text_block_tags:
            # Flatten only the outermost selected block. Common markup such as
            # ``<li><p>…</p></li>`` and ``<pre><code>…</code></pre>`` would
            # otherwise emit the same text once for every nested block.
            if el.find_parent(text_block_tags) is not None:
                continue
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
                        record = anchor_records.get(id(el))
                        next_heading_text = record["text"] if record else _heading_text(el)
                        next_heading_anchor = record["anchor"] if record else None
                    break
            txt = el.get_text(" ", strip=True) if hasattr(el, "get_text") else ""
            if txt and txt not in collected:
                collected.append(txt)
        body_text = "\n\n".join(collected)
    else:
        body_text = "\n\n".join(pieces)

    body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()

    result = {
        "matched_heading": matched_text,
        "anchor": matched_record["anchor"],
        "level": matched_level,
        "text": body_text,
        "next_heading": next_heading_text,
        "next_heading_anchor": next_heading_anchor,
    }
    for field in ("source_fragment", "citation_url"):
        if field in matched_record:
            result[field] = matched_record[field]
    return result
