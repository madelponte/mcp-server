"""
Reddit helper for `fetch_page`.

Reddit is no longer a normal web page: anonymous `.json` is refused, old.reddit
is login-walled, and www.reddit.com HTML sits behind a bot challenge. This
module acquires Reddit posts, comment trees, and listings through the sources
that still work, in order: authenticated OAuth JSON (when configured), the
Arctic Shift archive, RSS (until Reddit actually stops serving it), then, for
individual posts, Brave search-index excerpts and oEmbed metadata. Every source
is compacted to one shared shape (`post` + `comments`, or `items`).

This is not a registered MCP tool. `fetch_page` routes a Reddit content URL
here (`is_reddit_url`), calls `acquire_reddit`, and applies its usual
`query=` / `offset` / truncation handling to the result, using
`reddit_query_text` for `query=` and `reddit_source_url` / `reddit_notes` for
the payload's URL and model-facing caveats. `RedditPostDeletedError` means the
post is gone and no copy survives.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import anyio
import httpx
from bs4 import BeautifulSoup

from config import web_search_settings as cfg
from .cache import TTLCache
from .serialize import to_json
from .page_acquire import UnexpectedResourceHTMLError, acquire_page as _acquire_page
from .web_extract import _markdown_from_soup, _plain_text_from_soup
from .web_fetch import DownloadTooLargeError, _decode_body, _redact_backend_secrets
from .web_search import (
    _BRAVE_SAFESEARCH,
    _brave_query,
    _resolve_country,
    _resolve_optional_choice,
    _resolve_search_lang,
)


# ---------------------------------------------------------------------------
# Source chain
#
# Authenticated JSON is the only complete source, but Reddit stopped granting
# new Data API access on 2026-10-31 and closes the public API in March 2027.
# Anonymous access is shrinking: `.json` returns 403, old.reddit redirects
# logged-out visitors to a login page, www.reddit.com HTML sits behind a
# "prove your humanity" challenge even for a headless browser. RSS was announced
# to end on 2026-11-13 but still answered on 2026-11-16, throttled to about one
# request per minute per IP. The anonymous chain is therefore: the Arctic Shift
# archive (third-party, near-real-time, posts and listings), then RSS until it
# actually fails, then (for posts) Brave's search-index excerpts plus official
# oEmbed metadata.
# ---------------------------------------------------------------------------

# Hosts that serve Reddit posts and listings. Other *.reddit.com hosts
# (developers., support., redditinc. …) are ordinary sites and must use the
# normal page-acquisition path.
_REDDIT_CONTENT_HOSTS = frozenset({
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "new.reddit.com",
    "np.reddit.com",
    "m.reddit.com",
    "i.reddit.com",
    "sh.reddit.com",
    "amp.reddit.com",
})

# Reddit announced RSS would end on 2026-11-13 but kept serving it past that
# date, so RSS is gated on observed failures rather than the calendar: a
# response that is no longer a feed at all (HTML, an empty 403, HTTP 410) marks
# RSS as retired for this long, so later fetches stop spending a request on it.
# A 429 is only the per-minute budget, and an Atom-bodied error (a missing
# subreddit/post, Reddit's search-feed refusal) is a per-URL failure.
_REDDIT_RSS_RETIRED_TTL_SECONDS = 6 * 3600.0
_reddit_rss_retired_until = 0.0

# Body/title placeholders Reddit leaves when a post or its text is gone.
_REDDIT_DELETED_MARKERS = frozenset({"[deleted]", "[removed]"})

# Listing sort suffixes the archive cannot reproduce (it returns newest first).
_REDDIT_LISTING_SORTS = frozenset({
    "hot", "new", "top", "rising", "controversial", "best",
})
# Archive `fields=` is a strict whitelist: one unsupported name rejects the
# whole request. These were verified against the live API on 2026-11-16
# (`permalink`, `_meta`, `removed_by_category`, `is_self`, and `parent_id` are
# not accepted). Single-post lookups omit `fields=` to get the full record.
_ARCHIVE_POST_LISTING_FIELDS = (
    "id,title,author,subreddit,score,num_comments,created_utc,url,selftext,"
    "over_18,link_flair_text"
)
_ARCHIVE_COMMENT_LISTING_FIELDS = "id,author,body,created_utc,link_id,subreddit,score"
# Per-item text excerpt in archived listings; the full thread is one fetch away.
_ARCHIVE_LISTING_EXCERPT_CHARS = 300

# Compacted archive results, keyed by archive request URL, so paging through a
# long thread with `offset` (or a re-fetch) does not re-download it.
_reddit_archive_cache = TTLCache(cfg.cache_ttl_seconds, cfg.cache_max_entries)


class RedditPostDeletedError(RuntimeError):
    """The post was deleted/removed and no copy of its content survives."""

_REDDIT_POST_ID_RE = re.compile(
    r"^/(?:(?:r|user|u)/[^/]+/)?comments/(?P<id>[a-z0-9]+)(?=/|$)"
    r"(?:/(?P<slug>[^/]+))?",
    re.I,
)
_REDDIT_SUBREDDIT_RE = re.compile(r"^/r/(?P<sub>[^/]+)", re.I)
# Brave allows 75 words; leave room for the site:/subreddit prefix.
_REDDIT_SEARCH_MAX_TITLE_WORDS = 60

_REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_reddit_access_token: str | None = None
_reddit_access_token_expires_at = 0.0
_reddit_access_token_credentials: tuple[str, str] | None = None


class _RedditRequestQueue:
    """Serialize Reddit acquisitions and leave a quiet period between them.

    Reddit's anonymous RSS/HTML endpoints throttle bursts aggressively on hosted
    IPs. One slot covers the complete fallback chain so concurrent tool calls do
    not each fan out into RSS, oEmbed, and Brave requests at the same time.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._next_request_at = 0.0

    async def acquire(self, url: str) -> tuple[dict, Any, str, list[str]]:
        # A cached archive result makes no upstream request, so it neither
        # waits for nor consumes the quiet period.
        cached = _reddit_archive_cached(url)
        if cached is not None:
            return cached
        delay = cfg.reddit_request_delay_seconds
        if delay <= 0:
            return await _acquire_reddit_unqueued(url)

        async with self._lock:
            wait = self._next_request_at - time.monotonic()
            if wait > 0:
                await anyio.sleep(wait)
            try:
                return await _acquire_reddit_unqueued(url)
            finally:
                self._next_request_at = time.monotonic() + delay


_reddit_request_queue = _RedditRequestQueue()


def is_reddit_url(url: str) -> bool:
    """Whether a URL targets a Reddit content host (posts, comments, listings)."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    # Compare the parsed hostname (not the netloc, which carries any port)
    # against an explicit allowlist. A suffix match would also capture
    # look-alikes like `notreddit.com` and documentation hosts such as
    # `developers.reddit.com`, which are normal pages, not Reddit content.
    return (p.hostname or "").lower() in _REDDIT_CONTENT_HOSTS


def _reddit_rss_available() -> bool:
    """Whether RSS has not been observed retired within the latch window."""
    return time.monotonic() >= _reddit_rss_retired_until


def _mark_reddit_rss_retired() -> None:
    global _reddit_rss_retired_until
    _reddit_rss_retired_until = time.monotonic() + _REDDIT_RSS_RETIRED_TTL_SECONDS


def _rss_response_means_retired(fetched: dict) -> bool:
    """Whether an RSS response shows Reddit no longer serves feeds at all."""
    status = fetched.get("status") or 0
    if status == 429 or status >= 500:
        # Per-minute budget, or a transient server/CDN failure.
        return False
    if status == 410:
        return True
    text = (fetched.get("text") or "")[:4096].lower()
    # Reddit returns Atom even for per-URL errors (404 for an unknown
    # subreddit/post, 403 for search feeds); anything else is not a feed.
    return "<feed" not in text and "<rss" not in text


def _is_reddit_search_url(url: str) -> bool:
    segments = [s for s in _reddit_clean_path(url).split("/") if s]
    return bool(segments) and segments[-1].lower() == "search"


def _normalize_reddit_url(url: str) -> str:
    """Return the legacy unauthenticated JSON representation of a Reddit URL."""
    if not is_reddit_url(url):
        return url
    p = urlparse(url)
    path = p.path or "/"
    if path.endswith("/"):
        path = path[:-1]
    if not path.endswith(".json"):
        path = path + ".json"
    return urlunparse((p.scheme or "https", "www.reddit.com", path, "", p.query, ""))


def _reddit_clean_path(url: str) -> str:
    """Strip a Reddit representation suffix and return its canonical path."""
    path = urlparse(url).path or "/"
    for suffix in (".json", ".rss"):
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)]
            break
    return path


def _is_reddit_post_url(url: str) -> bool:
    return _reddit_post_id(url) is not None


def _reddit_post_id(url: str) -> str | None:
    """Return the lowercase base-36 submission id of a Reddit post URL."""
    if not is_reddit_url(url):
        return None
    match = _REDDIT_POST_ID_RE.search(_reddit_clean_path(url))
    return match.group("id").lower() if match else None


def _reddit_subreddit(url: str) -> str | None:
    match = _REDDIT_SUBREDDIT_RE.match(_reddit_clean_path(url))
    return match.group("sub") if match else None


def _reddit_search_query(url: str, title: str | None) -> str | None:
    """Build a Brave query that should surface one specific Reddit post.

    The post title (from oEmbed) is the best key; the URL slug is a lossy
    fallback (Reddit truncates and strips it). Without either there is nothing
    a search index can match on, so return None.
    """
    words = (title or "").split()
    if not words:
        match = _REDDIT_POST_ID_RE.search(_reddit_clean_path(url))
        slug = match.group("slug") if match else None
        words = unquote(slug or "").replace("_", " ").split()
    if not words:
        return None
    parts = ["site:reddit.com"]
    subreddit = _reddit_subreddit(url)
    if subreddit:
        parts.append(f"r/{subreddit}")
    parts.extend(words[:_REDDIT_SEARCH_MAX_TITLE_WORDS])
    return " ".join(parts)[:590]


def _reddit_oauth_url(url: str) -> str:
    """Build the OAuth Data API URL corresponding to a public Reddit URL."""
    parsed = urlparse(url)
    path = _reddit_clean_path(url).rstrip("/") or "/"
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["raw_json"] = "1"
    if _is_reddit_post_url(url):
        params.setdefault("limit", "500")
    return urlunparse(("https", "oauth.reddit.com", path, "", urlencode(params), ""))


def _reddit_rss_url(url: str) -> str:
    """Build a Reddit Atom feed URL for a post or listing.

    Post query parameters such as ``context``, ``sort``, and share-tracking
    tokens do not change the Atom snapshot. Dropping them gives every variant of
    one post the same cache key and prevents duplicate anonymous Reddit requests.
    Listing/search parameters remain significant and are preserved.
    """
    parsed = urlparse(url)
    is_post = _is_reddit_post_url(url)
    path = _reddit_clean_path(url).rstrip("/")
    path += "/.rss" if is_post else ".rss"
    query = "" if is_post else parsed.query
    return urlunparse(("https", "www.reddit.com", path or "/.rss", "", query, ""))


def _reddit_oembed_url(url: str) -> str:
    """Metadata-only fallback when Reddit's post/comments JSON is blocked."""
    p = urlparse(url)
    path = _reddit_clean_path(url)
    # oEmbed identifies the submission from its path. Omitting context/sort/share
    # parameters canonicalizes the fallback cache key just like the post RSS URL.
    post_url = urlunparse((p.scheme or "https", "www.reddit.com", path, "", "", ""))
    return "https://www.reddit.com/oembed?" + urlencode({"url": post_url})


def _reddit_fragment_text(fragment: str, base_url: str) -> str:
    """Render Reddit's HTML-in-XML/body fragment without surrounding chrome."""
    soup = BeautifulSoup(fragment or "", "lxml")
    root = soup.select_one(".md") or soup.body or soup
    if cfg.markdown:
        return _markdown_from_soup(root, base_url).strip()
    return _plain_text_from_soup(root).strip()


def _compact_reddit_rss(xml: str, url: str) -> dict | None:
    """Convert a Reddit Atom feed into the compact Reddit result shape."""
    try:
        soup = BeautifulSoup(xml, "xml")
        entries = soup.find_all("entry")
        if not entries:
            return None

        parsed_entries = []
        for entry in entries:
            id_tag = entry.find("id")
            author_tag = entry.find("author")
            name_tag = author_tag.find("name") if author_tag else None
            author = name_tag.get_text(strip=True) if name_tag else None
            if author and author.startswith("/u/"):
                author = author[3:]
            title_tag = entry.find("title")
            content_tag = entry.find("content")
            link_tag = entry.find("link", href=True)
            updated_tag = entry.find("updated")
            parsed_entries.append(
                {
                    "id": id_tag.get_text(strip=True) if id_tag else None,
                    "title": title_tag.get_text(" ", strip=True) if title_tag else None,
                    "author": author,
                    "url": link_tag.get("href") if link_tag else None,
                    "body": _reddit_fragment_text(
                        content_tag.get_text() if content_tag else "", url
                    ),
                    "updated": updated_tag.get_text(strip=True) if updated_tag else None,
                }
            )

        # A post feed begins with the submission (t3), followed by comments (t1).
        # Subreddit feeds are all t3 entries, so decide from the URL, not the
        # first entry's kind.
        if _is_reddit_post_url(url) and (parsed_entries[0].get("id") or "").startswith("t3_"):
            first = parsed_entries[0]
            post = {
                "title": first["title"],
                "author": first["author"],
                "permalink": first["url"],
                "selftext": first["body"],
                "created": first["updated"],
            }
            comments = [
                {
                    "id": item["id"],
                    "author": item["author"],
                    "body": item["body"],
                    "permalink": item["url"],
                    "created": item["updated"],
                }
                for item in parsed_entries[1:]
                if (item.get("id") or "").startswith("t1_")
            ]
            return {
                "post": post,
                "comments": comments,
                "comments_returned": len(comments),
                "comments_incomplete": True,
            }

        return {"items": parsed_entries, "items_returned": len(parsed_entries)}
    except Exception:
        return None


def _compact_reddit_json(data: Any) -> Any:
    """Compact Reddit's verbose post+comments JSON to just what the model needs."""
    try:
        if isinstance(data, list) and len(data) == 2:
            post_listing, comments_listing = data

            def child_of(listing):
                return (listing or {}).get("data", {}).get("children", []) or []

            post = None
            kids = child_of(post_listing)
            if kids:
                pd = kids[0].get("data", {})
                post = {
                    "title": pd.get("title"),
                    "author": pd.get("author"),
                    "subreddit": pd.get("subreddit"),
                    "score": pd.get("score"),
                    "num_comments": pd.get("num_comments"),
                    "permalink": pd.get("permalink"),
                    "url": pd.get("url"),
                    "selftext": pd.get("selftext"),
                    "created_utc": pd.get("created_utc"),
                }

            comments = []
            more_children = 0

            def walk(node, depth=0):
                nonlocal more_children
                if not isinstance(node, dict):
                    return
                kind = node.get("kind")
                d = node.get("data") or {}
                if kind == "more":
                    more_children += len(d.get("children") or [])
                    return
                if kind == "t1":
                    comment = {
                        "author": d.get("author"),
                        "score": d.get("score"),
                        "depth": depth,
                        "body": d.get("body"),
                    }
                    # The short base-36 id lets a comment be linked as
                    # <post permalink><id>/ without a long per-comment URL.
                    if isinstance(d.get("id"), str) and d["id"]:
                        comment = {"id": d["id"], **comment}
                    comments.append(comment)
                    replies = d.get("replies")
                    if isinstance(replies, dict):
                        for c in (replies.get("data") or {}).get("children", []) or []:
                            walk(c, depth + 1)

            for c in child_of(comments_listing):
                walk(c, 0)

            result = {
                "post": post,
                "comments": comments,
                "comments_returned": len(comments),
            }
            if post and post.get("num_comments") is not None:
                result["comments_reported"] = post["num_comments"]
                result["comments_incomplete"] = len(comments) < post["num_comments"]
            if more_children:
                result["more_children"] = more_children
            return result
    except Exception:
        pass
    return data


async def _reddit_token() -> str:
    """Return a cached application-only Reddit OAuth token."""
    global _reddit_access_token
    global _reddit_access_token_expires_at
    global _reddit_access_token_credentials

    client_id = cfg.reddit_client_id.strip()
    client_secret = cfg.reddit_client_secret.strip()
    user_agent = cfg.reddit_user_agent.strip()
    missing = [
        name
        for name, value in (
            ("WEB_SEARCH_REDDIT_CLIENT_ID", client_id),
            ("WEB_SEARCH_REDDIT_CLIENT_SECRET", client_secret),
            ("WEB_SEARCH_REDDIT_USER_AGENT", user_agent),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Incomplete Reddit OAuth configuration: " + ", ".join(missing)
        )

    credentials = (client_id, client_secret)
    now = time.monotonic()
    if (
        _reddit_access_token
        and _reddit_access_token_credentials == credentials
        and now < _reddit_access_token_expires_at - 30
    ):
        return _reddit_access_token

    async with httpx.AsyncClient(verify=cfg.verify_ssl) as client:
        try:
            response = await client.post(
                _REDDIT_TOKEN_URL,
                auth=(client_id, client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": user_agent, "Accept": "application/json"},
                timeout=cfg.http_timeout_seconds,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Reddit OAuth network error: {exc.__class__.__name__} - {str(exc)[:200]}"
            ) from exc
    if response.status_code >= 400:
        # Distinguish auth failures from other server errors
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Reddit OAuth authentication failed (HTTP {response.status_code}). "
                "Check that client_id and client_secret are correct and the app is approved."
            )
        raise RuntimeError(
            f"Reddit OAuth token endpoint returned HTTP {response.status_code}"
        )
    try:
        data = response.json()
        token = str(data["access_token"]).strip()
        expires_in = max(60, int(data.get("expires_in", 3600)))
    except Exception as exc:
        raise RuntimeError("Reddit OAuth token endpoint returned invalid JSON") from exc
    if not token:
        raise RuntimeError("Reddit OAuth token endpoint returned an empty access token")

    _reddit_access_token = token
    _reddit_access_token_expires_at = time.monotonic() + expires_in
    _reddit_access_token_credentials = credentials
    return token


async def _fetch_reddit_oauth(url: str) -> dict:
    """Fetch one Reddit JSON representation using application-only OAuth."""
    token = await _reddit_token()
    fetch_url = _reddit_oauth_url(url)
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": cfg.reddit_user_agent.strip(),
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(verify=cfg.verify_ssl) as client:
        async with client.stream(
            "GET", fetch_url, headers=headers, timeout=cfg.http_timeout_seconds
        ) as response:
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if cfg.max_download_bytes and total > cfg.max_download_bytes:
                    raise DownloadTooLargeError(
                        f"Reddit OAuth response exceeds the {cfg.max_download_bytes}-byte "
                        "WEB_SEARCH_MAX_DOWNLOAD_BYTES cap."
                    )
                chunks.append(chunk)
            status = response.status_code
            ctype = response.headers.get("content-type", "application/json")
    if status >= 400:
        raise RuntimeError(f"Reddit OAuth JSON returned HTTP {status}")
    body = b"".join(chunks)
    return {
        "url": fetch_url,
        "status": status,
        "content_type": ctype,
        "text": _decode_body(body, ctype),
        "bytes": None,
        "via": "reddit_oauth",
        "blocked_detected": False,
    }


def _reddit_oauth_requested() -> bool:
    return any(
        value.strip()
        for value in (
            cfg.reddit_client_id,
            cfg.reddit_client_secret,
            cfg.reddit_user_agent,
        )
    )


def _reddit_failure_trace(name: str, exc: BaseException) -> str:
    """Bound one provider failure so a debug note cannot dominate the result."""
    detail = " ".join((str(exc).strip() or type(exc).__name__).split())
    if len(detail) > 500:
        detail = detail[:497] + "..."
    return f"{name} failed ({detail})"


def _brave_configured() -> bool:
    return bool(cfg.brave_api_key.strip() and cfg.brave_api_url.strip())


async def _fetch_reddit_oembed(url: str) -> tuple[dict, dict]:
    """Fetch Reddit's official oEmbed record (title/author, no body) for a post."""
    fetched = await _acquire_page(_reddit_oembed_url(url))
    if fetched.get("status") == 404:
        raise RedditPostDeletedError(
            "Reddit reports that this post does not exist: it was deleted, or the "
            "URL is wrong. No archived copy is available."
        )
    if fetched.get("status", 0) >= 400:
        raise RuntimeError(f"HTTP {fetched['status']}")
    data = json.loads(fetched.get("text") or "")
    if not isinstance(data, dict):
        raise RuntimeError("oEmbed response was not a JSON object")
    if _reddit_text_gone(data.get("title")):
        raise RedditPostDeletedError(_REDDIT_DELETED_MESSAGE)
    return fetched, data


# ---------------------------------------------------------------------------
# Arctic Shift archive
# ---------------------------------------------------------------------------

_ARCHIVE_USER_AGENT = "mcp-server-fetch_page/1.0 (Reddit archive client)"
_REDDIT_DELETED_MESSAGE = (
    "This Reddit post was deleted or removed, and no archived copy of its "
    "content is available."
)


def _reddit_text_gone(value: Any) -> bool:
    """Whether a Reddit text field is empty or a deletion placeholder."""
    if not isinstance(value, str):
        return value is None
    text = value.strip()
    return not text or text in _REDDIT_DELETED_MARKERS


def _iso_utc(timestamp: Any) -> str | None:
    try:
        return (
            datetime.fromtimestamp(float(timestamp), timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return None


class _ArchiveTimeoutError(RuntimeError):
    """The archive's own query timed out; it usually succeeds on a retry."""


# The archive's query engine reports an internal timeout ("Timeout. Maybe slow
# down a bit") on cold queries, as HTTP 422 or as HTTP 200 with `data: null`.
# Observed on 2026-11-16, an identical request usually succeeded moments later.
_ARCHIVE_TIMEOUT_RETRY_SECONDS = 1.0


def _archive_error_text(body: bytes) -> str | None:
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if isinstance(parsed, dict) and parsed.get("error"):
        return " ".join(str(parsed["error"]).split())[:200]
    return None


def _raise_archive_error(error: str, prefix: str) -> None:
    message = f"{prefix}archive error: {error}"
    if "timeout" in error.lower():
        raise _ArchiveTimeoutError(message)
    raise RuntimeError(message)


async def _reddit_archive_get(path: str, params: dict) -> tuple[str, Any]:
    """GET one Arctic Shift endpoint; return (request URL, the `data` member).

    The API reports some failures ("Timeout. Maybe slow down a bit", an invalid
    `fields=` name) as a JSON `error` string, with HTTP 200 or 4xx, so that is
    treated as a failure exactly like an HTTP error. An archive-side timeout is
    retried once after a short pause.
    """
    try:
        return await _reddit_archive_get_once(path, params)
    except _ArchiveTimeoutError:
        await anyio.sleep(_ARCHIVE_TIMEOUT_RETRY_SECONDS)
        return await _reddit_archive_get_once(path, params)


async def _reddit_archive_get_once(path: str, params: dict) -> tuple[str, Any]:
    base = urlparse(cfg.reddit_archive_api_url.strip().rstrip("/"))
    auth = (unquote(base.username), unquote(base.password or "")) if base.username else None
    # Use Basic auth separately so neither provenance nor HTTP URLs include userinfo.
    public_base = base._replace(netloc=base.netloc.rsplit("@", 1)[-1]).geturl()
    request_url = f"{public_base}{path}?{urlencode(params)}"
    headers = {"User-Agent": _ARCHIVE_USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(verify=cfg.verify_ssl, auth=auth) as client:
            async with client.stream(
                "GET", request_url, headers=headers, timeout=cfg.http_timeout_seconds
            ) as response:
                status = response.status_code
                if status >= 400:
                    detail = f"HTTP {status}"
                    reset = response.headers.get("x-ratelimit-reset")
                    if status == 429 and reset:
                        detail += f", rate limit resets in {reset}s"
                    body = b""
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > 4096:
                            break
                    error = _archive_error_text(body)
                    if error:
                        _raise_archive_error(error, f"{detail}, ")
                    raise RuntimeError(detail)
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if cfg.max_download_bytes and total > cfg.max_download_bytes:
                        raise DownloadTooLargeError(
                            f"Archive response exceeds the {cfg.max_download_bytes}-byte "
                            "WEB_SEARCH_MAX_DOWNLOAD_BYTES cap."
                        )
                    chunks.append(chunk)
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"network error: {exc.__class__.__name__} - {str(exc)[:200]}"
        ) from exc
    try:
        body = json.loads(b"".join(chunks))
    except ValueError as exc:
        raise RuntimeError("response was not valid JSON") from exc
    if not isinstance(body, dict):
        raise RuntimeError("response was not a JSON object")
    if body.get("error"):
        _raise_archive_error(" ".join(str(body["error"]).split())[:200], "")
    return request_url, body.get("data")


def _archive_fetched(request_url: str) -> dict:
    return {
        "url": request_url,
        "status": 200,
        "content_type": "application/json",
        "text": None,
        "bytes": None,
        "via": "arctic_shift",
        "blocked_detected": False,
    }


async def _archive_gather(requests: dict[str, tuple[str, dict]]) -> dict[str, Any]:
    """Run independent archive requests together; failures become exceptions."""
    results: dict[str, Any] = {}

    async def run_one(key: str, path: str, params: dict) -> None:
        try:
            results[key] = await _reddit_archive_get(path, params)
        except Exception as exc:  # collected per request, never cancels siblings
            results[key] = exc

    async with anyio.create_task_group() as tg:
        for key, (path, params) in requests.items():
            tg.start_soon(run_one, key, path, params)
    return results


async def _fetch_reddit_archive_post(url: str) -> tuple[dict, dict]:
    """Return the archived post and comment tree in the compact Reddit shape.

    Raises RuntimeError when the archive has no copy yet (so later fallbacks
    run) and RedditPostDeletedError when its copy has no surviving content.
    """
    post_id = _reddit_post_id(url)
    if not post_id:
        raise RuntimeError("no submission id in the URL")
    limit = cfg.reddit_archive_max_comments
    cache_key = _archive_post_cache_key(post_id)
    cached = _reddit_archive_cache.get(cache_key)
    if cached is not None:
        return cached

    results = await _archive_gather({
        # No `fields=`: permalink, _meta, and removed_by_category are needed
        # but are not accepted by the field whitelist.
        "post": ("/posts/ids", {"ids": post_id}),
        "tree": ("/comments/tree", {"link_id": f"t3_{post_id}", "limit": limit}),
    })
    post_result = results["post"]
    if isinstance(post_result, Exception):
        raise post_result
    post_url, posts = post_result
    if not isinstance(posts, list) or not posts or not isinstance(posts[0], dict):
        raise RuntimeError(f"archive has no copy of post {post_id} yet")
    post = posts[0]
    if _reddit_text_gone(post.get("title")) and _reddit_text_gone(post.get("selftext")):
        raise RedditPostDeletedError(_REDDIT_DELETED_MESSAGE)

    tree_result = results["tree"]
    tree_error: Exception | None = None
    tree = []
    if isinstance(tree_result, Exception):
        tree_error = tree_result
    elif not isinstance(tree_result[1], list):
        tree_error = RuntimeError("expected an array in archive comment-tree data")
    else:
        tree = tree_result[1]
    compact = _compact_reddit_json([
        {"data": {"children": [{"kind": "t3", "data": post}]}},
        {"data": {"children": tree}},
    ])
    if not isinstance(compact, dict) or not isinstance(compact.get("post"), dict):
        raise RuntimeError("archive returned an unexpected post shape")

    compact_post = compact["post"]
    permalink = compact_post.get("permalink")
    if isinstance(permalink, str) and permalink.startswith("/"):
        compact_post["permalink"] = "https://www.reddit.com" + permalink
    if post.get("removed_by_category"):
        compact_post["removed_by_category"] = post["removed_by_category"]

    meta = post.get("_meta") if isinstance(post.get("_meta"), dict) else {}
    archived_at = _iso_utc(meta.get("retrieved_2nd_on") or post.get("retrieved_on"))

    returned = compact.get("comments_returned", 0)
    reported = post.get("num_comments")
    if isinstance(reported, int) and reported < returned:
        # A first-ingest snapshot often reports 0 comments for a thread whose
        # tree the archive has since collected; the count is just stale.
        compact.pop("comments_reported", None)
    if tree_error is not None:
        compact["comments_incomplete"] = True
        compact["comments_unavailable"] = _redact_backend_secrets(
            _reddit_failure_trace("Archive comment tree", tree_error)
        )
    elif compact.get("more_children"):
        compact["comments_incomplete"] = True
    elif isinstance(reported, int) and reported > returned:
        compact["comments_incomplete"] = True
    elif not meta:
        # Only the first-ingest snapshot exists (no re-crawl yet), so whether
        # comments are missing is unknown.
        compact["comments_incomplete"] = None
    else:
        compact["comments_incomplete"] = False

    ordered = {"post": compact.pop("post")}
    if archived_at:
        ordered["archived_at"] = archived_at
    ordered.update(compact)
    result = (_archive_fetched(post_url), ordered)
    if tree_error is None:
        _reddit_archive_cache.set(cache_key, result)
    return result


def _reddit_archive_listing_plan(url: str) -> tuple[dict, dict[str, tuple[str, dict]]]:
    """Map a Reddit listing URL to archive requests.

    Returns (listing metadata, {"posts"|"comments": (path, params)}) or raises
    RuntimeError for pages the archive cannot list.
    """
    segments = [s for s in _reddit_clean_path(url).split("/") if s]
    lowered = [s.lower() for s in segments]
    limit = cfg.reddit_archive_max_listing
    common = {"sort": "desc", "limit": limit}

    if len(segments) >= 2 and lowered[0] == "r":
        sub = segments[1]
        if "+" in sub or sub.lower() in {"all", "popular", "friends", "mod"}:
            raise RuntimeError(
                "archive does not cover multi-subreddit, r/all, or r/popular listings"
            )
        rest = lowered[2:]
        if not rest or (len(rest) == 1 and rest[0] in _REDDIT_LISTING_SORTS):
            meta = {"listing": f"r/{sub}"}
            params = {"subreddit": sub, **common}
        elif rest[0] == "search":
            q = dict(parse_qsl(urlparse(url).query)).get("q", "").strip()
            if not q:
                raise RuntimeError("subreddit search URL has no q= query")
            meta = {"listing": f"r/{sub} search", "query": q, "match": "title"}
            params = {"subreddit": sub, "title": q, **common}
        else:
            raise RuntimeError(f"archive does not cover this r/{sub} page")
        params["fields"] = _ARCHIVE_POST_LISTING_FIELDS
        return meta, {"posts": ("/posts/search", params)}

    if len(segments) >= 2 and lowered[0] in {"user", "u"}:
        name = segments[1]
        view = lowered[2] if len(lowered) > 2 else "overview"
        requests: dict[str, tuple[str, dict]] = {}
        if view in {"overview", "submitted", "posts"}:
            requests["posts"] = ("/posts/search", {
                "author": name, **common, "fields": _ARCHIVE_POST_LISTING_FIELDS,
            })
        if view in {"overview", "comments"}:
            requests["comments"] = ("/comments/search", {
                "author": name, **common, "fields": _ARCHIVE_COMMENT_LISTING_FIELDS,
            })
        if not requests:
            raise RuntimeError(f"archive does not cover this u/{name} page")
        return {"listing": f"u/{name}"}, requests

    raise RuntimeError(
        "archive lists only subreddits, subreddit searches, and users "
        "(not the front page or site-wide search)"
    )


def _archive_post_cache_key(post_id: str) -> str:
    return f"post:{post_id}:{cfg.reddit_archive_max_comments}"


def _archive_listing_cache_key(requests: dict[str, tuple[str, dict]]) -> str:
    return "listing:" + to_json(sorted(
        (key, path, sorted(params.items())) for key, (path, params) in requests.items()
    ))


def _reddit_archive_cached(url: str) -> tuple[dict, Any, str, list[str]] | None:
    """Return a cached archive acquisition for `url` without any network I/O.

    Only used when OAuth is not configured: with OAuth, the authenticated API
    stays first in the chain and the archive cache must not pre-empt it.
    """
    if _reddit_oauth_requested() or not cfg.reddit_archive_api_url.strip():
        return None
    try:
        if _is_reddit_post_url(url):
            post_id = _reddit_post_id(url)
            if not post_id:
                return None
            key = _archive_post_cache_key(post_id)
        else:
            key = _archive_listing_cache_key(_reddit_archive_listing_plan(url)[1])
    except Exception:
        return None
    cached = _reddit_archive_cache.get(key)
    if cached is None:
        return None
    fetched, compact = cached
    trace = ["OAuth JSON skipped (not configured)", "Archive succeeded (cached)"]
    return fetched, compact, "archive", trace


def _is_placeholder_text(text: Any) -> bool:
    """Whether a body is only Reddit's [deleted]/[removed] placeholder."""
    return isinstance(text, str) and text.strip() in _REDDIT_DELETED_MARKERS


def _archive_excerpt(text: Any) -> str | None:
    if not isinstance(text, str) or not text.strip() or _is_placeholder_text(text):
        return None
    text = " ".join(text.split())
    if len(text) > _ARCHIVE_LISTING_EXCERPT_CHARS:
        text = text[: _ARCHIVE_LISTING_EXCERPT_CHARS - 1].rstrip() + "…"
    return text


def _archive_post_item(item: dict) -> dict:
    sub = item.get("subreddit") or ""
    post_id = item.get("id") or ""
    permalink = f"https://www.reddit.com/r/{sub}/comments/{post_id}/"
    out = {
        "type": "post",
        "title": item.get("title"),
        "author": item.get("author"),
        "subreddit": sub or None,
        "score": item.get("score"),
        "num_comments": item.get("num_comments"),
        "created": _iso_utc(item.get("created_utc")),
        "permalink": permalink,
    }
    link = item.get("url")
    if isinstance(link, str) and link and _reddit_post_id(link) != post_id.lower():
        out["url"] = link
    if item.get("link_flair_text"):
        out["flair"] = item["link_flair_text"]
    if item.get("over_18"):
        out["nsfw"] = True
    excerpt = _archive_excerpt(item.get("selftext"))
    if excerpt:
        out["text"] = excerpt
    elif _is_placeholder_text(item.get("selftext")):
        out["text_removed"] = True
    return {k: v for k, v in out.items() if v is not None}


def _archive_comment_item(item: dict) -> dict:
    sub = item.get("subreddit") or ""
    link_id = str(item.get("link_id") or "")
    if link_id.startswith("t3_"):
        link_id = link_id[3:]
    out = {
        "type": "comment",
        "author": item.get("author"),
        "subreddit": sub or None,
        "score": item.get("score"),
        "created": _iso_utc(item.get("created_utc")),
        "text": _archive_excerpt(item.get("body")),
        "text_removed": True if _is_placeholder_text(item.get("body")) else None,
        "permalink": (
            f"https://www.reddit.com/r/{sub}/comments/{link_id}/_/{item.get('id')}/"
            if sub and link_id else None
        ),
    }
    return {k: v for k, v in out.items() if v is not None}


async def _fetch_reddit_archive_listing(url: str) -> tuple[dict, dict]:
    """Return a newest-first archive listing for a subreddit, search, or user."""
    meta, requests = _reddit_archive_listing_plan(url)
    cache_key = _archive_listing_cache_key(requests)
    cached = _reddit_archive_cache.get(cache_key)
    if cached is not None:
        return cached

    results = await _archive_gather(requests)
    items: list[tuple[float, dict]] = []
    failures: dict[str, str] = {}
    request_url = None
    for key, result in results.items():
        if isinstance(result, Exception) or not isinstance(result[1], list):
            if isinstance(result, Exception):
                detail = " ".join((str(result).strip() or type(result).__name__).split())
                detail = _redact_backend_secrets(detail)
            else:
                detail = "expected an array in archive listing data"
            failures[key] = detail if len(requests) == 1 else f"{key}: {detail}"
            continue
        request_url = request_url or result[0]
        render = _archive_post_item if key == "posts" else _archive_comment_item
        for raw in result[1]:
            if isinstance(raw, dict):
                items.append((float(raw.get("created_utc") or 0), render(raw)))
    if request_url is None:
        raise RuntimeError("; ".join(failures.values()))

    items.sort(key=lambda pair: pair[0], reverse=True)
    rendered = [item for _, item in items[: cfg.reddit_archive_max_listing]]
    if len(requests) == 1:
        for item in rendered:
            item.pop("type", None)
    compact: dict = {
        **meta,
        "sort": "newest_first",
        "items": rendered,
        "items_returned": len(rendered),
    }
    if failures:
        compact["unavailable"] = failures
    result = (_archive_fetched(request_url), compact)
    if not failures:
        _reddit_archive_cache.set(cache_key, result)
    return result


def _archive_note(compact: Any) -> str:
    """Model-facing caveats for an Arctic Shift result."""
    if isinstance(compact, dict) and "items" in compact:
        note = (
            "This listing comes from Arctic Shift, a third-party Reddit archive, "
            "because Reddit's API is unavailable. It is ordered newest first "
            "(Reddit's hot/top ranking is not available), and scores and comment "
            "counts are snapshots that may be out of date. Fetch an item's "
            "permalink to read the full thread."
        )
        if compact.get("match") == "title":
            note += " Search results match post titles only."
        if any(item.get("text_removed") for item in compact.get("items") or []):
            note += (
                " text_removed marks items whose text Reddit deleted or removed "
                "before the archive captured it; only their metadata survives."
            )
        return note
    note = (
        "This thread comes from Arctic Shift, a third-party Reddit archive, "
        "because Reddit's API is unavailable. Scores and comment counts are as "
        "of archived_at, and very recent comments may be missing."
    )
    post = compact.get("post") if isinstance(compact, dict) else None
    if isinstance(post, dict) and post.get("removed_by_category"):
        category = post["removed_by_category"]
        if _is_placeholder_text(post.get("selftext")):
            note += (
                f" Reddit removed or deleted this post (removed_by_category: "
                f"{category}) before the archive captured its text, so the body "
                "is only a placeholder; the title, metadata, and any archived "
                "comments are all that survive."
            )
        else:
            note += (
                " Reddit has since removed or deleted this post "
                f"(removed_by_category: {category}); the content shown is the "
                "archived copy."
            )
    return note


def _reddit_heading_text(text: Any) -> str:
    """One-line heading text that cannot be mistaken for an anchor marker."""
    return " ".join(str(text or "").replace("{#", "{ #").split())


def _reddit_absolute_url(link: Any) -> str | None:
    if not isinstance(link, str) or not link.strip():
        return None
    link = link.strip()
    return "https://www.reddit.com" + link if link.startswith("/") else link


def reddit_query_text(
    compact: Any,
) -> tuple[str, dict[str, dict], list[str | None] | None]:
    """Render a compact Reddit result as headed text for `query=` matching.

    Every post, comment, listing item, and excerpt gets its own `{#anchor}`
    heading line, so match windows never run across two items and each
    window's heading names the author. Only titles and body text are matched:
    bylines, scores, dates, comment headers, and link lines are context that
    appears in a matching passage but never produces a match itself (so
    `query="python"` on r/Python does not match every item's subreddit).

    Returns (text, heading lookup, match_segments). The lookup adds a
    `citation_url` for items with a link; match_segments is aligned with
    `_segment_text(text)`, holding the text to match or None for context-only
    lines, and is None when the whole text should be searched.
    """
    if not isinstance(compact, dict):
        return to_json(compact), {}, None

    lines: list[str] = []
    searchable: list[str | None] = []
    lookup: dict[str, dict] = {}

    def add(line: str, match: str | None) -> None:
        line = line.strip()
        if line:
            lines.append(line)
            searchable.append(match.strip() if match and match.strip() else None)

    def body(text: Any) -> None:
        if isinstance(text, str):
            for line in text.split("\n"):
                add(line, line)

    def details(*parts: str) -> str:
        joined = ", ".join(part for part in parts if part)
        return f" ({joined})" if joined else ""

    post = compact.get("post")
    post_link = None
    if isinstance(post, dict):
        title = _reddit_heading_text(post.get("title"))
        byline = details(
            f"u/{_reddit_heading_text(post['author'])}" if post.get("author") else "",
            f"r/{_reddit_heading_text(post['subreddit'])}" if post.get("subreddit") else "",
            f"score {post['score']}" if post.get("score") is not None else "",
        )
        add(f"# {title or 'Reddit post'}{byline} {{#post}}", title)
        post_link = _reddit_absolute_url(post.get("permalink"))
        if post_link:
            lookup["post"] = {"citation_url": post_link}
        body(post.get("selftext"))

    for number, comment in enumerate(compact.get("comments") or [], 1):
        if not isinstance(comment, dict):
            continue
        anchor = f"comment-{number}"
        add(
            f"## Comment {number}"
            + details(
                f"u/{_reddit_heading_text(comment.get('author')) or '[unknown]'}",
                f"score {comment['score']}" if comment.get("score") is not None else "",
                f"reply depth {comment['depth']}" if comment.get("depth") else "",
            )
            + f" {{#{anchor}}}",
            None,
        )
        link = _reddit_absolute_url(comment.get("permalink"))
        comment_id = comment.get("id")
        if not link and post_link and isinstance(comment_id, str) and comment_id:
            link = post_link.rstrip("/") + f"/{comment_id}/"
        if link:
            lookup[anchor] = {"citation_url": link}
        body(comment.get("body"))

    for number, item in enumerate(compact.get("items") or [], 1):
        if not isinstance(item, dict):
            continue
        anchor = f"item-{number}"
        title = _reddit_heading_text(item.get("title"))
        author = _reddit_heading_text(item.get("author"))
        label = title or f"Comment by u/{author or '[unknown]'}"
        add(
            f"## {label}"
            + details(
                f"u/{author}" if author and title else "",
                f"r/{_reddit_heading_text(item['subreddit'])}" if item.get("subreddit") else "",
                _reddit_heading_text(item["created"]) if item.get("created") else "",
                f"score {item['score']}" if item.get("score") is not None else "",
            )
            + f" {{#{anchor}}}",
            title or None,
        )
        link = _reddit_absolute_url(item.get("permalink") or item.get("url"))
        if link:
            add(_reddit_heading_text(link), None)
            lookup[anchor] = {"citation_url": link}
        body(item.get("text") or item.get("body"))

    for number, excerpt in enumerate(compact.get("excerpts") or [], 1):
        add(f"## Excerpt {number} {{#excerpt-{number}}}", None)
        body(excerpt)

    if not lines:
        # oEmbed metadata and any unexpected shape: search the JSON itself.
        return to_json(compact), lookup, None
    return "\n".join(lines), lookup, searchable


def _reddit_rss_post_deleted(compact: dict) -> bool:
    post = compact.get("post") if isinstance(compact, dict) else None
    if not isinstance(post, dict) or not _reddit_text_gone(post.get("title")):
        return False
    body = (post.get("selftext") or "").strip()
    return not body or any(body.startswith(marker) for marker in _REDDIT_DELETED_MARKERS)


async def _fetch_reddit_brave(url: str, oembed: dict | None) -> tuple[dict, dict]:
    """Return Brave LLM Context excerpts of one Reddit post.

    Brave's index holds pre-extracted text of many Reddit threads (post body and
    a selection of comments). Results are kept only when their URL names the
    same submission id, so an unrelated thread with a similar title is never
    returned as this post.
    """
    post_id = _reddit_post_id(url)
    title = (oembed or {}).get("title")
    title = title.strip() if isinstance(title, str) and title.strip() else None
    query = _reddit_search_query(url, title)
    if not post_id or not query:
        raise RuntimeError("no post title or URL slug to search for")

    results = await _brave_query(
        api_url=cfg.brave_api_url.strip(),
        api_key=cfg.brave_api_key.strip(),
        query=query,
        num_results=cfg.max_num_results,
        country=_resolve_country(None, cfg.brave_country),
        search_lang=_resolve_search_lang(None, cfg.brave_search_lang),
        # Any age: a configured search_web freshness default must not hide the
        # older thread the model explicitly asked for.
        freshness="",
        safesearch=_resolve_optional_choice(
            None, cfg.brave_safesearch, _BRAVE_SAFESEARCH, "safesearch"
        ),
        # Results are filtered to the exact submission id below, so favor
        # coverage over Brave's relevance threshold.
        context_threshold_mode="lenient",
        max_tokens=cfg.max_context_tokens,
        search_count=cfg.brave_search_count,
        max_tokens_per_url=cfg.brave_max_tokens_per_url,
        timeout=cfg.brave_timeout_seconds,
        verify_ssl=cfg.verify_ssl,
        user_agent=cfg.user_agent,
        request_delay_seconds=cfg.brave_request_delay_seconds,
        max_retries=cfg.brave_max_retries,
        retry_backoff_seconds=cfg.brave_retry_backoff_seconds,
    )
    matches = [r for r in results if _reddit_post_id(r.get("url") or "") == post_id]
    excerpts = list(dict.fromkeys(
        snippet for result in matches for snippet in result.get("snippets") or []
    ))
    if not excerpts:
        raise RuntimeError(f"search index returned no excerpts for post {post_id}")

    first = matches[0]
    post = {
        "title": title or first.get("title"),
        "author": (oembed or {}).get("author_name"),
        "subreddit": _reddit_subreddit(url),
        "permalink": urlunparse(
            ("https", "www.reddit.com", _reddit_clean_path(url), "", "", "")
        ),
    }
    compact: dict = {
        "post": {key: value for key, value in post.items() if value},
        "excerpts": excerpts,
        "excerpts_source": "brave_llm_context",
        "complete": False,
    }
    if first.get("page_date"):
        compact["page_date"] = first["page_date"]
    fetched = {
        "url": first["url"],
        "status": 200,
        "content_type": "application/json",
        "text": None,
        "bytes": None,
        "via": "brave_llm_context",
        "blocked_detected": False,
    }
    return fetched, compact


async def _fetch_reddit_rss(rss_url: str) -> dict:
    try:
        return await _acquire_page(rss_url)
    except UnexpectedResourceHTMLError as exc:
        # Known feed URLs don't download HTML bodies, but their status still
        # distinguishes throttling/server errors from definitive non-feeds.
        return {"status": exc.status, "text": ""}


async def _acquire_reddit_unqueued(url: str) -> tuple[dict, Any, str, list[str]]:
    """Try OAuth JSON -> archive -> RSS -> Brave excerpts -> oEmbed.

    RedditPostDeletedError ends the chain: once a source shows the post's
    content is gone, later sources cannot recover it.
    """
    errors: list[str] = []
    trace: list[str] = []
    is_post = _is_reddit_post_url(url)

    if _reddit_oauth_requested():
        try:
            fetched = await _fetch_reddit_oauth(url)
            parsed = json.loads(fetched.get("text") or "")
            trace.append("OAuth JSON succeeded")
            return fetched, _compact_reddit_json(parsed), "oauth", trace
        except Exception as exc:
            failure = _reddit_failure_trace("OAuth JSON", exc)
            errors.append(failure)
            trace.append(failure)
    else:
        trace.append("OAuth JSON skipped (not configured)")

    if cfg.reddit_archive_api_url.strip():
        try:
            if is_post:
                fetched, compact = await _fetch_reddit_archive_post(url)
            else:
                fetched, compact = await _fetch_reddit_archive_listing(url)
            trace.append("Archive succeeded")
            return fetched, compact, "archive", trace
        except RedditPostDeletedError:
            raise
        except Exception as exc:
            failure = _reddit_failure_trace("Archive", exc)
            errors.append(failure)
            trace.append(failure)
    else:
        trace.append("Archive skipped (not configured)")

    if _is_reddit_search_url(url):
        skipped = "RSS skipped (Reddit refuses anonymous search feeds)"
        errors.append(skipped)
        trace.append(skipped)
    elif not _reddit_rss_available():
        skipped = "RSS skipped (Reddit recently stopped serving feeds)"
        errors.append(skipped)
        trace.append(skipped)
    else:
        rss_url = _reddit_rss_url(url)
        try:
            fetched = await _fetch_reddit_rss(rss_url)
            if (
                fetched.get("status") == 429
                and cfg.reddit_rate_limit_retry_seconds > 0
            ):
                trace.append(
                    "RSS returned HTTP 429; retrying after "
                    f"{cfg.reddit_rate_limit_retry_seconds:g}s"
                )
                await anyio.sleep(cfg.reddit_rate_limit_retry_seconds)
                fetched = await _fetch_reddit_rss(rss_url)
            if _rss_response_means_retired(fetched):
                _mark_reddit_rss_retired()
                if fetched.get("status", 0) < 400:
                    raise RuntimeError("response was not a feed")
            if fetched.get("status", 0) >= 400:
                raise RuntimeError(f"HTTP {fetched['status']}")
            compact = await anyio.to_thread.run_sync(
                _compact_reddit_rss, fetched.get("text") or "", url
            )
            if compact is None:
                raise RuntimeError("feed contained no Reddit entries")
            if is_post and _reddit_rss_post_deleted(compact):
                raise RedditPostDeletedError(_REDDIT_DELETED_MESSAGE)
            trace.append("RSS succeeded")
            return fetched, compact, "rss", trace
        except RedditPostDeletedError:
            raise
        except Exception as exc:
            failure = _reddit_failure_trace("RSS", exc)
            errors.append(failure)
            trace.append(failure)

    if not is_post:
        errors.append(
            "no source could list this Reddit page; use search_web with a "
            "site:reddit.com/r/<subreddit> query to find posts, then fetch an "
            "individual post URL"
        )
        raise RuntimeError("; ".join(errors))

    oembed_fetched: dict | None = None
    oembed: dict | None = None
    try:
        oembed_fetched, oembed = await _fetch_reddit_oembed(url)
        trace.append("oEmbed metadata succeeded")
    except RedditPostDeletedError:
        raise
    except Exception as exc:
        failure = _reddit_failure_trace("oEmbed", exc)
        errors.append(failure)
        trace.append(failure)

    if not cfg.reddit_search_fallback:
        skipped = "Brave excerpts skipped (reddit_search_fallback disabled)"
    elif not _brave_configured():
        skipped = "Brave excerpts skipped (Brave Search not configured)"
    else:
        skipped = None
        try:
            fetched, compact = await _fetch_reddit_brave(url, oembed)
            trace.append("Brave excerpts succeeded")
            return fetched, compact, "brave", trace
        except Exception as exc:
            failure = _reddit_failure_trace("Brave excerpts", exc)
            errors.append(failure)
            trace.append(failure)
    if skipped:
        errors.append(skipped)
        trace.append(skipped)

    if oembed_fetched is not None and oembed is not None:
        return oembed_fetched, oembed, "oembed", trace

    raise RuntimeError("; ".join(errors) or "no Reddit acquisition method succeeded")


async def acquire_reddit(url: str) -> tuple[dict, Any, str, list[str]]:
    """Acquire Reddit content through the process-wide anti-burst queue."""
    return await _reddit_request_queue.acquire(url)


# ---------------------------------------------------------------------------
# Result presentation (used by fetch_page)
# ---------------------------------------------------------------------------

def reddit_source_url(url: str, source: str, fetched: dict) -> str:
    """The URL fetch_page reports for a Reddit result from `source`."""
    return {
        "oauth": _reddit_oauth_url(url),
        "rss": _reddit_rss_url(url),
        "oembed": _reddit_oembed_url(url),
    }.get(source) or fetched.get("url") or url


def reddit_notes(compact: Any, source: str, trace: list[str], *, debug: bool) -> list[str]:
    """Model-facing caveats for a Reddit result, in the order they are shown.

    Covers which source answered (and what that source can omit), any rate
    limiting seen along the way, and, in debug mode, the redacted fallback trace.
    """
    notes: list[str] = []
    if source == "archive":
        notes.append(_archive_note(compact))
    elif source == "rss":
        notes.append(
            "Returned Reddit's RSS feed; it may contain only an initial "
            "snapshot of the comments."
        )
    elif source == "brave":
        notes.append(
            "No full-thread Reddit source (API, archive, or RSS) returned this "
            "post, so this is search-index "
            "excerpts of the post from Brave, not the full thread: excerpts "
            "may omit parts of the post and most comments, and may lag "
            "recent edits."
        )
    elif source == "oembed":
        notes.append(
            "No full-thread Reddit source (API, archive, or RSS) or "
            "search-index excerpts were available; "
            "returning official oEmbed post metadata without the post body "
            "or comments."
        )
    if any("HTTP 429" in step for step in trace):
        if source == "rss" and any("retrying after" in step for step in trace):
            notes.append(
                "Reddit rate-limited the initial RSS request (HTTP 429); "
                "the automatic retry succeeded."
            )
        else:
            notes.append(
                "At least one Reddit source was rate-limited (HTTP 429). "
                "Retry later for a fuller result."
            )
    if debug:
        notes.append(_redact_backend_secrets(
            "Reddit fallback trace: " + "; ".join(trace) + ".",
            cfg.reddit_client_secret,
        ))
    return notes
