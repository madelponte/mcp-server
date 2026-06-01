"""
YouTube Transcript MCP tool.

Fetches the transcript/captions of a YouTube video using the open-source
youtube-transcript-api library (no API key required). Translated from the
Open WebUI tool; status-emitter calls were removed.
"""

import functools
import re
from typing import List
from urllib.parse import urlparse, parse_qs

import anyio
from mcp.server.fastmcp import FastMCP

# Compatible with youtube-transcript-api >= 1.0.0
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

from config import youtube_settings as cfg

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _extract_video_id(url_or_id: str) -> str:
    """Extract an 11-character YouTube video ID from a URL or pass through a bare ID."""
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("No URL or video ID provided.")

    if _VIDEO_ID_RE.match(s):
        return s

    if not s.startswith(("http://", "https://")):
        s = "https://" + s

    parsed = urlparse(s)
    host = (parsed.hostname or "").lower().lstrip("www.")
    path = parsed.path or ""

    if host == "youtu.be":
        candidate = path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(candidate):
            return candidate

    if host.endswith("youtube.com") or host == "youtube-nocookie.com":
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            candidate = qs["v"][0]
            if _VIDEO_ID_RE.match(candidate):
                return candidate

        m = re.match(r"^/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})", path)
        if m:
            return m.group(1)

    raise ValueError(
        f"Could not extract a YouTube video ID from: {url_or_id!r}. "
        "Pass a full YouTube URL or an 11-character video ID."
    )


def _format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _build_client() -> YouTubeTranscriptApi:
    if cfg.webshare_proxy_username and cfg.webshare_proxy_password:
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=cfg.webshare_proxy_username,
                proxy_password=cfg.webshare_proxy_password,
            )
        )
    if cfg.http_proxy_url:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=cfg.http_proxy_url,
                https_url=cfg.http_proxy_url,
            )
        )
    return YouTubeTranscriptApi()


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def get_youtube_transcript(url: str, languages: str | None = None) -> str:
        """
        Fetch the transcript / closed captions of a YouTube video and return it
        as plain text so you can summarize, quote, translate, or answer
        questions about its contents.

        USE THIS when the user:
          - Pastes a YouTube URL and asks anything about the video
          - Asks you to summarize, transcribe, or "tell me what this video says"
          - Wants to quote, search, or translate spoken content from a video
          - References a specific YouTube video by URL or 11-char video ID
          - You need to get information about a Youtube video that came up in a search

        DO NOT use this for:
          - Generic web pages (use a web fetch/search tool instead)
          - Non-YouTube videos (Vimeo, TikTok, etc. — not supported)
          - Downloading audio/video files

        Notes:
          - The transcript may be auto-generated and contain mis-transcriptions;
            treat exact wording with mild skepticism but the substance is reliable.


        :param url: A YouTube URL (youtube.com/watch, youtu.be, /shorts/, /embed/,
                    /live/) or a bare 11-character video ID.
        :param languages: Optional comma-separated language codes to prefer
                          (e.g. "en,es"). Overrides the default for this call.
        :return: The transcript as a single string (optionally with timestamps),
                 prefixed by a short metadata header, or an error message.
        """
        try:
            video_id = _extract_video_id(url)

            lang_str = languages if languages else cfg.default_languages
            lang_list: List[str] = [
                code.strip() for code in lang_str.split(",") if code.strip()
            ] or ["en"]

            client = _build_client()

            # Try preferred languages first, then fall back to anything available.
            try:
                fetched = await anyio.to_thread.run_sync(
                    functools.partial(client.fetch, video_id, languages=lang_list)
                )
            except NoTranscriptFound:
                transcript_list = await anyio.to_thread.run_sync(
                    functools.partial(client.list, video_id)
                )
                any_transcript = None
                for t in transcript_list:
                    any_transcript = t
                    break
                if any_transcript is None:
                    raise
                fetched = await anyio.to_thread.run_sync(any_transcript.fetch)

            snippets = list(fetched)
            if not snippets:
                return f"❌ The transcript for video {video_id} is empty."

            language = getattr(fetched, "language", None) or "unknown"
            language_code = getattr(fetched, "language_code", None) or "?"
            is_generated = getattr(fetched, "is_generated", None)
            kind = (
                "auto-generated"
                if is_generated
                else ("manually-created" if is_generated is False else "unknown source")
            )

            include_ts = cfg.include_timestamps
            lines: List[str] = []
            for snip in snippets:
                text = (snip.text or "").replace("\n", " ").strip()
                if not text:
                    continue
                if include_ts:
                    ts = _format_timestamp(snip.start)
                    lines.append(f"[{ts}] {text}")
                else:
                    lines.append(text)

            body = "\n".join(lines) if include_ts else " ".join(lines)

            truncated_note = ""
            max_chars = cfg.max_characters
            if max_chars and len(body) > max_chars:
                body = body[:max_chars].rsplit(" ", 1)[0] + " …"
                truncated_note = (
                    f"\n\n[Note: transcript truncated to {max_chars} characters "
                    "by tool configuration.]"
                )

            header = (
                f"Transcript for YouTube video {video_id}\n"
                f"Language: {language} ({language_code}) — {kind}\n"
                f"Segments: {len(snippets)}\n"
                f"Source: https://www.youtube.com/watch?v={video_id}\n"
                "---"
            )

            return f"{header}\n{body}{truncated_note}"

        except ValueError as ve:
            return f"❌ {ve}"
        except TranscriptsDisabled:
            return "❌ This video has subtitles/transcripts disabled by the uploader."
        except NoTranscriptFound:
            return (
                "❌ No transcript was found for this video in any language. "
                "It may not have captions at all."
            )
        except VideoUnavailable:
            return "❌ This video is unavailable (private, removed, or region-blocked)."
        except (RequestBlocked, IpBlocked):
            return (
                "❌ YouTube is blocking requests from this server's IP address. "
                "This is common on cloud providers (AWS, GCP, Azure, etc.). "
                "Configure a residential proxy via the YOUTUBE_WEBSHARE_PROXY_* or "
                "YOUTUBE_HTTP_PROXY_URL environment variables to work around it."
            )
        except Exception as exc:
            return f"❌ Error fetching transcript: {type(exc).__name__}: {exc}"
