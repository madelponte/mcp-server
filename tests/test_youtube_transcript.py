"""Tests for tools/youtube_transcript.py — URL parsing & formatting helpers."""

import pytest

from tools.youtube_transcript import (
    _extract_video_id,
    is_youtube_video_url,
    _format_timestamp,
)

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        VIDEO_ID,  # bare id
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}&t=10s",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}",
        f"youtube.com/watch?v={VIDEO_ID}",  # scheme added automatically
    ],
)
def test_extract_video_id_variants(url):
    assert _extract_video_id(url) == VIDEO_ID


def test_extract_video_id_empty_raises():
    with pytest.raises(ValueError):
        _extract_video_id("")


def test_extract_video_id_unparseable_raises():
    with pytest.raises(ValueError):
        _extract_video_id("https://example.com/not-a-video")


def test_extract_video_id_channel_raises():
    with pytest.raises(ValueError):
        _extract_video_id("https://www.youtube.com/@somechannel")


# --------------------------- is_youtube_video_url ---------------------------

def test_is_youtube_video_url_true():
    assert is_youtube_video_url(f"https://youtu.be/{VIDEO_ID}") is True
    assert is_youtube_video_url(f"https://www.youtube.com/watch?v={VIDEO_ID}") is True


def test_is_youtube_video_url_requires_scheme():
    # A bare id is a valid id but not a URL this predicate accepts.
    assert is_youtube_video_url(VIDEO_ID) is False
    assert is_youtube_video_url(f"youtube.com/watch?v={VIDEO_ID}") is False


def test_is_youtube_video_url_non_video_pages_false():
    assert is_youtube_video_url("https://www.youtube.com/@channel") is False
    assert is_youtube_video_url("https://www.youtube.com/playlist?list=PL123") is False
    assert is_youtube_video_url("https://example.com/video") is False


def test_is_youtube_video_url_lookalike_domains_false():
    # A domain that merely ends in "youtube.com" (notyoutube.com, …) is not
    # YouTube; routing it to the transcript fetcher would skip its real content.
    for host in ("notyoutube.com", "myyoutube.com", "evilyoutube.com"):
        assert is_youtube_video_url(f"https://{host}/watch?v={VIDEO_ID}") is False


def test_extract_video_id_lookalike_domain_raises():
    with pytest.raises(ValueError):
        _extract_video_id(f"https://notyoutube.com/watch?v={VIDEO_ID}")


def test_extract_video_id_youtube_subdomains():
    for host in ("m.youtube.com", "music.youtube.com"):
        assert _extract_video_id(f"https://{host}/watch?v={VIDEO_ID}") == VIDEO_ID


# --------------------------- _format_timestamp ---------------------------

def test_format_timestamp_minutes_seconds():
    assert _format_timestamp(0) == "0:00"
    assert _format_timestamp(5) == "0:05"
    assert _format_timestamp(65) == "1:05"
    assert _format_timestamp(599) == "9:59"


def test_format_timestamp_hours():
    assert _format_timestamp(3600) == "1:00:00"
    assert _format_timestamp(3661) == "1:01:01"


def test_format_timestamp_truncates_fractional():
    assert _format_timestamp(65.9) == "1:05"
