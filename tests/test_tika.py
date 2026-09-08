"""Offline Tika 4 wire-contract, Markdown, OCR fallback and backpressure tests."""

import json
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest

from tools import web_fetch as wf


@pytest.fixture
def tika_transport(monkeypatch):
    calls = []
    sleeps = []
    replies = []
    monkeypatch.setattr(wf.cfg, "max_concurrent_tika", 1)
    monkeypatch.setattr(wf.time, "sleep", sleeps.append)

    def handler(request):
        envelope = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + request.headers["Content-Type"].encode()
            + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.content
        )
        parts = {
            part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in envelope.iter_parts()
        }
        config_part = parts["config"]
        assert isinstance(config_part, bytes)
        calls.append((request, json.loads(config_part), parts["file"]))
        assert replies, "Unexpected additional Tika request"
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

    @contextmanager
    def stream(method, url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream(method, url, **kwargs) as response:
                yield response

    monkeypatch.setattr(wf.httpx, "stream", stream)
    return replies, calls, sleeps


def extract(**kwargs):
    return wf._tika_extract(b"%PDF-1.4 scan bytes", "http://tika:9998/", **kwargs)


def strategies(calls):
    return [config["pdf-parser"]["ocr"]["strategy"] for _, config, _ in calls]


def assert_capacity_released():
    sema = wf._tika_semaphore()
    assert sema.acquire(blocking=False), "Tika capacity leaked on an error/retry"
    sema.release()


def test_native_markdown_is_preserved_without_ocr(tika_transport):
    replies, calls, _ = tika_transport
    markdown = "# Report\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n- Item"
    replies.append({"tk:content": markdown, "pdf:chars-per-page": "20"})
    assert extract() == markdown
    assert strategies(calls) == ["NO_OCR"]
    request, config, data = calls[0]
    assert request.method == "POST"
    assert request.url.path == "/tika/config/json/md"
    assert not any(key.lower().startswith("x-tika-") for key in request.headers)
    assert config["tesseract-ocr-parser"] == {"skipOcr": True}
    assert 0 < config["timeout-limits"]["totalTaskTimeoutMillis"] <= 81000
    assert data == b"%PDF-1.4 scan bytes"
    assert_capacity_released()


@pytest.mark.parametrize("counts", ["0", 0, ["0", "0"]])
def test_scanned_pdf_with_metadata_title_retries_ocr(tika_transport, counts):
    replies, calls, _ = tika_transport
    replies.extend([
        {"tk:content": "Presentation1  \n", "dc:title": "Presentation1", "pdf:chars-per-page": counts},
        {"tk:content": "Presentation1\n\nHappy New Year 2003!", "dc:title": "Presentation1", "pdf:chars-per-page": "0"},
    ])
    assert "Happy New Year" in extract()
    assert strategies(calls) == ["NO_OCR", "OCR_AND_TEXT_EXTRACTION"]
    assert calls[1][1]["tesseract-ocr-parser"] == {"skipOcr": False}
    assert calls[0][2] == calls[1][2]
    assert_capacity_released()


@pytest.mark.parametrize("empty", [{}, {"tk:content": " \n\t"}, {"tk:content": "![](embedded:image1.png)"}])
def test_empty_or_image_only_markdown_retries(tika_transport, empty):
    replies, calls, _ = tika_transport
    replies.extend([empty, {"tk:content": "Recovered OCR text"}])
    assert extract() == "Recovered OCR text"
    assert len(calls) == 2


def test_mixed_pdf_with_some_native_text_is_not_retried(tika_transport):
    replies, calls, _ = tika_transport
    replies.append({"tk:content": "Text on page one", "pdf:chars-per-page": ["16", "0"]})
    assert extract() == "Text on page one"
    assert len(calls) == 1


def test_non_pdf_markdown_without_pdf_metadata(tika_transport):
    replies, calls, _ = tika_transport
    replies.append({"tk:content": "# Word document\n\nText", "Content-Type": "application/msword"})
    assert extract().startswith("# Word document")
    assert len(calls) == 1


@pytest.mark.parametrize("strategy", ["auto", "ocr_only", "ocr_and_text_extraction"])
def test_explicit_ocr_strategy_runs_once(tika_transport, strategy):
    replies, calls, _ = tika_transport
    replies.append({"tk:content": "OCR text", "pdf:chars-per-page": "0"})
    assert extract(ocr_strategy=strategy) == "OCR text"
    assert strategies(calls) == [strategy.upper()]
    assert calls[0][1]["tesseract-ocr-parser"]["skipOcr"] is False


def test_retry_can_be_disabled(tika_transport):
    replies, calls, _ = tika_transport
    replies.append({})
    with pytest.raises(RuntimeError, match="no extractable text"):
        extract(ocr_retry=False)
    assert len(calls) == 1
    assert_capacity_released()


@pytest.mark.parametrize("strategy", ["no_ocr", "auto", "ocr_only"])
def test_textless_result_still_fails_after_available_passes(tika_transport, strategy):
    replies, calls, _ = tika_transport
    title_only = {"tk:content": "Scan title  \n", "dc:title": "Scan title", "pdf:chars-per-page": "0"}
    replies.extend([title_only, title_only])
    with pytest.raises(RuntimeError, match="no extractable text"):
        extract(ocr_strategy=strategy)
    assert len(calls) == (2 if strategy == "no_ocr" else 1)
    assert_capacity_released()


@pytest.mark.parametrize("status", [400, 413, 422, 500, 503])
def test_http_failure_is_not_an_ocr_retry(tika_transport, status):
    replies, calls, _ = tika_transport
    replies.append(httpx.Response(status, json={"status": "ERROR"}))
    with pytest.raises(httpx.HTTPStatusError):
        extract()
    assert len(calls) == 1
    assert_capacity_released()


def test_disabled_per_request_config_is_actionable(tika_transport):
    replies, calls, _ = tika_transport
    replies.append(httpx.Response(403))
    with pytest.raises(RuntimeError, match="allowPerRequestConfig=true"):
        extract()
    assert len(calls) == 1
    assert_capacity_released()


@pytest.mark.parametrize("key", ["tk:exception:container-exception", "tk:exception:write-limit-reached"])
def test_http_200_parser_failure_is_not_returned_or_ocr_retried(tika_transport, key):
    replies, calls, _ = tika_transport
    replies.append({"tk:content": "Partial content", key: "true"})
    with pytest.raises(RuntimeError, match="incomplete or failed extraction"):
        extract()
    assert len(calls) == 1
    assert_capacity_released()


@pytest.mark.parametrize("reply", [
    httpx.Response(200, text="not JSON"),
    httpx.Response(200, json=[]),
    httpx.Response(200, json={"X-TIKA:content": "Tika 3"}),
    httpx.Response(200, json={"tk:content": ["invalid"]}),
    httpx.ReadTimeout("read timed out"),
])
def test_invalid_response_or_timeout_does_not_trigger_ocr(tika_transport, reply):
    replies, calls, _ = tika_transport
    replies.append(reply)
    with pytest.raises((RuntimeError, ValueError, httpx.ReadTimeout)):
        extract()
    assert len(calls) == 1
    assert_capacity_released()


@pytest.mark.parametrize("fallback", [False, True])
def test_download_cap_applies_to_both_passes(tika_transport, fallback):
    replies, calls, _ = tika_transport
    if fallback:
        replies.append({})
    replies.append({"tk:content": "x" * 1000})
    with pytest.raises(wf.DownloadTooLargeError):
        extract(max_output_bytes=100)
    assert len(calls) == (2 if fallback else 1)
    assert_capacity_released()


def test_busy_worker_honors_retry_after_without_enabling_ocr(tika_transport):
    replies, calls, sleeps = tika_transport
    replies.extend([httpx.Response(429, headers={"Retry-After": "2"}), {"tk:content": "Native text"}])
    assert extract() == "Native text"
    assert sleeps == [2]
    assert strategies(calls) == ["NO_OCR", "NO_OCR"]
    assert_capacity_released()


def test_busy_retries_are_bounded(tika_transport):
    replies, calls, sleeps = tika_transport
    replies.extend(httpx.Response(429, headers={"Retry-After": "0"}) for _ in range(3))
    with pytest.raises(httpx.HTTPStatusError):
        extract()
    assert len(calls) == 3
    assert sleeps == [0, 0]
    assert_capacity_released()


def test_huge_retry_after_does_not_exceed_budget(tika_transport):
    replies, calls, sleeps = tika_transport
    replies.append(httpx.Response(429, headers={"Retry-After": "999999"}))
    with pytest.raises(httpx.HTTPStatusError):
        extract(timeout=5)
    assert len(calls) == 1
    assert sleeps == []
    assert_capacity_released()


def test_local_capacity_wait_is_bounded(monkeypatch):
    class BusySemaphore:
        def acquire(self, *, timeout):
            assert timeout == 5
            return False

        def release(self):
            pytest.fail("Must not release capacity we did not acquire")

    monkeypatch.setattr(wf, "_tika_semaphore", BusySemaphore)
    with pytest.raises(TimeoutError, match="local Tika extraction capacity"):
        extract(timeout=5)
