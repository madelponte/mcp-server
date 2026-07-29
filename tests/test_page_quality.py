"""Tests for the hybrid rendered-page quality gate."""

import json

import httpx

from conftest import run
from tools import page_quality as pq


def test_substantive_page_is_accepted():
    html = "<main><h1>Story</h1><p>" + "A useful article sentence. " * 8 + "</p></main>"
    out = pq.deterministic_assessment(200, html)
    assert out.verdict is pq.PageVerdict.ACCEPT


def test_empty_render_is_unusable():
    out = pq.deterministic_assessment(200, "<html><div id='root'></div></html>")
    assert out.verdict is pq.PageVerdict.UNUSABLE


def test_generic_challenge_structure_is_blocked():
    html = (
        "<html><body><form id='human-verification'>"
        "<p>Verify that you are human</p><button>Continue</button></form></body></html>"
    )
    out = pq.deterministic_assessment(200, html)
    assert out.verdict is pq.PageVerdict.BLOCKED


def test_article_discussing_captchas_is_not_blocked():
    html = "<article><h1>CAPTCHA accessibility</h1>" + "".join(
        f"<p>Paragraph {i} discusses accessible web design and practical alternatives.</p>"
        for i in range(4)
    ) + "</article>"
    out = pq.deterministic_assessment(200, html)
    assert out.verdict is pq.PageVerdict.ACCEPT


def test_sparse_page_is_uncertain():
    out = pq.deterministic_assessment(200, "<html><body>Service operational</body></html>")
    assert out.verdict is pq.PageVerdict.UNCERTAIN


def test_optional_openai_classifier_resolves_uncertain(monkeypatch, patch_httpx):
    monkeypatch.setattr(pq.cfg, "classifier_api_url", "http://classifier:8000/v1")
    monkeypatch.setattr(pq.cfg, "classifier_api_key", "secret")
    monkeypatch.setattr(pq.cfg, "classifier_model", "small-4b")
    monkeypatch.setattr(pq.cfg, "classifier_min_confidence", 0.7)
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({
                    "verdict": "blocked", "confidence": 0.95, "reason": "verification interstitial"
                })}}]
            },
        )

    patch_httpx(handler)
    out = run(pq.assess_page("https://example.com", 200, "<body>Please wait here</body>"))
    assert out.verdict is pq.PageVerdict.BLOCKED
    assert out.source == "llm"
    assert seen["url"] == "http://classifier:8000/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "small-4b"


def test_classifier_failure_leaves_deterministic_uncertain(monkeypatch, patch_httpx):
    monkeypatch.setattr(pq.cfg, "classifier_api_url", "http://classifier/v1")
    monkeypatch.setattr(pq.cfg, "classifier_model", "small-4b")
    patch_httpx(lambda request: httpx.Response(500, text="down"))
    out = run(pq.assess_page("https://example.com", 200, "<body>Short page</body>"))
    assert out.verdict is pq.PageVerdict.UNCERTAIN
    assert out.source == "deterministic"
