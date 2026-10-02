"""MCP output schemas for structured tool results.

Pi's codemode preserves an MCP ``CallToolResult`` and types its
``structuredContent`` field from the schema advertised by the server.  Keep
these schemas object-shaped (as required by MCP) and broad enough for optional
provider fields while still exposing the stable envelope each tool returns.
"""

SEARCH_WEB_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "provider": {"type": "string"},
        "time_range": {"type": "string"},
        "country": {"type": "string"},
        "search_lang": {"type": "string"},
        "safesearch": {"type": "string"},
        "context_threshold_mode": {"type": "string"},
        "max_tokens": {"type": "integer"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "snippets": {"type": "array", "items": {"type": "string"}},
                    "page_date": {"type": "string"},
                    "date_source": {"type": "string"},
                    "description": {"type": "string"},
                    "site_name": {"type": "string"},
                },
                "required": ["url", "title", "snippets"],
            },
        },
    },
    "required": [
        "query",
        "provider",
        "time_range",
        "country",
        "search_lang",
        "max_tokens",
        "results",
    ],
}

FETCH_PAGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "format": {"type": "string"},
        # Provenance fields from fetch_page._provenance: present only when the
        # fetch deviated from a direct HTTP 200 on the requested URL, or in debug.
        "original_url": {"type": "string"},
        "status": {"type": ["integer", "null"]},
        "content_type": {"type": "string"},
        "via": {"type": "string"},
        "content": {},
        # HTML text/section pages.
        "title": {"type": ["string", "null"]},
        # section= results.
        "matched_heading": {"type": "string"},
        "anchor": {"type": "string"},
        "level": {"type": "integer"},
        "next_heading": {"type": ["string", "null"]},
        "next_heading_anchor": {"type": ["string", "null"]},
        "source_fragment": {"type": "string"},
        "citation_url": {"type": "string"},
        # Standalone images.
        "media_type": {"type": "string"},
        "filename": {"type": "string"},
        "description": {"type": "string"},
        "description_source": {"type": "string"},
        # query= results.
        "query": {"type": "string"},
        "context_lines": {"type": "integer"},
        "line_numbering": {"type": "string"},
        "match_count": {"type": "integer"},
        "match_metadata": {"type": "array", "items": {"type": "object"}},
        "matching_toc": {"type": "array", "items": {}},
        "sections": {"type": "integer"},
        "truncated": {"type": "boolean"},
        "offset": {"type": "integer"},
        "continuation_anchor": {"type": "string"},
        "next_offset": {"type": "integer"},
        "content_length": {"type": "integer"},
        "content_format": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["url", "format", "content"],
}

COMPANY_DATA_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "sections": {"type": "array", "items": {"type": "string"}},
        "data": {"type": "object"},
        "resolved_from": {},
        "errors": {"type": "object"},
        "results": {"type": "array", "items": {"type": "object"}},
        "note": {"type": "string"},
    },
}

WOLFRAM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "data": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "result": {"type": "string"},
        "assumptions": {
            "type": "object",
            "properties": {
                "used": {"type": "string"},
                "alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "assumption": {"type": "string"},
                        },
                        "required": ["description", "assumption"],
                    },
                },
            },
            "required": ["alternatives"],
        },
        "url": {"type": "string"},
    },
    "required": ["query"],
}

PLACES_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "query_category": {"type": "string"},
        "center": {"type": "object"},
        "radius_m": {"type": "integer"},
        "count": {"type": "integer"},
        "results": {"type": "array", "items": {"type": "object"}},
        "nearby_towns_radius_m": {"type": "integer"},
        "nearby_towns": {"type": "array", "items": {"type": "object"}},
        "place": {"type": "object"},
        "alternatives": {"type": "array", "items": {"type": "object"}},
    },
}

EMAIL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"enum": ["sent", "partial", "failed"], "type": "string"},
        "subject": {"type": "string"},
        "recipients": {
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}},
                "cc": {"type": "array", "items": {"type": "string"}},
                "bcc": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["to", "cc", "bcc"],
        },
        "attempted_recipients": {"type": "array", "items": {"type": "string"}},
        "accepted_recipients": {"type": "array", "items": {"type": "string"}},
        "refused_recipients": {"type": "array", "items": {"type": "object"}},
        "invalid_recipients": {"type": "array", "items": {"type": "object"}},
        "dropped_recipients": {"type": "array", "items": {"type": "object"}},
        "attachments": {"type": "array", "items": {"type": "object"}},
        "dropped": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "status",
        "subject",
        "recipients",
        "attempted_recipients",
        "accepted_recipients",
        "refused_recipients",
        "invalid_recipients",
        "dropped_recipients",
        "attachments",
        "dropped",
    ],
}
