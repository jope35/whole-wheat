"""Contract tests for PageRow ingest. No Spark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whole_wheat.contracts import PAGE_ROW_KEYS, page_row_from_dict, page_row_to_dict
from whole_wheat.ingest import TEXT_BYTE_CAP, document_to_page_rows, load_json_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixture.json"


def test_page_row_round_trip_json() -> None:
    row = {
        "chunk_id": "doc::page-1",
        "source_file": "doc",
        "page_id": 1,
        "year": 1872,
        "text": "hello",
    }
    loaded = page_row_from_dict(json.loads(json.dumps(row)))
    assert page_row_to_dict(loaded) == row


def test_page_row_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing"):
        page_row_from_dict({"chunk_id": "x", "source_file": "y", "page_id": 0})


def test_fixture_json_yields_page_rows() -> None:
    rows = load_json_file(FIXTURE)
    assert rows
    for row in rows:
        for key in PAGE_ROW_KEYS:
            assert key in row
        page_row_from_dict(dict(row))


def test_header_only_page_produces_no_row() -> None:
    doc = {
        "document": {
            "elements": [
                {
                    "type": "page_header",
                    "content": "only a header",
                    "bbox": [{"page_id": 0, "coord": [0, 0, 1, 1]}],
                }
            ]
        }
    }
    assert document_to_page_rows(doc, "header_only") == []


def test_text_truncated_at_byte_cap() -> None:
    huge = "a" * (TEXT_BYTE_CAP + 500)
    doc = {
        "document": {
            "elements": [
                {
                    "type": "text",
                    "content": huge,
                    "bbox": [{"page_id": 0, "coord": [0, 0, 1, 1]}],
                }
            ]
        }
    }
    rows = document_to_page_rows(doc, "long_doc")
    assert len(rows) == 1
    assert len(rows[0]["text"].encode("utf-8")) <= TEXT_BYTE_CAP
