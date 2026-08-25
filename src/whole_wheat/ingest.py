from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from whole_wheat.constants import (
    KEEP_TYPES,
    MAX_CHUNK_TOKENS,
    PARSED_JSON_DIR,
)

YEAR_RE = re.compile(r"\d{4}")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    source: str
    year: str
    text: str


def list_json_paths(limit: int | None) -> list[Path]:
    paths = sorted(PARSED_JSON_DIR.glob("*.json"))
    if not paths:
        sys.exit("no files")
    if limit is None:
        return paths
    return paths[:limit]


def parse_source(stem: str) -> str:
    if stem.startswith("govinfo_receipts"):
        return "govinfo_receipts"
    if stem.startswith("combined_statement"):
        return "combined_statement"
    return stem.split("__", 1)[0]


def parse_year(stem: str) -> str:
    match = YEAR_RE.search(stem)
    if match is None:
        return "unknown"
    return match.group(0)


def file_title(elements: list[dict], document_id: str) -> str:
    for element in elements:
        if element.get("type") in {"title", "section_header"}:
            content = (element.get("content") or "").strip()
            if content:
                return content
    return document_id


def page_id(element: dict, missing_key: int) -> int:
    bbox = element.get("bbox") or []
    if bbox and isinstance(bbox[0], dict) and "page_id" in bbox[0]:
        return int(bbox[0]["page_id"])
    return missing_key


def split_non_table(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_CHUNK_TOKENS:
        return [text]
    pieces = []
    for start in range(0, len(words), MAX_CHUNK_TOKENS):
        pieces.append(" ".join(words[start : start + MAX_CHUNK_TOKENS]))
    return pieces


def flush(window: list[str]) -> str:
    return "\n\n".join(window)


def chunk_file(path: Path) -> list[Chunk]:
    payload = json.loads(path.read_text())
    elements = payload.get("document", {}).get("elements", [])
    document_id = path.stem
    title = file_title(elements, document_id)
    source = parse_source(document_id)
    year = parse_year(document_id)

    texts: list[str] = []
    window: list[str] = []
    window_tokens = 0
    window_page: int | None = None
    missing_seq = -1

    def emit() -> None:
        nonlocal window, window_tokens, window_page
        if window:
            texts.append(flush(window))
        window = []
        window_tokens = 0
        window_page = None

    for element in elements:
        if element.get("type") not in KEEP_TYPES:
            continue
        content = (element.get("content") or "").strip()
        if not content:
            continue

        missing_seq -= 1
        page = page_id(element, missing_seq)
        if element.get("type") == "table":
            pieces = [content]
        else:
            pieces = split_non_table(content)

        for piece in pieces:
            n = len(piece.split())
            same = window_page is not None and page == window_page
            fits = window_tokens + n <= MAX_CHUNK_TOKENS
            if window and same and fits:
                window.append(piece)
                window_tokens += n
            else:
                emit()
                window = [piece]
                window_tokens = n
                window_page = page

    emit()

    chunks = []
    for index, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}::chunk-{index}",
                document_id=document_id,
                title=title,
                source=source,
                year=year,
                text=text,
            )
        )
    return chunks
