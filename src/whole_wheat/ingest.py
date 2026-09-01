"""Flatten OfficeQA-shaped JSON into PageRow lists and write Delta."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from whole_wheat.contracts import PageRow

TEXT_BYTE_CAP = 32_764
KEEP_TYPES = frozenset({"text", "table"})
# ponytail: year from filename patterns only; use metadata later if nulls pile up
_YEAR_RE = re.compile(r"(?:^|[_\-])((?:17|18|19|20)\d{2})(?:$|[_\-])")


def parse_year(source_file: str, doc: dict[str, Any] | None = None) -> int | None:
    match = _YEAR_RE.search(source_file)
    if match:
        return int(match.group(1))
    if doc:
        meta = doc.get("metadata") or {}
        year = meta.get("year")
        if year is not None:
            return int(year)
    return None


def _truncate(text: str) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= TEXT_BYTE_CAP:
        return text
    return raw[:TEXT_BYTE_CAP].decode("utf-8", errors="ignore")


def _page_texts(doc: dict[str, Any]) -> dict[int, list[str]]:
    pages: dict[int, list[str]] = {}
    elements = (doc.get("document") or {}).get("elements") or []
    for el in elements:
        if el.get("type") not in KEEP_TYPES:
            continue
        content = el.get("content")
        if not content:
            continue
        bboxes = el.get("bbox") or []
        page_ids = {b["page_id"] for b in bboxes if "page_id" in b}
        if not page_ids:
            continue
        for page_id in page_ids:
            pages.setdefault(page_id, []).append(str(content))
    return pages


def document_to_page_rows(doc: dict[str, Any], source_file: str) -> list[PageRow]:
    year = parse_year(source_file, doc)
    rows: list[PageRow] = []
    for page_id, parts in sorted(_page_texts(doc).items()):
        text = _truncate("\n".join(parts))
        if not text:
            continue
        rows.append(
            PageRow(
                chunk_id=f"{source_file}::page-{page_id}",
                source_file=source_file,
                page_id=page_id,
                year=year,
                text=text,
            )
        )
    return rows


def load_json_file(path: Path) -> list[PageRow]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    source_file = path.stem
    return document_to_page_rows(doc, source_file)


def load_json_dir(dir_path: Path) -> list[PageRow]:
    rows: list[PageRow] = []
    for path in sorted(dir_path.glob("*.json")):
        rows.extend(load_json_file(path))
    return rows


def write_page_rows(spark: Any, rows: list[PageRow], table: str) -> None:
    if not rows:
        raise ValueError("no page rows to write")
    df = spark.createDataFrame([dict(r) for r in rows])
    df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(table)
    spark.sql(
        f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
