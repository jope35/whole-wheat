"""Data contracts for whole-wheat."""

from __future__ import annotations

from typing import Any, TypedDict


PAGE_ROW_KEYS = ("chunk_id", "source_file", "page_id", "year", "text")


class PageRow(TypedDict):
    chunk_id: str
    source_file: str
    page_id: int
    year: int | None
    text: str


def page_row_from_dict(data: dict[str, Any]) -> PageRow:
    missing = [k for k in PAGE_ROW_KEYS if k not in data]
    if missing:
        raise ValueError(f"PageRow missing keys: {missing}")
    return PageRow(
        chunk_id=str(data["chunk_id"]),
        source_file=str(data["source_file"]),
        page_id=int(data["page_id"]),
        year=None if data["year"] is None else int(data["year"]),
        text=str(data["text"]),
    )


def page_row_to_dict(row: PageRow) -> dict[str, Any]:
    return {k: row[k] for k in PAGE_ROW_KEYS}
