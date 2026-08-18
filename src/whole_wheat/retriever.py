from __future__ import annotations

import sqlite3
from collections.abc import Callable
from functools import cache
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from sqlite_vec import serialize_float32

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


@cache
def embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL)


def default_embed_query(query: str) -> list[float]:
    for attempt in range(3):
        try:
            return next(
                embedding_model().embed([f"{QUERY_PREFIX}{query}"])
            ).tolist()
        except Exception:
            if attempt == 2:
                raise


class Retriever:
    def __init__(
        self,
        db_path: Path,
        embed_query: Callable[[str], list[float]] | None = None,
    ) -> None:
        if not db_path.exists():
            raise FileNotFoundError(str(db_path))
        self.db_path = db_path
        self.embed_query = embed_query or default_embed_query

    def search(self, query: str, k: int = 8) -> list[dict]:
        db = open_db(self.db_path)
        rows = db.execute(
            """
            SELECT chunks.*
            FROM vec_chunks
            JOIN chunks ON chunks.chunk_id = vec_chunks.chunk_id
            WHERE vec_chunks.embedding MATCH ? AND k = ?
            """,
            (serialize_float32(self.embed_query(query)), k),
        ).fetchall()
        db.close()
        return [dict(row) for row in rows]

    def grep(self, pattern: str, k: int = 8) -> list[dict]:
        db = open_db(self.db_path)
        escaped = (
            pattern.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        rows = db.execute(
            "SELECT * FROM chunks WHERE text LIKE ? ESCAPE '\\'",
            (f"%{escaped}%",),
        ).fetchall()
        db.close()
        return [dict(row) for row in rows[:k]]

    def duplicate_titles(self) -> set[str]:
        db = open_db(self.db_path)
        rows = db.execute(
            "SELECT DISTINCT title, article_id FROM chunks"
        ).fetchall()
        db.close()
        articles_by_title = {}
        duplicates = set()
        for row in rows:
            title = " ".join(row["title"].casefold().split())
            if title in articles_by_title and articles_by_title[title] != row["article_id"]:
                duplicates.add(title)
            articles_by_title[title] = row["article_id"]
        return duplicates
