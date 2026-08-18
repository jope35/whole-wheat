from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from fastembed import TextEmbedding
from sqlite_vec import serialize_float32

from whole_wheat.retriever import DOC_PREFIX, EMBED_MODEL, open_db


def make_article_id(title: str, source: str, published_at: str) -> str:
    return hashlib.sha256(f"{title}\n{source}\n{published_at}".encode()).hexdigest()[:16]


def chunk_article(article: dict) -> list[dict]:
    article_id = make_article_id(article["title"], article["source"], article["published_at"])
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", article["body"]):
        words = paragraph.split()
        if len(words) <= 800:
            if words:
                paragraphs.append(paragraph.strip())
            continue
        paragraphs.extend(
            " ".join(words[start : start + 800])
            for start in range(0, len(words), 800)
        )
    windows: list[list[str]] = []
    current: list[str] = []
    n = 0
    for paragraph in paragraphs:
        words = max(1, len(paragraph.split()))
        if current and n + words > 800:
            windows.append(current)
            current, n = [paragraph], words
            continue
        current.append(paragraph)
        n += words
    if current:
        windows.append(current)
    return [
        {
            "chunk_id": f"{article_id}::chunk-{i}",
            "article_id": article_id,
            "title": article["title"],
            "source": article["source"],
            "published_at": article["published_at"],
            "text": "\n\n".join(parts),
        }
        for i, parts in enumerate(windows)
    ]


def default_embed_documents(texts: list[str]) -> list[list[float]]:
    model = TextEmbedding(model_name=EMBED_MODEL)
    prefixed = [f"{DOC_PREFIX}{text}" for text in texts]
    for attempt in range(3):
        try:
            return [vector.tolist() for vector in model.embed(prefixed)]
        except Exception:
            if attempt == 2:
                raise


def ingest(
    corpus_path: Path,
    index_path: Path,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> int:
    chunks = []
    for article in json.loads(corpus_path.read_text()):
        chunks.extend(chunk_article(article))
    vectors = (embed_fn or default_embed_documents)([c["text"] for c in chunks])
    dim = len(vectors[0])
    tmp_path = index_path.with_suffix(".tmp")
    tmp_path.unlink(missing_ok=True)
    db = open_db(tmp_path)
    try:
        db.execute(
            "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, article_id TEXT, "
            "title TEXT, source TEXT, published_at TEXT, text TEXT)"
        )
        db.execute(
            f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
            f"chunk_id TEXT PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
        )
        for chunk, vector in zip(chunks, vectors, strict=True):
            db.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chunk["chunk_id"],
                    chunk["article_id"],
                    chunk["title"],
                    chunk["source"],
                    chunk["published_at"],
                    chunk["text"],
                ),
            )
            db.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (chunk["chunk_id"], serialize_float32(vector)),
            )
        db.commit()
    except Exception:
        db.close()
        tmp_path.unlink(missing_ok=True)
        raise
    else:
        db.close()
    os.replace(tmp_path, index_path)
    return len(chunks)
