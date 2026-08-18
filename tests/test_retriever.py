import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from whole_wheat.retriever import Retriever


def fake_embed(text: str) -> list[float]:
    v = [0.0, 0.0, 0.0, 0.0]
    t = text.casefold()
    if "apple" in t:
        v[0] = 1.0
    if "banana" in t:
        v[1] = 1.0
    if "paris" in t:
        v[2] = 1.0
    if "2023-10-01" in t:
        v[3] = 1.0
    return v


def write_index(path: Path) -> None:
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, article_id TEXT, "
        "title TEXT, source TEXT, published_at TEXT, text TEXT)"
    )
    db.execute(
        "CREATE VIRTUAL TABLE vec_chunks USING vec0("
        "chunk_id TEXT PRIMARY KEY, embedding float[4] distance_metric=cosine)"
    )
    rows = [
        ("a::chunk-0", "a", "Apples", "The Verge", "2023-09-01T00:00:00+00:00", "An apple a day."),
        ("b::chunk-0", "b", "Bananas", "Fortune", "2023-09-02T00:00:00+00:00", "Banana bread recipe."),
        ("c::chunk-0", "c", "Cars", "TechCrunch", "2023-10-01T12:00:00+00:00", "The automobile shipped on 2023-10-01."),
        ("d::chunk-0", "d", "Paris", "The Age", "2023-09-03T00:00:00+00:00", "Paris fashion week."),
    ]
    for row in rows:
        db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", row)
        db.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (row[0], serialize_float32(fake_embed(row[5]))),
        )
    db.commit()
    db.close()


def test_search_returns_semantic_neighbor(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    hits = Retriever(path, embed_query=fake_embed).search("tell me about apple fruit", k=2)
    assert hits[0]["title"] == "Apples"


def test_grep_hits_date_and_misses_synonym(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    r = Retriever(path, embed_query=fake_embed)
    assert any("2023-10-01" in h["text"] for h in r.grep("2023-10-01", k=5))
    assert any(h["title"] == "Cars" for h in r.grep("automobile", k=5))
    assert all(h["title"] != "Cars" for h in r.grep("car", k=5))


def test_grep_bad_regex_is_literal(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    assert Retriever(path, embed_query=fake_embed).grep("(unclosed", k=5) == []


def test_grep_escapes_like_wildcards(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    assert Retriever(path, embed_query=fake_embed).grep("%", k=5) == []
