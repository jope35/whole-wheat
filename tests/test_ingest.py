import json
from pathlib import Path

import pytest

from whole_wheat.ingest import chunk_article, ingest, make_article_id
from whole_wheat.retriever import Retriever


def test_article_id_is_stable() -> None:
    a = make_article_id("Hello", "The Verge", "2023-10-01T12:00:00+00:00")
    assert a == make_article_id("Hello", "The Verge", "2023-10-01T12:00:00+00:00")
    assert len(a) == 16


def test_keeps_short_paragraphs_together() -> None:
    article = {
        "title": "T",
        "source": "S",
        "published_at": "2023-10-01T12:00:00+00:00",
        "body": ("Alpha " * 200) + "\n\n" + ("Bravo " * 200),
    }
    chunks = chunk_article(article)
    assert len(chunks) == 1
    assert "Alpha" in chunks[0]["text"] and "Bravo" in chunks[0]["text"]


def test_starts_new_chunk_after_800_words() -> None:
    article = {
        "title": "T",
        "source": "S",
        "published_at": "2023-10-01T12:00:00+00:00",
        "body": ("Apple " * 500) + "\n\n" + ("Banana " * 400),
    }
    chunks = chunk_article(article)
    assert len(chunks) == 2
    assert chunks[1]["chunk_id"].endswith("::chunk-1")


def test_splits_a_paragraph_over_800_words() -> None:
    article = {
        "title": "T",
        "source": "S",
        "published_at": "2023-10-01T12:00:00+00:00",
        "body": "Apple " * 900,
    }
    chunks = chunk_article(article)
    assert len(chunks) == 2
    assert all(len(chunk["text"].split()) <= 800 for chunk in chunks)


def fake_embed_many(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        v = [0.0, 0.0]
        if "alpha" in text.casefold():
            v[0] = 1.0
        if "beta" in text.casefold():
            v[1] = 1.0
        out.append(v)
    return out


def fail_embed(texts: list[str]) -> list[list[float]]:
    raise RuntimeError("embedding failed")


def test_ingest_is_searchable(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    index = tmp_path / "index.sqlite"
    corpus.write_text(
        json.dumps(
            [
                {
                    "title": "Alpha news",
                    "source": "The Verge",
                    "published_at": "2023-10-01T12:00:00+00:00",
                    "body": "Alpha event in the valley.",
                },
                {
                    "title": "Beta news",
                    "source": "Fortune",
                    "published_at": "2023-10-02T12:00:00+00:00",
                    "body": "Beta markets moved.",
                },
            ]
        )
    )
    assert ingest(corpus, index, embed_fn=fake_embed_many) == ingest(
        corpus, index, embed_fn=fake_embed_many
    )
    with pytest.raises(RuntimeError):
        ingest(corpus, index, embed_fn=fail_embed)
    retriever = Retriever(index, embed_query=lambda q: fake_embed_many([q])[0])
    assert retriever.search("alpha", k=1)[0]["title"] == "Alpha news"
