# whole-wheat Search Subagent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a short blog-demo searcher: index MultiHop-RAG, run a LangGraph loop, return a ranked evidence package, score hit@k on 20 questions.

**Architecture:** One SQLite file. `Retriever.search` / `grep` read it. A 3-node graph (`seed_search` → `agent` → `tools`) fills a pool and stops on `submit_ranking` or after 4 rounds. CLI `ask` does one extra chat call. Eval scores the searcher only.

**Tech Stack:** Python 3.12, uv, FastEmbed `nomic-ai/nomic-embed-text-v1.5-Q`, sqlite-vec, LangChain, LangGraph, ChatOpenAI, MLflow, `datasets`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-whole-wheat-search-subagent-design.md`

## Global Constraints

- This is a blog post, not production. Prefer a short file a reader can hold in their head.
- uv only. Do not document pip, Poetry, or conda.
- Searcher model: `gpt-5.6-luna` (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`).
- Parent model: same, or `WHOLE_WHEAT_PARENT_MODEL`.
- Embed with FastEmbed `nomic-ai/nomic-embed-text-v1.5-Q` (768-d). Prefix chunks `search_document: ` and queries `search_query: `. Use `.embed()`, not `query_embed`.
- Max 4 agent rounds. Seed / tool `k` = 8. Rank at most 10 chunks.
- Graph state keys only: `messages`, `pool`, `ranking`, `rounds`.
- Graph must not import `sqlite3` or `sqlite_vec`.
- No live model or Hugging Face calls in tests. Pass fakes.
- No extra tools, no auto-ingest, no second graph, no BM25, no `types.py` / `config.py` / `utils.py`.
- Do not commit `data/eval.json` or `data/index.sqlite`.
- Token count is `len(text.split())`. Do not add tiktoken.
- Keep the implementation small, but keep four contract safeguards: 3 embedding attempts, parallel tool calls, atomic index replacement, and duplicate-title scoring.
- Use ChatOpenAI's built-in `max_retries=2`. Do not add a retry dependency.
- `sqlite-vec` is pre-v1. Commit `uv.lock`, and run the SQLite extension smoke check before implementation.

---

## File map

| File                           | Responsibility                      |
| ------------------------------ | ----------------------------------- |
| `pyproject.toml`               | Package, deps, script `whole-wheat` |
| `src/whole_wheat/ingest.py`    | Chunk + write `index.sqlite`        |
| `src/whole_wheat/retriever.py` | `search` / `grep`                   |
| `src/whole_wheat/tools.py`     | Three LangChain tools               |
| `src/whole_wheat/graph.py`     | The loop + evidence package         |
| `src/whole_wheat/parent.py`    | One completion                      |
| `src/whole_wheat/eval.py`      | hit@k, fetch, run                   |
| `src/whole_wheat/cli.py`       | argparse                            |
| `tests/test_*.py`              | One file per module above           |
| `data/corpus.json`             | 609 articles (git)                  |
| `data/qids.json`               | 20 `{id, query}` rows (git)         |

---
 
## Names to keep

```python
# retriever.Hit
{"chunk_id", "article_id", "title", "source", "published_at", "text"}

# graph.EvidencePackage
{"query", "ranked", "rounds", "tool_calls", "no_evidence"}
# ranked items are Hit + "reason"

# graph.SearchState
{"messages", "pool", "ranking", "rounds"}
# pool: dict[chunk_id, Hit]
# ranking: list[tuple[chunk_id, reason]] | None
```

`make_article_id` = `sha256(f"{title}\n{source}\n{published_at}").hexdigest()[:16]`
`chunk_id` = `f"{article_id}::chunk-{i}"`
`squeeze` = `" ".join(text.casefold().split())`

`run_search(question, retriever, model=None) -> dict`
`answer(question, package, model=None) -> str`
`main(argv=None) -> int`

---

### Task 1: Index (chunk, search, grep, ingest)

**Files:**
- Create: `pyproject.toml`, `.python-version`, `src/whole_wheat/__init__.py`, `src/whole_wheat/ingest.py`, `src/whole_wheat/retriever.py`, `tests/test_ingest.py`, `tests/test_retriever.py`
- Modify: `.gitignore` (append `data/index.sqlite`, `data/eval.json`, `mlruns/`)

**Interfaces:**
- Consumes: nothing
- Produces: `make_article_id`, `chunk_article`, `ingest(corpus_path, index_path, embed_fn=None) -> int`, `Retriever(db_path, embed_query=None)` with `search(query, k=8)`, `grep(pattern, k=8)`, and `duplicate_titles()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest.py
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
```

```python
# tests/test_retriever.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py tests/test_retriever.py -v`

Expected: fail (no package yet).

- [ ] **Step 3: Write the scaffold and the code**

`.python-version`:

```text
3.12
```

Append to `.gitignore`:

```gitignore
data/index.sqlite
data/eval.json
mlruns/
```

`pyproject.toml`:

```toml
[project]
name = "whole-wheat"
version = "0.1.0"
description = "Toast-1-style search subagent demo"
readme = "README.md"
requires-python = ">=3.12"
license = "Apache-2.0"
dependencies = [
    "datasets",
    "fastembed",
    "langchain",
    "langchain-openai",
    "langgraph",
    "mlflow",
    "sqlite-vec",
]

[project.scripts]
whole-wheat = "whole_wheat.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/whole_wheat"]

[dependency-groups]
dev = ["pytest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`src/whole_wheat/__init__.py` is empty.

```python
# src/whole_wheat/retriever.py
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
```

```python
# src/whole_wheat/ingest.py
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
```

Then:

```bash
uv lock
uv sync
uv run python -c "import sqlite3, sqlite_vec; db=sqlite3.connect(':memory:'); db.enable_load_extension(True); sqlite_vec.load(db); print(db.execute('select vec_version()').fetchone()[0])"
```

Expected: the smoke check prints a sqlite-vec version. On macOS, this catches a Python build that cannot load SQLite extensions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py tests/test_retriever.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore src/whole_wheat/__init__.py src/whole_wheat/ingest.py src/whole_wheat/retriever.py tests/test_ingest.py tests/test_retriever.py
git commit -m "$(cat <<'EOF'
feat: add chunking, sqlite-vec search, and ingest

EOF
)"
```

---

### Task 2: Graph and tools

**Files:**
- Create: `src/whole_wheat/tools.py`, `src/whole_wheat/graph.py`, `tests/test_graph.py`

**Interfaces:**
- Consumes: `Retriever.search` / `grep`
- Produces: `build_tools(retriever)`, `run_search(question, retriever, model=None) -> dict`. Default model is `ChatOpenAI(model=os.environ.get("WHOLE_WHEAT_SEARCHER_MODEL", "gpt-5.6-luna"))`.

Nodes: `START → seed_search → agent → tools|END`, `tools → agent|END`.

No `finalize` node. `package_from_state` fills `ranking` from pool order when it is missing.

`submit_ranking` args: `items: list[dict]` with `chunk_id` and `reason`.

Unknown ids are dropped. If none remain, use pool order. Empty pool → `ranked: []`, `no_evidence: true`.

Run every tool call from one model turn with one `ThreadPoolExecutor`. Merge results in call order.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph.py
import sqlite3
from pathlib import Path

import sqlite_vec
from langchain.messages import AIMessage
from sqlite_vec import serialize_float32

from whole_wheat.graph import run_search
from whole_wheat.retriever import Retriever


def fake_embed(text: str) -> list[float]:
    v = [0.0, 0.0]
    t = text.casefold()
    if "apple" in t or "fruit" in t:
        v[0] = 1.0
    if "paris" in t:
        v[1] = 1.0
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
        "chunk_id TEXT PRIMARY KEY, embedding float[2] distance_metric=cosine)"
    )
    rows = [
        ("seed::chunk-0", "seed", "Apples", "The Verge", "2023-09-01T00:00:00+00:00", "Apple harvest."),
        ("g::chunk-0", "g", "Paris", "The Age", "2023-09-02T00:00:00+00:00", "Paris fashion week."),
    ]
    for row in rows:
        db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", row)
        db.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (row[0], serialize_float32(fake_embed(row[5]))),
        )
    db.commit()
    db.close()


class ScriptedChat:
    def __init__(self, turns: list[AIMessage]) -> None:
        self.turns = list(turns)
        self.seen = []

    def bind_tools(self, tools: list) -> "ScriptedChat":
        return self

    def invoke(self, messages: list, **kwargs: object) -> AIMessage:
        self.seen.append(messages)
        return self.turns.pop(0)


def test_one_turn_search_and_grep_then_submit(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    retriever = Retriever(path, embed_query=fake_embed)
    model = ScriptedChat(
        [
            AIMessage(
                content="find fruit and the city",
                tool_calls=[
                    {"name": "search", "args": {"query": "fruit"}, "id": "t1"},
                    {"name": "grep", "args": {"pattern": "Paris"}, "id": "t2"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_ranking",
                        "args": {
                            "items": [
                                {"chunk_id": "g::chunk-0", "reason": "names Paris"},
                                {"chunk_id": "seed::chunk-0", "reason": "apple harvest"},
                            ]
                        },
                        "id": "t3",
                    }
                ],
            ),
        ]
    )
    package = run_search("Where is the fruit show?", retriever, model=model)
    assert [r["chunk_id"] for r in package["ranked"]] == ["g::chunk-0", "seed::chunk-0"]
    assert package["rounds"] == 2
    assert package["tool_calls"] == 3
    assert package["no_evidence"] is False
    first_prompt = " ".join(str(message.content) for message in model.seen[0])
    assert "seed::chunk-0" in first_prompt


def test_force_submit_after_four_rounds(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    ping = AIMessage(
        content="again",
        tool_calls=[{"name": "search", "args": {"query": "apple"}, "id": "x"}],
    )
    package = run_search(
        "apples",
        Retriever(path, embed_query=fake_embed),
        model=ScriptedChat([ping, ping, ping, ping]),
    )
    assert package["rounds"] == 4
    assert package["ranked"][0]["reason"] == "force: pool order"


def test_unknown_ids_drop_then_fallback(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    retriever = Retriever(path, embed_query=fake_embed)
    drop = run_search(
        "apples",
        retriever,
        model=ScriptedChat(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "submit_ranking",
                            "args": {
                                "items": [
                                    {"chunk_id": "missing::chunk-9", "reason": "nope"},
                                    {"chunk_id": "seed::chunk-0", "reason": "seed hit"},
                                ]
                            },
                            "id": "t1",
                        }
                    ],
                )
            ]
        ),
    )
    assert [r["chunk_id"] for r in drop["ranked"]] == ["seed::chunk-0"]
    fallback = run_search(
        "apples",
        retriever,
        model=ScriptedChat(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "submit_ranking",
                            "args": {"items": [{"chunk_id": "nope::chunk-0", "reason": "x"}]},
                            "id": "t1",
                        }
                    ],
                )
            ]
        ),
    )
    assert fallback["ranked"][0]["chunk_id"] == "seed::chunk-0"
    assert fallback["ranked"][0]["reason"] == "force: pool order"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph.py -v`

Expected: FAIL (`whole_wheat.graph` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# src/whole_wheat/tools.py
from __future__ import annotations

import json

from langchain.tools import tool

from whole_wheat.retriever import Retriever


def build_tools(retriever: Retriever) -> list:
    @tool
    def search(query: str, k: int = 8) -> str:
        """Semantic search. Use for meaning."""
        return json.dumps(retriever.search(query, k=k))

    @tool
    def grep(pattern: str, k: int = 8) -> str:
        """Lexical search. Use for names, dates, titles, exact tokens."""
        return json.dumps(retriever.grep(pattern, k=k))

    @tool
    def submit_ranking(items: list[dict]) -> str:
        """End the search. items is best-first [{chunk_id, reason}] (max 10)."""
        return json.dumps(items[:10])

    return [search, grep, submit_ranking]
```

```python
# src/whole_wheat/graph.py
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, TypedDict

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from whole_wheat.retriever import Retriever
from whole_wheat.tools import build_tools

MAX_ROUNDS = 4
SYSTEM = (
    "You are a retrieval subagent. Return evidence, not an answer. "
    "Write one concise sentence of what you want to find. "
    "Use search for meaning. Use grep for names, dates, titles, and exact tokens. "
    "Call submit_ranking when the pool is good enough. "
    "Remaining agent rounds: {remaining}. "
    "Current pool: {pool}"
)


class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    pool: dict[str, dict]
    ranking: list[tuple[str, str]] | None
    rounds: int


def resolve_ranking(items: list[tuple[str, str]], pool: dict[str, dict]) -> list[dict]:
    ranked = []
    seen = set()
    for chunk_id, reason in items:
        hit = pool.get(chunk_id)
        if hit and chunk_id not in seen:
            ranked.append({**hit, "reason": reason})
            seen.add(chunk_id)
        if len(ranked) == 10:
            return ranked
    if not ranked:
        for hit in pool.values():
            ranked.append({**hit, "reason": "force: pool order"})
            if len(ranked) == 10:
                break
    return ranked


def package_from_state(question: str, state: SearchState) -> dict:
    ranked = resolve_ranking(state["ranking"] or [], state["pool"])
    tool_calls = 0
    for message in state["messages"]:
        tool_calls += len(getattr(message, "tool_calls", None) or [])
    return {
        "query": question,
        "ranked": ranked,
        "rounds": state["rounds"],
        "tool_calls": tool_calls,
        "no_evidence": len(ranked) == 0,
    }


def build_graph(retriever: Retriever, model: Any):
    tools = build_tools(retriever)
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)

    def seed_search(state: SearchState) -> dict:
        question = next(
            m.content for m in state["messages"] if isinstance(m, HumanMessage)
        )
        hits = retriever.search(question, k=8)
        return {"pool": {h["chunk_id"]: h for h in hits}}

    def agent(state: SearchState) -> dict:
        rounds = state["rounds"] + 1
        response = bound.invoke(
            [
                SystemMessage(
                    content=SYSTEM.format(
                        remaining=max(0, MAX_ROUNDS - rounds),
                        pool=json.dumps(list(state["pool"].values())),
                    )
                )
            ]
            + state["messages"]
        )
        return {"messages": [response], "rounds": rounds}

    def tools_node(state: SearchState) -> dict:
        last = state["messages"][-1]
        pool = dict(state["pool"])
        ranking = state["ranking"]
        messages = []

        def run_tool(call: dict) -> tuple[dict, str]:
            tool = by_name.get(call["name"])
            try:
                payload = tool.invoke(call["args"]) if tool else f"unknown tool: {call['name']}"
            except Exception as exc:
                payload = f"bad arguments: {exc}"
            return call, payload

        with ThreadPoolExecutor(max_workers=len(last.tool_calls)) as executor:
            results = executor.map(run_tool, last.tool_calls)
        for call, payload in results:
            messages.append(
                ToolMessage(content=payload, tool_call_id=call["id"], name=call["name"])
            )
            if call["name"] == "submit_ranking":
                try:
                    ranking = [(i["chunk_id"], i.get("reason", "")) for i in json.loads(payload)]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
                continue
            if call["name"] in {"search", "grep"}:
                try:
                    for hit in json.loads(payload):
                        pool.setdefault(hit["chunk_id"], hit)
                except json.JSONDecodeError:
                    pass
        return {"messages": messages, "pool": pool, "ranking": ranking}

    def after_agent(state: SearchState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def after_tools(state: SearchState) -> Literal["agent", "__end__"]:
        if state["ranking"] is not None or state["rounds"] >= MAX_ROUNDS:
            return END
        return "agent"

    g = StateGraph(SearchState)
    g.add_node("seed_search", seed_search)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_edge(START, "seed_search")
    g.add_edge("seed_search", "agent")
    g.add_conditional_edges("agent", after_agent)
    g.add_conditional_edges("tools", after_tools)
    return g.compile()


def run_search(question: str, retriever: Retriever, model: object | None = None) -> dict:
    if model is None:
        model = ChatOpenAI(
            model=os.environ.get("WHOLE_WHEAT_SEARCHER_MODEL", "gpt-5.6-luna"),
            max_retries=2,
        )
    state = build_graph(retriever, model).invoke(
        {
            "messages": [HumanMessage(content=question)],
            "pool": {},
            "ranking": None,
            "rounds": 0,
        }
    )
    return package_from_state(question, state)
```

If LangGraph wants a string target instead of `END` in the `Literal`, return `"__end__"` and map it in `add_conditional_edges`. Do not add a fourth node.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whole_wheat/tools.py src/whole_wheat/graph.py tests/test_graph.py
git commit -m "$(cat <<'EOF'
feat: add the three-tool search loop

EOF
)"
```

---

### Task 3: Parent, eval, CLI

**Files:**
- Create: `src/whole_wheat/parent.py`, `src/whole_wheat/eval.py`, `src/whole_wheat/cli.py`, `tests/test_parent.py`, `tests/test_eval.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `run_search`, `Retriever`, `ingest`
- Produces: `answer(question, package, model=None) -> str`, `squeeze`, `hit_at_k(ranked, gold, k, duplicate_titles) -> bool`, `rows_to_eval(qids, hf_rows)`, `fetch_eval(qids_path, out_path)`, `run_eval(eval_path, retriever, model=None) -> tuple[list[dict], bool]`, `main(argv=None) -> int`.

A chunk hits gold when its squeezed title matches. For duplicate corpus titles, source and publication time must also match.

Empty package → `answer` returns `No evidence in the package.` and does not call the model.

CLI paths: `data/corpus.json`, `data/index.sqlite`, `data/qids.json`, `data/eval.json`.
Commands: `ingest`, `search [QUESTION]`, `search --qid`, `ask QUESTION`, `eval fetch`, `eval run`.
Missing index: `Index missing. Run: uv run whole-wheat ingest` on stderr, exit 1.
Missing eval: `Eval bundle missing. Run: uv run whole-wheat eval fetch` on stderr, exit 1.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parent.py
from langchain.messages import AIMessage

from whole_wheat.parent import answer

PACKAGE = {
    "query": "Who grew apples?",
    "ranked": [
        {
            "chunk_id": "a::chunk-0",
            "article_id": "a",
            "title": "Apples",
            "source": "The Verge",
            "published_at": "2023-10-01T12:00:00+00:00",
            "text": "Farmers grew apples in the valley.",
            "reason": "states the crop",
        }
    ],
    "rounds": 1,
    "tool_calls": 1,
    "no_evidence": False,
}


class Recorder:
    def __init__(self) -> None:
        self.seen = None

    def invoke(self, messages: list, **kwargs: object) -> AIMessage:
        self.seen = messages
        return AIMessage(content="demo answer")


def test_prompt_has_rule_and_snippet() -> None:
    rec = Recorder()
    assert answer("Who grew apples?", PACKAGE, model=rec) == "demo answer"
    blob = " ".join(str(m.content) for m in rec.seen)
    assert "answer only from this package" in blob.casefold()
    assert "Farmers grew apples in the valley." in blob


def test_empty_package_skips_model() -> None:
    rec = Recorder()
    empty = {"query": "??", "ranked": [], "rounds": 1, "tool_calls": 0, "no_evidence": True}
    assert answer("??", empty, model=rec) == "No evidence in the package."
    assert rec.seen is None
```

```python
# tests/test_eval.py
from whole_wheat.eval import hit_at_k, rows_to_eval, squeeze

GOLD = [
    {
        "title": "The FTX trial",
        "source": "The Verge",
        "published_at": "2023-09-28T12:00:00+00:00",
    }
]


def test_squeeze() -> None:
    assert squeeze("  The   FTX Trial ") == "the ftx trial"


def test_hit_at_k() -> None:
    ranked = [
        {"title": "Unrelated", "source": "X", "published_at": "2023-01-01T00:00:00+00:00"},
        {
            "title": "The FTX trial",
            "source": "The Verge",
            "published_at": "2023-09-28T12:00:00+00:00",
        },
    ]
    assert hit_at_k(ranked, GOLD, 1) is False
    assert hit_at_k(ranked, GOLD, 2) is True
    assert hit_at_k(ranked, GOLD, 5) is True
    wrong_source = [{**ranked[1], "source": "Fortune"}]
    assert hit_at_k(
        wrong_source, GOLD, 1, duplicate_titles={"the ftx trial"}
    ) is False


def test_rows_to_eval() -> None:
    rows = rows_to_eval(
        [{"id": "q01", "query": "Who ran FTX?"}],
        [
            {
                "query": "Who ran FTX?",
                "question_type": "inference_query",
                "answer": "Sam Bankman-Fried",
                "evidence_list": [
                    {
                        "title": "The FTX trial",
                        "source": "The Verge",
                        "published_at": "2023-09-28T12:00:00+00:00",
                    }
                ],
            }
        ],
    )
    assert rows[0]["id"] == "q01"
    assert rows[0]["gold"][0]["title"] == "The FTX trial"
```

```python
# tests/test_cli.py
from whole_wheat.cli import main


def test_missing_index(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["search", "hello"]) == 1
    assert "whole-wheat ingest" in capsys.readouterr().err


def test_missing_eval(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "index.sqlite").write_bytes(b"x")
    assert main(["eval", "run"]) == 1
    assert "eval fetch" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_parent.py tests/test_eval.py tests/test_cli.py -v`

Expected: FAIL (modules missing).

- [ ] **Step 3: Write minimal implementation**

```python
# src/whole_wheat/parent.py
from __future__ import annotations

import json
import os

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

PARENT = "Answer only from this package. If the snippets do not support an answer, say so."


def answer(question: str, package: dict, model: object | None = None) -> str:
    if package.get("no_evidence") or not package.get("ranked"):
        return "No evidence in the package."
    if model is None:
        model = ChatOpenAI(
            model=os.environ.get("WHOLE_WHEAT_PARENT_MODEL")
            or os.environ.get("WHOLE_WHEAT_SEARCHER_MODEL", "gpt-5.6-luna"),
            max_retries=2,
        )
    snippets = [row["text"] for row in package["ranked"]]
    response = model.invoke(
        [
            SystemMessage(content=PARENT),
            HumanMessage(
                content=f"Question: {question}\nSnippets: {json.dumps(snippets)}"
            ),
        ]
    )
    return str(response.content)
```

```python
# src/whole_wheat/eval.py
from __future__ import annotations

import json
from pathlib import Path

import mlflow
from datasets import load_dataset

from whole_wheat.graph import run_search
from whole_wheat.retriever import Retriever


def squeeze(text: str) -> str:
    return " ".join(text.casefold().split())


def _match(chunk: dict, gold: dict, duplicate_titles: set[str]) -> bool:
    title = squeeze(chunk["title"])
    if title != squeeze(gold["title"]):
        return False
    if title not in duplicate_titles:
        return True
    return squeeze(chunk["source"]) == squeeze(gold["source"]) and squeeze(
        chunk["published_at"]
    ) == squeeze(gold["published_at"])


def hit_at_k(
    ranked: list[dict],
    gold: list[dict],
    k: int,
    duplicate_titles: set[str] | None = None,
) -> bool:
    duplicates = duplicate_titles or set()
    return any(
        _match(chunk, gold_row, duplicates)
        for chunk in ranked[:k]
        for gold_row in gold
    )


def rows_to_eval(qids: list[dict], hf_rows: list[dict]) -> list[dict]:
    by_query = {row["query"]: row for row in hf_rows}
    out = []
    for item in qids:
        row = by_query[item["query"]]
        gold = [
            {
                "title": ev["title"],
                "source": ev["source"],
                "published_at": ev["published_at"],
            }
            for ev in row["evidence_list"]
        ]
        out.append(
            {
                "id": item["id"],
                "query": row["query"],
                "question_type": row["question_type"],
                "answer": row["answer"],
                "gold": gold,
            }
        )
    return out


def fetch_eval(qids_path: Path, out_path: Path) -> int:
    qids = json.loads(qids_path.read_text())
    dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    rows = rows_to_eval(qids, [dict(r) for r in dataset])
    out_path.write_text(json.dumps(rows, indent=2))
    return len(rows)


def run_eval(
    eval_path: Path, retriever: Retriever, model: object | None = None
) -> tuple[list[dict], bool]:
    skipped = False
    mlflow_ready = True
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.langchain.autolog()
    except Exception:
        skipped = True
        mlflow_ready = False
    duplicate_titles = retriever.duplicate_titles()
    rows = []
    for item in json.loads(eval_path.read_text()):
        tracing = False
        if mlflow_ready:
            try:
                mlflow.start_run(run_name=item["id"])
                tracing = True
            except Exception:
                skipped = True
        try:
            package = run_search(item["query"], retriever, model=model)
            record = {
                "id": item["id"],
                "hit@5": hit_at_k(
                    package["ranked"], item["gold"], 5, duplicate_titles
                ),
                "hit@10": hit_at_k(
                    package["ranked"], item["gold"], 10, duplicate_titles
                ),
                "rounds": package["rounds"],
                "tool_calls": package["tool_calls"],
            }
            rows.append(record)
            if tracing:
                try:
                    mlflow.log_param("qid", item["id"])
                    mlflow.log_metric("hit@5", int(record["hit@5"]))
                    mlflow.log_metric("hit@10", int(record["hit@10"]))
                    mlflow.log_metric("rounds", record["rounds"])
                    mlflow.log_metric("tool_calls", record["tool_calls"])
                except Exception:
                    skipped = True
        finally:
            if tracing:
                try:
                    mlflow.end_run()
                except Exception:
                    skipped = True
    return rows, skipped
```

```python
# src/whole_wheat/cli.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from whole_wheat.eval import fetch_eval, run_eval
from whole_wheat.graph import run_search
from whole_wheat.ingest import ingest
from whole_wheat.parent import answer
from whole_wheat.retriever import Retriever

CORPUS, INDEX, QIDS, EVAL = (
    Path("data/corpus.json"),
    Path("data/index.sqlite"),
    Path("data/qids.json"),
    Path("data/eval.json"),
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="whole-wheat")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    s = sub.add_parser("search")
    s.add_argument("question", nargs="?")
    s.add_argument("--qid")
    a = sub.add_parser("ask")
    a.add_argument("question")
    ev = sub.add_parser("eval")
    evs = ev.add_subparsers(dest="eval_cmd", required=True)
    evs.add_parser("fetch")
    evs.add_parser("run")
    args = p.parse_args(argv)

    if args.cmd == "ingest":
        print(f"wrote {ingest(CORPUS, INDEX)} chunks")
        return 0

    if args.cmd == "eval" and args.eval_cmd == "fetch":
        print(f"wrote {fetch_eval(QIDS, EVAL)} rows")
        return 0

    if not INDEX.exists():
        print("Index missing. Run: uv run whole-wheat ingest", file=sys.stderr)
        return 1
    retriever = Retriever(INDEX)

    if args.cmd == "search":
        question = args.question
        if args.qid:
            if not EVAL.exists():
                print("Eval bundle missing. Run: uv run whole-wheat eval fetch", file=sys.stderr)
                return 1
            question = next(
                (
                    row["query"]
                    for row in json.loads(EVAL.read_text())
                    if row["id"] == args.qid
                ),
                None,
            )
            if question is None:
                print(f"Unknown qid: {args.qid}", file=sys.stderr)
                return 1
        if not question:
            print("Pass a question or --qid.", file=sys.stderr)
            return 1
        print(json.dumps(run_search(question, retriever), indent=2))
        return 0

    if args.cmd == "ask":
        package = run_search(args.question, retriever)
        print(answer(args.question, package))
        return 0

    if not EVAL.exists():
        print("Eval bundle missing. Run: uv run whole-wheat eval fetch", file=sys.stderr)
        return 1
    rows, skipped = run_eval(EVAL, retriever)
    print("id\thit@5\thit@10\trounds\ttool_calls")
    h5 = h10 = 0
    for row in rows:
        print(f"{row['id']}\t{int(row['hit@5'])}\t{int(row['hit@10'])}\t{row['rounds']}\t{row['tool_calls']}")
        h5 += int(row["hit@5"])
        h10 += int(row["hit@10"])
    n = max(len(rows), 1)
    print(f"mean\t{h5 / n:.3f}\t{h10 / n:.3f}")
    if skipped:
        print("warning: MLflow tracing was skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whole_wheat/parent.py src/whole_wheat/eval.py src/whole_wheat/cli.py tests/test_parent.py tests/test_eval.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat: add parent, eval, and CLI

EOF
)"
```

---

### Task 4: Corpus, qids, NOTICE, README

**Files:**
- Create: `data/corpus.json`, `data/qids.json`, `NOTICE`
- Modify: `README.md`

**Interfaces:**
- Consumes: Hugging Face `yixuantt/MultiHopRAG`
- Produces: 609 articles in git; 20 qids (7 inference, 7 comparison, 6 temporal, no null). Queries have no native id, so store `{id, query}`.

- [ ] **Step 1: Vendor corpus and pick qids**

```bash
mkdir -p data
uv run python -c "$(cat <<'EOF'
import json
from pathlib import Path
from datasets import load_dataset

Path("data").mkdir(exist_ok=True)

corpus = [dict(r) for r in load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")]
assert len(corpus) == 609
Path("data/corpus.json").write_text(json.dumps(corpus))

questions = [
    dict(row)
    for row in load_dataset(
        "yixuantt/MultiHopRAG", "MultiHopRAG", split="train"
    )
]
want = {"inference_query": 7, "comparison_query": 7, "temporal_query": 6}
categories = ["technology", "sports", "business", "entertainment", "science", "health"]
selected = []
for qtype, count in want.items():
    candidates = [
        row
        for row in questions
        if row["question_type"] == qtype and row["evidence_list"]
    ]
    for index in range(count):
        category = categories[index % len(categories)]
        row = next(
            (
                candidate
                for candidate in candidates
                if any(
                    evidence.get("category") == category
                    for evidence in candidate["evidence_list"]
                )
            ),
            candidates[0],
        )
        selected.append(row)
        candidates.remove(row)
picked = [
    {"id": f"q{index:02d}", "query": row["query"]}
    for index, row in enumerate(selected, start=1)
]
assert len(picked) == 20
assert {row["question_type"] for row in selected} == set(want)
assert set(categories) <= {
    evidence.get("category")
    for row in selected
    for evidence in row["evidence_list"]
}
Path("data/qids.json").write_text(json.dumps(picked, indent=2) + "\n")
print(len(corpus), "articles,", len(picked), "qids")
EOF
)"
```

Expected: `data/corpus.json` has 609 rows. `data/qids.json` has `q01`–`q20`.

- [ ] **Step 2: Write NOTICE and README**

`NOTICE`:

```text
The MultiHop-RAG corpus is made available under the Open Data Commons
Attribution License: https://opendatacommons.org/licenses/by/1-0/

Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG
Subset: corpus (609 news articles)
Citation: Tang and Yang, "MultiHop-RAG: Benchmarking Retrieval-Augmented
Generation for Multi-Hop Queries", arXiv:2401.15391, 2024.
```

`README.md`:

~~~markdown
# whole-wheat

A small Toast-1-style search subagent for a blog post. The searcher returns a
ranked evidence package. It does not write the final answer.

## Install

```bash
uv sync
```

## Run

```bash
uv run whole-wheat ingest
uv run whole-wheat search "Who is on trial for FTX fraud?"
uv run whole-wheat ask "Who is on trial for FTX fraud?"
uv run whole-wheat eval fetch   # HF_TOKEN is optional for this public dataset
uv run whole-wheat eval run     # needs OPENAI_API_KEY
```

Default model: `gpt-5.6-luna`. Optional: `OPENAI_BASE_URL`, `WHOLE_WHEAT_PARENT_MODEL`.

```bash
uv run pytest
```

The loop is `seed_search` → `agent` → `tools`. Tools: `search`, `grep`, `submit_ranking`.
Corpus license: ODC-BY (see `NOTICE`). Answers stay out of git.
~~~

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add data/corpus.json data/qids.json NOTICE README.md
git commit -m "$(cat <<'EOF'
feat: vendor MultiHop-RAG corpus and freeze 20 qids

EOF
)"
```

Do not add `data/eval.json`.

Manual (not CI):

1. Run `uv run whole-wheat ingest`.
2. Run one real `search` query. Stop if model tool calls or MLflow traces fail.
3. Run `uv run whole-wheat eval fetch`.
4. Run one qid with `search --qid q01`. Then run all 20 with `eval run`.

The full eval can make at most 80 searcher model turns. Check API quota before step 4.

---

## Self-review

### Spec coverage

| Spec item                                             | Task |
| ----------------------------------------------------- | ---- |
| uv package                                            | 1    |
| Evidence package                                      | 2    |
| Chunk 400–800 words, no mid-paragraph split under 800 | 1    |
| Vendor corpus + 7/7/6 qids                            | 4    |
| `eval fetch` / hit@k / MLflow warn                    | 3    |
| Nomic prefixes, grep LIKE + regex                     | 1    |
| Graph does not import SQLite                          | 2    |
| Three tools, 4 rounds, force-submit, drop unknown ids | 2    |
| Parent from package only                              | 3    |
| CLI commands + missing-file messages                  | 3    |
| NOTICE, no answers in git                             | 4    |

The plan keeps the three-node graph and omits a `finalize` node. It uses small helpers for the four contract safeguards.

### Placeholder scan

No TBD / TODO / "similar to Task N".

### Type consistency

`run_search` → dict package. `answer` and `run_eval` consume that dict. CLI calls `ingest`, `run_search`, `answer`, `fetch_eval`, `run_eval`.
