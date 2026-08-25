# whole-wheat ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reader can install Python with uv, download OfficeQA Pro v2 parsed JSON, and build `data/index.sqlite` (chunks plus sqlite-vec embeddings) with `uv run whole-wheat ingest`.

**Architecture:** A small installable package. `scripts/download-corpus.sh` writes JSON under `data/parsed_corpus/jsons/`. `ingest.py` globs those files, chunks them, embeds with FastEmbed in batches of 64, and writes SQLite in one transaction. `cli.py` exposes only `ingest`. The cached embedder lives in `ingest.py` for this slice (the searcher plan can move it later).

**Tech Stack:** uv, CPython 3.12+, `huggingface_hub.snapshot_download`, FastEmbed `TextEmbedding`, sqlite-vec `vec0`, tenacity, stdlib `argparse` / `sqlite3` / `json` / `re` / `pathlib`.

**Spec:** `docs/superpowers/specs/2026-08-18-whole-wheat-search-subagent-design.md` (ingest, corpus download, index, tooling, error handling, layout). Do not implement search, grep, LangGraph, parent, or eval in this plan.

## Global Constraints

- Blog-demo code: keep it short and easy to follow. Convey the idea. Do not bulletproof.
- uv installs the runtime and dependencies. Pin `requires-python = ">=3.12"` and a `.python-version` file. Lock deps in `uv.lock`. The reader runs `uv python install`, then `uv sync`, then `uv run`.
- Do not use Apple `/usr/bin/python3`. That build cannot load `sqlite-vec`. Do not document pip, Poetry, or conda as the project workflow.
- The CLI uses standard-library `argparse`. Do not add Typer or Click.
- v1 has no automated tests. Do not add pytest. Check with commands in each task.
- Shared static config (paths, model name, prefixes) lives in `constants.py`.
- Use tenacity for retries (not a custom retry loop). Embed calls: 3 attempts (first call plus two retries), then fail.
- Do not vendor JSON in git. Do not download from `ingest`.
- `scripts/download-corpus.sh` runs `uv run python` and calls `huggingface_hub.snapshot_download` with `repo_type="dataset"`. Set `local_dir` to the repo `data/` directory (not `data/parsed_corpus/jsons/`). `allow_patterns` is `parsed_corpus/jsons/*.json`.
- Hugging Face repo is gated: `HF_TOKEN` is required. A token is not enough. The reader must request access on the dataset page, then set `HF_TOKEN`.
- Ingest glob `data/parsed_corpus/jsons/*.json` and index every match. Optional `--limit N` keeps the first N paths after **sort**. Omit `--limit` to ingest every JSON file on disk. Zero files → generic “no files” error. Do not read a cited-file list.
- Keep element types: `text`, `title`, `section_header`, `table`, `caption`, `footnote`. Drop `figure`, `page_header`, `page_footer`, `page_number`. Drop unknown types. Skip elements whose `content` is missing or only whitespace.
- Merge consecutive kept elements on the same page if the window stays at or under 800 whitespace tokens (`len(text.split())`). Page is the first `bbox` entry’s `page_id`. If `bbox` is missing or empty, that element starts a new unique page (it does not merge with the previous window, and two missing-bbox neighbors do not merge with each other).
- Do not split a `table` element. If a table is longer than 800 tokens, it is still one chunk. If a non-table kept element is longer than 800 tokens, split that element’s text into chunks of at most 800 whitespace tokens.
- Join texts inside one merged window with `"\n\n"`.
- `chunk_id` is `{document_id}::chunk-{n}` with **0-based** `n` in file order.
- `document_id`: JSON basename without `.json`. `title`: first `title` or `section_header` `content` in that file; if none, `document_id`. `source`: `combined_statement` or `govinfo_receipts` from the filename prefix. `year`: first four-digit year in the filename when present, else `"unknown"`.
- FastEmbed `nomic-ai/nomic-embed-text-v1.5-Q` (768-d, full dims). Prefix chunks with `search_document: ` at ingest. FastEmbed does not add Nomic prefixes. Construct FastEmbed on first embed through `get_embedder()` in `ingest.py` (lru_cache). Do not construct at import or CLI startup. Embed ingest chunks in batches of 64.
- Store metadata and full chunk text in `chunks`. Store only the embedding in `vec0` table `chunk_vectors`. One integer primary key in `chunks` and the same rowid in `chunk_vectors`. `distance_metric=cosine`. Do not put long HTML in a vec0 metadata column.
- Rebuild uses one SQLite connection and one explicit transaction on a **temp file** (`data/index.sqlite.tmp`). Create tables, insert all rows, commit, then `Path.replace` onto `data/index.sqlite`. Do not DROP the live index. If embed or insert fails, roll back the temp file and delete it. The previous valid index remains.
- Connect with `isolation_level=None` and issue `BEGIN` yourself. Default Python `sqlite3` isolation does not put DDL in an implicit transaction. `INSERT` would otherwise start a second `BEGIN`.
- Stream files: chunk one JSON, append to a 64-item embed batch, insert, discard. Do not hold every chunk in a list.
- `enable_load_extension` missing: catch `AttributeError` and `sqlite3.NotSupportedError`. Tell the user to use `uv run` (uv-managed CPython), not Apple Python, then exit.
- This plan’s `pyproject.toml` dependencies are ingest-only: `huggingface_hub`, `fastembed`, `sqlite-vec`, `tenacity`. Do not add LangChain, LangGraph, MLflow, or OpenAI clients here.
- Do not add `retriever.py` in this plan.

## Planning decisions (not in the spec, locked in grill)

- `--limit N` uses sorted paths, then the first N.
- Same-page merge uses the first `bbox` `page_id`.
- `get_embedder()` stays in `ingest.py`.
- First chunk id suffix is `chunk-0`.

## Validation notes (2026-08-25)

Checked Hatch, FastEmbed, sqlite-vec, huggingface_hub, tenacity, and Python `sqlite3` docs. Changes below are required. Do not implement the original live-file DROP rebuild.

## File structure

| File                           | Responsibility                                                          |
| ------------------------------ | ----------------------------------------------------------------------- |
| `pyproject.toml`               | Package name, Python floor, ingest deps, `whole-wheat` script entry.    |
| `.python-version`              | CPython pin for `uv python install`.                                    |
| `uv.lock`                      | Locked ingest deps (created by `uv lock` / `uv sync`).                  |
| `NOTICE`                       | OfficeQA Pro v2 attribution (CC-BY-SA questions; public-domain parses). |
| `.gitignore`                   | Ignore corpus, index, HF local cache under `data/`.                     |
| `README.md`                    | How to request access, download, and ingest.                            |
| `scripts/download-corpus.sh`   | Gated snapshot into `data/`.                                            |
| `src/whole_wheat/__init__.py`  | Empty package marker.                                                   |
| `src/whole_wheat/constants.py` | Paths, model id, prefixes, token window, batch size.                    |
| `src/whole_wheat/ingest.py`    | Glob, chunk, embed helper, one-transaction SQLite write.                |
| `src/whole_wheat/cli.py`       | `argparse` command `ingest` (`--limit` optional).                       |

Do not create `retriever.py`, `tools.py`, `graph.py`, `parent.py`, `eval.py`, or `data/qids.json` here.

---

### Task 1: uv package, constants, ignore rules, NOTICE, README

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `NOTICE`
- Create: `src/whole_wheat/__init__.py`
- Create: `src/whole_wheat/constants.py`
- Modify: `.gitignore` (append data ignores)
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing
- Produces: `REPO_ROOT`, `DATA_DIR`, `PARSED_JSON_DIR`, `INDEX_PATH`, `HF_DATASET_ID`, `EMBED_MODEL`, `EMBED_DIM`, `DOCUMENT_PREFIX`, `MAX_CHUNK_TOKENS`, `EMBED_BATCH_SIZE` in `constants.py`

- [ ] **Step 1: Write `.python-version`**

```text
3.12
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "whole-wheat"
version = "0.1.0"
description = "Toast-1-style search subagent (blog demo)"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastembed",
    "huggingface_hub",
    "sqlite-vec",
    "tenacity",
]

[project.scripts]
whole-wheat = "whole_wheat.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/whole_wheat"]
```

- [ ] **Step 3: Write `src/whole_wheat/__init__.py`**

Empty file.

- [ ] **Step 4: Write `src/whole_wheat/constants.py`**

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PARSED_JSON_DIR = DATA_DIR / "parsed_corpus" / "jsons"
INDEX_PATH = DATA_DIR / "index.sqlite"

HF_DATASET_ID = "databricks/officeqa-pro-v2"
HF_JSON_GLOB = "parsed_corpus/jsons/*.json"

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
EMBED_DIM = 768
DOCUMENT_PREFIX = "search_document: "
MAX_CHUNK_TOKENS = 800
EMBED_BATCH_SIZE = 64

KEEP_TYPES = frozenset(
    {"text", "title", "section_header", "table", "caption", "footnote"}
)
```

Unknown types and the spec drop list (`figure`, `page_header`, `page_footer`, `page_number`) are skipped because they are not in `KEEP_TYPES`. Do not add an unused `DROP_TYPES`.

`REPO_ROOT` is two parents up from `src/whole_wheat/constants.py` (`whole_wheat` → `src` → repo root).

- [ ] **Step 5: Append to `.gitignore`**

Add at the end of the existing file (do not remove the rest):

```gitignore
# whole-wheat local data (corpus, index, eval answers)
data/parsed_corpus/
data/index.sqlite
data/index.sqlite.tmp
data/eval.json
data/.cache/
```

- [ ] **Step 6: Write `NOTICE`**

```text
whole-wheat
Copyright holders of this software: see LICENSE (Apache-2.0).

OfficeQA Pro v2 (https://huggingface.co/datasets/databricks/officeqa-pro-v2)
- Questions and answers: CC-BY-SA 4.0. Keep this NOTICE, the dataset citation,
  and the Hugging Face access terms. Do not use answer keys to train models
  that you then score on OfficeQA.
- Source PDFs and their parses: public domain, per the dataset NOTICE.

This project does not commit data/eval.json (answers) or data/parsed_corpus/
(corpus files).
```

- [ ] **Step 7: Replace `README.md` with ingest-only setup**

Keep the Toast-1 pointer. Add commands a reader can run. Do not document pip, Poetry, conda, or Apple Python.

```markdown
# whole-wheat

Open implementation of the Toast-1 *contract* (a cheap retrieval subagent).
See https://www.mixedbread.com/blog/toast-1

This repo is a blog demo. It is not production software.

## Setup

1. Request access on https://huggingface.co/datasets/databricks/officeqa-pro-v2
2. Set `HF_TOKEN` (a token without access is not enough).
3. Install a uv-managed CPython and the package:

```bash
uv python install
uv sync
```

Do not use Apple `/usr/bin/python3`. That build cannot load `sqlite-vec`.

## Download parsed JSON (~794MB)

```bash
./scripts/download-corpus.sh
```

Files land in `data/parsed_corpus/jsons/`. Do not set `local_dir` to the jsons folder (the Hub paths already include `parsed_corpus/jsons/`).

## Index

```bash
uv run whole-wheat ingest
```

Optional demo slice (sorted paths, first N files):

```bash
uv run whole-wheat ingest --limit 20
```

`--limit` can omit gold documents for later eval. A full 1,435-file index is harder than a cited-only index.

Output: gitignored `data/index.sqlite`.

Full ingest of 1,435 files is slow (local ONNX embed). The first FastEmbed call downloads `nomic-ai/nomic-embed-text-v1.5-Q`. Use `--limit` for a smoke check.
```

- [ ] **Step 8: Add a temporary `cli.py` so the package installs**

Hatchling / uv will fail `uv sync` if `whole_wheat.cli:main` is missing. Create `src/whole_wheat/cli.py`:

```python
def main() -> None:
    raise SystemExit("ingest CLI is not ready")
```

Task 4 replaces this file.

- [ ] **Step 9: Install the Python runtime and lock deps**

Run:

```bash
uv python install
uv lock
uv sync
```

Expected: `uv.lock` exists. `uv run python -c "from whole_wheat.constants import INDEX_PATH; print(INDEX_PATH)"` prints a path that ends with `data/index.sqlite`.

Also run this load check (fail now if this CPython cannot load extensions):

```bash
uv run python - <<'PY'
import sqlite3
import sqlite_vec

conn = sqlite3.connect(":memory:")
try:
    conn.enable_load_extension(True)
except (AttributeError, sqlite3.NotSupportedError) as exc:
    raise SystemExit(f"sqlite extensions unavailable: {exc}")
sqlite_vec.load(conn)
print(conn.execute("select vec_version()").fetchone()[0])
PY
```

Expected: a sqlite-vec version string. If this fails, stop. Later ingest cannot work.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock .python-version NOTICE README.md .gitignore src/whole_wheat/__init__.py src/whole_wheat/constants.py src/whole_wheat/cli.py
git commit -m "$(cat <<'EOF'
Add uv package scaffold and ingest constants.

EOF
)"
```

---

### Task 2: Gated corpus download script

**Files:**
- Create: `scripts/download-corpus.sh`

**Interfaces:**
- Consumes: `DATA_DIR`, `HF_DATASET_ID`, `HF_JSON_GLOB` from `whole_wheat.constants`
- Produces: JSON files at `data/parsed_corpus/jsons/*.json` (gitignored)

- [ ] **Step 1: Write `scripts/download-corpus.sh`**

Use `uv run python` so the script uses the project environment (and `huggingface_hub`). Source `.env` if it exists so a local `HF_TOKEN=` line works. Hub still reads `HF_TOKEN` from the environment.

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN. Request access on https://huggingface.co/datasets/databricks/officeqa-pro-v2 first." >&2
  exit 1
fi

uv run python - <<'PY'
from huggingface_hub import snapshot_download

from whole_wheat.constants import DATA_DIR, HF_DATASET_ID, HF_JSON_GLOB

DATA_DIR.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=HF_DATASET_ID,
    repo_type="dataset",
    local_dir=str(DATA_DIR),
    allow_patterns=HF_JSON_GLOB,
)
print(f"downloaded parsed JSON under {DATA_DIR / 'parsed_corpus' / 'jsons'}")
PY
```

Do not pass `local_dir` as the jsons folder.

- [ ] **Step 2: Make the script executable**

Run:

```bash
chmod +x scripts/download-corpus.sh
```

- [ ] **Step 3: Check the no-token path**

Run:

```bash
env -u HF_TOKEN bash -c 'cd /Users/joostdetheije/Developer/personal/whole-wheat && HF_TOKEN= ./scripts/download-corpus.sh' ; true
```

Safer check when `.env` is **absent** or has no `HF_TOKEN` (if `.env` sets `HF_TOKEN`, this prefix does not win — the script sources `.env` after startup):

```bash
cd /Users/joostdetheije/Developer/personal/whole-wheat
# temporarily hide .env so this check is meaningful
if [[ -f .env ]]; then mv .env .env.bak; fi
HF_TOKEN= ./scripts/download-corpus.sh; status=$?
if [[ -f .env.bak ]]; then mv .env.bak .env; fi
exit $status
```

Expected: exit code 1 and a message that tells the user to set `HF_TOKEN` and request access. No download.

If you already have access and want the real corpus, run `./scripts/download-corpus.sh` once (~794MB). That is optional for the rest of the plan. Task 3 and Task 4 can use a tiny JSON file you write under `data/parsed_corpus/jsons/` yourself.

- [ ] **Step 4: Commit**

```bash
git add scripts/download-corpus.sh
git commit -m "$(cat <<'EOF'
Add Hugging Face snapshot script for parsed OfficeQA JSON.

EOF
)"
```

---

### Task 3: Chunk parsed JSON (no SQLite yet)

**Files:**
- Modify: `src/whole_wheat/ingest.py` (create)

**Interfaces:**
- Consumes: constants (`PARSED_JSON_DIR`, `KEEP_TYPES`, `MAX_CHUNK_TOKENS`)
- Produces:
  - `Chunk` dataclass with fields `chunk_id: str`, `document_id: str`, `title: str`, `source: str`, `year: str`, `text: str`
  - `list_json_paths(limit: int | None) -> list[Path]`
  - `chunk_file(path: Path) -> list[Chunk]`

- [ ] **Step 1: Create `src/whole_wheat/ingest.py` with chunking only**

```python
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
```

Leave `lru_cache` unused until Task 4.

The import block above is the one to keep. Do not add `lru_cache` in Task 3.

- [ ] **Step 2: Check chunking on a throwaway JSON**

Write `/tmp/ww-chunk-fixture.json` (do not commit it) and run a one-off interpreter. The fixture must live **outside** `PARSED_JSON_DIR` so `list_json_paths` is not required yet.

```bash
uv run python - <<'PY'
from pathlib import Path
import json
from whole_wheat.ingest import chunk_file, parse_source, parse_year

path = Path("/tmp/ww-chunk-fixture.json")
path.write_text(json.dumps({
    "document": {
        "elements": [
            {"type": "page_header", "content": "skip me", "bbox": [{"page_id": 0}]},
            {"type": "title", "content": "Receipts 1872", "bbox": [{"page_id": 0}]},
            {"type": "text", "content": "alpha beta", "bbox": [{"page_id": 0}]},
            {"type": "table", "content": "<table>big</table>", "bbox": [{"page_id": 0}, {"page_id": 1}]},
            {"type": "text", "content": "gamma", "bbox": [{"page_id": 1}]},
            {"type": "figure", "content": "not indexed", "bbox": [{"page_id": 1}]},
        ]
    }
}))

# rename copy with a real stem
real = Path("/tmp/combined_statement__historical__cs-1872.json")
real.write_text(path.read_text())
chunks = chunk_file(real)
assert parse_source(real.stem) == "combined_statement"
assert parse_year(real.stem) == "1872"
assert chunks[0].title == "Receipts 1872"
assert chunks[0].chunk_id == "combined_statement__historical__cs-1872::chunk-0"
# title + alpha beta + table share first-box page 0 and are small → one chunk
assert len(chunks) == 2
assert "alpha beta" in chunks[0].text and "<table>big</table>" in chunks[0].text
assert chunks[1].text == "gamma"
assert chunks[1].chunk_id.endswith("::chunk-1")
print("ok", len(chunks))
PY
```

Expected: prints `ok 2`.

Also check a long non-table split and a year-less stem:

```bash
uv run python - <<'PY'
from pathlib import Path
import json
from whole_wheat.constants import MAX_CHUNK_TOKENS
from whole_wheat.ingest import chunk_file, parse_year

assert parse_year("combined_statement__transition__appendix99__foo") == "unknown"
words = " ".join(f"w{i}" for i in range(MAX_CHUNK_TOKENS + 10))
path = Path("/tmp/govinfo_receipts__1793__SERIALSET-1.json")
path.write_text(json.dumps({
    "document": {"elements": [{"type": "text", "content": words, "bbox": [{"page_id": 0}]}]}
}))
chunks = chunk_file(path)
assert len(chunks) == 2
assert len(chunks[0].text.split()) == MAX_CHUNK_TOKENS
assert len(chunks[1].text.split()) == 10
assert chunks[0].source == "govinfo_receipts"
assert chunks[0].year == "1793"
print("ok split")
PY
```

Expected: prints `ok split`.

- [ ] **Step 3: Commit**

```bash
git add src/whole_wheat/ingest.py
git commit -m "$(cat <<'EOF'
Add OfficeQA JSON chunking for ingest.

EOF
)"
```

---

### Task 4: Embed, sqlite-vec write, ingest CLI

**Files:**
- Modify: `src/whole_wheat/ingest.py`
- Modify: `src/whole_wheat/cli.py`

**Interfaces:**
- Consumes: `list_json_paths`, `chunk_file`, `Chunk`; constants `INDEX_PATH`, `EMBED_MODEL`, `EMBED_DIM`, `DOCUMENT_PREFIX`, `EMBED_BATCH_SIZE`
- Produces:
  - `get_embedder() -> TextEmbedding` (cached, first call only)
  - `ingest(limit: int | None) -> None`
  - `cli.main()` with subcommand `ingest` and optional `--limit`

- [ ] **Step 1: Add embedder, SQLite helpers, and `ingest()` to `ingest.py`**

Keep the Task 3 functions. Add these imports at the **top** of the file (no inline imports):

```python
from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from sqlite_vec import serialize_float32
from tenacity import retry, stop_after_attempt

from whole_wheat.constants import (
    DOCUMENT_PREFIX,
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    INDEX_PATH,
    KEEP_TYPES,
    MAX_CHUNK_TOKENS,
    PARSED_JSON_DIR,
)
```

Add after the chunking functions (not inside a function):

```python
@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL)


@retry(stop=stop_after_attempt(3), reraise=True)
def embed_texts(texts: list[str]) -> list:
    return list(get_embedder().embed(texts, batch_size=EMBED_BATCH_SIZE))


def connect_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.NotSupportedError):
        raise SystemExit(
            "This Python cannot load SQLite extensions. Use uv run "
            "(uv-managed CPython), not Apple Python."
        )
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def rebuild_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            year TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            embedding float[{EMBED_DIM}] distance_metric=cosine
        )
        """
    )


def insert_batch(conn: sqlite3.Connection, batch: list[Chunk]) -> None:
    prefixed = [DOCUMENT_PREFIX + chunk.text for chunk in batch]
    vectors = embed_texts(prefixed)
    for chunk, vector in zip(batch, vectors, strict=True):
        cur = conn.execute(
            """
            INSERT INTO chunks (chunk_id, document_id, title, source, year, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.title,
                chunk.source,
                chunk.year,
                chunk.text,
            ),
        )
        rowid = cur.lastrowid
        conn.execute(
            "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
            (rowid, serialize_float32(vector.tolist())),
        )


def ingest(limit: int | None) -> None:
    paths = list_json_paths(limit)
    tmp_path = INDEX_PATH.with_name("index.sqlite.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = connect_index(tmp_path)
    written = 0
    batch: list[Chunk] = []
    ok = False
    try:
        conn.execute("BEGIN")
        rebuild_schema(conn)
        for path in paths:
            for chunk in chunk_file(path):
                batch.append(chunk)
                if len(batch) == EMBED_BATCH_SIZE:
                    insert_batch(conn, batch)
                    written += len(batch)
                    batch = []
        if batch:
            insert_batch(conn, batch)
            written += len(batch)
            batch = []
        conn.commit()
        ok = True
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
        if not ok:
            tmp_path.unlink(missing_ok=True)

    tmp_path.replace(INDEX_PATH)
    print(f"wrote {written} chunks from {len(paths)} files to {INDEX_PATH}")
```

`zip(..., strict=True)` needs Python 3.12, which matches the pin.

`get_embedder` is not called until `embed_texts` runs inside the open transaction. First FastEmbed construct happens on first ingest embed, not at import.

`isolation_level=None` puts SQLite in autocommit until `BEGIN`. Then DROP is not needed: the temp file is new. `Path.replace` swaps the live index only after commit.

If `ROLLBACK` runs after SQLite already aborted the transaction, the extra `ROLLBACK` is still valid SQL (no-op or harmless). `tmp_path.unlink` runs so a half temp file does not remain.

- [ ] **Step 2: Replace `src/whole_wheat/cli.py`**

```python
from __future__ import annotations

import argparse

from whole_wheat.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(prog="whole-wheat")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest", help="build data/index.sqlite")
    ingest_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="index only the first N sorted JSON paths (demo)",
    )
    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.limit)
```

- [ ] **Step 3: Check the zero-file error**

If `data/parsed_corpus/jsons/` is empty or missing:

```bash
uv run whole-wheat ingest
```

Expected: process exits with message `no files` (generic). No stack trace required. Do not auto-download.

- [ ] **Step 4: Check a one-file ingest**

Create a tiny file on the glob path (gitignored directory). Then ingest with `--limit 1`.

```bash
mkdir -p data/parsed_corpus/jsons
cat > data/parsed_corpus/jsons/combined_statement__historical__cs-1872.json <<'EOF'
{"document":{"elements":[
  {"type":"title","content":"Combined Statement 1872","bbox":[{"page_id":0}]},
  {"type":"text","content":"Receipts were recorded.","bbox":[{"page_id":0}]}
]}}
EOF
uv run whole-wheat ingest --limit 1
uv run python - <<'PY'
import sqlite3
import sqlite_vec
from whole_wheat.constants import EMBED_DIM, INDEX_PATH

conn = sqlite3.connect(INDEX_PATH)
conn.enable_load_extension(True)
sqlite_vec.load(conn)
n, = conn.execute("select count(*) from chunks").fetchone()
assert n == 1
row = conn.execute("select chunk_id, document_id, title, source, year, text from chunks").fetchone()
assert row[0] == "combined_statement__historical__cs-1872::chunk-0"
assert row[1] == "combined_statement__historical__cs-1872"
assert row[2] == "Combined Statement 1872"
assert row[3] == "combined_statement"
assert row[4] == "1872"
v, = conn.execute("select vec_length(embedding) from chunk_vectors").fetchone()
assert v == EMBED_DIM
print("ok index")
PY
```

Expected: ingest prints a “wrote 1 chunks…” line. The check prints `ok index`.

First run may download the FastEmbed ONNX model into the Hugging Face cache. That is expected.

- [ ] **Step 5: Commit**

```bash
git add src/whole_wheat/ingest.py src/whole_wheat/cli.py
git commit -m "$(cat <<'EOF'
Write sqlite-vec index from parsed JSON in one transaction.

EOF
)"
```

---

## Self-review

**Spec coverage (ingest slice):**
- Tooling (uv, Python 3.12, argparse, no pip/Poetry/conda) → Task 1
- Download script, `local_dir=data/`, `allow_patterns`, gated token copy → Task 2
- Glob, sorted `--limit`, generic no-files error, no cited-file list → Task 3–4
- Element keep/drop, 800-token merge, first-box page, table no-split, non-table split → Task 3
- Chunk metadata (`document_id`, `title`, `source`, `year`, `chunk_id`) → Task 3
- FastEmbed model, prefixes, batch 64, lazy construct, tenacity 3 attempts → Task 4
- `chunks` + `chunk_vectors` cosine 768-d, shared rowid, one transaction on a temp file, `Path.replace` onto the live index → Task 4
- `enable_load_extension` message, no download from ingest → Task 4
- Layout paths, NOTICE, gitignore corpus/index → Task 1
- Out of scope left out: searcher graph, parent, eval, pytest, langchain

**Placeholder scan:** none. Commands and function bodies are complete.

**Type consistency:** `list_json_paths(limit: int | None) -> list[Path]`, `chunk_file(path: Path) -> list[Chunk]`, `connect_index(path: Path) -> sqlite3.Connection`, `ingest(limit: int | None) -> None`, `get_embedder() -> TextEmbedding`. CLI passes `args.limit` into `ingest`.
