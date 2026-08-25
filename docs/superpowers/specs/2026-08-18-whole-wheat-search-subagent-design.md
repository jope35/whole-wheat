# whole-wheat: a Toast-1-style search subagent

Date: 2026-08-18  
Updated: 2026-08-25  
Status: implementation decisions locked  
Repo: `whole-wheat` (blog-sized open implementation, not production)

This spec is the approved design. It is a teaching clone of Mixedbread Toast 1’s *contract* (a cheap retrieval subagent that returns ranked evidence). It is not a clone of the Toast-1 weights, Mixedbread Search, or `toast-harness`.

Inspired by: [Introducing Toast 1](https://www.mixedbread.com/blog/toast-1), [toast-harness](https://github.com/mixedbread-ai/toast-harness), [SID-1](https://www.sid.ai/research/sid-1), [Chroma Context-1](https://www.trychroma.com/research/context-1).

## Goal

Ship a small Python repo a reader can run after clone:

- Download OfficeQA Pro v2 parsed JSON (not PDFs) and index it into SQLite + `sqlite-vec`.
- Run a LangGraph searcher that returns a ranked **evidence package** (snippets, not an answer).
- Optionally run a 30-line **parent** that answers only from that package.
- Score the searcher on 20 frozen questions with document-level hit@k, with MLflow traces.
- Success: a reader can follow the graph in one sitting, see `search` vs `grep` vs `submit_ranking`, and understand why a frontier model should not own the search loop.

## Tooling

[uv](https://docs.astral.sh/uv/) installs the Python runtime and manages the project dependencies. Pin `requires-python = ">=3.12"` and a `.python-version` file. Lock deps in `uv.lock`. The reader runs `uv python install` so that uv installs a CPython that can load SQLite extensions. Then the reader runs `uv sync` and `uv run`. Do not use Apple `/usr/bin/python3` with this project. That build cannot load `sqlite-vec`. Do not document pip, Poetry, or conda as the project workflow.

The CLI uses standard-library `argparse`. Do not add Typer or Click.

## Non-goals

- Training a search model (no SFT, no RL).
- Matching Toast quality, latency, or cost.
- PDF or OCR pipelines. Ingest reads Databricks parsed JSON only. Skip `figure` elements (no vision).
- Plugin frameworks, hosted APIs, or a second LangGraph for the parent.
- Null-query evaluation, prune/read tools, official `reward.py` answer scoring, or a public OfficeQA leaderboard.
- Live model calls in CI.
- Automated tests in v1.

## Contract

The searcher is a retrieval subagent.

- **In:** a natural-language question.
- **Out:** a ranked evidence package. No final answer.
- The parent (CLI `ask` only) may write an answer. It has no search tools. The CLI calls searcher then parent. The parent does not call the searcher as a tool.
- `search` and `ask` take question text only. Frozen qids are for `eval fetch` and `eval run` only.

### Evidence package

```json
{
  "query": "…",
  "ranked": [
    {
      "chunk_id": "document-id::chunk-3",
      "document_id": "combined_statement__historical__cs-1872",
      "title": "Combined Statement of Receipts, Outlays, and Balances, 1872",
      "source": "combined_statement",
      "year": "1872",
      "text": "table or text snippet",
      "reason": "one line why this chunk is in the ranking"
    }
  ],
  "rounds": 2,
  "tool_calls": 4,
  "no_evidence": false
}
```

`ranked` is best-first and has at most 10 items. `no_evidence` is true only when the force-submit path has an empty pool.

## Corpus and eval set

**Corpus:** [databricks/officeqa-pro-v2](https://huggingface.co/datasets/databricks/officeqa-pro-v2) parsed JSON (`parsed_corpus/jsons/*.json`). The full set is 1,435 U.S. receipts-and-expenditures documents (1793–2024). About 249 of those files appear in the 90-question `source_files` column. Do not download the 13.3GB PDF tree. The Hugging Face repo is gated: `HF_TOKEN` is required.

Do not vendor the JSON in git (about 794MB for the full parse set). `scripts/download-corpus.sh` runs `uv run python` and calls `huggingface_hub.snapshot_download` with `repo_type="dataset"`. Set `local_dir` to the repo `data/` directory (not `data/parsed_corpus/jsons/`). Hub paths keep the prefix `parsed_corpus/jsons/…`, so files land at `data/parsed_corpus/jsons/`. If `local_dir` is the jsons folder, the tree nests twice. Download the full parsed JSON tree (`allow_patterns` is `parsed_corpus/jsons/*.json`). Do not download from `ingest`.

A token is not enough. The reader must request access on the Hugging Face dataset page, then set `HF_TOKEN`.

Each JSON has the shape `{"document": {"elements": [...], "pages": [...]}}`. Each element has `type`, `content`, `confidence`, and `bbox` with 0-indexed `page_id`.

**Ingest inputs:** glob `data/parsed_corpus/jsons/*.json` and index every file that matches. Optional CLI `--limit N` keeps the first N paths from that glob (demo only). If you omit `--limit`, ingest every JSON file on disk. `--limit` can omit gold documents for eval. If the glob finds zero files, fail with a generic “no files” error. Do not read a cited-file list.

**Chunking:** walk elements in order. Keep `text`, `title`, `section_header`, `table`, `caption`, and `footnote`. Drop `figure`, `page_header`, `page_footer`, and `page_number`. Use `content` as the chunk text (tables stay as the parse already stored them, usually HTML. Do not re-OCR). Merge consecutive kept elements on the same page with a greedy rule: add the next same-page element if the window stays at or under 800 whitespace tokens (`len(text.split())`). Page-end windows can be smaller than 400 tokens. Do not add tiktoken.

Do not split a `table` element. If a table is longer than 800 tokens, it is still one chunk. FastEmbed truncates Nomic inputs at 8192 tokens. A huge table can lose its tail at embed time. If a non-table kept element is longer than 800 tokens, split that element’s text into chunks of at most 800 whitespace tokens.

Each chunk stores `chunk_id`, `document_id`, `title`, `source`, `year`, `text`.

- `document_id`: JSON basename without `.json` (same stem as `source_files`, for example `combined_statement__historical__cs-1872`).
- `title`: the first `title` or `section_header` `content` in that file. If none, the `document_id`.
- `source`: `combined_statement` or `govinfo_receipts` from the filename prefix.
- `year`: a four-digit year parsed from the filename when present. Otherwise `"unknown"` (transition `appendix{YY}` files have no four-digit year in the name).

**Eval questions:** 20 of the 90 OfficeQA Pro v2 rows, frozen in `data/qids.json` as `{id, query}` with `id` equal to dataset `uid` (for example `qid_7`) and `query` copied from the CSV `question` column. Stratify by how many source documents the gold list cites: 7 questions with 1–3 sources, 7 with 4–8, 6 with 9 or more. No empty gold list. Prefer rows that do not need figures or live web search (about 7% of the 90 need vision. About 10% need web). Spread across document families (`combined_statement__historical`, `combined_statement__modern`, `combined_statement__transition`, `govinfo_receipts`) so the slice is not one era. Choose concrete uids once with a throwaway sampling script. Commit `data/qids.json` only. Do not keep the sampling script in the repo. Do not change the frozen slice without a spec update.

**Answers:** not in git. `whole-wheat eval fetch` uses `HF_TOKEN` to load those 20 rows from `officeqa_pro_v2.csv` into gitignored `data/eval.json` (`id`, `query`, `answer`, `source_files`, `source_docs`). Do not call Databricks `reward.py` in `eval run`.

**Gold for retrieval:** document ids from `source_files` (parse a JSON list if needed. Strip a trailing `.txt` / `.json` / `.pdf` if present). **hit@k** is 1 when at least one gold `document_id` appears among the top-k ranked chunks, else 0. Average over scored questions. This is not gold-document recall and not chunk nDCG. We do not score short answers on the searcher. The parent is demo-only and is not part of `eval run`. A full 1,435-file index is harder than a 249-file cited-only index. Say that in the README.

## Architecture

Two programs, one index.

```text
question
    ├─► seed vector search
    ├─► LangGraph (search | grep | submit_ranking)  ──► evidence package
    └─► parent Responses API(question + package)    ──► answer (ask only)
```

- **Index:** [FastEmbed](https://qdrant.github.io/fastembed/) `TextEmbedding` with `nomic-ai/nomic-embed-text-v1.5-Q` (768-d, full dims, no Matryoshka truncate) stored in SQLite via `sqlite-vec`. FastEmbed does not add Nomic prefixes. Prefix chunks with `search_document: ` at ingest and queries with `search_query: ` at search time.

  Store metadata and full chunk text in a normal `chunks` table. Store only the embedding in a `vec0` table named `chunk_vectors`. Use one integer primary key in `chunks` and the same rowid in `chunk_vectors`. Do not put long HTML in a vec0 metadata column. KNN queries must include `k = ?` (do not rely on `LIMIT` alone). Use `distance_metric=cosine`.

  Rebuild is one SQLite connection and one transaction. Drop and recreate tables, insert all rows, then commit. If embed or insert fails, roll back. The previous valid index remains. No half index.

  Construct FastEmbed on the first ingest embed or `search()` call through a small cached helper. Do not construct it at import or at CLI startup. Embed ingest chunks in batches of 64. Write those rows in the open transaction. Commit once at the end.

  `grep`: if `re.compile(pattern)` succeeds, scan chunk `text` with that regex in Python and return up to `k` hits. If compile fails, return no hits. Do not use SQL `LIKE`. Do not raise. One SQLite connection per retriever call. `check_same_thread=False` is not required if tools run in sequence.

- **LLM:** `langchain_openai.ChatOpenAI` against an OpenAI-compatible API (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`). The client always uses `use_responses_api=True`. The searcher and the parent send every LLM call to `/v1/responses`. Do not use `/v1/chat/completions`. The searcher **must use a reasoning model** (effort not `none`). Do not hard-code a vendor slug: read `WHOLE_WHEAT_SEARCHER_MODEL` (default example `gpt-5.6-luna`). Default `reasoning_effort="low"` in `constants.py`. The `base_url` must accept reasoning plus tools on the Responses API. Parent uses the same client unless `WHOLE_WHEAT_PARENT_MODEL` is set. Retry LLM and embed calls with tenacity (3 attempts: the first call plus two retries), then fail.

- **Retriever protocol:** `search(query, k) -> list[Hit]` and `grep(pattern, k) -> list[Hit]`. `tools.py` calls those functions. The graph never imports SQLite.

- **Context size:** eight calls times eight hits times long snippets can overflow the model context. v1 does not cap pool size or truncate tool text. A four-round run can fail at the model boundary. That is an accepted demo failure.

## Components

| Module         | Responsibility                                                                                                                               |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `constants.py` | Paths, model names, prefixes, prompts, round and `k` limits.                                                                                 |
| `ingest.py`    | Glob JSON, chunk, embed in batches of 64, write `data/index.sqlite` in one transaction. Does not run implicitly from `search`.               |
| `retriever.py` | `search` / `grep` over SQLite. `Hit = {chunk_id, document_id, title, source, year, text}`. Cached FastEmbed helper.                          |
| `tools.py`     | LangChain tools `search`, `grep`, `submit_ranking`. `submit_ranking` writes the package onto graph state and ends the loop.                  |
| `graph.py`     | Explicit `StateGraph`: `seed_search` → `agent` → custom tools node → `agent` …                                                               |
| `parent.py`    | One Responses API call on the same OpenAI-compatible client. Answer only from the package. If `no_evidence` or the package is empty, say so. |
| `eval.py`      | `fetch` and `run` (searcher only).                                                                                                           |
| `cli.py`       | `argparse` commands: `ingest` (`--limit` optional), `search`, `ask`, `eval fetch`, `eval run`.                                               |

Graph state is only: `messages`, `pool`, `ranking`, `rounds`.

Pool items keep a first-seen retrieval source (`seed`, `search`, or `grep`) for fallback reasons. First seen wins for hit data, order, and source.

## Data flow

1. `download-corpus` **(once).** Gated Hugging Face snapshot of the full parsed JSON tree into `data/parsed_corpus/jsons/`.
2. `ingest` **(once).** Glob JSON → chunks → embeddings (batches of 64) → `data/index.sqlite` in one transaction. Optional `--limit N` for demos.
3. `search`**.**
   a. `seed_search` runs the raw question through `retriever.search` and puts hits in `pool` (the original phrasing is always represented). The source is `seed`.
   b. The agent sees the question, seed hits, and remaining rounds. System prompt: write one concise sentence of what you want to find. Use `search` for meaning. Use `grep` for filenames, years, agency names, dollar amounts, and exact tokens.
   c. One model turn can emit several `search` and `grep` calls. Run them in **sequence** on one process (v1). Merge into `pool` keyed by `chunk_id`, first seen wins. Cap at 8 tool calls per turn. Drop the rest with a short tool error string. Default `k` is 8 when omitted. If `k` is greater than 8, return a short tool error and do not retrieve.
   d. The agent calls `submit_ranking` with ordered `chunk_id`s and a one-line reason each. A custom tools node inspects the tool calls first. If a turn includes `submit_ranking` and other tools, run `submit_ranking` only and skip the rest.
   e. Stop when a ranking is submitted, or after **4** agent rounds (the final ranking turn included). If round 4 emits only `search` or `grep` calls, run those calls, merge hits, then force-submit from pool order. Do not call the model a fifth time. If the last AI message has no tool calls and `ranking` is empty, a graph node builds the package from current `pool` order. Do not call the model again for that fallback.
   f. Keep the first 10 valid `chunk_id`s. Drop unknown ids. Drop later duplicates of the same `chunk_id`. An empty list, or a list with no remaining valid ids, falls back to the first 10 items in pool order. Fallback `reason` is a short line from the first retrieval source only. If the pool is empty, return `ranked: []` and `no_evidence: true`.
4. `ask`**.** Run `search`, then `parent.py` with `{question, ranked}`.
5. `eval fetch`**.** Write gitignored `data/eval.json` for the 20 qids.
6. `eval run`**.** For each eval question, run the searcher, log the MLflow trace, compute document **hit@5** and **hit@10**. Print a table. If MLflow is unset or down, still print the table and warn that tracing was skipped. If a question fails after three LLM attempts, count that row as hit@5=0 and hit@10=0, then continue.

## Error handling

- `enable_load_extension` missing: tell the user to use `uv run` (uv-managed CPython), not Apple Python, then exit.
- Zero JSON files from the ingest glob: fail with a generic “no files” error, then exit.
- Missing `data/index.sqlite`: tell the user to run `ingest` and exit. No auto-ingest.
- Missing `data/eval.json`: tell the user to run `eval fetch`.
- LLM or embedding call failures: tenacity, 3 attempts, then fail that query (ingest: fail the run and roll back the index transaction).
- Bad tool arguments: return a short error string in the tool message. Do not crash the graph.
- No custom exception types, no queues.

## Observability

- `mlflow.langchain.autolog()` (or the LangGraph tracer) around `search` / `eval run`.
- Each eval item is one MLflow run (or nested span) with query id, hit@5, hit@10, rounds, tool_call count.
- Default tracking URI: local `./mlruns` (`file:./mlruns` if you set a URI). `mlflow.langchain.autolog()` traces LangGraph. If MLflow is unset or down, still print the eval table.

## Testing

v1 has no automated tests and no CI model calls.

Manual check: `uv run whole-wheat eval run` against a real index.

## Defaults and configuration

| Knob              | Default                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Searcher model    | `WHOLE_WHEAT_SEARCHER_MODEL` (default `gpt-5.6-luna`), reasoning on (`reasoning_effort="low"`), always `use_responses_api=True` |
| Parent model      | same, or `WHOLE_WHEAT_PARENT_MODEL`. Always `use_responses_api=True`                                                            |
| Embedding model   | FastEmbed `nomic-ai/nomic-embed-text-v1.5-Q` (768-d), cached on first use                                                       |
| Embed prefixes    | `search_document: ` (chunks), `search_query: ` (queries)                                                                        |
| Embed batch size  | 64                                                                                                                              |
| Ingest `--limit`  | omitted: all JSON files on disk                                                                                                 |
| Max agent rounds  | 4                                                                                                                               |
| Seed / tool `k`   | 8 (omit → 8. `k > 8` → tool error)                                                                                              |
| Ranking size      | up to 10 chunks                                                                                                                 |
| Tool calls / turn | up to 8, in sequence                                                                                                            |

## Layout

```text
whole-wheat/
  pyproject.toml
  uv.lock                     # locked deps; uv is the runtime + package manager
  .python-version             # CPython pin; `uv python install` uses this
  NOTICE                      # CC-BY-SA + public-domain parse attribution for OfficeQA Pro v2
  scripts/
    download-corpus.sh        # gated HF snapshot of the full parsed JSON tree
  data/
    qids.json                 # 20 frozen uids + query text (in git)
    parsed_corpus/jsons/      # gitignored parsed JSON
    index.sqlite              # gitignored
    eval.json                 # gitignored (questions, answers, gold files)
  src/whole_wheat/
    constants.py
    ingest.py
    retriever.py
    tools.py
    graph.py
    parent.py
    eval.py
    cli.py
  docs/superpowers/specs/     # this file
```

## License notes

- Code: Apache-2.0 (existing repo license).
- OfficeQA Pro v2 questions and answers: CC-BY-SA 4.0 — keep `NOTICE`, the dataset citation, and the Hugging Face access terms (do not use answer keys to train models that you then score on OfficeQA).
- Source PDFs and their parses: public domain, per the dataset `NOTICE`.
- Do not commit `data/eval.json` (answers) or `data/parsed_corpus/` (corpus files).

## Out of scope for v1 (explicit)

`prune_context`, `read_document`, `filter_chunks`, hybrid BM25+dense, auto-ingest, a parent-as-tool graph, null queries, automated tests, context caps, a cited-only download, qid flags on `search`/`ask`, and any claim of Toast-1 parity.
