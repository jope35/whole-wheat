# whole-wheat: a Toast-1-style search subagent

Date: 2026-08-18  
Status: draft for review  
Repo: `whole-wheat` (blog-sized open implementation, not production)

This spec is the approved design. It is a teaching clone of Mixedbread Toast 1’s *contract* (a cheap retrieval subagent that returns ranked evidence). It is not a clone of the Toast-1 weights, Mixedbread Search, or `toast-harness`.

Inspired by: [Introducing Toast 1](https://www.mixedbread.com/blog/toast-1), [toast-harness](https://github.com/mixedbread-ai/toast-harness), [SID-1](https://www.sid.ai/research/sid-1), [Chroma Context-1](https://www.trychroma.com/research/context-1).

## Goal

Ship a small Python repo a reader can run after clone:

- Index 609 MultiHop-RAG news articles into SQLite + `sqlite-vec`.
- Run a LangGraph searcher that returns a ranked **evidence package** (snippets, not an answer).
- Optionally run a 30-line **parent** that answers only from that package.
- Score the searcher on 20 frozen questions with article-level hit@k, with MLflow traces.
- Success: a reader can follow the graph in one sitting, see `search` vs `grep` vs `submit_ranking`, and understand why a frontier model should not own the search loop.

## Tooling

[uv](https://docs.astral.sh/uv/) manages the Python runtime and the project dependencies. Pin the interpreter with `requires-python` in `pyproject.toml` (and `.python-version` if useful). Lock deps in `uv.lock`. Readers install and run with `uv sync` and `uv run`. Do not document pip, Poetry, or conda as the project workflow.

## Non-goals

- Training a search model (no SFT, no RL).
- Matching Toast quality, latency, or cost.
- PDF / OCR pipelines (OfficeQA Pro V2 was considered and rejected).
- Plugin frameworks, hosted APIs, or a second LangGraph for the parent.
- Null-query evaluation, prune/read tools, or a public OfficeQA leaderboard.
- Live model calls in CI.



## Contract

The searcher is a retrieval subagent.

- **In:** a natural-language question.
- **Out:** a ranked evidence package. No final answer.
- The parent (CLI `ask` only) may write an answer. It has no search tools. The CLI calls searcher then parent; the parent does not call the searcher as a tool.



### Evidence package

```json
{
  "query": "…",
  "ranked": [
    {
      "chunk_id": "article-id::chunk-3",
      "article_id": "article-id",
      "title": "…",
      "source": "The Verge",
      "published_at": "2023-10-01T12:00:00",
      "text": "paragraph snippet",
      "reason": "one line why this chunk is in the ranking"
    }
  ],
  "rounds": 2,
  "tool_calls": 4,
  "no_evidence": false
}
```

`ranked` is best-first. `no_evidence` is true only when the force-submit path has an empty pool.

## Corpus and eval set

**Corpus:** [yixuantt/MultiHopRAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) `corpus` split — 609 news articles, already text, ODC-BY. Vendor `data/corpus.json` in git so ingest works offline after clone. Average article length is about 2,000 tokens.

**Chunking:** split each article into paragraph-sized windows of about 400–800 tokens. Do not split mid-paragraph if the paragraph is under 800 tokens. Each chunk stores `chunk_id`, `article_id`, `title`, `source`, `published_at`, `text`.

**Eval questions:** 20 MultiHop-RAG queries, frozen in `data/qids.json`. Stratify as 7 `inference_query`, 7 `comparison_query`, 6 `temporal_query`. No `null_query`. Spread across news categories (technology, sports, business, entertainment, science, health) so the slice is not one beat. Concrete ids are chosen once during implementation with a small sampling script, then committed and not changed without a spec update.

**Answers:** not in git. `whole-wheat eval fetch` uses `HF_TOKEN` to load those 20 rows from Hugging Face into gitignored `data/eval.json` (`id`, `query`, `question_type`, `answer`, gold article titles from `evidence_list`). The CLI can take a `qid` and load the question text from that file.

**Gold for retrieval:** article `title` values from each query’s `evidence_list` (the dataset has no separate article id). A ranked chunk is a hit if its `title` matches a gold title after Unicode casefold and whitespace squeeze. If two corpus rows share a title, also require `source` and `published_at` to match the evidence row. Ingest may still assign an internal `article_id` (hash of title + source + published_at) for chunk keys. We do not score short answers on the searcher. The parent is demo-only and is not part of `eval run`.

## Architecture

Two programs, one index.

```text
question
    ├─► seed vector search
    ├─► LangGraph (search | grep | submit_ranking)  ──► evidence package
    └─► parent completion(question + package)       ──► answer (ask only)
```

- **Index:** [FastEmbed](https://qdrant.github.io/fastembed/) `TextEmbedding` with `nomic-ai/nomic-embed-text-v1.5-Q` (768-d, full dims, no Matryoshka truncate) stored in SQLite via `sqlite-vec`. Prefix chunks with `search_document: ` at ingest and queries with `search_query: ` at search time (Nomic task instructions). `grep` is SQL `LIKE` over the same text rows. If the pattern is a valid regex, `grep` may also apply it in Python on the SQL candidate set; if the regex is invalid, treat the pattern as a literal `LIKE` substring. Do not raise.

- **LLM**: langchain-openai.ChatOpenAI against the OpenAI Chat Completions API, or any compatible base_url. Default searcher model: `gpt-5.6-luna`. Parent may use the same model or WHOLE_WHEAT_PARENT_MODEL.

- **Retriever protocol:** `search(query, k) -> list[Hit]` and `grep(pattern, k) -> list[Hit]`. The graph never imports SQLite.



## Components


| Module         | Responsibility                                                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ingest.py`    | Read `data/corpus.json`, chunk, embed, write `data/index.sqlite`. Idempotent: rebuilds the index cleanly. Does not run implicitly from `search`. |
| `retriever.py` | `search` / `grep` over SQLite. `Hit = {chunk_id, article_id, title, source, published_at, text}`.                                                |
| `tools.py`     | LangChain tools `search`, `grep`, `submit_ranking`. `submit_ranking` writes the package onto graph state and ends the loop.                      |
| `graph.py`     | Explicit `StateGraph`: `seed_search` → `agent` → `tools` → `agent` …                                                                             |
| `parent.py`    | One Chat Completions call. Answer only from the package; if `no_evidence` or the package is empty, say so.                                       |
| `eval.py`      | `fetch` and `run` (searcher only).                                                                                                               |
| `cli.py`       | Commands: `ingest`, `search`, `ask`, `eval fetch`, `eval run`.                                                                                   |


Graph state is only: `messages`, `pool`, `ranking`, `rounds`.

## Data flow

1. `ingest` **(once).** Articles → chunks → embeddings → `data/index.sqlite`. Fail the whole run if embed or SQLite write fails. No half index.
2. `search`**.**
  a. `seed_search` runs the raw question through `retriever.search` and puts hits in `pool` (the original phrasing is always represented).  
   b. The agent sees the question, seed hits, and remaining rounds. System prompt: write one concise sentence of what you want to find; use `search` for meaning; use `grep` for names, dates, titles, and exact tokens.  
   c. One model turn may emit several `search` and `grep` calls. Run those calls in parallel, then merge into `pool` keyed by `chunk_id`.  
   d. The agent calls `submit_ranking` with ordered `chunk_id`s and a one-line reason each.  
   e. Stop when a ranking is submitted, or after **4** agent rounds (the final ranking turn included). If the model stops without ranking, force `submit_ranking` from current `pool` order (seed hits if that is all there is).  
   f. Unknown `chunk_id`s in a ranking are dropped. If none remain, fall back to pool order. If the pool is empty, return `ranked: []` and `no_evidence: true`.
3. `ask`**.** Run `search`, then `parent.py` with `{question, ranked}`.
4. `eval fetch`**.** Write gitignored `data/eval.json` for the 20 qids.
5. `eval run`**.** For each eval question, run the searcher, log the MLflow trace, compute article **hit@5** and **hit@10**. Print a table. If MLflow is unset or down, still print the table and warn that tracing was skipped.



## Error handling

- Missing `data/index.sqlite`: tell the user to run `ingest` and exit. No auto-ingest.
- Missing `data/eval.json`: tell the user to run `eval fetch`.
- LLM or embedding call failures: retry twice, then fail that query (ingest: fail the run).
- Bad tool arguments: return a short error string in the tool message; do not crash the graph.
- No custom exception types, no queues.



## Observability

- `mlflow.langchain.autolog()` (or the LangGraph tracer) around `search` / `eval run`.
- Each eval item is one MLflow run (or nested span) with query id, hit@5, hit@10, rounds, tool_call count.
- Default tracking URI: local `./mlruns`.



## Testing

No live API in CI.

- **Retriever:** temp SQLite with 4–5 fake paragraphs. `search` returns the semantic neighbor. `grep` hits a date or name and misses a synonym.
- **Graph + fake model:** (1) seed → parallel `search`+`grep` → `submit_ranking`; assert merge, dedupe, stop. (2) 4 rounds and no ranking → force-submit uses the pool. (3) unknown chunk ids are dropped.
- **Metric:** canned ranking + gold titles → exact hit@5 / hit@10.
- **Parent:** fake OpenAI client; assert the request includes “answer only from this package” and the snippet texts.

Manual: `uv run whole-wheat eval run` against the real index.

## Defaults and configuration


| Knob             | Default                                                       |
| ---------------- | ------------------------------------------------------------- |
| Searcher model   | `gpt-5.6-luna` (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`) |
| Parent model     | same, or `WHOLE_WHEAT_PARENT_MODEL`                           |
| Embedding model  | FastEmbed `nomic-ai/nomic-embed-text-v1.5-Q` (768-d)          |
| Embed prefixes   | `search_document: ` (chunks), `search_query: ` (queries)      |
| Max agent rounds | 4                                                             |
| Seed / tool `k`  | 8                                                             |
| Ranking size     | up to 10 chunks                                               |
| Parallelism      | all `search`/`grep` calls in one model turn                   |




## Layout

```text
whole-wheat/
  pyproject.toml
  uv.lock                     # locked deps; uv is the runtime + package manager
  .python-version             # optional pin for `uv python`
  NOTICE                      # ODC-BY attribution for MultiHop-RAG
  data/
    corpus.json               # 609 articles (in git)
    qids.json                 # 20 frozen ids (in git)
    index.sqlite              # gitignored
    eval.json                 # gitignored (questions + answers)
  src/whole_wheat/
    ingest.py
    retriever.py
    tools.py
    graph.py
    parent.py
    eval.py
    cli.py
  tests/
  docs/superpowers/specs/     # this file
```



## License notes

- Code: Apache-2.0 (existing repo license).
- MultiHop-RAG articles: ODC-BY — keep `NOTICE` and the dataset citation.
- Do not commit `data/eval.json` (answers).



## Out of scope for v1 (explicit)

`prune_context`, `read_document`, `filter_chunks`, hybrid BM25+dense, auto-ingest, a parent-as-tool graph, null queries, and any claim of Toast-1 parity.