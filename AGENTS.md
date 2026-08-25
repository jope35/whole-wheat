# 
Always load the following skills:
- ponytail
- simple english


the code is intended for a blog post, keep it super simple. the code is not going into production, it should convey the idea not bulletproof.

## Learned User Preferences

- Keep the code super simple and easy to follow; this repo is a blog-post demo, not production.
- Grill with one clarifying question at a time before writing a spec or code.
- Use LangChain, LangGraph, MLflow, and an OpenAI-compatible API for LLM calls. The searcher is a reasoning model (`WHOLE_WHEAT_SEARCHER_MODEL`, default `gpt-5.6-luna`, `reasoning_effort=low`). Searcher and parent send every LLM call to `/v1/responses` (`use_responses_api=True`); do not use `/v1/chat/completions` or pin a no-reasoning Chat Completions path.
- Use uv for the Python runtime and dependencies (`uv.lock`, `uv python install`, `uv sync`, `uv run`); do not use pip, Poetry, conda, or Apple `/usr/bin/python3` (that build cannot load sqlite-vec).
- Prefer local embeddings via FastEmbed (`nomic-ai/nomic-embed-text-v1.5-Q`, full 768-d) stored in SQLite with sqlite-vec over BM25-only search; prefix chunks with `search_document:` and queries with `search_query:`.
- Prefer text-native Hugging Face corpora that do not need OCR or PDF processing.
- Put shared static config (paths, model name, prefixes, prompts) in `constants.py`.
- Use tenacity for retries (not a custom retry loop).
- Use standard-library `argparse` for the CLI; do not add Typer or Click.
- v1 has no automated tests; check with a manual `uv run whole-wheat eval run`.

## Learned Workspace Facts

- whole-wheat is an open-source Toast 1-style retrieval subagent: it returns a ranked evidence package (snippets, source ids, short relevance notes), not a final answer.
- The agent is an explicit LangGraph loop: `seed_search` → `agent` → custom tools node → loop; one turn may emit several `search`/`grep` calls and they run in sequence on one process; if a turn mixes `submit_ranking` with other tools, run `submit_ranking` only; omit `k` → 8, `k > 8` is a tool error; ranking is at most 10 unique first-seen `chunk_id`s (empty or invalid lists fall back to pool order); stop on ranking or after 4 rounds (run that turn’s searches, then force-submit; no fifth model call).
- Tool surface is three tools: `search`, `grep`, and `submit_ranking`.
- `search` is semantic (FastEmbed Nomic embeddings in SQLite via sqlite-vec); `grep` is Python regex over chunk text (invalid regex returns no hits; do not use SQL `LIKE`); the index is gitignored `data/index.sqlite`: a `chunks` table holds text and metadata, `chunk_vectors` (`vec0`) holds embeddings only, shared integer rowid; rebuild is one connection and one transaction (commit after all inserts); construct FastEmbed on first ingest embed or `search()` through a cached helper; embed ingest batches of 64.
- MLflow traces the loop; eval scores document hit@5 and hit@10 on 20 fixture questions; after three failed LLM attempts, count that row as 0 and continue; no live model calls in CI.
- Corpus is OfficeQA Pro v2 from Hugging Face `databricks/officeqa-pro-v2` (gated, CC-BY-SA): parsed JSON only, no PDF/OCR; `scripts/download-corpus.sh` snapshots the full parsed JSON tree with `local_dir` set to `data/` (Hub paths already include `parsed_corpus/jsons/`); ingest globs `data/parsed_corpus/jsons/*.json` and indexes every match (optional `--limit N` for demos; omit `--limit` to ingest all files; zero files → generic “no files” error); do not use `cited-files.txt`; skip `figure`, `page_header`, `page_footer`, and `page_number`; gold is `source_files` document id.
- Eval slice lives in `data/qids.json` as frozen `{id, query}` rows matching dataset `uid`s (7 with 1–3 gold docs, 7 with 4–8, 6 with 9+); answers stay off git and are fetched into a gitignored bundle; `search` and `ask` take question text only (qids are for `eval fetch` / `eval run`).
- A thin parent demo (one Responses API call) answers only from the evidence package; it has no search tools.
- Layout is a small Python package (`constants`, `ingest`, `retriever`, `tools`, `graph`, `parent`, `eval`, `cli`) run with uv; the graph never imports SQLite.
