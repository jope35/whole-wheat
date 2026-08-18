# 
Always load the following skills:
- ponytail
- simple english

## Learned User Preferences

- Keep the code super simple and easy to follow; this repo is a blog-post demo, not production.
- Grill with one clarifying question at a time before writing a spec or code.
- Use LangChain, LangGraph, MLflow, and the OpenAI-compatible chat API for LLM calls.
- Use uv for the Python runtime and dependencies (`uv.lock`, `uv sync`, `uv run`); do not use pip, Poetry, or conda as the project workflow.
- Prefer local embeddings via FastEmbed (`nomic-ai/nomic-embed-text-v1.5-Q`, full 768-d) stored in SQLite with sqlite-vec over BM25-only search; prefix chunks with `search_document:` and queries with `search_query:`.
- Prefer text-native Hugging Face corpora that do not need OCR or PDF processing.

## Learned Workspace Facts

- whole-wheat is an open-source Toast 1-style retrieval subagent: it returns a ranked evidence package (snippets, source ids, short relevance notes), not a final answer.
- The agent is an explicit LangGraph loop: `seed_search` → `agent` → `tools` → loop; one turn may run several `search`/`grep` calls in parallel; stop on `submit_ranking` or after 4 rounds, then force a ranking.
- Tool surface is three tools: `search`, `grep`, and `submit_ranking`.
- `search` is semantic (FastEmbed Nomic embeddings in SQLite via sqlite-vec); `grep` stays lexical.
- MLflow traces the loop; eval scores article hit@k on 20 fixture questions.
- Corpus is MultiHop-RAG (609 news articles, vendored): paragraph-sized chunks, gold is article title; ODC-BY license.
- Eval slice is 7 inference, 7 comparison, and 6 temporal queries (no nulls), spread across news categories; answers stay off git and are fetched into a gitignored bundle.
- A thin parent demo (one OpenAI chat completion) answers only from the evidence package; it has no search tools.
- Layout is a small Python package (ingest, tools, graph, parent, eval, CLI) run with uv.
