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
