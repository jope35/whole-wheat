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
