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
