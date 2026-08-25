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
