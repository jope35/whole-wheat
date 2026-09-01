# Databricks notebook source
# Download OfficeQA Pro V2 corpus from Hugging Face into the Volume.
# Pro V2 ships parsed JSON (no .txt). source_files use a .txt basename only.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("volume", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
assert catalog and schema and volume, "set catalog, schema, volume (bundle vars)"

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
json_dir = f"{volume_path}/jsons"
print(volume_path)

# COMMAND ----------

import json
import shutil
from pathlib import Path

for root in (Path.cwd(), Path.cwd().parent):
    data = (root / "data").resolve()
    if data.is_dir():
        DATA = data
        break
else:
    raise FileNotFoundError("data/ not found next to the notebook")

token = dbutils.secrets.get(catalog=catalog, schema=schema, key="hf_token")

basenames = json.loads((DATA / "subset.json").read_text(encoding="utf-8"))

# COMMAND ----------

from huggingface_hub import hf_hub_download, snapshot_download

out_dir = Path(json_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if basenames:
    for name in basenames:
        local = hf_hub_download(
            repo_id="databricks/officeqa-pro-v2",
            repo_type="dataset",
            filename=f"parsed_corpus/jsons/{name}.json",
            token=token,
        )
        dest = out_dir / f"{name}.json"
        dest.write_bytes(Path(local).read_bytes())
        print(f"wrote {dest.name}")
else:
    # Full corpus (~794MB). Fill data/subset.json to download only the frozen subset.
    local_dir = snapshot_download(
        repo_id="databricks/officeqa-pro-v2",
        repo_type="dataset",
        allow_patterns="parsed_corpus/jsons/*.json",
        token=token,
    )
    src = Path(local_dir) / "parsed_corpus" / "jsons"
    for path in sorted(src.glob("*.json")):
        shutil.copy2(path, out_dir / path.name)
    print(f"wrote {len(list(out_dir.glob('*.json')))} files to {json_dir}")

display(dbutils.fs.ls(json_dir)[:20])
