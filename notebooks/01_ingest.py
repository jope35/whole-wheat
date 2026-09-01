# Databricks notebook source
# Plan 1 ingest: fixture → Volume → Delta page rows. Optional HF cells at the end.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("volume", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
assert catalog and schema and volume, "set catalog, schema, volume (bundle vars)"

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
table = f"{catalog}.{schema}.page_rows"
json_dir = f"{volume_path}/jsons"
print(volume_path, table)

# COMMAND ----------

import sys
from pathlib import Path

for root in (Path.cwd(), Path.cwd().parent):
    src = (root / "src").resolve()
    if (src / "whole_wheat").is_dir():
        sys.path.insert(0, str(src))
        DATA = (root / "data").resolve()
        break
else:
    raise FileNotFoundError("src/whole_wheat not found next to the notebook")

from whole_wheat.ingest import load_json_dir, load_json_file, write_page_rows

# COMMAND ----------

# Fixture path: copy fixture into the Volume, write Delta, enable CDF, show rows.

dbutils.fs.mkdirs(json_dir)
fixture_src = DATA / "fixture.json"
dbutils.fs.cp(f"file:{fixture_src}", f"{json_dir}/fixture.json")

rows = load_json_file(fixture_src)
write_page_rows(spark, rows, table)
display(spark.table(table).limit(20))

# COMMAND ----------

# Optional HF path. Stop cleanly when the token or subset list is missing.

import json

basenames = json.loads((DATA / "subset.json").read_text(encoding="utf-8"))
try:
    token = dbutils.secrets.get(catalog=catalog, schema=schema, key="hf_token")
except Exception:
    dbutils.notebook.exit("hf_token secret missing — fixture path done")

if not basenames:
    dbutils.notebook.exit("data/subset.json empty — fixture path done")

# COMMAND ----------

from huggingface_hub import hf_hub_download

out_dir = Path(json_dir)
out_dir.mkdir(parents=True, exist_ok=True)

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

all_rows = load_json_file(fixture_src) + [
    r for r in load_json_dir(out_dir) if r["source_file"] != "fixture"
]
write_page_rows(spark, all_rows, table)
display(spark.table(table).limit(20))
