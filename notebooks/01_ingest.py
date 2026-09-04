# Databricks notebook source
# Batch JSON from the Volume → Delta table with a VARIANT column.

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
table = f"{catalog}.{schema}.bronze_corpus"
print(json_dir, table)

# COMMAND ----------

import pyspark.sql.functions as F

(
    spark.read.format("json")
    .option("singleVariantColumn", "data")
    .option("multiLine", "true")
    .load(json_dir)
    .select(
        F.col("_metadata.file_path").alias("source_file"),
        F.col("_metadata.file_name").alias("file_name"),
        F.col("_metadata.file_size").alias("file_size"),
        F.current_timestamp().alias("ingestion_time"),
        F.col("data"),
    )
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(table)
)

# COMMAND ----------

display(spark.table(table).limit(5))
display(spark.sql(f"DESCRIBE TABLE {table}"))
