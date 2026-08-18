from __future__ import annotations

import json
from pathlib import Path

import mlflow
from datasets import load_dataset

from whole_wheat.graph import run_search
from whole_wheat.retriever import Retriever


def squeeze(text: str) -> str:
    return " ".join(text.casefold().split())


def _match(chunk: dict, gold: dict, duplicate_titles: set[str]) -> bool:
    title = squeeze(chunk["title"])
    if title != squeeze(gold["title"]):
        return False
    if title not in duplicate_titles:
        return True
    return squeeze(chunk["source"]) == squeeze(gold["source"]) and squeeze(
        chunk["published_at"]
    ) == squeeze(gold["published_at"])


def hit_at_k(
    ranked: list[dict],
    gold: list[dict],
    k: int,
    duplicate_titles: set[str] | None = None,
) -> bool:
    duplicates = duplicate_titles or set()
    return any(
        _match(chunk, gold_row, duplicates)
        for chunk in ranked[:k]
        for gold_row in gold
    )


def rows_to_eval(qids: list[dict], hf_rows: list[dict]) -> list[dict]:
    by_query = {row["query"]: row for row in hf_rows}
    out = []
    for item in qids:
        row = by_query[item["query"]]
        gold = [
            {
                "title": ev["title"],
                "source": ev["source"],
                "published_at": ev["published_at"],
            }
            for ev in row["evidence_list"]
        ]
        out.append(
            {
                "id": item["id"],
                "query": row["query"],
                "question_type": row["question_type"],
                "answer": row["answer"],
                "gold": gold,
            }
        )
    return out


def fetch_eval(qids_path: Path, out_path: Path) -> int:
    qids = json.loads(qids_path.read_text())
    dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    rows = rows_to_eval(qids, [dict(r) for r in dataset])
    out_path.write_text(json.dumps(rows, indent=2))
    return len(rows)


def run_eval(
    eval_path: Path, retriever: Retriever, model: object | None = None
) -> tuple[list[dict], bool]:
    skipped = False
    mlflow_ready = True
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.langchain.autolog()
    except Exception:
        skipped = True
        mlflow_ready = False
    duplicate_titles = retriever.duplicate_titles()
    rows = []
    for item in json.loads(eval_path.read_text()):
        tracing = False
        if mlflow_ready:
            try:
                mlflow.start_run(run_name=item["id"])
                tracing = True
            except Exception:
                skipped = True
        try:
            package = run_search(item["query"], retriever, model=model)
            record = {
                "id": item["id"],
                "hit@5": hit_at_k(
                    package["ranked"], item["gold"], 5, duplicate_titles
                ),
                "hit@10": hit_at_k(
                    package["ranked"], item["gold"], 10, duplicate_titles
                ),
                "rounds": package["rounds"],
                "tool_calls": package["tool_calls"],
            }
            rows.append(record)
            if tracing:
                try:
                    mlflow.log_param("qid", item["id"])
                    mlflow.log_metric("hit@5", int(record["hit@5"]))
                    mlflow.log_metric("hit@10", int(record["hit@10"]))
                    mlflow.log_metric("rounds", record["rounds"])
                    mlflow.log_metric("tool_calls", record["tool_calls"])
                except Exception:
                    skipped = True
        finally:
            if tracing:
                try:
                    mlflow.end_run()
                except Exception:
                    skipped = True
    return rows, skipped
