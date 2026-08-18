from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from whole_wheat.eval import fetch_eval, run_eval
from whole_wheat.graph import run_search
from whole_wheat.ingest import ingest
from whole_wheat.parent import answer
from whole_wheat.retriever import Retriever

CORPUS, INDEX, QIDS, EVAL = (
    Path("data/corpus.json"),
    Path("data/index.sqlite"),
    Path("data/qids.json"),
    Path("data/eval.json"),
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="whole-wheat")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    s = sub.add_parser("search")
    s.add_argument("question", nargs="?")
    s.add_argument("--qid")
    a = sub.add_parser("ask")
    a.add_argument("question")
    ev = sub.add_parser("eval")
    evs = ev.add_subparsers(dest="eval_cmd", required=True)
    evs.add_parser("fetch")
    evs.add_parser("run")
    args = p.parse_args(argv)

    if args.cmd == "ingest":
        print(f"wrote {ingest(CORPUS, INDEX)} chunks")
        return 0

    if args.cmd == "eval" and args.eval_cmd == "fetch":
        print(f"wrote {fetch_eval(QIDS, EVAL)} rows")
        return 0

    if not INDEX.exists():
        print("Index missing. Run: uv run whole-wheat ingest", file=sys.stderr)
        return 1
    retriever = Retriever(INDEX)

    if args.cmd == "search":
        question = args.question
        if args.qid:
            if not EVAL.exists():
                print("Eval bundle missing. Run: uv run whole-wheat eval fetch", file=sys.stderr)
                return 1
            question = next(
                (
                    row["query"]
                    for row in json.loads(EVAL.read_text())
                    if row["id"] == args.qid
                ),
                None,
            )
            if question is None:
                print(f"Unknown qid: {args.qid}", file=sys.stderr)
                return 1
        if not question:
            print("Pass a question or --qid.", file=sys.stderr)
            return 1
        print(json.dumps(run_search(question, retriever), indent=2))
        return 0

    if args.cmd == "ask":
        package = run_search(args.question, retriever)
        print(answer(args.question, package))
        return 0

    if not EVAL.exists():
        print("Eval bundle missing. Run: uv run whole-wheat eval fetch", file=sys.stderr)
        return 1
    rows, skipped = run_eval(EVAL, retriever)
    print("id\thit@5\thit@10\trounds\ttool_calls")
    h5 = h10 = 0
    for row in rows:
        print(f"{row['id']}\t{int(row['hit@5'])}\t{int(row['hit@10'])}\t{row['rounds']}\t{row['tool_calls']}")
        h5 += int(row["hit@5"])
        h10 += int(row["hit@10"])
    n = max(len(rows), 1)
    print(f"mean\t{h5 / n:.3f}\t{h10 / n:.3f}")
    if skipped:
        print("warning: MLflow tracing was skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
