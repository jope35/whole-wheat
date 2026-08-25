from __future__ import annotations

import argparse

from whole_wheat.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(prog="whole-wheat")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest", help="build data/index.sqlite")
    ingest_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="index only the first N sorted JSON paths (demo)",
    )
    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.limit)
