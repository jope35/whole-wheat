from __future__ import annotations

import json

from langchain.tools import tool

from whole_wheat.retriever import Retriever


def build_tools(retriever: Retriever) -> list:
    @tool
    def search(query: str, k: int = 8) -> str:
        """Semantic search. Use for meaning."""
        return json.dumps(retriever.search(query, k=k))

    @tool
    def grep(pattern: str, k: int = 8) -> str:
        """Lexical search. Use for names, dates, titles, exact tokens."""
        return json.dumps(retriever.grep(pattern, k=k))

    @tool
    def submit_ranking(items: list[dict]) -> str:
        """End the search. items is best-first [{chunk_id, reason}] (max 10)."""
        return json.dumps(items[:10])

    return [search, grep, submit_ranking]
