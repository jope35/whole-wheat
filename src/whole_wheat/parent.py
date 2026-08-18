from __future__ import annotations

import json
import os

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

PARENT = "Answer only from this package. If the snippets do not support an answer, say so."


def answer(question: str, package: dict, model: object | None = None) -> str:
    if package.get("no_evidence") or not package.get("ranked"):
        return "No evidence in the package."
    if model is None:
        model = ChatOpenAI(
            model=os.environ.get("WHOLE_WHEAT_PARENT_MODEL")
            or os.environ.get("WHOLE_WHEAT_SEARCHER_MODEL", "gpt-5.6-luna"),
            max_retries=2,
        )
    snippets = [row["text"] for row in package["ranked"]]
    response = model.invoke(
        [
            SystemMessage(content=PARENT),
            HumanMessage(
                content=f"Question: {question}\nSnippets: {json.dumps(snippets)}"
            ),
        ]
    )
    return str(response.content)
