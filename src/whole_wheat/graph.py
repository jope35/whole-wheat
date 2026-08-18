from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, TypedDict

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from whole_wheat.retriever import Retriever
from whole_wheat.tools import build_tools

MAX_ROUNDS = 4
SYSTEM = (
    "You are a retrieval subagent. Return evidence, not an answer. "
    "Write one concise sentence of what you want to find. "
    "Use search for meaning. Use grep for names, dates, titles, and exact tokens. "
    "Call submit_ranking when the pool is good enough. "
    "Remaining agent rounds: {remaining}. "
    "Current pool: {pool}"
)


class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    pool: dict[str, dict]
    ranking: list[tuple[str, str]] | None
    rounds: int


def resolve_ranking(items: list[tuple[str, str]], pool: dict[str, dict]) -> list[dict]:
    ranked = []
    seen = set()
    for chunk_id, reason in items:
        hit = pool.get(chunk_id)
        if hit and chunk_id not in seen:
            ranked.append({**hit, "reason": reason})
            seen.add(chunk_id)
        if len(ranked) == 10:
            return ranked
    if not ranked:
        for hit in pool.values():
            ranked.append({**hit, "reason": "force: pool order"})
            if len(ranked) == 10:
                break
    return ranked


def package_from_state(question: str, state: SearchState) -> dict:
    ranked = resolve_ranking(state["ranking"] or [], state["pool"])
    tool_calls = 0
    for message in state["messages"]:
        tool_calls += len(getattr(message, "tool_calls", None) or [])
    return {
        "query": question,
        "ranked": ranked,
        "rounds": state["rounds"],
        "tool_calls": tool_calls,
        "no_evidence": len(ranked) == 0,
    }


def build_graph(retriever: Retriever, model: Any):
    tools = build_tools(retriever)
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)

    def seed_search(state: SearchState) -> dict:
        question = next(
            m.content for m in state["messages"] if isinstance(m, HumanMessage)
        )
        hits = retriever.search(question, k=8)
        return {"pool": {h["chunk_id"]: h for h in hits}}

    def agent(state: SearchState) -> dict:
        rounds = state["rounds"] + 1
        response = bound.invoke(
            [
                SystemMessage(
                    content=SYSTEM.format(
                        remaining=max(0, MAX_ROUNDS - rounds),
                        pool=json.dumps(list(state["pool"].values())),
                    )
                )
            ]
            + state["messages"]
        )
        return {"messages": [response], "rounds": rounds}

    def tools_node(state: SearchState) -> dict:
        last = state["messages"][-1]
        pool = dict(state["pool"])
        ranking = state["ranking"]
        messages = []

        def run_tool(call: dict) -> tuple[dict, str]:
            tool = by_name.get(call["name"])
            try:
                payload = tool.invoke(call["args"]) if tool else f"unknown tool: {call['name']}"
            except Exception as exc:
                payload = f"bad arguments: {exc}"
            return call, payload

        with ThreadPoolExecutor(max_workers=len(last.tool_calls)) as executor:
            results = executor.map(run_tool, last.tool_calls)
        for call, payload in results:
            messages.append(
                ToolMessage(content=payload, tool_call_id=call["id"], name=call["name"])
            )
            if call["name"] == "submit_ranking":
                try:
                    ranking = [(i["chunk_id"], i.get("reason", "")) for i in json.loads(payload)]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
                continue
            if call["name"] in {"search", "grep"}:
                try:
                    for hit in json.loads(payload):
                        pool.setdefault(hit["chunk_id"], hit)
                except json.JSONDecodeError:
                    pass
        return {"messages": messages, "pool": pool, "ranking": ranking}

    def after_agent(state: SearchState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def after_tools(state: SearchState) -> Literal["agent", "__end__"]:
        if state["ranking"] is not None or state["rounds"] >= MAX_ROUNDS:
            return END
        return "agent"

    g = StateGraph(SearchState)
    g.add_node("seed_search", seed_search)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_edge(START, "seed_search")
    g.add_edge("seed_search", "agent")
    g.add_conditional_edges("agent", after_agent)
    g.add_conditional_edges("tools", after_tools)
    return g.compile()


def run_search(question: str, retriever: Retriever, model: object | None = None) -> dict:
    if model is None:
        model = ChatOpenAI(
            model=os.environ.get("WHOLE_WHEAT_SEARCHER_MODEL", "gpt-5.6-luna"),
            max_retries=2,
        )
    state = build_graph(retriever, model).invoke(
        {
            "messages": [HumanMessage(content=question)],
            "pool": {},
            "ranking": None,
            "rounds": 0,
        }
    )
    return package_from_state(question, state)
