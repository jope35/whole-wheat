import sqlite3
from pathlib import Path

import sqlite_vec
from langchain.messages import AIMessage
from sqlite_vec import serialize_float32

from whole_wheat.graph import run_search
from whole_wheat.retriever import Retriever


def fake_embed(text: str) -> list[float]:
    v = [0.0, 0.0]
    t = text.casefold()
    if "apple" in t or "fruit" in t:
        v[0] = 1.0
    if "paris" in t:
        v[1] = 1.0
    return v


def write_index(path: Path) -> None:
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, article_id TEXT, "
        "title TEXT, source TEXT, published_at TEXT, text TEXT)"
    )
    db.execute(
        "CREATE VIRTUAL TABLE vec_chunks USING vec0("
        "chunk_id TEXT PRIMARY KEY, embedding float[2] distance_metric=cosine)"
    )
    rows = [
        ("seed::chunk-0", "seed", "Apples", "The Verge", "2023-09-01T00:00:00+00:00", "Apple harvest."),
        ("g::chunk-0", "g", "Paris", "The Age", "2023-09-02T00:00:00+00:00", "Paris fashion week."),
    ]
    for row in rows:
        db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", row)
        db.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (row[0], serialize_float32(fake_embed(row[5]))),
        )
    db.commit()
    db.close()


class ScriptedChat:
    def __init__(self, turns: list[AIMessage]) -> None:
        self.turns = list(turns)
        self.seen = []

    def bind_tools(self, tools: list) -> "ScriptedChat":
        return self

    def invoke(self, messages: list, **kwargs: object) -> AIMessage:
        self.seen.append(messages)
        return self.turns.pop(0)


def test_one_turn_search_and_grep_then_submit(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    retriever = Retriever(path, embed_query=fake_embed)
    model = ScriptedChat(
        [
            AIMessage(
                content="find fruit and the city",
                tool_calls=[
                    {"name": "search", "args": {"query": "fruit"}, "id": "t1"},
                    {"name": "grep", "args": {"pattern": "Paris"}, "id": "t2"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_ranking",
                        "args": {
                            "items": [
                                {"chunk_id": "g::chunk-0", "reason": "names Paris"},
                                {"chunk_id": "seed::chunk-0", "reason": "apple harvest"},
                            ]
                        },
                        "id": "t3",
                    }
                ],
            ),
        ]
    )
    package = run_search("Where is the fruit show?", retriever, model=model)
    assert [r["chunk_id"] for r in package["ranked"]] == ["g::chunk-0", "seed::chunk-0"]
    assert package["rounds"] == 2
    assert package["tool_calls"] == 3
    assert package["no_evidence"] is False
    first_prompt = " ".join(str(message.content) for message in model.seen[0])
    assert "seed::chunk-0" in first_prompt


def test_force_submit_after_four_rounds(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    pings = [
        AIMessage(
            content="again",
            tool_calls=[{"name": "search", "args": {"query": "apple"}, "id": f"x{i}"}],
        )
        for i in range(4)
    ]
    package = run_search(
        "apples",
        Retriever(path, embed_query=fake_embed),
        model=ScriptedChat(pings),
    )
    assert package["rounds"] == 4
    assert package["ranked"][0]["reason"] == "force: pool order"


def test_unknown_ids_drop_then_fallback(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    write_index(path)
    retriever = Retriever(path, embed_query=fake_embed)
    drop = run_search(
        "apples",
        retriever,
        model=ScriptedChat(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "submit_ranking",
                            "args": {
                                "items": [
                                    {"chunk_id": "missing::chunk-9", "reason": "nope"},
                                    {"chunk_id": "seed::chunk-0", "reason": "seed hit"},
                                ]
                            },
                            "id": "t1",
                        }
                    ],
                )
            ]
        ),
    )
    assert [r["chunk_id"] for r in drop["ranked"]] == ["seed::chunk-0"]
    fallback = run_search(
        "apples",
        retriever,
        model=ScriptedChat(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "submit_ranking",
                            "args": {"items": [{"chunk_id": "nope::chunk-0", "reason": "x"}]},
                            "id": "t1",
                        }
                    ],
                )
            ]
        ),
    )
    assert fallback["ranked"][0]["chunk_id"] == "seed::chunk-0"
    assert fallback["ranked"][0]["reason"] == "force: pool order"
