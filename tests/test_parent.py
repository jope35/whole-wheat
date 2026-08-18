from langchain.messages import AIMessage

from whole_wheat.parent import answer

PACKAGE = {
    "query": "Who grew apples?",
    "ranked": [
        {
            "chunk_id": "a::chunk-0",
            "article_id": "a",
            "title": "Apples",
            "source": "The Verge",
            "published_at": "2023-10-01T12:00:00+00:00",
            "text": "Farmers grew apples in the valley.",
            "reason": "states the crop",
        }
    ],
    "rounds": 1,
    "tool_calls": 1,
    "no_evidence": False,
}


class Recorder:
    def __init__(self) -> None:
        self.seen = None

    def invoke(self, messages: list, **kwargs: object) -> AIMessage:
        self.seen = messages
        return AIMessage(content="demo answer")


def test_prompt_has_rule_and_snippet() -> None:
    rec = Recorder()
    assert answer("Who grew apples?", PACKAGE, model=rec) == "demo answer"
    blob = " ".join(str(m.content) for m in rec.seen)
    assert "answer only from this package" in blob.casefold()
    assert "Farmers grew apples in the valley." in blob


def test_empty_package_skips_model() -> None:
    rec = Recorder()
    empty = {"query": "??", "ranked": [], "rounds": 1, "tool_calls": 0, "no_evidence": True}
    assert answer("??", empty, model=rec) == "No evidence in the package."
    assert rec.seen is None
