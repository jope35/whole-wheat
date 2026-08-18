from whole_wheat.eval import hit_at_k, rows_to_eval, squeeze

GOLD = [
    {
        "title": "The FTX trial",
        "source": "The Verge",
        "published_at": "2023-09-28T12:00:00+00:00",
    }
]


def test_squeeze() -> None:
    assert squeeze("  The   FTX Trial ") == "the ftx trial"


def test_hit_at_k() -> None:
    ranked = [
        {"title": "Unrelated", "source": "X", "published_at": "2023-01-01T00:00:00+00:00"},
        {
            "title": "The FTX trial",
            "source": "The Verge",
            "published_at": "2023-09-28T12:00:00+00:00",
        },
    ]
    assert hit_at_k(ranked, GOLD, 1) is False
    assert hit_at_k(ranked, GOLD, 2) is True
    assert hit_at_k(ranked, GOLD, 5) is True
    wrong_source = [{**ranked[1], "source": "Fortune"}]
    assert hit_at_k(
        wrong_source, GOLD, 1, duplicate_titles={"the ftx trial"}
    ) is False


def test_rows_to_eval() -> None:
    rows = rows_to_eval(
        [{"id": "q01", "query": "Who ran FTX?"}],
        [
            {
                "query": "Who ran FTX?",
                "question_type": "inference_query",
                "answer": "Sam Bankman-Fried",
                "evidence_list": [
                    {
                        "title": "The FTX trial",
                        "source": "The Verge",
                        "published_at": "2023-09-28T12:00:00+00:00",
                    }
                ],
            }
        ],
    )
    assert rows[0]["id"] == "q01"
    assert rows[0]["gold"][0]["title"] == "The FTX trial"
