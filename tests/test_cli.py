from whole_wheat.cli import main


def test_missing_index(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["search", "hello"]) == 1
    assert "whole-wheat ingest" in capsys.readouterr().err


def test_missing_eval(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "index.sqlite").write_bytes(b"x")
    assert main(["eval", "run"]) == 1
    assert "eval fetch" in capsys.readouterr().err
