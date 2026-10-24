import os

from dfd.env import load_dotenv


def test_load_dotenv_fills_unset_and_empty_but_keeps_real_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nexport A_KEY="from-file"\nB_KEY=from-file\nC_KEY=from-file\nBAD LINE\n')
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.setenv("B_KEY", "")  # empty export must not shadow the file (the bug seen in practice)
    monkeypatch.setenv("C_KEY", "from-shell")
    loaded = load_dotenv(env)
    assert sorted(loaded) == ["A_KEY", "B_KEY"]
    assert os.environ["A_KEY"] == "from-file" and os.environ["B_KEY"] == "from-file"
    assert os.environ["C_KEY"] == "from-shell"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []
