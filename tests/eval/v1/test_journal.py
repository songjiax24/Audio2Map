import json

import pytest

from audio2map.eval.v1.journal import Journal


def test_resume_discards_only_an_uncommitted_final_record(tmp_path):
    path = tmp_path / "records.jsonl"
    row = {"relpath": "a.osu", "scenario": "none", "mode": "teacher", "metrics": {}}
    path.write_text(json.dumps(row) + '\n{"relpath":')
    with Journal(path, resume=True) as journal:
        cached = journal.get(row, "teacher")
        assert cached == row
        assert journal.get({**row, "condition": {"mask": [1]}}, "teacher") is None
        journal.add(cached)
        assert journal.reused == 1
    assert path.read_text() == json.dumps(row) + "\n"
    path.write_text("broken\n")
    with pytest.raises(json.JSONDecodeError):
        Journal(path, resume=True)
