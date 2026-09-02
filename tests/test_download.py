"""
Unit tests for src/data/download.py (Module 1, W1.2). Never invokes real
git -- subprocess.run is monkeypatched so the test suite stays offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import REPO_URL, download


def test_skips_clone_when_target_exists(tmp_path, monkeypatch):
    target = tmp_path / "3W"
    target.mkdir()

    called = []
    monkeypatch.setattr("src.data.download.subprocess.run", lambda *a, **k: called.append((a, k)))

    download(str(target))
    assert called == []


def test_clones_when_target_missing(tmp_path, monkeypatch):
    target = tmp_path / "3W"

    calls = []
    monkeypatch.setattr("src.data.download.subprocess.run", lambda *a, **k: calls.append((a, k)))

    download(str(target))
    assert len(calls) == 1
    args, kwargs = calls[0]
    cmd = args[0]
    assert cmd == ["git", "clone", "--depth", "1", REPO_URL, str(target)]
    assert kwargs.get("check") is True
