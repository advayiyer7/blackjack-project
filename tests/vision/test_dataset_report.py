"""A failed label refresh must preserve the previous per-session labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import dataset_report


def setup_session(tmp_path, monkeypatch):
    session = tmp_path / "raw" / "session_test"
    session.mkdir(parents=True)
    (session / "session_meta.json").write_text("{}", encoding="utf-8")
    (session / "frame_00000.png").write_bytes(b"frame")
    labels = session / "detections.json"
    labels.write_text('{"previous": true}', encoding="utf-8")
    monkeypatch.setattr(dataset_report, "RAW", tmp_path / "raw")
    monkeypatch.setattr(
        dataset_report,
        "analyze_session",
        lambda *_args, **_kwargs: {
            "frames_valid": 1,
            "frames_total": 1,
            "frames_skipped": 0,
            "mean_score": 0.95,
            "detections": {"frame_00000.png": []},
        },
    )
    return session, labels


def test_failed_label_replacement_keeps_previous_file(tmp_path, monkeypatch):
    session, labels = setup_session(tmp_path, monkeypatch)

    def fail_replace(_self, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        dataset_report.analyze_real_sessions(limit=None)

    assert labels.read_text(encoding="utf-8") == '{"previous": true}'
    assert list(session.glob(".detections.json.*")) == []


def test_partial_label_write_keeps_previous_file(tmp_path, monkeypatch):
    session, labels = setup_session(tmp_path, monkeypatch)
    original_write_text = Path.write_text

    def partial_write(path, data, *args, **kwargs):
        if path.name.startswith(".detections.json."):
            original_write_text(path, data[:10], *args, **kwargs)
            raise OSError("disk full")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", partial_write)
    with pytest.raises(OSError, match="disk full"):
        dataset_report.analyze_real_sessions(limit=None)

    assert labels.read_text(encoding="utf-8") == '{"previous": true}'
    assert list(session.glob(".detections.json.*")) == []


def test_successful_label_refresh_replaces_previous_file(tmp_path, monkeypatch):
    _session, labels = setup_session(tmp_path, monkeypatch)

    dataset_report.analyze_real_sessions(limit=None)

    assert json.loads(labels.read_text(encoding="utf-8"))["frames_valid"] == 1
