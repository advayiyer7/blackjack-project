"""M4 gate results must reflect which ranks the held-out split actually tests."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
from scripts import eval_detector

from bjcounter.types import Detection
from bjcounter.vision.dataset import RealFrame


def setup_eval(tmp_path, monkeypatch, hits, predictions):
    (tmp_path / "models").mkdir()
    model = tmp_path / "models" / "best.onnx"
    model.write_bytes(b"fake model")
    frame = RealFrame(
        session="session_a",
        name="frame_00001.png",
        path=tmp_path / "frame_00001.png",
        scale=1.0,
        frame_w=960,
        frame_h=676,
        hits=hits,
        split="test",
    )

    class FakeDetector:
        imgsz = 640

        def __init__(self, *args, **kwargs):
            pass

        def detect(self, image):
            return predictions

    monkeypatch.setattr(eval_detector, "REPO", tmp_path)
    monkeypatch.setattr(
        eval_detector,
        "parse_args",
        lambda: argparse.Namespace(onnx=model, split="test", imgsz=None, conf=0.8),
    )
    monkeypatch.setattr(eval_detector, "iter_real_frames", lambda _: iter((frame,)))
    monkeypatch.setattr(eval_detector, "OnnxYoloDetector", FakeDetector)
    monkeypatch.setattr(
        eval_detector.cv2, "imread", lambda _: np.zeros((676, 960, 3), dtype=np.uint8)
    )
    return tmp_path / "models" / "metrics.json"


def test_missing_ranks_cannot_pass_per_rank_recall_gate(tmp_path, monkeypatch):
    prediction = Detection(label="2c", bbox=(100, 100, 67, 94), conf=0.99)
    metrics_path = setup_eval(tmp_path, monkeypatch, ((0, 100, 100, 0.99),), (prediction,))

    with pytest.raises(SystemExit, match="1"):
        eval_detector.main()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["per_rank_recall_at_conf"] == {"2": 1.0}
    assert "A" in metrics["missing_ranks"]
    assert "back" in metrics["missing_ranks"]
    assert metrics["gates"]["per_rank_recall"] is False


def test_no_ground_truth_cards_reports_inadequate_split(tmp_path, monkeypatch):
    setup_eval(tmp_path, monkeypatch, (), ())

    with pytest.raises(SystemExit, match="no ground-truth cards"):
        eval_detector.main()


def test_all_represented_ranks_can_pass_recall_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_detector, "CLASS_NAMES", ("2c", "Ah"))
    hits = ((0, 100, 100, 0.99), (1, 300, 100, 0.99))
    predictions = (
        Detection(label="2c", bbox=(100, 100, 67, 94), conf=0.99),
        Detection(label="Ah", bbox=(300, 100, 67, 94), conf=0.99),
    )
    metrics_path = setup_eval(tmp_path, monkeypatch, hits, predictions)

    eval_detector.main()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["missing_ranks"] == []
    assert metrics["gates"]["per_rank_recall"] is True
