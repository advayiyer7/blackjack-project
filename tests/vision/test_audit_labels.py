"""Audit flags must leave reviewable crops before suggesting exclusions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scripts import audit_labels

from bjcounter.vision.autolabel import CLASS_NAMES
from bjcounter.vision.dataset import RealFrame


def test_missed_ground_truth_card_gets_review_crop(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "session_a" / "frame_00000.png"
    image_path.parent.mkdir()
    frame = RealFrame(
        session="session_a",
        name=image_path.name,
        path=image_path,
        scale=1.0,
        frame_w=960,
        frame_h=676,
        hits=((0, 100, 100, 0.99),),
        split="val",
    )

    class EmptyDetector:
        def __init__(self, *_args, **_kwargs):
            pass

        def detect(self, _image):
            return ()

    crops = []
    monkeypatch.setattr(audit_labels, "parse_args", lambda: SimpleNamespace(onnx=None, imgsz=None))
    monkeypatch.setattr(audit_labels, "OnnxYoloDetector", EmptyDetector)
    monkeypatch.setattr(audit_labels, "iter_real_frames", lambda *_args, **_kwargs: iter((frame,)))
    monkeypatch.setattr(
        audit_labels.cv2, "imread", lambda *_: np.zeros((676, 960, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(audit_labels, "save_crop", lambda *_args: crops.append(_args))

    audit_labels.main()

    assert f"MISS {CLASS_NAMES[0]} x1" in capsys.readouterr().out
    assert len(crops) == 1
    assert crops[0][1] == (100, 100, 67, 94)
    assert "MISS" in crops[0][3]


def test_failed_audit_crop_write_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_labels.cv2, "imwrite", lambda *_: False)
    with pytest.raises(OSError, match="crop"):
        audit_labels.save_crop(
            np.zeros((100, 100, 3), dtype=np.uint8),
            (10, 10, 20, 20),
            tmp_path,
            "flag",
        )
