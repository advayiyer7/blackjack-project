"""ONNX detector math: letterbox geometry, v8 output decoding, per-class NMS.

Pure-math tests with hand-built tensors — real-inference evaluation happens in
scripts/eval_detector.py once the trained weights exist (M4).
"""

from __future__ import annotations

import numpy as np
import pytest

from bjcounter.vision.autolabel import CLASS_NAMES
from bjcounter.vision.detector import (
    LETTERBOX_FILL,
    decode_predictions,
    letterbox,
    nms_per_class,
)

N_CLASSES = len(CLASS_NAMES)


def raw_output(rows: list[tuple[float, float, float, float, int, float]]) -> np.ndarray:
    """Build a (1, 4+nc, N) v8-style output from (cx, cy, w, h, class_id, score)."""
    arr = np.zeros((len(rows), 4 + N_CLASSES), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(rows):
        arr[i, :4] = (cx, cy, w, h)
        arr[i, 4 + class_id] = score
    return arr.T[np.newaxis]


class TestLetterbox:
    def test_wide_frame_pads_top_and_bottom(self):
        frame = np.full((845, 1200, 3), 200, dtype=np.uint8)
        padded, ratio, (dw, dh) = letterbox(frame, 1280)
        assert padded.shape == (1280, 1280, 3)
        assert ratio == pytest.approx(1280 / 1200)
        assert dw == 0
        assert dh > 0
        assert (padded[0] == LETTERBOX_FILL).all()  # top padding row
        assert (padded[640] == 200).all()  # centre row is image

    def test_roundtrip_maps_a_box_back_to_frame_coords(self):
        frame = np.zeros((845, 1200, 3), dtype=np.uint8)
        _, ratio, (dw, dh) = letterbox(frame, 1280)
        x, y = 422, 110  # a frame-space point
        lx, ly = x * ratio + dw, y * ratio + dh  # into letterbox space
        assert (lx - dw) / ratio == pytest.approx(x)
        assert (ly - dh) / ratio == pytest.approx(y)


class TestDecode:
    def test_decodes_box_back_into_frame_coordinates(self):
        frame_shape = (845, 1200)
        _, ratio, pad = letterbox(np.zeros((*frame_shape, 3), np.uint8), 1280)
        # A card at frame (422, 110), 84x118 -> centre in letterbox space:
        cx = (422 + 42) * ratio + pad[0]
        cy = (110 + 59) * ratio + pad[1]
        out = raw_output([(cx, cy, 84 * ratio, 118 * ratio, 7, 0.97)])
        dets = decode_predictions(out, ratio, pad, frame_shape, 0.8, 0.6)
        assert len(dets) == 1
        det = dets[0]
        assert det.label == CLASS_NAMES[7]
        assert det.conf == pytest.approx(0.97)
        x, y, w, h = det.bbox
        assert (x, y) == pytest.approx((422, 110), abs=1)
        assert (w, h) == pytest.approx((84, 118), abs=2)

    def test_below_threshold_is_dropped(self):
        out = raw_output([(640, 640, 80, 110, 3, 0.5)])
        assert decode_predictions(out, 1.0, (0, 0), (1280, 1280), 0.8, 0.6) == ()

    def test_empty_output_is_fine(self):
        out = np.zeros((1, 4 + N_CLASSES, 0), dtype=np.float32)
        assert decode_predictions(out, 1.0, (0, 0), (845, 1200), 0.8, 0.6) == ()

    def test_boxes_clip_to_frame(self):
        out = raw_output([(5, 5, 80, 110, 0, 0.95)])  # spills past the origin
        dets = decode_predictions(out, 1.0, (0, 0), (845, 1200), 0.8, 0.6)
        x, y, w, h = dets[0].bbox
        assert x == 0 and y == 0 and w > 0 and h > 0


class TestNms:
    def test_duplicate_boxes_of_same_class_are_suppressed(self):
        boxes = np.array([[100, 100, 184, 218], [102, 101, 186, 220]], dtype=float)
        kept = nms_per_class(boxes, np.array([0.99, 0.95]), np.array([7, 7]), 0.6)
        assert kept == [0]

    def test_fan_neighbours_of_same_class_survive(self):
        # Two identical cards fanned 27px apart: IoU ~0.47, must both survive.
        boxes = np.array([[100, 100, 184, 218], [127, 100, 211, 218]], dtype=float)
        kept = nms_per_class(boxes, np.array([0.99, 0.95]), np.array([7, 7]), 0.6)
        assert sorted(kept) == [0, 1]

    def test_different_classes_never_suppress_each_other(self):
        boxes = np.array([[100, 100, 184, 218], [101, 101, 185, 219]], dtype=float)
        kept = nms_per_class(boxes, np.array([0.99, 0.95]), np.array([7, 8]), 0.6)
        assert sorted(kept) == [0, 1]
