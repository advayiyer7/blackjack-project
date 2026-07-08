"""CardDetector protocol, the template-matching fallback, and the ONNX detector
(ARCHITECTURE §2).

The protocol keeps the model swappable: M5's tracker development and replay tests run
against TemplateMatchDetector before the M4 YOLO weights exist; OnnxYoloDetector
implements the same protocol for the fine-tuned weights. Template matching runs
~100-300 ms/frame — over the 100 ms budget, accepted for development only
(ARCHITECTURE §12); the latency gate applies to the ONNX detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from bjcounter.types import Detection
from bjcounter.vision.autolabel import (
    CARD_H,
    CARD_W,
    CLASS_NAMES,
    MATCH_THRESHOLD,
    detect_cards,
    load_templates,
)

# Runtime operating point: matches tracker/state.py CONF_THRESHOLD, so a detection the
# assembly would gate as low-confidence is not emitted at all by default. The eval
# script passes a much lower threshold to sweep the PR curve.
ONNX_CONF_THRESHOLD = 0.80
# Per-class NMS: fan neighbours of the SAME class sit at IoU ~0.47, duplicate boxes for
# one card at ~0.9 — 0.6 separates the two regimes with margin on both sides.
ONNX_IOU_THRESHOLD = 0.6
LETTERBOX_FILL = 114  # ultralytics convention


@runtime_checkable
class CardDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection, ...]:
        """All cards visible in the frame, as full-card bboxes in frame coords."""
        ...


class TemplateMatchDetector:
    """Corner-strip template matching against the trainer's own sprites.

    `scale` is the capture scale (card width / 67); templates are prepared once per
    instance, so create one detector per capture session.
    """

    def __init__(self, deck_png: Path, scale: float, threshold: float = MATCH_THRESHOLD) -> None:
        self._templates = load_templates(deck_png, scale)
        self._scale = scale
        self._threshold = threshold

    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection, ...]:
        frame_h, frame_w = frame_bgr.shape[:2]
        hits = detect_cards(frame_bgr, self._templates, self._scale, self._threshold)
        return tuple(
            Detection(
                label=CLASS_NAMES[hit.class_id],
                bbox=(
                    hit.x,
                    hit.y,
                    round(min(CARD_W * self._scale, frame_w - hit.x)),
                    round(min(CARD_H * self._scale, frame_h - hit.y)),
                ),
                conf=hit.score,
            )
            for hit in hits
        )


def letterbox(frame_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect, pad to (size, size). Returns (image, ratio, (dw, dh))."""
    h, w = frame_bgr.shape[:2]
    ratio = min(size / w, size / h)
    new_w, new_h = round(w * ratio), round(h * ratio)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(LETTERBOX_FILL,) * 3,
    )
    return padded, ratio, (left, top)


def nms_per_class(
    boxes_xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float
) -> list[int]:
    """Greedy per-class NMS; returns kept indices ordered by descending score."""
    kept: list[int] = []
    for order_i in np.argsort(-scores):
        i = int(order_i)
        suppressed = False
        for j in kept:
            if classes[i] != classes[j]:
                continue
            if _iou(boxes_xyxy[i], boxes_xyxy[j]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(i)
    return kept


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def decode_predictions(
    output: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    frame_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[Detection, ...]:
    """YOLOv8 ONNX raw output -> Detections in original frame coordinates.

    `output` is (1, 4+nc, N): rows 0-3 are cx,cy,w,h in letterbox pixels, the rest are
    per-class scores (already sigmoid-ed in the v8 export).
    """
    expected_rows = 4 + len(CLASS_NAMES)
    if output.shape[:2] != (1, expected_rows):
        raise ValueError(
            f"unexpected detector output shape {output.shape}; "
            f"expected (1, {expected_rows}, N) — wrong weights for this class list?"
        )
    arr = output[0].T  # (N, 4+nc)
    if arr.shape[0] == 0:
        return ()
    cls_scores = arr[:, 4:]
    classes = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(classes)), classes]
    keep = scores >= conf_threshold
    if not keep.any():
        return ()
    boxes, scores, classes = arr[keep, :4], scores[keep], classes[keep]

    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    kept = nms_per_class(xyxy, scores, classes, iou_threshold)

    frame_h, frame_w = frame_shape
    dw, dh = pad
    detections = []
    for i in kept:
        x1 = max(0, round((xyxy[i, 0] - dw) / ratio))
        y1 = max(0, round((xyxy[i, 1] - dh) / ratio))
        x2 = min(frame_w, round((xyxy[i, 2] - dw) / ratio))
        y2 = min(frame_h, round((xyxy[i, 3] - dh) / ratio))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                label=CLASS_NAMES[int(classes[i])],
                bbox=(x1, y1, x2 - x1, y2 - y1),
                conf=float(scores[i]),
            )
        )
    return tuple(detections)


class OnnxYoloDetector:
    """Fine-tuned YOLOv8 weights via onnxruntime, CPU-only (zero network egress)."""

    def __init__(
        self,
        onnx_path: Path,
        imgsz: int | None = None,
        conf_threshold: float = ONNX_CONF_THRESHOLD,
        iou_threshold: float = ONNX_IOU_THRESHOLD,
    ) -> None:
        import onnxruntime  # deferred: template-matcher users don't pay for it

        self._session = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        # Exports are static-shape (1, 3, S, S): default to the model's own size so a
        # weights swap (1280 -> 640 retrain) can never silently mismatch the letterbox.
        if imgsz is None:
            dim = model_input.shape[2]
            if not isinstance(dim, int):  # dynamic axes report symbolic names/None
                raise ValueError(
                    f"{onnx_path} has a dynamic input shape ({dim!r}) — export with "
                    "static shapes (the notebook does) or pass imgsz explicitly"
                )
            imgsz = dim
        self._imgsz = imgsz
        self._conf = conf_threshold
        self._iou = iou_threshold

    @property
    def imgsz(self) -> int:
        return self._imgsz

    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection, ...]:
        padded, ratio, pad = letterbox(frame_bgr, self._imgsz)
        blob = padded[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        (output,) = self._session.run(None, {self._input_name: np.ascontiguousarray(blob)})
        return decode_predictions(output, ratio, pad, frame_bgr.shape[:2], self._conf, self._iou)
