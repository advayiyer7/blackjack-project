"""CardDetector protocol + the template-matching fallback (ARCHITECTURE §2).

The protocol keeps the model swappable: M5's tracker development and replay tests run
against TemplateMatchDetector before the M4 YOLO weights exist; OnnxYoloDetector joins
at M4 implementing the same protocol. Template matching runs ~100-300 ms/frame — over
the 100 ms budget, accepted for development only (ARCHITECTURE §12); the latency gate
applies to the ONNX detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

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

    def __init__(
        self, deck_png: Path, scale: float, threshold: float = MATCH_THRESHOLD
    ) -> None:
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
