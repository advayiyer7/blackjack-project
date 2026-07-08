"""Worker thread lifecycle: a mocked screen feeds a composited synthetic frame
through the real detector, tracker, and count — verifying the queue plumbing and the
QUIT shutdown ordering without a display or the live trainer.

Needs the trainer assets and the trained ONNX weights; skips cleanly without them.
"""

from __future__ import annotations

import queue
from pathlib import Path
from unittest.mock import patch

import pytest

from bjcounter.types import Event, OverlayModel

REPO = Path(__file__).resolve().parents[2]
DECK_PNG = REPO / "data" / "assets" / "deck.png"
TABLE_PNG = REPO / "data" / "assets" / "table.png"
ONNX = REPO / "models" / "best.onnx"

pytestmark = pytest.mark.skipif(
    not (DECK_PNG.exists() and TABLE_PNG.exists() and ONNX.exists()),
    reason="needs trainer assets + trained weights",
)


class FakeGrabber:
    """Stands in for mss: always 'sees' one composited deal frame."""

    frame = None

    def __init__(self) -> None:
        pass

    def grab(self, region):
        return FakeGrabber.frame

    def close(self) -> None:
        pass


def test_worker_processes_a_deal_and_shuts_down_cleanly():
    import cv2

    from bjcounter.app.config import AppConfig
    from bjcounter.app.worker import CAPTURE, QUIT, RESET, Worker
    from bjcounter.vision.autolabel import BACK_CLASS, CLASS_NAMES
    from bjcounter.vision.synthesize import COUNT_BAR_CSS_PX, FrameSpec, render

    scale = 1.12
    spec = FrameSpec(
        dealer=(CLASS_NAMES.index("Th"), BACK_CLASS),
        hands=((CLASS_NAMES.index("9s"), CLASS_NAMES.index("8h")),),
        scale=scale,
    )
    frame, _ = render(spec, cv2.imread(str(TABLE_PNG)), cv2.imread(str(DECK_PNG)))
    FakeGrabber.frame = frame

    config = AppConfig(
        region=(0, 0, frame.shape[1], frame.shape[0]),
        scale=scale,
        onnx_path=str(ONNX),
    )
    assert config.table_origin == (0, round(COUNT_BAR_CSS_PX * scale))

    jobs: queue.Queue = queue.Queue()
    ui: queue.Queue = queue.Queue()
    with patch("bjcounter.vision.capture.Grabber", FakeGrabber):
        worker = Worker(config, jobs, ui)
        worker.start()
        startup = ui.get(timeout=30)  # detector load can take a moment
        assert isinstance(startup, OverlayModel)

        jobs.put(CAPTURE)
        model = ui.get(timeout=30)
        assert isinstance(model, OverlayModel)
        assert Event.NEW_ROUND in model.events, model.warnings
        assert model.running_count == -1  # 9,8 tag 0; T tags -1
        assert model.per_hand and model.per_hand[0][1].action is not None

        jobs.put(RESET)
        reset_model = ui.get(timeout=10)
        assert reset_model.running_count == 0

        jobs.put(QUIT)
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert ui.get(timeout=5) == QUIT  # forwarded sentinel for the overlay pump
