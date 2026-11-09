"""The worker's 'never die silently' branch — runs without assets or weights."""

from __future__ import annotations

import queue
import sys
from unittest.mock import patch

import numpy as np

from bjcounter.app.config import AppConfig
from bjcounter.app.worker import CAPTURE, QUIT, RESET, Worker
from bjcounter.types import OverlayModel
from tests.app.test_worker_thread import FakeGrabber


class ExplodingDetector:
    def detect(self, frame_bgr):
        raise RuntimeError("boom")


def test_worker_survives_a_detector_error_and_reports_it(capsys):
    """A raising detector produces a visible overlay warning, a stderr traceback, and
    the loop keeps serving jobs (a dead worker would silently freeze the count)."""
    FakeGrabber.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    config = AppConfig(region=(0, 0, 100, 100), scale=1.0)
    jobs: queue.Queue = queue.Queue()
    ui: queue.Queue = queue.Queue()
    with (
        patch("bjcounter.vision.capture.Grabber", FakeGrabber),
        patch.object(Worker, "_build_detector", lambda self: ExplodingDetector()),
    ):
        worker = Worker(config, jobs, ui)
        worker.start()
        assert isinstance(ui.get(timeout=10), OverlayModel)  # startup status

        jobs.put(CAPTURE)
        model = ui.get(timeout=10)
        assert any("worker error" in w for w in model.warnings)
        assert worker.is_alive()

        jobs.put(RESET)  # the loop still serves jobs after the error
        assert ui.get(timeout=10).running_count == 0

        jobs.put(QUIT)
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert ui.get(timeout=5) == QUIT
    assert "RuntimeError: boom" in capsys.readouterr().err


def test_worker_logs_grabber_startup_failure_and_stops_ui(capsys):
    config = AppConfig(region=(0, 0, 100, 100), scale=1.0)
    jobs: queue.Queue = queue.Queue()
    ui: queue.Queue = queue.Queue()

    with patch("bjcounter.vision.capture.Grabber", side_effect=RuntimeError("screen unavailable")):
        worker = Worker(config, jobs, ui)
        worker.start()
        assert ui.get(timeout=10) == QUIT
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert ui.empty()
    assert "RuntimeError: screen unavailable" in capsys.readouterr().err


def test_worker_stops_ui_when_capture_import_fails(capsys):
    config = AppConfig(region=(0, 0, 100, 100), scale=1.0)
    jobs: queue.Queue = queue.Queue()
    ui: queue.Queue = queue.Queue()

    with patch.dict(sys.modules, {"bjcounter.vision.capture": None}):
        worker = Worker(config, jobs, ui)
        worker.start()
        assert ui.get(timeout=10) == QUIT
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert ui.empty()
    assert "ModuleNotFoundError" in capsys.readouterr().err
