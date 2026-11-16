"""Capture failures must not be reported as saved frames or leak the grabber."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
from scripts import capture_session


class FakeGrabber:
    def __init__(self):
        self.closed = False

    def grab(self, region):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_failed_frame_write_stops_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_session, "START_COUNTDOWN_S", 0)
    monkeypatch.setattr(capture_session.cv2, "imwrite", lambda *_: False)

    def stop_after_one_iteration(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(capture_session.time, "sleep", stop_after_one_iteration)

    with pytest.raises(OSError, match="frame_00000.png"):
        capture_session.capture_loop(FakeGrabber(), (0, 0, 4, 4), tmp_path, 0.5)


def test_failed_preview_write_does_not_prompt_for_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_session.cv2, "imwrite", lambda *_: False)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("preview was not saved"))

    with pytest.raises(OSError, match="region_preview.png"):
        capture_session.confirm_preview(FakeGrabber(), (0, 0, 4, 4), tmp_path)


def test_main_closes_grabber_when_capture_fails(tmp_path, monkeypatch):
    grabber = FakeGrabber()
    monkeypatch.setattr(capture_session, "Grabber", lambda: grabber)
    monkeypatch.setattr(capture_session, "set_dpi_awareness", lambda: None)
    monkeypatch.setattr(
        capture_session,
        "parse_args",
        lambda: argparse.Namespace(region="0,0,960,676", out=tmp_path, interval=0.5),
    )
    monkeypatch.setattr(capture_session, "confirm_preview", lambda *_: True)
    monkeypatch.setattr(
        capture_session, "capture_loop", lambda *_: (_ for _ in ()).throw(OSError("disk full"))
    )

    with pytest.raises(OSError, match="disk full"):
        capture_session.main()

    assert grabber.closed
