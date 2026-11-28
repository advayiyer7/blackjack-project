"""A worker startup failure must make the CLI exit unsuccessfully."""

from __future__ import annotations

import argparse

import pytest

from bjcounter.app import main as app_main
from bjcounter.app.config import AppConfig


def test_main_reports_worker_startup_failure(monkeypatch):
    class FailedWorker:
        def __init__(self, *_args):
            self.startup_error = None

        def start(self):
            pass

        def join(self, timeout=None):
            if timeout is None:
                self.startup_error = "screen unavailable"

    class FakeHotkeys:
        def __init__(self, _bindings):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class FakeOverlay:
        def __init__(self, _ui, on_quit):
            pass

        def run(self):
            pass

    monkeypatch.setattr(app_main, "set_dpi_awareness", lambda: None)
    monkeypatch.setattr(app_main, "parse_args", lambda: argparse.Namespace())
    monkeypatch.setattr(
        app_main,
        "build_config",
        lambda _args: AppConfig(region=(0, 0, 960, 676), scale=1.0, detector="template"),
    )
    monkeypatch.setattr(app_main, "Worker", FailedWorker)
    monkeypatch.setattr("bjcounter.app.hotkeys.Hotkeys", FakeHotkeys)
    monkeypatch.setattr("bjcounter.app.overlay.Overlay", FakeOverlay)

    with pytest.raises(SystemExit) as exc:
        app_main.main()

    assert exc.value.code != 0
