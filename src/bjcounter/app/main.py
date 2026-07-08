"""Entrypoint: DPI awareness FIRST, then wiring per ARCHITECTURE §7.

    python -m bjcounter.app.main [--relocate] [--template] [--region X,Y,W,H]

First run auto-locates the trainer table on screen (same approach as
scripts/capture_session.py) and persists region+scale to config.json (gitignored).
Shutdown: quit hotkey -> QUIT job -> worker drains, closes mss, forwards QUIT to the
UI queue -> overlay pump stops hotkeys, joins the worker, destroys the root.
"""

from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

from bjcounter.app.config import (
    CONFIG_PATH,
    COUNT_BAR_CSS_PX,
    AppConfig,
    load_config,
    save_config,
)
from bjcounter.app.worker import CAPTURE, QUIT, RESET, Worker
from bjcounter.vision.capture import Grabber, find_table, set_dpi_awareness

TABLE_PNG = Path("data/assets/table.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relocate", action="store_true", help="redo table location")
    parser.add_argument(
        "--template", action="store_true", help="use the template-match detector instead of ONNX"
    )
    parser.add_argument("--region", type=str, default=None, help="manual X,Y,W,H")
    return parser.parse_args()


def locate_region() -> tuple[tuple[int, int, int, int], float]:
    """One in-memory virtual-screen grab to find the felt; never written to disk."""
    if not TABLE_PNG.exists():
        raise SystemExit(
            f"missing {TABLE_PNG} for auto-location — see data/assets/README.md, "
            "or pass --region X,Y,W,H"
        )
    import cv2

    grabber = Grabber()
    try:
        print("Locating the trainer table (bring it on screen)...")
        screen, (off_x, off_y) = grabber.virtual_screen()
        match = find_table(screen, cv2.imread(str(TABLE_PNG)))
        x, y, w, h = match.region
        bar = round(COUNT_BAR_CSS_PX * match.scale)
        region = (x + off_x, max(y + off_y - bar, 0), w, h + bar)
        print(f"Found region={region} scale={match.scale:.2f} score={match.score:.3f}")
        if match.score < 0.45:
            print("WARNING: low match confidence — overlay warnings will show if wrong.")
        return region, match.scale
    finally:
        grabber.close()


def parse_region(raw: str) -> tuple[tuple[int, int, int, int], float]:
    try:
        x, y, w, h = (int(v) for v in raw.split(","))
    except ValueError:
        raise SystemExit(f"--region must be four integers X,Y,W,H, got {raw!r}") from None
    if w <= 0 or h <= 0:
        raise SystemExit(f"--region needs positive W,H, got {raw!r}")
    return (x, y, w, h), w / 960


def build_config(args: argparse.Namespace) -> AppConfig:
    config = None if args.relocate else load_config()
    if config is None:
        if args.region:
            region, scale = parse_region(args.region)
        else:
            region, scale = locate_region()
        config = AppConfig(region=region, scale=scale)
        save_config(config)
        print(f"config saved to {CONFIG_PATH} (delete it or --relocate to redo)")
    if args.template:
        from dataclasses import replace

        config = replace(config, detector="template")
    return config


def main() -> None:
    set_dpi_awareness()  # must precede any mss/tkinter work
    args = parse_args()
    config = build_config(args)
    if config.detector == "onnx" and not Path(config.onnx_path).exists():
        raise SystemExit(
            f"{config.onnx_path} not found — run the Colab notebook (M4) or use --template"
        )

    jobs: queue.Queue = queue.Queue()
    ui: queue.Queue = queue.Queue()
    worker = Worker(config, jobs, ui)
    worker.start()
    # Everything after start() runs under this finally: the worker is non-daemon (it
    # must close mss cleanly), so ANY failure below — a bad hotkey combo string, a tk
    # init error — must still deliver a QUIT job or the blocked worker would keep the
    # process alive forever (review finding).
    try:
        from bjcounter.app.hotkeys import Hotkeys
        from bjcounter.app.overlay import Overlay

        hotkeys = Hotkeys(
            {
                config.hotkey_capture: lambda: jobs.put(CAPTURE),
                config.hotkey_reset: lambda: jobs.put(RESET),
                config.hotkey_quit: lambda: jobs.put(QUIT),
            }
        )

        def on_quit() -> None:
            hotkeys.stop()
            worker.join(timeout=3)

        overlay = Overlay(ui, on_quit=on_quit)
        hotkeys.start()
        print(
            f"bjcounter active — {config.hotkey_capture}=capture  "
            f"{config.hotkey_reset}=reset count  {config.hotkey_quit}=quit"
        )
        overlay.run()
    finally:
        jobs.put(QUIT)  # idempotent: unblocks the worker on every exit path
        worker.join(timeout=3)
    sys.exit(0)


if __name__ == "__main__":
    main()
