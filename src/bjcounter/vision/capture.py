"""Screen capture and table-region location. Effectful edge (ARCHITECTURE §2/§7).

Capture is region-scoped by design (security rule: screenshots stay local and narrow).
The one exception is a single in-memory virtual-screen grab used to LOCATE the table at
session start — it is never written to disk.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
from dataclasses import dataclass

import cv2
import mss
import numpy as np

from bjcounter.types import BBox

TABLE_W, TABLE_H = 960, 640  # trainer canvas at scale 1.0 (trainer-notes §4)


def set_dpi_awareness() -> None:
    """Per-monitor DPI awareness. MUST run first in main(), before mss/tkinter init
    (stack-notes §3 — the call only succeeds once per process)."""
    if sys.platform == "win32":
        with contextlib.suppress(OSError):  # OSError = already set for this process
            ctypes.windll.shcore.SetProcessDpiAwareness(2)


class Grabber:
    """Wraps one mss instance; create one per thread (mss is thread-affine)."""

    def __init__(self) -> None:
        self._sct = mss.mss()

    def grab(self, region: BBox) -> np.ndarray:
        """Region screenshot as BGR ndarray (drops mss's alpha channel)."""
        x, y, w, h = region
        shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        return np.asarray(shot)[:, :, :3]

    def virtual_screen(self) -> tuple[np.ndarray, tuple[int, int]]:
        """One full virtual-screen grab (all monitors) + its absolute (left, top) offset.

        Used only in-memory for table location; never persisted.
        """
        monitor = self._sct.monitors[0]
        shot = self._sct.grab(monitor)
        return np.asarray(shot)[:, :, :3], (monitor["left"], monitor["top"])

    def close(self) -> None:
        self._sct.close()


@dataclass(frozen=True, slots=True)
class TableMatch:
    region: BBox  # in the coordinate frame of the searched image
    scale: float
    score: float  # TM_CCOEFF_NORMED peak, 1.0 = perfect


def find_table(
    screen_bgr: np.ndarray,
    table_bgr: np.ndarray,
    coarse_scales: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(31)),
) -> TableMatch:
    """Locate the trainer's 960x640 felt in a screenshot via a two-pass scale sweep.

    Coarse pass at half resolution over `coarse_scales` (0.5..2.0), then a full-resolution
    refinement around the best scale. Robust to browser scale (0.9/0.74 CSS transforms)
    combined with Windows display scaling.
    """
    half_screen = cv2.resize(screen_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    coarse = _best_over_scales(half_screen, table_bgr, coarse_scales, pre_shrink=0.5)
    refine_scales = tuple(
        round(coarse.scale + 0.01 * i, 3) for i in range(-5, 6) if coarse.scale + 0.01 * i > 0.2
    )
    return _best_over_scales(screen_bgr, table_bgr, refine_scales, pre_shrink=1.0)


def _best_over_scales(
    screen: np.ndarray, table: np.ndarray, scales: tuple[float, ...], pre_shrink: float
) -> TableMatch:
    best: TableMatch | None = None
    for scale in scales:
        w = int(TABLE_W * scale * pre_shrink)
        h = int(TABLE_H * scale * pre_shrink)
        if w < 40 or h < 40 or w >= screen.shape[1] or h >= screen.shape[0]:
            continue
        template = cv2.resize(table, (w, h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if best is None or score > best.score:
            x, y = (int(loc[0] / pre_shrink), int(loc[1] / pre_shrink))
            full_w, full_h = int(TABLE_W * scale), int(TABLE_H * scale)
            best = TableMatch(region=(x, y, full_w, full_h), scale=scale, score=float(score))
    if best is None:
        raise ValueError("screen too small to contain the table at any searched scale")
    return best
