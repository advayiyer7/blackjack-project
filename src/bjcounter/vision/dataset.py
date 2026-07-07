"""Real-capture dataset helpers shared by the M4 export and eval scripts.

The auto-labeler's per-session detections.json (written by scripts/dataset_report.py)
is the label source for real frames. Real captures form the VAL/TEST pools only —
training runs on the synthetic pool (ARCHITECTURE §14 M3 amendment). The val/test
split is deterministic: within each session's valid frames, even index -> val, odd ->
test, so both pools cover every session's capture scale.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from bjcounter.types import BBox
from bjcounter.vision.autolabel import CARD_H, CARD_W

Hit = tuple[int, int, int, float]  # class_id, x, y, score — detections.json entry


@dataclass(frozen=True, slots=True)
class RealFrame:
    session: str
    name: str  # e.g. "frame_00042.png"
    path: Path
    scale: float
    frame_w: int
    frame_h: int
    hits: tuple[Hit, ...]
    split: str  # "val" | "test"

    @property
    def export_name(self) -> str:
        return f"{self.session}_{self.name}"


def hit_to_bbox(hit: Hit, scale: float, frame_w: int, frame_h: int) -> BBox:
    """Full-card bbox for a corner hit, clipped at frame edges (autolabel convention)."""
    _, x, y, _ = hit
    w = min(round(CARD_W * scale), frame_w - x)
    h = min(round(CARD_H * scale), frame_h - y)
    return (x, y, w, h)


def yolo_line(hit: Hit, scale: float, frame_w: int, frame_h: int) -> str:
    class_id = hit[0]
    x, y, w, h = hit_to_bbox(hit, scale, frame_w, frame_h)
    cx, cy = (x + w / 2) / frame_w, (y + h / 2) / frame_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w / frame_w:.6f} {h / frame_h:.6f}"


def iter_real_frames(raw_dir: Path) -> Iterator[RealFrame]:
    """Every valid labeled frame across all analyzed capture sessions.

    Requires detections.json + session_meta.json per session (run
    scripts/dataset_report.py first). Frames the auto-labeler skipped as invalid are
    absent from detections.json and therefore excluded here too.
    """
    for session_dir in sorted(raw_dir.glob("session_*")):
        detections_path = session_dir / "detections.json"
        meta_path = session_dir / "session_meta.json"
        if not (detections_path.exists() and meta_path.exists()):
            continue
        report = json.loads(detections_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _, _, frame_w, frame_h = meta["region"]
        scale = float(report["scale"])
        for i, name in enumerate(sorted(report["detections"])):
            path = session_dir / name
            if not path.exists():
                continue
            yield RealFrame(
                session=session_dir.name,
                name=name,
                path=path,
                scale=scale,
                frame_w=int(frame_w),
                frame_h=int(frame_h),
                hits=tuple(tuple(h) for h in report["detections"][name]),
                split="val" if i % 2 == 0 else "test",
            )
