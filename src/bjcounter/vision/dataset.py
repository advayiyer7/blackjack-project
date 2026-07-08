"""Real-capture dataset helpers shared by the M4 export and eval scripts.

The auto-labeler's per-session detections.json (written by scripts/dataset_report.py)
is the label source for real frames. Real captures form the VAL/TEST pools only —
training runs on the synthetic pool (ARCHITECTURE §14 M3 amendment). The val/test
split is deterministic: within each session's valid frames, even index -> val, odd ->
test, so both pools cover every session's capture scale.

Curation: a session may carry an `exclude.json` ({"frame_NNNNN.png": "reason", ...})
listing frames verified as mid-animation transients (cards captured mid-slide or
mid-flip render distorted sprites — ambiguous ground truth for a detection benchmark;
the runtime tracker quarantines such frames as SUSPECT anyway). Flagging candidates
is automated by scripts/audit_labels.py; a human confirms each exclusion visually.
Excluded frames leave the export AND the eval — the split parity of the remaining
frames is unchanged (parity is assigned before exclusion).
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
    excluded: str | None = None  # exclusion reason; only set with include_excluded=True

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


def iter_real_frames(raw_dir: Path, include_excluded: bool = False) -> Iterator[RealFrame]:
    """Every valid labeled frame across all analyzed capture sessions.

    Requires detections.json + session_meta.json per session (run
    scripts/dataset_report.py first). Frames the auto-labeler skipped as invalid are
    absent from detections.json and therefore excluded here too. Curated exclusions
    (exclude.json) are skipped unless `include_excluded`, which yields them with
    their reason set (audit tooling reviews past decisions this way).
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
        exclude_path = session_dir / "exclude.json"
        excluded = (
            json.loads(exclude_path.read_text(encoding="utf-8")) if exclude_path.exists() else {}
        )
        # NB: split parity comes from the position in the FULL frame list — an
        # exclusion never shifts the val/test assignment of surviving frames.
        for i, name in enumerate(sorted(report["detections"])):
            path = session_dir / name
            if not path.exists():
                continue
            if name in excluded and not include_excluded:
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
                excluded=excluded.get(name),
            )
