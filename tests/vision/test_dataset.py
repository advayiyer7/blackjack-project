"""dataset.py owns the val/test split and the ground-truth boxes both the M4 export
and the gate eval depend on — a silent regression here corrupts either the training
labels or the gate verdict, so it gets direct coverage (review finding)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bjcounter.vision.autolabel import CARD_H, CARD_W, CardHit, yolo_lines
from bjcounter.vision.dataset import hit_to_bbox, iter_real_frames, yolo_line


def write_session(root: Path, name: str, scale: float, frames: dict[str, list[list]]) -> None:
    session = root / name
    session.mkdir(parents=True)
    (session / "session_meta.json").write_text(
        json.dumps({"region": [10, 20, round(960 * scale), round(676 * scale)], "scale": scale}),
        encoding="utf-8",
    )
    (session / "detections.json").write_text(
        json.dumps({"scale": scale, "detections": frames}), encoding="utf-8"
    )
    for frame_name in frames:
        (session / frame_name).write_bytes(b"png-stub")


class TestIterRealFrames:
    def test_split_alternates_val_test_within_a_session(self, tmp_path):
        frames = {f"frame_{i:05d}.png": [] for i in range(5)}
        write_session(tmp_path, "session_a", 1.25, frames)
        got = list(iter_real_frames(tmp_path))
        assert [f.split for f in got] == ["val", "test", "val", "test", "val"]
        assert [f.name for f in got] == sorted(frames)  # deterministic order

    def test_sessions_without_labels_or_missing_images_are_skipped(self, tmp_path):
        write_session(tmp_path, "session_a", 1.0, {"frame_00000.png": []})
        (tmp_path / "session_bare").mkdir()  # no detections.json
        write_session(tmp_path, "session_c", 1.0, {"frame_00000.png": []})
        (tmp_path / "session_c" / "frame_00000.png").unlink()  # labeled but image gone
        got = list(iter_real_frames(tmp_path))
        assert [f.session for f in got] == ["session_a"]

    def test_frame_carries_meta_dims_scale_and_hits(self, tmp_path):
        hits = [[7, 100, 200, 0.97]]
        write_session(tmp_path, "session_a", 1.25, {"frame_00000.png": hits})
        (frame,) = iter_real_frames(tmp_path)
        assert (frame.frame_w, frame.frame_h) == (1200, 845)
        assert frame.scale == 1.25
        assert frame.hits == ((7, 100, 200, 0.97),)
        assert frame.export_name == "session_a_frame_00000.png"


class TestExcludeJson:
    """Curated transient exclusions must drop frames WITHOUT shifting the val/test
    parity of survivors — the gate's ground truth depends on this staying true."""

    def write_excluded_session(self, tmp_path):
        frames = {f"frame_{i:05d}.png": [] for i in range(5)}
        write_session(tmp_path, "session_a", 1.0, frames)
        (tmp_path / "session_a" / "exclude.json").write_text(
            json.dumps({"frame_00001.png": "mid-animation transient"}), encoding="utf-8"
        )

    def test_excluded_frame_is_dropped(self, tmp_path):
        self.write_excluded_session(tmp_path)
        names = [f.name for f in iter_real_frames(tmp_path)]
        assert "frame_00001.png" not in names
        assert len(names) == 4

    def test_survivors_keep_their_original_parity(self, tmp_path):
        self.write_excluded_session(tmp_path)
        splits = {f.name: f.split for f in iter_real_frames(tmp_path)}
        # frame_00002 stays "val" (index 2 in the FULL list) even though only one
        # frame precedes it after the exclusion.
        assert splits == {
            "frame_00000.png": "val",
            "frame_00002.png": "val",
            "frame_00003.png": "test",
            "frame_00004.png": "val",
        }

    def test_include_excluded_yields_them_with_the_reason(self, tmp_path):
        self.write_excluded_session(tmp_path)
        frames = {f.name: f for f in iter_real_frames(tmp_path, include_excluded=True)}
        assert len(frames) == 5
        assert frames["frame_00001.png"].excluded == "mid-animation transient"
        assert frames["frame_00000.png"].excluded is None


class TestGroundTruthBoxes:
    def test_full_card_bbox_from_corner_hit(self):
        assert hit_to_bbox((7, 100, 200, 0.97), 1.0, 1200, 845) == (100, 200, 67, 94)
        assert hit_to_bbox((7, 100, 200, 0.97), 1.25, 1200, 845) == (100, 200, 84, 118)

    def test_bbox_clips_at_frame_edges(self):
        x, y, w, h = hit_to_bbox((7, 1150, 800, 0.97), 1.25, 1200, 845)
        assert (x, y) == (1150, 800)
        assert (w, h) == (50, 45)  # clipped, not 84x118

    def test_yolo_line_matches_the_autolabel_convention(self):
        """Real-frame labels (export/eval) must agree with the labels synthetic
        training frames get via autolabel.yolo_lines — within the 1px rounding the
        two paths differ by."""
        scale, frame_w, frame_h = 1.25, 1200, 845
        hit = (13, 422, 110, 0.95)
        ours = yolo_line(hit, scale, frame_w, frame_h).split()
        theirs = yolo_lines([CardHit(13, 422, 110, 0.95)], frame_w, frame_h, scale)[0].split()
        assert ours[0] == theirs[0] == "13"
        for a, b, span in zip(ours[1:], theirs[1:], (frame_w, frame_h) * 2, strict=True):
            assert float(a) * span == pytest.approx(float(b) * span, abs=1.0)

    def test_yolo_line_is_normalized_and_in_bounds(self):
        line = yolo_line((52, 1150, 800, 0.9), 1.25, 1200, 845)
        _, cx, cy, w, h = (float(v) for v in line.split())
        assert 0 < cx < 1 and 0 < cy < 1 and 0 < w < 1 and 0 < h < 1
        assert cx + w / 2 <= 1.0001 and cy + h / 2 <= 1.0001

    def test_sprite_dims_are_the_trainer_constants(self):
        assert (CARD_W, CARD_H) == (67, 94)
