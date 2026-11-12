"""A failed dataset export must leave the last complete Colab archive usable."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest
from scripts import export_dataset

from bjcounter.vision.dataset import RealFrame


def setup_export(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    synthetic = tmp_path / "synthetic"
    image = synthetic / "images" / "frame_00000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    real_image = tmp_path / "raw" / "session_a" / "frame_00000.png"
    real_image.parent.mkdir(parents=True)
    real_image.write_bytes(b"real image")
    frame = RealFrame(
        session="session_a",
        name=real_image.name,
        path=real_image,
        scale=1.0,
        frame_w=960,
        frame_h=676,
        hits=(),
        split="val",
    )
    output = tmp_path / "dataset_yolo.zip"
    output.write_bytes(b"last complete archive")
    monkeypatch.setattr(export_dataset, "SYNTHETIC", synthetic)
    monkeypatch.setattr(export_dataset, "RAW", tmp_path / "raw")
    monkeypatch.setattr(export_dataset, "OUT_ZIP", output)
    monkeypatch.setattr(export_dataset, "iter_real_frames", lambda _: iter((frame,)))
    return synthetic, output


def test_missing_synthetic_label_preserves_previous_archive(tmp_path, monkeypatch):
    _, output = setup_export(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match="missing synthetic label"):
        export_dataset.main()

    assert output.read_bytes() == b"last complete archive"


def test_mid_export_error_preserves_previous_archive(tmp_path, monkeypatch):
    synthetic, output = setup_export(tmp_path, monkeypatch)
    label = synthetic / "labels" / "frame_00000.txt"
    label.parent.mkdir()
    label.write_text("", encoding="utf-8")
    monkeypatch.setattr(export_dataset, "yolo_line", lambda *args: 1 / 0)
    frame = next(export_dataset.iter_real_frames(export_dataset.RAW))
    monkeypatch.setattr(
        export_dataset,
        "iter_real_frames",
        lambda _: iter((replace(frame, hits=((0, 0, 0, 1.0),)),)),
    )

    with pytest.raises(ZeroDivisionError):
        export_dataset.main()

    assert output.read_bytes() == b"last complete archive"


def test_complete_export_replaces_archive(tmp_path, monkeypatch):
    synthetic, output = setup_export(tmp_path, monkeypatch)
    label = synthetic / "labels" / "frame_00000.txt"
    label.parent.mkdir()
    label.write_text("", encoding="utf-8")

    export_dataset.main()

    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "dataset/data.yaml",
            "dataset/images/train/frame_00000.png",
            "dataset/labels/train/frame_00000.txt",
            "dataset/images/val/session_a_frame_00000.png",
            "dataset/labels/val/session_a_frame_00000.txt",
        }
