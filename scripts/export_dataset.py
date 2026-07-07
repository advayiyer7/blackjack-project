"""Package the YOLO dataset for the Colab M4 fine-tune into one uploadable zip.

Usage: python scripts/export_dataset.py

Assembles data/dataset_yolo.zip (ZIP_STORED — PNGs don't recompress) containing:

    dataset/data.yaml
    dataset/images/train/*.png   + labels/train/*.txt   (synthetic pool, all of it)
    dataset/images/val/*.png     + labels/val/*.txt     (real captures, even frames)
    dataset/images/test/*.png    + labels/test/*.txt    (real captures, odd frames)

Prereqs: scripts/make_synthetic.py (train pool) and scripts/dataset_report.py
(detections.json per session) have been run. The zip is gitignored (regenerable,
contains third-party-art-derived images).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from bjcounter.vision.autolabel import CLASS_NAMES
from bjcounter.vision.dataset import iter_real_frames, yolo_line

REPO = Path(__file__).resolve().parents[1]
SYNTHETIC = REPO / "data" / "synthetic"
RAW = REPO / "data" / "raw"
OUT_ZIP = REPO / "data" / "dataset_yolo.zip"
COLAB_ROOT = "/content/dataset"  # keep in sync with notebooks/train_yolo.ipynb


def data_yaml() -> str:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    return (
        f"path: {COLAB_ROOT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )


def main() -> None:
    synth_images = sorted((SYNTHETIC / "images").glob("frame_*.png"))
    if not synth_images:
        raise SystemExit("no synthetic pool — run scripts/make_synthetic.py first")
    real_frames = list(iter_real_frames(RAW))
    if not real_frames:
        raise SystemExit("no labeled real frames — run scripts/dataset_report.py first")

    counts = {"train": 0, "val": 0, "test": 0}
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("dataset/data.yaml", data_yaml())

        for image in synth_images:
            label = SYNTHETIC / "labels" / image.with_suffix(".txt").name
            zf.write(image, f"dataset/images/train/{image.name}")
            zf.write(label, f"dataset/labels/train/{label.name}")
            counts["train"] += 1

        for frame in real_frames:
            lines = [
                yolo_line(hit, frame.scale, frame.frame_w, frame.frame_h)
                for hit in frame.hits
            ]
            stem = Path(frame.export_name).stem
            zf.write(frame.path, f"dataset/images/{frame.split}/{frame.export_name}")
            zf.writestr(
                f"dataset/labels/{frame.split}/{stem}.txt",
                "\n".join(lines) + ("\n" if lines else ""),
            )
            counts[frame.split] += 1

    size_mb = OUT_ZIP.stat().st_size / 2**20
    print(f"{OUT_ZIP} ({size_mb:.0f} MB)")
    print(f"frames: train={counts['train']} (synthetic), "
          f"val={counts['val']}, test={counts['test']} (real captures)")
    print("Next: upload the zip in notebooks/train_yolo.ipynb on Colab (T4 runtime).")


if __name__ == "__main__":
    main()
