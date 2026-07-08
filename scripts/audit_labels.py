"""Cross-model label audit: flag real frames where the YOLO detector and the
template-matcher ground truth disagree — the systematic form of M3's "hand-verify a
sample" step.

Usage: python scripts/audit_labels.py [--onnx models/best.onnx] [--imgsz N]

Flags, per frame (both splits, past exclusions shown so decisions stay reviewable):
  FP>=0.90   model is confident about a card the GT doesn't contain (label miss?)
  TP<0.80    model found a GT card but below the runtime threshold (distorted sprite?)
  MISS       a GT card the model cannot find even at conf 0.01

Writes crops for every flag to data/raw/<session>/audit_crops/ (gitignored) for human
review, and prints exclude.json-ready lines. A human confirms each exclusion — this
script never writes exclude.json itself. Mid-animation transients (cards captured
mid-slide/mid-flip) are the expected cause; see vision/dataset.py.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

from bjcounter.types import BBox
from bjcounter.vision.autolabel import CLASS_NAMES
from bjcounter.vision.dataset import hit_to_bbox, iter_real_frames
from bjcounter.vision.detector import OnnxYoloDetector
from bjcounter.vision.matching import match_frame

REPO = Path(__file__).resolve().parents[1]
SWEEP_CONF = 0.01
SURE_FP = 0.90
WEAK_TP = 0.80
CROP_PAD = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=REPO / "models" / "best.onnx")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="letterbox size; default = the model's own static input size",
    )
    return parser.parse_args()


def save_crop(image, bbox: BBox, out_dir: Path, tag: str) -> None:
    x, y, w, h = bbox
    crop = image[
        max(0, y - CROP_PAD) : y + h + CROP_PAD, max(0, x - CROP_PAD) : x + w + CROP_PAD
    ].copy()
    cv2.rectangle(
        crop,
        (min(CROP_PAD, x), min(CROP_PAD, y)),
        (min(CROP_PAD, x) + w, min(CROP_PAD, y) + h),
        (0, 0, 255),
        2,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{tag}.png"), crop)


def main() -> None:
    args = parse_args()
    detector = OnnxYoloDetector(args.onnx, imgsz=args.imgsz, conf_threshold=SWEEP_CONF)
    flagged: dict[str, list[str]] = defaultdict(list)

    for frame in iter_real_frames(REPO / "data" / "raw", include_excluded=True):
        image = cv2.imread(str(frame.path))
        preds = detector.detect(image)
        gt = [
            (CLASS_NAMES[h[0]], hit_to_bbox(h, frame.scale, frame.frame_w, frame.frame_h))
            for h in frame.hits
        ]
        records, missed = match_frame(gt, preds)
        audit_dir = frame.path.parent / "audit_crops"
        stem = frame.name[:-4]
        flags = []
        # match_frame iterates predictions confidence-descending; mirror that order
        # so records and detections line up one-to-one.
        for det, (conf, is_tp, label) in zip(
            sorted(preds, key=lambda d: -d.conf), records, strict=True
        ):
            if not is_tp and conf >= SURE_FP:
                flags.append(f"FP {label}@{conf:.2f}")
                save_crop(image, det.bbox, audit_dir, f"{stem}_FP_{label}_{conf:.2f}")
            elif is_tp and conf < WEAK_TP:
                flags.append(f"weak-TP {label}@{conf:.2f}")
                save_crop(image, det.bbox, audit_dir, f"{stem}_weakTP_{label}_{conf:.2f}")
        for label, n in missed.items():
            flags.append(f"MISS {label} x{n}")
        if flags:
            marker = " [already excluded]" if frame.excluded else ""
            flagged[frame.session].append(
                f"{frame.name} ({frame.split}){marker}: " + "; ".join(flags)
            )

    if not flagged:
        print("No disagreements — labels and detector agree everywhere.")
        return
    for session, lines in flagged.items():
        print(f"\n{session}  (crops -> data/raw/{session}/audit_crops/)")
        for line in lines:
            print(f"  {line}")
    print(
        "\nReview each crop; for confirmed transients add the frame to "
        'data/raw/<session>/exclude.json as {"frame_NNNNN.png": "reason"}.'
    )


if __name__ == "__main__":
    main()
