"""Check the M4 gates for the trained ONNX detector on held-out REAL frames.

Usage: python scripts/eval_detector.py [--onnx models/best.onnx] [--split test]
                                       [--imgsz 1280] [--conf 0.80]

Runs the project's own pre/post-processing (vision/detector.OnnxYoloDetector) — the
Colab numbers are indicative, these are authoritative. Gates (BUILD-GUIDE M4):

    mAP50 >= 0.99        per-rank recall >= 0.995 (at --conf)        < 100 ms/frame CPU

Writes models/metrics.json and exits non-zero if any gate fails. Ground truth comes
from the auto-labeler detections (same source the dataset export used).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2

from bjcounter.types import BBox, Detection
from bjcounter.vision.autolabel import CLASS_NAMES
from bjcounter.vision.dataset import hit_to_bbox, iter_real_frames
from bjcounter.vision.detector import OnnxYoloDetector, _iou

REPO = Path(__file__).resolve().parents[1]
GATE_MAP50 = 0.99
GATE_RANK_RECALL = 0.995
GATE_LATENCY_MS = 100.0
SWEEP_CONF = 0.001  # PR-curve floor for AP; the recall gate uses the operating --conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=REPO / "models" / "best.onnx")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.80, help="operating threshold")
    return parser.parse_args()


def to_xyxy(bbox: BBox) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def match_frame(
    gt: list[tuple[str, BBox]], preds: tuple[Detection, ...]
) -> tuple[list[tuple[float, bool, str]], dict[str, int]]:
    """Greedy IoU>=0.5 same-label matching, best-confidence first.

    Returns ([(conf, is_tp, label) per prediction], {label: n_missed_gt}).
    """
    unmatched = {i: (label, to_xyxy(bbox)) for i, (label, bbox) in enumerate(gt)}
    records = []
    for det in sorted(preds, key=lambda d: -d.conf):
        best_i, best_iou = None, 0.5
        det_box = to_xyxy(det.bbox)
        for i, (label, box) in unmatched.items():
            if label != det.label:
                continue
            iou = _iou(det_box, box)
            if iou >= best_iou:
                best_i, best_iou = i, iou
        if best_i is not None:
            del unmatched[best_i]
            records.append((det.conf, True, det.label))
        else:
            records.append((det.conf, False, det.label))
    missed: dict[str, int] = defaultdict(int)
    for label, _ in unmatched.values():
        missed[label] += 1
    return records, missed


def average_precision(records: list[tuple[float, bool]], n_gt: int) -> float:
    """VOC-style AP with precision envelope over the ranked prediction list."""
    if n_gt == 0:
        return float("nan")
    ranked = sorted(records, key=lambda r: -r[0])
    tp = fp = 0
    points = []
    for _, is_tp in ranked:
        tp, fp = tp + is_tp, fp + (not is_tp)
        points.append((tp / n_gt, tp / (tp + fp)))  # (recall, precision)
    ap = 0.0
    prev_recall = 0.0
    for i, (recall, _) in enumerate(points):
        envelope = max(p for _, p in points[i:])
        ap += (recall - prev_recall) * envelope
        prev_recall = recall
    return ap


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise SystemExit(f"{args.onnx} not found — run the Colab notebook first "
                         "(notebooks/train_yolo.ipynb) and unzip the weights into models/")
    frames = [f for f in iter_real_frames(REPO / "data" / "raw") if f.split == args.split]
    if not frames:
        raise SystemExit("no labeled real frames — run scripts/dataset_report.py first")

    # Two detector instances on purpose: AP needs a low-threshold sweep, but the
    # latency gate must measure the DEPLOYED configuration — at 0.001 conf the pure-
    # Python NMS chews through hundreds of background anchors and would report a
    # spurious latency FAIL that has nothing to do with real inference cost.
    sweeper = OnnxYoloDetector(args.onnx, imgsz=args.imgsz, conf_threshold=SWEEP_CONF)
    deployed = OnnxYoloDetector(args.onnx, imgsz=args.imgsz, conf_threshold=args.conf)
    per_class_records: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    per_class_gt: dict[str, int] = defaultdict(int)
    rank_tp: dict[str, int] = defaultdict(int)
    rank_fn: dict[str, int] = defaultdict(int)
    latencies = []

    for frame in frames:
        image = cv2.imread(str(frame.path))
        preds = sweeper.detect(image)
        start = time.perf_counter()
        deployed.detect(image)
        latencies.append((time.perf_counter() - start) * 1000)

        gt = [
            (CLASS_NAMES[hit[0]], hit_to_bbox(hit, frame.scale, frame.frame_w, frame.frame_h))
            for hit in frame.hits
        ]
        for label, _ in gt:
            per_class_gt[label] += 1

        records, _ = match_frame(gt, preds)
        for conf, is_tp, label in records:
            per_class_records[label].append((conf, is_tp))

        # Recall at the operating threshold, aggregated by rank (count uses rank only).
        op_records, op_missed = match_frame(gt, tuple(p for p in preds if p.conf >= args.conf))
        rank = lambda label: label if label == "back" else label[0]  # noqa: E731
        for _conf, is_tp, label in op_records:
            if is_tp:
                rank_tp[rank(label)] += 1
        for label, n in op_missed.items():
            rank_fn[rank(label)] += n

    aps = {
        label: average_precision(per_class_records[label], n_gt)
        for label, n_gt in per_class_gt.items()
    }
    map50 = sum(aps.values()) / len(aps)
    # Predictions for classes with zero GT in this split can't be scored by AP —
    # surface them instead of silently ignoring hallucinated classes.
    unscored_fp = {
        label: sum(1 for conf, _ in recs if conf >= args.conf)
        for label, recs in per_class_records.items()
        if label not in per_class_gt
    }
    unscored_fp = {k: v for k, v in unscored_fp.items() if v}
    rank_recall = {
        r: rank_tp[r] / (rank_tp[r] + rank_fn[r])
        for r in sorted(set(rank_tp) | set(rank_fn))
    }
    latency_ms = statistics.median(latencies)

    gates = {
        "map50": map50 >= GATE_MAP50,
        "per_rank_recall": bool(rank_recall) and min(rank_recall.values()) >= GATE_RANK_RECALL,
        "latency": latency_ms < GATE_LATENCY_MS,
    }
    metrics = {
        "onnx": args.onnx.name,
        "split": args.split,
        "frames": len(frames),
        "imgsz": args.imgsz,
        "operating_conf": args.conf,
        "map50": round(map50, 5),
        "per_class_ap50": {k: round(v, 5) for k, v in sorted(aps.items())},
        "per_rank_recall_at_conf": {k: round(v, 5) for k, v in rank_recall.items()},
        "unscored_fp_for_absent_classes": unscored_fp,
        "latency_ms_median_cpu": round(latency_ms, 1),
        "gates": gates,
    }
    (REPO / "models" / "metrics.json").write_text(
        json.dumps(metrics, indent=1), encoding="utf-8"
    )

    print(f"\n{args.split} frames: {len(frames)}  |  imgsz {args.imgsz}  |  conf {args.conf}")
    print(f"mAP50                {map50:.4f}  (gate >= {GATE_MAP50})  "
          f"{'PASS' if gates['map50'] else 'FAIL'}")
    worst = min(rank_recall.items(), key=lambda kv: kv[1]) if rank_recall else ("-", 0.0)
    print(f"per-rank recall min  {worst[1]:.4f} ({worst[0]})  (gate >= {GATE_RANK_RECALL})  "
          f"{'PASS' if gates['per_rank_recall'] else 'FAIL'}")
    print(f"latency median       {latency_ms:.0f} ms  (gate < {GATE_LATENCY_MS:.0f})  "
          f"{'PASS' if gates['latency'] else 'FAIL'}")
    if unscored_fp:
        print(f"WARNING: confident detections of classes absent from this split "
              f"(unscored by mAP): {unscored_fp}")
    print("metrics -> models/metrics.json")
    if not all(gates.values()):
        if not gates["latency"]:
            print("hint: try the 960 export — python scripts/eval_detector.py "
                  "--onnx models/best_960.onnx --imgsz 960")
        raise SystemExit(1)
    print("\nAll M4 gates PASS.")


if __name__ == "__main__":
    main()
