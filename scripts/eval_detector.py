"""Check the M4 gates for the trained ONNX detector on held-out REAL frames.

Usage: python scripts/eval_detector.py [--onnx models/best.onnx] [--split test]
                                       [--imgsz N] [--conf 0.80]

Runs the project's own pre/post-processing (vision/detector.OnnxYoloDetector) — the
Colab numbers are indicative, these are authoritative. Gates (BUILD-GUIDE M4):

    mAP50 >= 0.99        per-rank recall >= 0.995 (at --conf)        < 100 ms/frame CPU

Writes models/metrics.json and exits non-zero if any gate fails. Ground truth comes
from the auto-labeler detections, minus human-verified transient exclusions
(vision/dataset.py; audit with scripts/audit_labels.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2

from bjcounter.vision.autolabel import CLASS_NAMES
from bjcounter.vision.dataset import hit_to_bbox, iter_real_frames
from bjcounter.vision.detector import OnnxYoloDetector
from bjcounter.vision.matching import average_precision, match_frame

REPO = Path(__file__).resolve().parents[1]
GATE_MAP50 = 0.99
GATE_RANK_RECALL = 0.995
GATE_LATENCY_MS = 100.0
SWEEP_CONF = 0.001  # PR-curve floor for AP; the recall gate uses the operating --conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=REPO / "models" / "best.onnx")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="letterbox size; default = the model's own static input size",
    )
    parser.add_argument("--conf", type=float, default=0.80, help="operating threshold")
    return parser.parse_args()


def rank_of(label: str) -> str:
    return label if label == "back" else label[0]


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise SystemExit(
            f"{args.onnx} not found — run the Colab notebook first "
            "(notebooks/train_yolo.ipynb) and unzip the weights into models/"
        )
    frames = [f for f in iter_real_frames(REPO / "data" / "raw") if f.split == args.split]
    if not frames:
        raise SystemExit("no labeled real frames — run scripts/dataset_report.py first")

    # Two detector instances on purpose: AP needs a low-threshold sweep, but the
    # latency gate must measure the DEPLOYED configuration — at 0.001 conf the pure-
    # Python NMS chews through hundreds of background anchors and would report a
    # spurious latency FAIL that has nothing to do with real inference cost.
    sweeper = OnnxYoloDetector(args.onnx, imgsz=args.imgsz, conf_threshold=SWEEP_CONF)
    deployed = OnnxYoloDetector(args.onnx, imgsz=args.imgsz, conf_threshold=args.conf)
    imgsz = sweeper.imgsz  # resolved from the model when --imgsz omitted
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
        for _conf, is_tp, label in op_records:
            if is_tp:
                rank_tp[rank_of(label)] += 1
        for label, n in op_missed.items():
            rank_fn[rank_of(label)] += n

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
        r: rank_tp[r] / (rank_tp[r] + rank_fn[r]) for r in sorted(set(rank_tp) | set(rank_fn))
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
        "imgsz": imgsz,
        "operating_conf": args.conf,
        "map50": round(map50, 5),
        "per_class_ap50": {k: round(v, 5) for k, v in sorted(aps.items())},
        "per_rank_recall_at_conf": {k: round(v, 5) for k, v in rank_recall.items()},
        "unscored_fp_for_absent_classes": unscored_fp,
        "latency_ms_median_cpu": round(latency_ms, 1),
        "gates": gates,
    }
    (REPO / "models" / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    print(f"\n{args.split} frames: {len(frames)}  |  imgsz {imgsz}  |  conf {args.conf}")
    print(
        f"mAP50                {map50:.4f}  (gate >= {GATE_MAP50})  "
        f"{'PASS' if gates['map50'] else 'FAIL'}"
    )
    worst = min(rank_recall.items(), key=lambda kv: kv[1]) if rank_recall else ("-", 0.0)
    print(
        f"per-rank recall min  {worst[1]:.4f} ({worst[0]})  (gate >= {GATE_RANK_RECALL})  "
        f"{'PASS' if gates['per_rank_recall'] else 'FAIL'}"
    )
    print(
        f"latency median       {latency_ms:.0f} ms  (gate < {GATE_LATENCY_MS:.0f})  "
        f"{'PASS' if gates['latency'] else 'FAIL'}"
    )
    if unscored_fp:
        print(
            f"WARNING: confident detections of classes absent from this split "
            f"(unscored by mAP): {unscored_fp}"
        )
    print("metrics -> models/metrics.json")
    if not all(gates.values()):
        if not gates["latency"]:
            print(
                "hint: try a smaller export, e.g. python scripts/eval_detector.py "
                "--onnx models/best_800.onnx"
            )
        raise SystemExit(1)
    print("\nAll M4 gates PASS.")


if __name__ == "__main__":
    main()
