"""Detection-vs-ground-truth matching and AP math shared by the M4 eval and the
label audit (scripts/eval_detector.py, scripts/audit_labels.py)."""

from __future__ import annotations

from collections import defaultdict

from bjcounter.types import BBox, Detection

MATCH_IOU = 0.5


def to_xyxy(bbox: BBox) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def match_frame(
    gt: list[tuple[str, BBox]], preds: tuple[Detection, ...]
) -> tuple[list[tuple[float, bool, str]], dict[str, int]]:
    """Greedy IoU>=0.5 same-label matching, best-confidence first.

    Returns ([(conf, is_tp, label) per prediction, confidence-descending],
    {label: n_missed_gt}). Each ground-truth box matches at most one prediction.
    """
    unmatched = {i: (label, to_xyxy(bbox)) for i, (label, bbox) in enumerate(gt)}
    records = []
    for det in sorted(preds, key=lambda d: -d.conf):
        best_i, best_iou = None, MATCH_IOU
        det_box = to_xyxy(det.bbox)
        for i, (label, box) in unmatched.items():
            if label != det.label:
                continue
            iou = iou_xyxy(det_box, box)
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
