"""Matching + AP math behind the M4 gate verdict (vision/matching.py)."""

from __future__ import annotations

from bjcounter.types import Detection
from bjcounter.vision.matching import average_precision, iou_xyxy, match_frame, to_xyxy

CARD = (84, 118)


def det(label: str, x: int, y: int, conf: float) -> Detection:
    return Detection(label=label, bbox=(x, y, *CARD), conf=conf)


class TestMatchFrame:
    def test_perfect_predictions_all_match(self):
        gt = [("8s", (100, 100, *CARD)), ("Th", (300, 100, *CARD))]
        records, missed = match_frame(gt, (det("8s", 101, 101, 0.99), det("Th", 299, 100, 0.97)))
        assert all(is_tp for _, is_tp, _ in records)
        assert not missed

    def test_wrong_position_is_fp_and_gt_counts_as_missed(self):
        gt = [("Th", (300, 100, *CARD))]
        records, missed = match_frame(gt, (det("Th", 700, 500, 0.9),))
        assert records == [(0.9, False, "Th")]
        assert missed == {"Th": 1}

    def test_label_must_match_not_just_position(self):
        gt = [("8s", (100, 100, *CARD))]
        records, missed = match_frame(gt, (det("8h", 100, 100, 0.95),))
        assert records == [(0.95, False, "8h")]
        assert missed == {"8s": 1}

    def test_each_gt_matches_at_most_one_prediction(self):
        gt = [("8s", (100, 100, *CARD))]
        records, _ = match_frame(gt, (det("8s", 101, 101, 0.99), det("8s", 102, 100, 0.95)))
        assert records[0][1] and not records[1][1]  # duplicate box is an FP

    def test_records_come_back_confidence_descending(self):
        gt = [("8s", (100, 100, *CARD)), ("Th", (300, 100, *CARD))]
        records, _ = match_frame(gt, (det("Th", 300, 100, 0.5), det("8s", 100, 100, 0.9)))
        assert [conf for conf, _, _ in records] == [0.9, 0.5]


class TestAveragePrecision:
    def test_perfect_detector_scores_one(self):
        assert average_precision([(0.99, True), (0.97, True)], n_gt=2) == 1.0

    def test_all_false_positives_score_zero(self):
        assert average_precision([(0.9, False)], n_gt=1) == 0.0

    def test_fp_above_a_tp_costs_precision(self):
        # FP at 0.9 ranked above the TP at 0.8: envelope precision at full recall is 1/2.
        ap = average_precision([(0.9, False), (0.8, True)], n_gt=1)
        assert ap == 0.5

    def test_no_ground_truth_is_nan(self):
        import math

        assert math.isnan(average_precision([], n_gt=0))

    def test_input_list_is_not_mutated(self):
        records = [(0.5, True), (0.9, True)]
        average_precision(records, n_gt=2)
        assert records == [(0.5, True), (0.9, True)]


class TestGeometry:
    def test_iou_of_fan_neighbours_is_below_point_five(self):
        a = to_xyxy((100, 100, *CARD))
        b = to_xyxy((130, 100, *CARD))  # 24px fan step at scale 1.25 (84x118 cards)
        assert 0.4 < iou_xyxy(a, b) < 0.5

    def test_disjoint_boxes_have_zero_iou(self):
        assert iou_xyxy(to_xyxy((0, 0, 10, 10)), to_xyxy((100, 100, 10, 10))) == 0.0
