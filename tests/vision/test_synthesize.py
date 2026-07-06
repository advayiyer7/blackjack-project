"""Compositor tests: geometry and planning run on fake assets; a round-trip test
against the real trainer sprites (data/assets/, gitignored) is skipped when absent.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from bjcounter.vision.autolabel import BACK_CLASS, CLASS_NAMES, detect_cards, load_templates
from bjcounter.vision.synthesize import (
    COUNT_BAR_CSS_PX,
    DEALER_ANCHOR,
    FAN_DX,
    PLAYER_ANCHORS,
    PLAYER_FAN_DY,
    TABLE_H,
    TABLE_W,
    FrameSpec,
    placements,
    plan_frames,
    render,
)

REPO = Path(__file__).resolve().parents[2]
DECK_PNG = REPO / "data" / "assets" / "deck.png"
TABLE_PNG = REPO / "data" / "assets" / "table.png"


@pytest.fixture(scope="module")
def fake_deck() -> np.ndarray:
    """Deterministic high-texture stand-in for deck.png (67x5452)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(5452, 67, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def fake_table() -> np.ndarray:
    """Solid felt-green 960x640 stand-in for table.png."""
    table = np.zeros((TABLE_H, TABLE_W, 3), dtype=np.uint8)
    table[:] = (40, 110, 30)  # BGR green, passes is_table_frame
    return table


def spec(dealer=(0, BACK_CLASS), hands=((13, 26),), scale=1.0) -> FrameSpec:
    return FrameSpec(dealer=tuple(dealer), hands=tuple(tuple(h) for h in hands), scale=scale)


class TestPlacements:
    def test_dealer_fan_steps_right_from_anchor(self):
        got = placements(spec(dealer=(0, BACK_CLASS, 5), hands=((13, 26),)))
        dealer = got[:3]
        ax, ay = DEALER_ANCHOR
        assert [(p.x, p.y) for p in dealer] == [(ax, ay), (ax + FAN_DX, ay), (ax + 2 * FAN_DX, ay)]
        assert [p.class_id for p in dealer] == [0, BACK_CLASS, 5]

    @pytest.mark.parametrize("n_hands", [1, 2, 3, 4])
    def test_player_hands_sit_at_layout_anchors(self, n_hands):
        hands = tuple((i, i + 13) for i in range(n_hands))
        got = placements(spec(hands=hands))
        player = got[2:]  # dealer took 2 slots
        for hand_idx in range(n_hands):
            ox, oy = PLAYER_ANCHORS[n_hands][hand_idx]
            first, second = player[hand_idx * 2], player[hand_idx * 2 + 1]
            assert (first.x, first.y) == (ox, oy)
            assert (second.x, second.y) == (ox + FAN_DX, oy + PLAYER_FAN_DY)

    def test_rejects_bad_hand_count(self):
        with pytest.raises(ValueError):
            placements(spec(hands=()))
        with pytest.raises(ValueError):
            placements(spec(hands=tuple((0,) for _ in range(5))))


class TestRender:
    def test_frame_shape_includes_count_bar(self, fake_table, fake_deck):
        for s in (1.0, 1.25):
            frame, labels = render(spec(scale=s), fake_table, fake_deck)
            bar = round(COUNT_BAR_CSS_PX * s)
            assert frame.shape == (round(TABLE_H * s) + bar, round(TABLE_W * s), 3)
            assert len(labels) == 4  # 2 dealer + 2 player cards

    def test_scale_one_pastes_sprites_pixel_exact(self, fake_table, fake_deck):
        """At scale 1.0 there is no resampling: the last (unoccluded) card of each fan
        must equal its sprite crop exactly."""
        sp = spec(dealer=(7,), hands=((20,),), scale=1.0)
        frame, _ = render(sp, fake_table, fake_deck)
        bar = COUNT_BAR_CSS_PX
        dx, dy = DEALER_ANCHOR
        sprite = fake_deck[7 * 94 : 8 * 94, 0:67]
        assert np.array_equal(frame[bar + dy : bar + dy + 94, dx : dx + 67], sprite)

    def test_yolo_labels_match_placements(self, fake_table, fake_deck):
        s = 1.12
        sp = spec(dealer=(0, BACK_CLASS), hands=((13, 26), (39, 40)), scale=s)
        frame, labels = render(sp, fake_table, fake_deck)
        fh, fw = frame.shape[:2]
        bar = round(COUNT_BAR_CSS_PX * s)
        parsed = []
        for line in labels:
            cls, cx, cy, w, h = line.split()
            parsed.append((int(cls), float(cx) * fw, float(cy) * fh, float(w) * fw, float(h) * fh))
        expect = placements(sp)
        assert [p[0] for p in parsed] == [e.class_id for e in expect]
        for (_, cx, cy, w, h), e in zip(parsed, expect, strict=True):
            assert cx - w / 2 == pytest.approx(round(e.x * s), abs=1.0)
            assert cy - h / 2 == pytest.approx(round(e.y * s) + bar, abs=1.0)
            assert w == pytest.approx(67 * s, abs=1.5)
            assert h == pytest.approx(94 * s, abs=1.5)

    def test_labels_stay_inside_frame(self, fake_table, fake_deck):
        _, labels = render(spec(hands=((0, 1, 2, 3, 4),), scale=0.9), fake_table, fake_deck)
        for line in labels:
            _, cx, cy, w, h = (float(v) for v in line.split()[0:5])
            assert cx - w / 2 >= 0 and cx + w / 2 <= 1.0001
            assert cy - h / 2 >= 0 and cy + h / 2 <= 1.0001


class TestPlan:
    def test_plan_covers_targets_and_is_deterministic(self):
        targets = dict.fromkeys(CLASS_NAMES, 8)
        plan_a = plan_frames(targets, random.Random(7))
        plan_b = plan_frames(targets, random.Random(7))
        assert plan_a == plan_b
        counts = dict.fromkeys(CLASS_NAMES, 0)
        for fs in plan_a:
            for card in fs.dealer:
                counts[CLASS_NAMES[card]] += 1
            for hand in fs.hands:
                for card in hand:
                    counts[CLASS_NAMES[card]] += 1
        for name, target in targets.items():
            assert counts[name] >= target, f"{name}: {counts[name]} < {target}"

    def test_plan_emphasizes_split_layouts(self):
        plan = plan_frames(dict.fromkeys(CLASS_NAMES, 20), random.Random(3))
        hand_counts = {len(fs.hands) for fs in plan}
        assert {2, 3, 4} <= hand_counts  # real capture already covers 1-hand play

    def test_plan_back_only_deficit_terminates(self):
        targets = dict.fromkeys(CLASS_NAMES, 0) | {"back": 5}
        plan = plan_frames(targets, random.Random(1))
        holes = sum(1 for fs in plan for c in fs.dealer if c == BACK_CLASS)
        assert holes >= 5

    def test_plan_rejects_unknown_class(self):
        with pytest.raises(ValueError):
            plan_frames({"XX": 4}, random.Random(0))


@pytest.mark.skipif(
    not (DECK_PNG.exists() and TABLE_PNG.exists()),
    reason="trainer assets not present (data/assets/ is gitignored)",
)
class TestRoundTripWithRealAssets:
    """The auto-labeler must recover exactly the cards the compositor placed — this
    validates both tools against each other at the scales used for generation."""

    @pytest.mark.parametrize("s", [0.9, 1.0, 1.12, 1.25])
    def test_autolabel_recovers_composited_cards(self, s):
        import cv2

        deck = cv2.imread(str(DECK_PNG))
        table = cv2.imread(str(TABLE_PNG))
        sp = spec(dealer=(3, BACK_CLASS), hands=((16, 29, 42), (8, 21)), scale=s)
        frame, _ = render(sp, table, deck)
        hits = detect_cards(frame, load_templates(DECK_PNG, s), s)
        got = sorted(h.class_id for h in hits)
        want = sorted(p.class_id for p in placements(sp))
        assert got == want
        bar = round(COUNT_BAR_CSS_PX * s)
        want_pos = {(p.class_id, round(p.x * s), round(p.y * s) + bar) for p in placements(sp)}
        for hit in hits:
            assert any(
                c == hit.class_id and abs(x - hit.x) <= 2 and abs(y - hit.y) <= 2
                for c, x, y in want_pos
            ), f"{hit} not at a placement"
