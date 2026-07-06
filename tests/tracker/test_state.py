"""TableState assembly: self-calibration, zone split, layout fitting, sanity gates."""

from __future__ import annotations

import pytest

from bjcounter.tracker.state import (
    CARD_H,
    CARD_W,
    DEALER_ANCHOR,
    FAN_DX,
    PLAYER_ANCHORS,
    PLAYER_FAN_DY,
    assemble_table,
)
from bjcounter.types import Detection, Rank, Suit


def det(label: str, ux: float, uy: float, scale: float = 1.0,
        origin: tuple[int, int] = (0, 0), conf: float = 0.99) -> Detection:
    """Detection whose bbox is the scaled/offset image of unscaled table coords."""
    return Detection(
        label=label,
        bbox=(
            round(ux * scale) + origin[0],
            round(uy * scale) + origin[1],
            round(CARD_W * scale),
            round(CARD_H * scale),
        ),
        conf=conf,
    )


def deal_detections(scale: float = 1.0, origin: tuple[int, int] = (0, 0)):
    """A deal-shaped frame: player 8s,7h; dealer Th + hole."""
    ax, ay = DEALER_ANCHOR
    px, py = PLAYER_ANCHORS[1][0]
    return (
        det("Th", ax, ay, scale, origin),
        det("back", ax + FAN_DX, ay, scale, origin),
        det("8s", px, py, scale, origin),
        det("7h", px + FAN_DX, py + PLAYER_FAN_DY, scale, origin),
    )


class TestAssembly:
    def test_empty_frame_is_a_clean_empty_table(self):
        table = assemble_table(3, ())
        assert table.player_hands == () and table.dealer_cards == ()
        assert not table.dealer_has_hole and table.warnings == ()

    @pytest.mark.parametrize("scale,origin", [(1.0, (0, 0)), (1.25, (0, 45)), (0.9, (12, 32))])
    def test_deal_frame_assembles_at_any_scale_and_origin(self, scale, origin):
        table = assemble_table(0, deal_detections(scale, origin), table_origin=origin)
        assert table.warnings == ()
        assert table.dealer_has_hole
        assert [(c.rank, c.suit) for c in table.dealer_cards] == [(Rank.TEN, Suit.HEARTS)]
        assert len(table.player_hands) == 1
        hand = table.player_hands[0]
        assert hand.slot == 0
        assert [(c.rank, c.suit) for c in hand.cards] == [
            (Rank.EIGHT, Suit.SPADES), (Rank.SEVEN, Suit.HEARTS)
        ]

    @pytest.mark.parametrize("n_hands", [2, 3, 4])
    def test_split_layouts_fit_their_anchor_sets(self, n_hands):
        labels = ["8s", "8h", "8d", "8c"]
        dets = [det("Th", *DEALER_ANCHOR)]
        for i in range(n_hands):
            ox, oy = PLAYER_ANCHORS[n_hands][i]
            dets += [det(labels[i], ox, oy), det("5c", ox + FAN_DX, oy + PLAYER_FAN_DY)]
        table = assemble_table(0, tuple(dets))
        assert table.warnings == ()
        assert len(table.player_hands) == n_hands
        for i, hand in enumerate(table.player_hands):
            assert hand.slot == i and len(hand.cards) == 2
            assert hand.cards[0].rank == Rank(labels[i][0])

    def test_cards_ordered_left_to_right_within_hand(self):
        px, py = PLAYER_ANCHORS[1][0]
        dets = (
            det("5c", px + 2 * FAN_DX, py + 2 * PLAYER_FAN_DY),
            det("8s", px, py),
            det("7h", px + FAN_DX, py + PLAYER_FAN_DY),
        )
        table = assemble_table(0, dets)
        assert [c.rank for c in table.player_hands[0].cards] == [
            Rank.EIGHT, Rank.SEVEN, Rank.FIVE
        ]

    def test_dealer_upcard_is_leftmost_face(self):
        ax, ay = DEALER_ANCHOR
        dets = (det("9d", ax + FAN_DX, ay), det("Th", ax, ay),
                det("Kc", ax + 2 * FAN_DX, ay))
        table = assemble_table(0, dets)
        assert table.dealer_cards[0].rank is Rank.TEN


class TestSanityGates:
    def test_low_confidence_warns(self):
        dets = deal_detections()
        low = Detection(label="8s", bbox=dets[2].bbox, conf=0.55)
        table = assemble_table(0, (dets[0], dets[1], low, dets[3]))
        assert any("low confidence" in w for w in table.warnings)

    def test_fan_gap_fails_cardinality(self):
        # NMS swallowed the middle card: extent says 3 cards, only 2 detected.
        px, py = PLAYER_ANCHORS[1][0]
        dets = (
            det("Th", *DEALER_ANCHOR),
            det("8s", px, py),
            det("5c", px + 2 * FAN_DX, py + 2 * PLAYER_FAN_DY),
        )
        table = assemble_table(0, dets)
        assert any("fan extent" in w for w in table.warnings)

    def test_dealer_fan_gap_fails_cardinality(self):
        ax, ay = DEALER_ANCHOR
        dets = (det("Th", ax, ay), det("9d", ax + 3 * FAN_DX, ay))
        table = assemble_table(0, dets)
        assert any("dealer fan extent" in w for w in table.warnings)

    def test_hole_occupies_a_dealer_fan_slot(self):
        # Upcard + back + revealed third card: extent 3, count 3 -> no warning.
        ax, ay = DEALER_ANCHOR
        dets = (det("Th", ax, ay), det("back", ax + FAN_DX, ay),
                det("9d", ax + 2 * FAN_DX, ay))
        table = assemble_table(0, dets)
        assert table.warnings == ()
        assert table.dealer_has_hole and len(table.dealer_cards) == 2

    def test_stray_card_far_from_all_anchors_is_ambiguous(self):
        dets = (det("Th", *DEALER_ANCHOR), det("8s", 300, 300))
        table = assemble_table(0, dets)
        assert any("ambiguous" in w for w in table.warnings)

    def test_back_in_player_zone_warns(self):
        px, py = PLAYER_ANCHORS[1][0]
        dets = (det("Th", *DEALER_ANCHOR), det("8s", px, py), det("back", px + FAN_DX, py))
        table = assemble_table(0, dets)
        assert any("player zone" in w for w in table.warnings)

    def test_two_lone_cards_fit_the_two_hand_layout(self):
        # A split captured before either hand draws: one card at each 2-hand anchor.
        dets = (
            det("Th", *DEALER_ANCHOR),
            det("8s", *PLAYER_ANCHORS[2][0]),
            det("8h", *PLAYER_ANCHORS[2][1]),
        )
        table = assemble_table(0, dets)
        assert table.warnings == ()
        assert len(table.player_hands) == 2
        assert all(len(h.cards) == 1 for h in table.player_hands)

    def test_unparseable_label_raises(self):
        bad = Detection(label="??", bbox=(100, 400, CARD_W, CARD_H), conf=0.99)
        with pytest.raises(ValueError, match="unparseable"):
            assemble_table(0, (bad,))


def test_geometry_constants_match_the_compositor():
    """tracker (stdlib-only) and vision/synthesize both transcribe the trainer
    geometry from trainer-notes §4/§6; this non-skipped test catches drift without
    needing the gitignored sprite assets."""
    from bjcounter.vision import synthesize

    assert dict(PLAYER_ANCHORS) == dict(synthesize.PLAYER_ANCHORS)
    assert DEALER_ANCHOR == synthesize.DEALER_ANCHOR
    assert FAN_DX == synthesize.FAN_DX
    assert PLAYER_FAN_DY == synthesize.PLAYER_FAN_DY
    assert (CARD_W, CARD_H) == (synthesize.CARD_W, synthesize.CARD_H)
