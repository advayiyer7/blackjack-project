"""Settledness rules and round-state contracts (ARCHITECTURE §4.5)."""

from __future__ import annotations

from bjcounter.tracker.rounds import frame_settles, fresh_shoe, is_natural
from tests.tracker.conftest import make_table


class TestFreshShoe:
    def test_fresh_shoe_has_no_table_and_nothing_pending(self):
        state = fresh_shoe()
        assert state.table is None
        assert state.counted == ()
        assert state.settled  # nothing pending -> no spurious PREV_ROUND_UNSETTLED
        assert not state.hole_was_revealed


class TestIsNatural:
    def test_ace_plus_each_ten_value_is_natural(self):
        for ten in ("Th", "Jc", "Qd", "Ks"):
            table = make_table(0, dealer=("5c",), hands=(("As", ten),), hole=True)
            assert is_natural(table.player_hands[0].cards)

    def test_order_does_not_matter(self):
        table = make_table(0, dealer=("5c",), hands=(("Kh", "Ad"),), hole=True)
        assert is_natural(table.player_hands[0].cards)

    def test_non_naturals(self):
        for hand in (("As", "9h"), ("Th", "Kh"), ("As", "Th", "5c"), ("As",)):
            table = make_table(0, dealer=("5c",), hands=(hand,), hole=True)
            assert not is_natural(table.player_hands[0].cards)


class TestFrameSettles:
    def test_hole_reveal_settles(self):
        table = make_table(0, dealer=("5c", "9h"), hands=(("8s", "7h"),))
        assert frame_settles(table, hole_was_revealed=True)

    def test_cleared_player_zone_settles(self):
        table = make_table(0, dealer=("5c",), hands=())
        assert frame_settles(table, hole_was_revealed=False)

    def test_sole_natural_settles(self):
        table = make_table(0, dealer=("5c",), hands=(("As", "Kh"),), hole=True)
        assert frame_settles(table, hole_was_revealed=False)

    def test_split_naturals_do_not_settle(self):
        # Post-split 2-card A+T hands are not blackjacks; the round is still live.
        table = make_table(
            0, dealer=("5c",), hands=(("As", "Kh"), ("Ad", "Qs")), hole=True
        )
        assert not frame_settles(table, hole_was_revealed=False)

    def test_live_hand_does_not_settle(self):
        table = make_table(0, dealer=("5c",), hands=(("8s", "7h"),), hole=True)
        assert not frame_settles(table, hole_was_revealed=False)
