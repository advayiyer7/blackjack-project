"""Hand math and legality predicates (strategy/hands.py)."""

from __future__ import annotations

import pytest

from bjcounter.strategy.engine import decide
from bjcounter.strategy.hands import can_double, can_split, can_surrender, hand_view
from bjcounter.types import Action, DoubleRule, Rank, Rules, Surrender, TableContext

R = Rank
CTX1 = TableContext(num_hands=1)
CTX2 = TableContext(num_hands=2)


class TestHandView:
    @pytest.mark.parametrize(
        ("ranks", "total", "is_soft"),
        [
            ((R.TWO, R.THREE), 5, False),
            ((R.TEN, R.SIX), 16, False),
            ((R.ACE, R.SIX), 17, True),
            ((R.ACE, R.ACE), 12, True),  # one ace demotes
            ((R.ACE, R.NINE), 20, True),
            ((R.ACE, R.TEN), 21, True),  # natural
            ((R.ACE, R.SIX, R.NINE), 16, False),  # ace forced hard
            ((R.ACE, R.TWO, R.THREE), 16, True),  # still soft
            ((R.TEN, R.SIX, R.NINE), 25, False),  # bust
            ((R.KING, R.QUEEN), 20, False),
            ((R.ACE, R.ACE, R.NINE), 21, True),
        ],
    )
    def test_totals(self, ranks, total, is_soft):
        view = hand_view(ranks)
        assert (view.total, view.is_soft) == (total, is_soft)
        assert view.n_cards == len(ranks)

    def test_pair_detection(self):
        assert hand_view((R.EIGHT, R.EIGHT)).pair_rank == R.EIGHT
        assert hand_view((R.ACE, R.ACE)).pair_rank == R.ACE
        assert hand_view((R.EIGHT, R.NINE)).pair_rank is None
        assert hand_view((R.EIGHT, R.EIGHT, R.EIGHT)).pair_rank is None  # 3 cards != pair

    def test_ten_value_cards_pair_as_ten(self):
        # Trainer groups ten-values for splitting; T,T deviations refer to any 20 pair.
        assert hand_view((R.KING, R.QUEEN)).pair_rank == R.TEN
        assert hand_view((R.TEN, R.JACK)).pair_rank == R.TEN


class TestLegality:
    def test_double_two_cards_only(self):
        rules = Rules()
        assert can_double(hand_view((R.FIVE, R.SIX)), rules)
        assert not can_double(hand_view((R.FIVE, R.THREE, R.THREE)), rules)

    def test_double_restrictions(self):
        h9_11 = Rules(double=DoubleRule.HARD_9_TO_11)
        h10_11 = Rules(double=DoubleRule.HARD_10_11)
        assert can_double(hand_view((R.FOUR, R.FIVE)), h9_11)  # hard 9
        assert not can_double(hand_view((R.THREE, R.FIVE)), h9_11)  # hard 8
        assert not can_double(hand_view((R.ACE, R.SIX)), h9_11)  # soft 17
        assert not can_double(hand_view((R.FOUR, R.FIVE)), h10_11)  # hard 9
        assert can_double(hand_view((R.FIVE, R.SIX)), h10_11)  # hard 11

    def test_split_budget(self):
        rules = Rules()  # max_hands=4, max_ace_hands=2
        pair8 = hand_view((R.EIGHT, R.EIGHT))
        assert can_split(pair8, rules, TableContext(num_hands=3))
        assert not can_split(pair8, rules, TableContext(num_hands=4))
        pair_a = hand_view((R.ACE, R.ACE))
        assert can_split(pair_a, rules, CTX1)
        assert not can_split(pair_a, rules, CTX2)  # aces split once only
        assert not can_split(hand_view((R.EIGHT, R.NINE)), rules, CTX1)

    def test_surrender_first_decision_single_hand_only(self):
        rules = Rules()  # ANY_CARD
        h16 = hand_view((R.TEN, R.SIX))
        assert can_surrender(h16, R.TEN, rules)
        assert not can_surrender(hand_view((R.TEN, R.THREE, R.THREE)), R.TEN, rules)  # 3 cards
        assert can_surrender(h16, R.ACE, rules)
        assert not can_surrender(h16, R.ACE, Rules(surrender=Surrender.NOT_VS_ACE))
        assert can_surrender(h16, R.TEN, Rules(surrender=Surrender.NOT_VS_ACE))
        assert not can_surrender(h16, R.TEN, Rules(surrender=Surrender.NONE))

    def test_surrender_illegal_after_split(self):
        # Post-split hands cannot surrender (ARCHITECTURE finding 6): 16vT post-split
        # falls to the count-aware stand/hit path, never SURRENDER.
        advice = decide(cards=(R.TEN, R.SIX), upcard=R.TEN, tc=0, rules=Rules(), ctx=CTX2)
        assert advice.action != Action.SURRENDER
