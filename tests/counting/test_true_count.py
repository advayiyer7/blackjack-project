"""True-count conversion: exact integer flooring, verified against Fraction ground truth.

The M2 review proved float intermediates put integer TCs one ULP on the wrong side of the
floor (RC -15 with 60 cards left is exactly -13; floats said -14). These tests verify the
implementation against an INDEPENDENT fractions.Fraction reference, not its own call chain.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from bjcounter.counting.true_count import decks_remaining, true_count


def reference_tc(rc: int, cards_seen: int, decks_total: int) -> int:
    """Exact-rational ground truth: floor(rc / clamp(remaining_decks, 1/2))."""
    remaining_decks = max(Fraction(decks_total * 52 - cards_seen, 52), Fraction(1, 2))
    return math.floor(Fraction(rc) / remaining_decks)


class TestTrueCountExact:
    @pytest.mark.parametrize(
        ("rc", "cards_seen", "decks", "expected"),
        [
            (3, 208, 6, 1),  # 2 decks left, 1.5 floors to 1
            (-3, 208, 6, -2),  # -1.5 floors to -2, NOT -1 (floor, not truncate)
            (0, 156, 6, 0),
            (-1, 26, 6, -1),  # -0.18 floors to -1
            (1, 26, 6, 0),  # +0.18 floors to 0
            (4, 208, 6, 2),
            (-4, 208, 6, -2),
            (7, 130, 6, 2),
            (-15, 252, 6, -13),  # M2 review counter-example: floats returned -14
            (-30, 192, 6, -13),  # in-penetration ULP case from the review sweep
            (-3, 300, 6, -6),  # clamp: 12 cards left -> divide by 26 (half deck)
            (-3, 312, 6, -6),  # exhausted shoe clamps the same way
        ],
    )
    def test_known_values(self, rc, cards_seen, decks, expected):
        assert true_count(rc, cards_seen, decks) == expected
        assert reference_tc(rc, cards_seen, decks) == expected  # table itself is verified

    def test_exhaustive_six_deck_sweep_matches_fraction_ground_truth(self):
        # Every reachable (cards_seen, rc) pair in a 6-deck shoe, exact match required.
        for cards_seen in range(0, 313):
            for rc in range(-30, 31):
                assert true_count(rc, cards_seen, 6) == reference_tc(rc, cards_seen, 6), (
                    f"rc={rc} seen={cards_seen}"
                )

    def test_random_sweep_other_deck_counts(self):
        rng = random.Random(42)
        for _ in range(5000):
            decks = rng.choice((1, 2, 4, 5, 8))
            cards_seen = rng.randint(0, decks * 52)
            rc = rng.randint(-40, 40)
            assert true_count(rc, cards_seen, decks) == reference_tc(rc, cards_seen, decks)

    def test_sign_property(self):
        # (rc >= 0) == (tc >= 0); any negative RC yields a strictly negative TC.
        rng = random.Random(7)
        for _ in range(2000):
            rc = rng.randint(-30, 30)
            cards_seen = rng.randint(0, 312)
            tc = true_count(rc, cards_seen, 6)
            assert (rc >= 0) == (tc >= 0), (rc, cards_seen, tc)
            if rc < 0:
                assert tc <= -1


class TestDecksRemainingDisplay:
    def test_full_and_half_shoe(self):
        assert decks_remaining(cards_seen=0, decks_total=6) == 6.0
        assert decks_remaining(cards_seen=156, decks_total=6) == 3.0

    def test_clamps_at_half_deck(self):
        assert decks_remaining(cards_seen=300, decks_total=6) == 0.5
        assert decks_remaining(cards_seen=312, decks_total=6) == 0.5
        assert decks_remaining(cards_seen=26, decks_total=1) == 0.5
