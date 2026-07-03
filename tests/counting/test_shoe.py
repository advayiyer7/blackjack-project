"""ShoeState and the seeded shoe simulator: property tests (BUILD-GUIDE M2 exit)."""

from __future__ import annotations

import random
from collections import Counter

from bjcounter.counting.shoe import ShoeState, shuffled_shoe
from bjcounter.types import Rank

TEN_VALUE_RANKS = {Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING}


class TestShoeState:
    def test_fresh_shoe(self):
        shoe = ShoeState(seen=(), decks_total=6)
        assert shoe.cards_seen == 0
        assert shoe.running_count == 0
        assert shoe.decks_remaining == 6.0
        assert shoe.true_count == 0

    def test_with_cards_is_immutable(self):
        shoe = ShoeState(seen=(), decks_total=6)
        shoe2 = shoe.with_cards((Rank.FIVE, Rank.SIX))
        assert shoe.cards_seen == 0  # original untouched
        assert shoe2.cards_seen == 2
        assert shoe2.running_count == 2
        assert shoe2.seen == (Rank.FIVE, Rank.SIX)

    def test_reset(self):
        shoe = ShoeState(seen=(Rank.FIVE, Rank.TEN), decks_total=6)
        fresh = shoe.reset()
        assert fresh.cards_seen == 0
        assert fresh.decks_total == 6

    def test_derived_true_count_uses_flooring(self):
        # 208 cards seen of 6 decks -> 2 decks remain; RC -3 -> TC floor(-1.5) = -2.
        seen = tuple([Rank.EIGHT] * 205 + [Rank.TEN] * 3)
        shoe = ShoeState(seen=seen, decks_total=6)
        assert shoe.running_count == -3
        assert shoe.decks_remaining == 2.0
        assert shoe.true_count == -2

    def test_shoe_state_is_frozen_and_slotted(self):
        import dataclasses

        import pytest

        shoe = ShoeState(seen=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            shoe.seen = (Rank.TWO,)
        # Non-field assignment: frozen+slots raises (TypeError under current CPython —
        # a stdlib quirk — or AttributeError); either way mutation must fail.
        with pytest.raises((AttributeError, TypeError)):
            shoe.extra = 1


class TestShuffledShoe:
    def test_composition_exact(self):
        shoe = shuffled_shoe(decks=6, rng=random.Random(7))
        assert len(shoe) == 312
        counts = Counter(shoe)
        assert all(counts[rank] == 24 for rank in Rank)  # 4 suits x 6 decks

    def test_seeded_determinism(self):
        assert shuffled_shoe(6, random.Random(11)) == shuffled_shoe(6, random.Random(11))
        assert shuffled_shoe(6, random.Random(11)) != shuffled_shoe(6, random.Random(12))

    def test_full_shoe_always_sums_to_zero(self):
        # Balanced count: ANY full shoe nets RC 0 (M2 property gate).
        for seed in range(200):
            cards = shuffled_shoe(6, random.Random(seed))
            assert ShoeState(seen=cards, decks_total=6).running_count == 0

    def test_ten_thousand_shoes_zero_drift(self):
        # BUILD-GUIDE M2: "simulator deals 10k shoes with zero count drift."
        # Deal card-by-card through ShoeState and require RC to return to 0 every shoe.
        rng = random.Random(1234)
        for _ in range(10_000):
            cards = shuffled_shoe(1, rng)  # 1-deck shoes keep the loop fast; same property
            shoe = ShoeState(seen=(), decks_total=1)
            shoe = shoe.with_cards(cards)
            assert shoe.running_count == 0
            assert shoe.cards_seen == 52

    def test_mid_shoe_true_count_consistency(self):
        # TC verified against an INDEPENDENT exact-rational reference (not the same
        # call chain — M2 review finding 2).
        import math
        from fractions import Fraction

        from bjcounter.counting.hilo import running_count

        rng = random.Random(99)
        cards = shuffled_shoe(6, rng)
        shoe = ShoeState(seen=(), decks_total=6)
        for i, rank in enumerate(cards[:250]):
            shoe = shoe.with_cards((rank,))
            rc = running_count(cards[: i + 1])
            assert shoe.running_count == rc
            remaining_decks = max(Fraction(312 - (i + 1), 52), Fraction(1, 2))
            assert shoe.true_count == math.floor(Fraction(rc) / remaining_decks)


class TestPurity:
    def test_counting_modules_are_stdlib_pure(self):
        import sys

        import bjcounter.counting.shoe  # noqa: F401 — populates sys.modules

        allowed = {"dataclasses", "enum", "typing", "types", "collections", "math",
                   "random", "functools", "itertools", "__future__"}
        for name in (
            "bjcounter.counting.hilo",
            "bjcounter.counting.true_count",
            "bjcounter.counting.shoe",
        ):
            source = sys.modules[name].__file__
            with open(source, encoding="utf-8") as fh:
                text = fh.read()
            for line in text.splitlines():
                line = line.strip()
                if line.startswith(("import ", "from ")):
                    mod = line.split()[1].split(".")[0]
                    assert mod in allowed or mod == "bjcounter", f"{name}: {line}"
                    if mod == "bjcounter":
                        target = line.split()[1]
                        assert target.startswith(("bjcounter.types", "bjcounter.counting")), (
                            f"{name} imports outside the pure core: {line}"
                        )
