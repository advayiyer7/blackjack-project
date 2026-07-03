"""Immutable shoe state + seeded shoe simulator. Pure, stdlib only (ARCHITECTURE §3/§5).

ShoeState stores every counted rank in reveal order; running/true count and decks
remaining are derived properties, so the numbers can never drift apart. The simulator
is reused by the M2 property tests, the RL environment, and the eval harness.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, replace

from bjcounter.counting.hilo import running_count as _running_count
from bjcounter.counting.true_count import decks_remaining as _decks_remaining
from bjcounter.counting.true_count import true_count as _true_count
from bjcounter.types import Rank


@dataclass(frozen=True, slots=True)
class ShoeState:
    """Immutable record of every counted rank this shoe; counts are derived."""

    seen: tuple[Rank, ...]
    decks_total: int = 6

    @property
    def cards_seen(self) -> int:
        return len(self.seen)

    @property
    def running_count(self) -> int:
        return _running_count(self.seen)

    @property
    def decks_remaining(self) -> float:
        """Display-only float; TC math uses exact integer arithmetic internally."""
        return _decks_remaining(self.cards_seen, self.decks_total)

    @property
    def true_count(self) -> int:
        return _true_count(self.running_count, self.cards_seen, self.decks_total)

    def with_cards(self, ranks: Iterable[Rank]) -> ShoeState:
        """New state with `ranks` appended to the seen sequence."""
        return replace(self, seen=self.seen + tuple(ranks))

    def reset(self) -> ShoeState:
        """Fresh shoe (manual shuffle-reset hotkey)."""
        return replace(self, seen=())


def shuffled_shoe(decks: int, rng: random.Random) -> tuple[Rank, ...]:
    """A full shoe (4 suits x `decks` of each rank), shuffled by the caller's RNG."""
    cards = [rank for rank in Rank for _ in range(4 * decks)]
    rng.shuffle(cards)
    return tuple(cards)
