"""Hi-Lo tag values and running-count arithmetic. Pure, stdlib only.

Tags per docs/research/strategy-tables.md §2: 2-6 = +1, 7-9 = 0, T/J/Q/K/A = -1.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from bjcounter.types import Rank

HILO_TAGS: Mapping[Rank, int] = MappingProxyType({
    Rank.TWO: 1,
    Rank.THREE: 1,
    Rank.FOUR: 1,
    Rank.FIVE: 1,
    Rank.SIX: 1,
    Rank.SEVEN: 0,
    Rank.EIGHT: 0,
    Rank.NINE: 0,
    Rank.TEN: -1,
    Rank.JACK: -1,
    Rank.QUEEN: -1,
    Rank.KING: -1,
    Rank.ACE: -1,
})


def running_count(seen: Iterable[Rank]) -> int:
    """Sum of Hi-Lo tags over the seen ranks."""
    return sum(HILO_TAGS[rank] for rank in seen)
