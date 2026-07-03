"""Hand math and legality predicates. Pure, stdlib only (ARCHITECTURE §6)."""

from __future__ import annotations

from dataclasses import dataclass

from bjcounter.types import DoubleRule, Rank, Rules, Surrender, TableContext

TEN_VALUES = frozenset({Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING})

_RANK_VALUE = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6,
    Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9,
    Rank.TEN: 10, Rank.JACK: 10, Rank.QUEEN: 10, Rank.KING: 10,
    Rank.ACE: 11,
}


@dataclass(frozen=True, slots=True)
class HandView:
    total: int
    is_soft: bool  # an ace is currently counted as 11
    pair_rank: Rank | None  # ten-values group as Rank.TEN
    n_cards: int


def normalize_ten(rank: Rank) -> Rank:
    """Face cards behave as tens everywhere in strategy space."""
    return Rank.TEN if rank in TEN_VALUES else rank


def hand_view(cards: tuple[Rank, ...]) -> HandView:
    """Derive total/softness/pair-ness from raw ranks (ten-values pair as Rank.TEN)."""
    total = sum(_RANK_VALUE[r] for r in cards)
    aces_as_eleven = sum(1 for r in cards if r is Rank.ACE)
    while total > 21 and aces_as_eleven:
        total -= 10
        aces_as_eleven -= 1
    pair: Rank | None = None
    if len(cards) == 2:
        a, b = (normalize_ten(cards[0]), normalize_ten(cards[1]))
        if a == b:
            pair = a
    return HandView(
        total=total, is_soft=aces_as_eleven > 0, pair_rank=pair, n_cards=len(cards)
    )


def can_double(view: HandView, rules: Rules) -> bool:
    """Doubling: exactly 2 cards, within the table's double-down restriction."""
    if view.n_cards != 2:
        return False
    if rules.double == DoubleRule.ANY_TWO:  # == not `is`: Rules fields may be config-loaded
        return True
    if view.is_soft:
        return False
    lo = 9 if rules.double == DoubleRule.HARD_9_TO_11 else 10
    return lo <= view.total <= 11


def can_split(view: HandView, rules: Rules, ctx: TableContext) -> bool:
    """Splitting: a pair with hand-count budget left (aces have their own budget)."""
    if view.pair_rank is None:
        return False
    budget = rules.max_ace_hands if view.pair_rank == Rank.ACE else rules.max_hands
    return ctx.num_hands < budget


def can_surrender(
    view: HandView, upcard: Rank, rules: Rules, ctx: TableContext | None = None
) -> bool:
    """Surrender: first decision (2 cards), never after a split, rules-gated vs ace."""
    if view.n_cards != 2:
        return False
    if ctx is not None and ctx.num_hands != 1:
        return False
    if rules.surrender == Surrender.NONE:
        return False
    return not (rules.surrender == Surrender.NOT_VS_ACE and normalize_ten(upcard) == Rank.ACE)
