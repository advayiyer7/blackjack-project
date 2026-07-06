"""Round-scoped tracker state and settledness rules (ARCHITECTURE §3, §4.5).

Pure, stdlib only; consumes types.py only. ReconcileResult lives here with RoundState
(both are tracker-owned contracts from ARCHITECTURE §3); reconcile.py produces them.
"""

from __future__ import annotations

from dataclasses import dataclass

from bjcounter.types import Card, Event, Rank, Suit, TableState

TEN_VALUE = frozenset({Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING})


@dataclass(frozen=True, slots=True)
class RoundState:
    round_index: int
    table: TableState | None  # last ACCEPTED frame (None = fresh shoe)
    counted: tuple[tuple[Rank, Suit], ...]  # sorted multiset of faces counted this round
    hole_was_revealed: bool  # this round reached a hole reveal
    settled: bool  # round reached a settled-looking frame (§4.5)


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    round_state: RoundState
    revealed: tuple[Card, ...]  # newly visible face-up cards -> count these
    events: tuple[Event, ...]
    accepted: bool  # False = SUSPECT frame, nothing changed
    warnings: tuple[str, ...]


def fresh_shoe() -> RoundState:
    """Initial state after startup or the manual shuffle-reset hotkey (§4.4)."""
    return RoundState(
        round_index=0, table=None, counted=(), hole_was_revealed=False, settled=True
    )


def is_natural(cards: tuple[Card, ...]) -> bool:
    """Two-card blackjack: an ace plus any ten-value card."""
    return (
        len(cards) == 2
        and any(c.rank is Rank.ACE for c in cards)
        and any(c.rank in TEN_VALUE for c in cards)
    )


def frame_settles(table: TableState, hole_was_revealed: bool) -> bool:
    """§4.5: a round is settled once any of these holds — the hole was revealed this
    round, the player zone cleared (surrender/table clear), or the sole player hand is
    a two-card natural."""
    if hole_was_revealed:
        return True
    if not table.player_hands:
        return True
    return len(table.player_hands) == 1 and is_natural(table.player_hands[0].cards)
