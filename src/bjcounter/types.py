"""Shared frozen contracts for bjcounter (ARCHITECTURE.md §3).

Stdlib only — this module is imported by every layer, including the pure core.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Rank(StrEnum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


class Suit(StrEnum):
    SPADES = "s"
    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"


class Action(StrEnum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"


class Surrender(StrEnum):
    NONE = "none"
    ANY_CARD = "any_card"
    NOT_VS_ACE = "not_vs_ace"


class DoubleRule(StrEnum):
    ANY_TWO = "any_two"
    HARD_9_TO_11 = "hard_9_to_11"
    HARD_10_11 = "hard_10_11"


class Event(StrEnum):
    NEW_ROUND = "new_round"
    HOLE_REVEALED = "hole_revealed"
    SPLIT = "split"
    ROUND_ENDED = "round_ended"
    DUPLICATE_FRAME = "duplicate"
    PREV_ROUND_UNSETTLED = "prev_round_unsettled"


BBox = tuple[int, int, int, int]  # x, y, w, h — capture-region pixel coords


@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    suit: Suit
    bbox: BBox
    conf: float


@dataclass(frozen=True, slots=True)
class Detection:
    """Raw detector output; label "back" has no rank."""

    label: str  # "2s".."Ah" or "back"
    bbox: BBox
    conf: float


@dataclass(frozen=True, slots=True)
class Hand:
    slot: int  # trainer hand slot 0..3
    cards: tuple[Card, ...]  # left-to-right fan order


@dataclass(frozen=True, slots=True)
class TableState:
    frame_id: int
    player_hands: tuple[Hand, ...]
    dealer_cards: tuple[Card, ...]  # fan order; face-down hole NOT included
    dealer_has_hole: bool
    last_hand: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Rules:
    """Trainer config mirror; defaults = the PRD reference config.

    NB: the trainer's own deck default is 8 — it must be set to 6 to match this.
    """

    decks: int = 6
    h17: bool = True
    das: bool = True
    surrender: Surrender = Surrender.ANY_CARD
    peek: bool = True
    double: DoubleRule = DoubleRule.ANY_TWO
    max_hands: int = 4  # SplitX "up to 3 times" -> 4 hands
    max_ace_hands: int = 2  # SplitA "once" -> 2 hands
    # Consumed by the RL env (P5); NOT part of M1 advice legality — vision cannot tell a
    # locked split-ace hand from a live soft hand (documented limitation, ARCHITECTURE §6).
    hit_split_aces: bool = False


@dataclass(frozen=True, slots=True)
class TableContext:
    """Table-level facts a single hand cannot know (ARCHITECTURE §13 finding 6)."""

    num_hands: int = 1  # 1 = no split has happened


@dataclass(frozen=True, slots=True)
class Advice:
    action: Action | None  # None = no legal decision to display (21/bust)
    fallback: Action | None = None  # e.g. DOUBLE falls back to HIT when illegal
    is_deviation: bool = False
    deviation: str | None = None  # e.g. "I18: 16vT stand @ TC>=0"
    insurance: bool = False  # dealer shows A and TC >= +3
    caveat: str | None = None  # e.g. "3+ cards: ignore if hand already doubled"


@dataclass(frozen=True, slots=True)
class OverlayModel:
    running_count: int
    true_count: int  # floored
    decks_remaining: float
    per_hand: tuple[tuple[int, Advice], ...]  # (slot, advice)
    bet_units: int
    events: tuple[Event, ...]
    warnings: tuple[str, ...]
    last_hand: bool
