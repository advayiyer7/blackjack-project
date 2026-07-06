"""Builders that place cards at exact trainer geometry (scale 1.0, origin (0, 0)) so
reconcile tests construct TableStates the same way assembly would produce them."""

from __future__ import annotations

from bjcounter.tracker.state import (
    CARD_H,
    CARD_W,
    DEALER_ANCHOR,
    FAN_DX,
    PLAYER_ANCHORS,
    PLAYER_FAN_DY,
)
from bjcounter.types import Card, Hand, Rank, Suit, TableState


def card_at(label: str, x: int, y: int, conf: float = 0.99) -> Card:
    return Card(
        rank=Rank(label[0]), suit=Suit(label[1]), bbox=(x, y, CARD_W, CARD_H), conf=conf
    )


def dealer_fan(labels: tuple[str, ...], hole_slot: int | None) -> tuple[Card, ...]:
    """Dealer faces in fan order; `hole_slot` marks which fan slot the face-down card
    occupies (faces after it shift right by one slot), None = no hole."""
    ax, ay = DEALER_ANCHOR
    cards = []
    slot = 0
    for label in labels:
        if slot == hole_slot:
            slot += 1  # the back occupies this slot; it is not a face
        cards.append(card_at(label, ax + slot * FAN_DX, ay))
        slot += 1
    return tuple(cards)


def make_table(
    frame_id: int,
    dealer: tuple[str, ...],
    hands: tuple[tuple[str, ...], ...],
    *,
    hole: bool = False,
    hole_slot: int | None = None,
    warnings: tuple[str, ...] = (),
) -> TableState:
    """TableState at trainer geometry. `hole=True` puts the face-down card at fan
    slot 1 (dealt second) unless `hole_slot` overrides it."""
    if hole and hole_slot is None:
        hole_slot = 1
    player_hands = []
    n = len(hands)
    for hand_idx, labels in enumerate(hands):
        ox, oy = PLAYER_ANCHORS[n][hand_idx]
        cards = tuple(
            card_at(label, ox + i * FAN_DX, oy + i * PLAYER_FAN_DY)
            for i, label in enumerate(labels)
        )
        player_hands.append(Hand(slot=hand_idx, cards=cards))
    return TableState(
        frame_id=frame_id,
        player_hands=tuple(player_hands),
        dealer_cards=dealer_fan(dealer, hole_slot if hole else None),
        dealer_has_hole=hole,
        warnings=warnings,
    )
