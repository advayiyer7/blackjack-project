"""True-count conversion (ARCHITECTURE §5). Pure, stdlib only.

    remaining_cards = max(decks_total*52 - cards_seen, 26)      # 26 = half-deck clamp
    true_count      = (running_count * 52) // remaining_cards   # exact, floors toward -inf

The division is done in EXACT INTEGER arithmetic: an IEEE-754 intermediate like
60/52 = 1.1538461538461537 puts mathematically-integer TCs one ULP on the wrong side of
the floor (e.g. RC -15 with 60 cards left is exactly TC -13, but floats yield -14) —
found by the M2 review; regression-tested against fractions.Fraction ground truth.
Flooring (not truncation) is the Schlesinger/CVData convention the Illustrious 18 indices
assume (research §3): a raw TC of -0.3 floors to -1, never to 0.

decks_remaining() is a float for DISPLAY ONLY — never feed it into index decisions.
"""

from __future__ import annotations

CARDS_PER_DECK = 52
MIN_CARDS_DIVISOR = 26  # never divide by less than half a deck


def true_count(running_count: int, cards_seen: int, decks_total: int) -> int:
    """Floored true count, computed exactly (no float intermediates)."""
    remaining = max(decks_total * CARDS_PER_DECK - cards_seen, MIN_CARDS_DIVISOR)
    return (running_count * CARDS_PER_DECK) // remaining


def decks_remaining(cards_seen: int, decks_total: int) -> float:
    """Decks left as a float — display only (overlay), not for TC math."""
    remaining = (decks_total * CARDS_PER_DECK - cards_seen) / CARDS_PER_DECK
    return max(remaining, MIN_CARDS_DIVISOR / CARDS_PER_DECK)
