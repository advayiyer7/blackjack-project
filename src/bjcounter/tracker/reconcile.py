"""Transition classifier + frame diff — the load-bearing algorithm (ARCHITECTURE §4).

step() classifies each captured frame as exactly one of DUPLICATE / NEW_DEAL /
CONTINUATION / ROUND_ENDED / SUSPECT (first match wins) and only then derives the
count delta from the table-wide (rank, suit) multiset. A frame that fits no positive
signal changes nothing (SUSPECT) — a skipped frame is recoverable, a silently wrong
count is not.

Implementation note (spec ambiguity resolved, recorded in ARCHITECTURE §14): the §4.2
step-0 label-flicker gate applies only to frames that are NOT deal-shaped — a new deal
legitimately reuses the fixed anchors with new faces, so a strict positional gate would
misclassify every consecutive round as SUSPECT. Deal-shaped frames are governed by the
NEW_DEAL positive signature instead ("two consecutive deal-shaped frames with ANY
difference = two rounds").
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from bjcounter.tracker.rounds import ReconcileResult, RoundState, frame_settles
from bjcounter.types import Card, Event, Rank, Suit, TableState

FaceKey = tuple[Rank, Suit]

# Positional tolerance for "same card, same place": a quarter card width, floored at a
# few pixels — comfortably under the 24px-scaled fan step, above per-frame jitter.
POS_TOLERANCE_DIVISOR = 4
MIN_POS_TOLERANCE_PX = 6


def _faces(table: TableState) -> tuple[Card, ...]:
    player = tuple(c for hand in table.player_hands for c in hand.cards)
    return player + table.dealer_cards


def _multiset(cards: tuple[Card, ...]) -> Counter[FaceKey]:
    return Counter((c.rank, c.suit) for c in cards)


def _same_pos(a: Card, b: Card) -> bool:
    tol = max(MIN_POS_TOLERANCE_PX, a.bbox[2] // POS_TOLERANCE_DIVISOR)
    return abs(a.bbox[0] - b.bbox[0]) <= tol and abs(a.bbox[1] - b.bbox[1]) <= tol


def _persists(card: Card, later: tuple[Card, ...]) -> bool:
    """The card is still visible: same label at (about) the same position."""
    return any(
        c.rank is card.rank and c.suit is card.suit and _same_pos(card, c) for c in later
    )


def _deal_shaped(table: TableState) -> bool:
    """§4.2 positive deal signature: exactly one player hand of exactly two cards at
    the 1-hand deal anchor, dealer showing exactly one face plus a hole. Assembly
    guarantees a single fitted hand sits at the 1-hand anchor (state.py)."""
    return (
        len(table.player_hands) == 1
        and len(table.player_hands[0].cards) == 2
        and len(table.dealer_cards) == 1
        and table.dealer_has_hole
    )


def _identical(prev: TableState, now: TableState) -> bool:
    if (
        prev.dealer_has_hole != now.dealer_has_hole
        or len(prev.dealer_cards) != len(now.dealer_cards)
        or len(prev.player_hands) != len(now.player_hands)
    ):
        return False
    prev_faces, now_faces = _faces(prev), _faces(now)
    return len(prev_faces) == len(now_faces) and all(
        _persists(c, now_faces) for c in prev_faces
    )


def _flicker(prev: TableState, now: TableState) -> tuple[str, ...]:
    """Label changes at continuing positions (suit/rank flicker) — §4.2 step 0.

    The player zone is exempt when the hand count increased: a split relocates cards
    and deals replacements onto the old fan positions, so a label change there is
    expected — multiset continuity governs splits (§4.2 CONTINUATION).
    """
    split_happened = len(now.player_hands) > len(prev.player_hands)
    prev_player = tuple(c for hand in prev.player_hands for c in hand.cards)
    checked = prev.dealer_cards if split_happened else prev.dealer_cards + prev_player
    now_faces = _faces(now)
    problems = []
    for card in checked:
        for other in now_faces:
            if _same_pos(card, other) and (card.rank, card.suit) != (other.rank, other.suit):
                problems.append(
                    f"label changed at a continuing position: "
                    f"{card.rank}{card.suit} -> {other.rank}{other.suit} at {other.bbox[:2]}"
                )
    return tuple(problems)


def _suspect(round_state: RoundState, warnings: tuple[str, ...]) -> ReconcileResult:
    return ReconcileResult(
        round_state=round_state,
        revealed=(),
        events=(),
        accepted=False,
        warnings=warnings + ("suspect frame: nothing counted — recapture",),
    )


def _new_deal(round_state: RoundState, now: TableState) -> ReconcileResult:
    faces = _faces(now)
    events: list[Event] = [Event.NEW_ROUND]
    warnings: tuple[str, ...] = ()
    if round_state.table is not None and not round_state.settled:
        events.append(Event.PREV_ROUND_UNSETTLED)
        warnings = ("previous round never settled — the count may be short",)
    new_round = RoundState(
        round_index=round_state.round_index + 1,
        table=now,
        counted=tuple(sorted((c.rank, c.suit) for c in faces)),
        hole_was_revealed=False,
        settled=frame_settles(now, hole_was_revealed=False),
    )
    return ReconcileResult(new_round, faces, tuple(events), True, warnings)


def _continuation_violation(
    round_state: RoundState, prev: TableState, now: TableState
) -> str | None:
    """None when `now` is a legal continuation of the round; else the reason it isn't."""
    if not _multiset(_faces(now)) >= Counter(round_state.counted):
        return "a counted card is no longer visible"
    n_prev, n_now = len(prev.player_hands), len(now.player_hands)
    if n_now < n_prev:
        return "a player hand disappeared without a round end"
    if now.dealer_has_hole and not prev.dealer_has_hole:
        return "a hole card appeared mid-round"
    for card in prev.dealer_cards:
        if not _persists(card, now.dealer_cards):
            return "a dealer card moved or vanished (fan may only append)"
    hole_flip = prev.dealer_has_hole and not now.dealer_has_hole
    if hole_flip and len(now.dealer_cards) <= len(prev.dealer_cards):
        return "the hole card vanished without a revealed face"
    if n_now == n_prev:  # split relocation is legal; otherwise positions persist
        prev_player = tuple(c for hand in prev.player_hands for c in hand.cards)
        now_player = tuple(c for hand in now.player_hands for c in hand.cards)
        for card in prev_player:
            if not _persists(card, now_player):
                return "a player card moved or vanished without a split"
    return None


def _continuation(round_state: RoundState, prev: TableState, now: TableState) -> ReconcileResult:
    faces = _faces(now)
    prev_faces = _faces(prev)
    delta = _multiset(faces) - Counter(round_state.counted)
    revealed: list[Card] = []
    need = delta.copy()
    for card in sorted(faces, key=lambda c: _persists(c, prev_faces)):  # unmatched first
        if need[(card.rank, card.suit)] > 0:
            revealed.append(card)
            need[(card.rank, card.suit)] -= 1

    events: list[Event] = []
    hole_flip = prev.dealer_has_hole and not now.dealer_has_hole
    if hole_flip:
        events.append(Event.HOLE_REVEALED)
    events += [Event.SPLIT] * (len(now.player_hands) - len(prev.player_hands))

    hole_was_revealed = round_state.hole_was_revealed or hole_flip
    new_round = replace(
        round_state,
        table=now,
        counted=tuple(sorted((c.rank, c.suit) for c in faces)),
        hole_was_revealed=hole_was_revealed,
        settled=round_state.settled or frame_settles(now, hole_was_revealed),
    )
    return ReconcileResult(new_round, tuple(revealed), tuple(events), True, ())


def _round_ended(round_state: RoundState, prev: TableState, now: TableState) -> bool:
    """Cards removed with no new faces: player zone cleared, dealer faces (if any)
    persisting at their old positions. Covers surrender and full table clears."""
    if now.player_hands:
        return False
    if _multiset(_faces(now)) - Counter(round_state.counted):
        return False
    return all(_persists(card, prev.dealer_cards) for card in now.dealer_cards)


def step(round_state: RoundState, now: TableState) -> ReconcileResult:
    """Classify the transition from the last accepted frame to `now` (§4.2)."""
    if now.warnings:
        return _suspect(round_state, now.warnings)

    prev = round_state.table
    if prev is None:
        if _deal_shaped(now):
            return _new_deal(round_state, now)
        return _suspect(
            round_state, ("fresh shoe: waiting for a deal-shaped frame to anchor the round",)
        )

    if _identical(prev, now):
        return ReconcileResult(
            round_state=round_state,
            revealed=(),
            events=(Event.DUPLICATE_FRAME,),
            accepted=True,
            warnings=("duplicate frame — if this was an identical re-deal, recapture "
                      "after the next card (ARCHITECTURE §12 residual risk)",),
        )

    if _deal_shaped(now):
        # A genuine deal replaces every card, but detector mishaps can also produce a
        # deal-shaped frame (a dropped hit card, a vanished split hand, flickered
        # labels). Discriminate by how many faces carry over from the previous frame,
        # with a stricter bar when the previous frame was itself deal-shaped: there the
        # three anchor positions necessarily coincide, so ANY label survival means a
        # recapture with misreads rather than a quick re-deal (a real re-deal replaces
        # all three cards; 1-2 surviving labels is far likelier flicker — the case a
        # lone >=2 threshold missed, silently recounting a still-live round). When the
        # previous frame was NOT deal-shaped, one exact repeat at a reused anchor is a
        # ~2%-per-round coincidence that MUST stay NEW_DEAL or whole rounds would go
        # uncounted silently; two repeats (~4e-4) is a mishap. Fail-safe direction:
        # mishaps stall loudly as SUSPECT.
        prev_faces = _faces(prev)
        persisting = sum(1 for c in _faces(now) if _persists(c, prev_faces))
        threshold = 1 if _deal_shaped(prev) else 2
        if persisting >= threshold:
            return _suspect(
                round_state,
                (f"deal-shaped frame, but {persisting} cards persist from the previous"
                 " frame — not a credible new deal",),
            )
        return _new_deal(round_state, now)

    flicker = _flicker(prev, now)
    if flicker:
        return _suspect(round_state, flicker)

    violation = _continuation_violation(round_state, prev, now)
    if violation is None:
        return _continuation(round_state, prev, now)

    if _round_ended(round_state, prev, now):
        new_round = replace(round_state, table=now, settled=True)
        return ReconcileResult(new_round, (), (Event.ROUND_ENDED,), True, ())

    return _suspect(round_state, (violation,))
