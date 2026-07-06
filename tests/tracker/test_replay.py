"""End-to-end replay on synthetic fixtures: compositor -> template detector ->
assembly -> transition classifier -> shoe count. The running count must be exact at
every frame (the M5 gate, run here on scripted synthetic shoes; recorded real shoes
join at the M5 exit once captured).

Skipped when the gitignored trainer assets are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bjcounter.counting.shoe import ShoeState
from bjcounter.tracker.reconcile import step
from bjcounter.tracker.rounds import fresh_shoe
from bjcounter.types import Event

REPO = Path(__file__).resolve().parents[2]
DECK_PNG = REPO / "data" / "assets" / "deck.png"
TABLE_PNG = REPO / "data" / "assets" / "table.png"

pytestmark = pytest.mark.skipif(
    not (DECK_PNG.exists() and TABLE_PNG.exists()),
    reason="trainer assets not present (data/assets/ is gitignored)",
)

SCALE = 1.12  # a real captured session's scale: exercises resampling + rounding


def cid(label: str) -> int:
    from bjcounter.vision.autolabel import BACK_CLASS, CLASS_NAMES

    return BACK_CLASS if label == "back" else CLASS_NAMES.index(label)


def make_spec(dealer: tuple[str, ...], hands: tuple[tuple[str, ...], ...]):
    from bjcounter.vision.synthesize import FrameSpec

    return FrameSpec(
        dealer=tuple(cid(c) for c in dealer),
        hands=tuple(tuple(cid(c) for c in hand) for hand in hands),
        scale=SCALE,
    )


# A scripted four-round shoe: normal round with hit and hole reveal, a split round,
# a surrender, and a natural. (frame, expected running count after it, expected events)
SHOE_SCRIPT = (
    # round 1: deal 8s,7h vs Th+hole; hit 5c; settle Th,9d,6c
    (make_spec(("Th", "back"), (("8s", "7h"),)), -1, (Event.NEW_ROUND,)),
    (make_spec(("Th", "back"), (("8s", "7h", "5c"),)), 0, ()),
    (make_spec(("Th", "9d", "6c"), (("8s", "7h", "5c"),)), 1, (Event.HOLE_REVEALED,)),
    # round 2: pair of eights split into two hands, then settle
    (make_spec(("2c", "back"), (("8h", "8d"),)), 2, (Event.NEW_ROUND,)),
    (make_spec(("2c", "back"), (("8h", "4c"), ("8d", "Kh"))), 2, (Event.SPLIT,)),
    (make_spec(("2c", "7s", "Jh"), (("8h", "4c"), ("8d", "Kh"))), 1, (Event.HOLE_REVEALED,)),
    # round 3: deal Ts,6d vs 9c, then surrender (player zone clears, upcard persists)
    (make_spec(("9c", "back"), (("Ts", "6d"),)), 1, (Event.NEW_ROUND,)),
    (make_spec(("9c", "back"), ((),)), 1, (Event.ROUND_ENDED,)),
    # round 4: a natural — settles without any hole reveal
    (make_spec(("5h", "back"), (("Ah", "Kd"),)), 0, (Event.NEW_ROUND,)),
)


def test_replay_reproduces_the_exact_count_at_every_frame():
    import cv2

    from bjcounter.tracker.state import assemble_table
    from bjcounter.vision.detector import TemplateMatchDetector
    from bjcounter.vision.synthesize import COUNT_BAR_CSS_PX, render

    deck = cv2.imread(str(DECK_PNG))
    table_bgr = cv2.imread(str(TABLE_PNG))
    detector = TemplateMatchDetector(DECK_PNG, SCALE)
    origin = (0, round(COUNT_BAR_CSS_PX * SCALE))

    round_state = fresh_shoe()
    shoe = ShoeState(seen=(), decks_total=6)
    for frame_id, (spec, want_rc, want_events) in enumerate(SHOE_SCRIPT):
        frame, _ = render(spec, table_bgr, deck)
        table = assemble_table(frame_id, detector.detect(frame), table_origin=origin)
        result = step(round_state, table)

        assert result.accepted, f"frame {frame_id} SUSPECT: {result.warnings}"
        for event in want_events:
            assert event in result.events, f"frame {frame_id}: missing {event}"
        assert Event.PREV_ROUND_UNSETTLED not in result.events

        round_state = result.round_state
        shoe = shoe.with_cards(c.rank for c in result.revealed)
        assert shoe.running_count == want_rc, (
            f"frame {frame_id}: RC {shoe.running_count} != {want_rc}"
        )

    assert round_state.round_index == 4
    assert round_state.settled  # the natural settled round 4 with no hole reveal


def test_replay_flags_a_skipped_settle_between_rounds():
    import cv2

    from bjcounter.tracker.state import assemble_table
    from bjcounter.vision.detector import TemplateMatchDetector
    from bjcounter.vision.synthesize import COUNT_BAR_CSS_PX, render

    deck = cv2.imread(str(DECK_PNG))
    table_bgr = cv2.imread(str(TABLE_PNG))
    detector = TemplateMatchDetector(DECK_PNG, SCALE)
    origin = (0, round(COUNT_BAR_CSS_PX * SCALE))

    frames = (
        make_spec(("Th", "back"), (("8s", "7h"),)),  # deal, never settled
        make_spec(("2c", "back"), (("9h", "3d"),)),  # next deal straight away
    )
    round_state = fresh_shoe()
    events_seen: list[Event] = []
    for frame_id, spec in enumerate(frames):
        frame, _ = render(spec, table_bgr, deck)
        table = assemble_table(frame_id, detector.detect(frame), table_origin=origin)
        result = step(round_state, table)
        assert result.accepted
        round_state = result.round_state
        events_seen += list(result.events)

    assert Event.PREV_ROUND_UNSETTLED in events_seen
