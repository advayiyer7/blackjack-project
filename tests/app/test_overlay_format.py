"""format_lines: the pure text layer between OverlayModel and tkinter."""

from __future__ import annotations

from bjcounter.app.overlay import format_lines
from bjcounter.types import Action, Advice, Event, OverlayModel


def model(**overrides) -> OverlayModel:
    base = {
        "running_count": 3,
        "true_count": 1,
        "decks_remaining": 4.2,
        "per_hand": (),
        "bet_units": 2,
        "events": (),
        "warnings": (),
        "last_hand": False,
    }
    return OverlayModel(**(base | overrides))


class TestFormatLines:
    def test_counts_and_bet_always_lead(self):
        lines = format_lines(model())
        assert lines[0] == "RC +3   TC +1   decks 4.2"
        assert lines[1] == "bet: 2 units"

    def test_single_hand_advice_with_fallback(self):
        advice = Advice(action=Action.DOUBLE, fallback=Action.HIT)
        lines = format_lines(model(per_hand=((0, advice),)))
        assert "play: DOUBLE (else HIT)" in lines

    def test_multiple_hands_are_numbered_from_one(self):
        hands = ((0, Advice(action=Action.STAND)), (1, Advice(action=Action.HIT)))
        lines = format_lines(model(per_hand=hands))
        assert "hand 1: STAND" in lines
        assert "hand 2: HIT" in lines

    def test_deviation_insurance_caveat_and_events_render(self):
        advice = Advice(
            action=Action.STAND,
            is_deviation=True,
            deviation="I18: 16vT stand @ TC>=0",
            insurance=True,
            caveat="3+ cards: ignore if hand already doubled",
        )
        lines = format_lines(
            model(
                per_hand=((0, advice),),
                events=(Event.HOLE_REVEALED,),
                warnings=("previous round never settled — the count may be short",),
                last_hand=True,
            )
        )
        joined = "\n".join(lines)
        assert "deviation — I18: 16vT stand @ TC>=0" in joined
        assert "INSURANCE: take it" in joined
        assert "(3+ cards: ignore if hand already doubled)" in joined
        assert "* hole revealed" in joined
        assert "! previous round never settled" in joined
        assert "LAST HAND" in joined

    def test_busted_hand_shows_its_caveat(self):
        advice = Advice(action=None, caveat="busted")
        lines = format_lines(model(per_hand=((0, advice),)))
        assert "play: busted" in lines
