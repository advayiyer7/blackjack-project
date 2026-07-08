"""process_frame: the pure capture->count->advice pipeline the worker thread wraps."""

from __future__ import annotations

from bjcounter.app.worker import process_frame, status_model
from bjcounter.counting.shoe import ShoeState
from bjcounter.tracker.rounds import fresh_shoe
from bjcounter.types import Action, Event, Rules
from tests.tracker.conftest import make_table

RULES = Rules()


def fresh():
    return ShoeState(seen=(), decks_total=6), fresh_shoe()


class TestProcessFrame:
    def test_deal_counts_cards_and_advises_each_hand(self):
        shoe, rs = fresh()
        deal = make_table(0, dealer=("Th",), hands=(("9s", "8h"),), hole=True)
        shoe, rs, model = process_frame(shoe, rs, deal, RULES, bet_cap=8)
        assert shoe.running_count == -1  # 9,8 are 0; T is -1
        assert model.running_count == -1
        assert Event.NEW_ROUND in model.events
        assert len(model.per_hand) == 1
        slot, advice = model.per_hand[0]
        assert slot == 0
        assert advice.action is Action.STAND  # hard 17 vs T, no count sensitivity

    def test_suspect_frame_changes_nothing_and_gives_no_advice(self):
        shoe, rs = fresh()
        deal = make_table(0, dealer=("Th",), hands=(("8s", "7h"),), hole=True)
        shoe, rs, _ = process_frame(shoe, rs, deal, RULES, bet_cap=8)
        dropped = make_table(1, dealer=("Th",), hands=(("8s",),), hole=True)
        shoe2, rs2, model = process_frame(shoe, rs, dropped, RULES, bet_cap=8)
        assert shoe2 == shoe and rs2 == rs
        assert model.per_hand == ()
        assert any("suspect" in w for w in model.warnings)

    def test_split_hands_each_get_advice(self):
        shoe, rs = fresh()
        deal = make_table(0, dealer=("6d",), hands=(("8s", "8h"),), hole=True)
        shoe, rs, model = process_frame(shoe, rs, deal, RULES, bet_cap=8)
        assert model.per_hand[0][1].action is Action.SPLIT
        split = make_table(1, dealer=("6d",), hands=(("8s", "5c"), ("8h", "9d")), hole=True)
        shoe, rs, model = process_frame(shoe, rs, split, RULES, bet_cap=8)
        assert Event.SPLIT in model.events
        assert [slot for slot, _ in model.per_hand] == [0, 1]
        assert model.per_hand[0][1].action is Action.STAND  # 13 vs 6
        assert model.per_hand[1][1].action is Action.STAND  # 17 vs 6

    def test_insurance_flag_surfaces_at_high_count(self):
        # Seed the shoe with enough low cards to push TC >= +3.
        from bjcounter.types import Rank

        shoe = ShoeState(seen=(Rank.FIVE,) * 40, decks_total=6)
        rs = fresh_shoe()
        deal = make_table(0, dealer=("Ah",), hands=(("9s", "9h"),), hole=True)
        shoe, rs, model = process_frame(shoe, rs, deal, RULES, bet_cap=8)
        assert model.true_count >= 3
        assert model.per_hand[0][1].insurance

    def test_bet_units_track_the_count_and_clamp(self):
        from bjcounter.types import Rank

        shoe, rs = fresh()
        _, _, model = process_frame(
            shoe,
            rs,
            make_table(0, dealer=("Th",), hands=(("8s", "7h"),), hole=True),
            RULES,
            bet_cap=8,
        )
        assert model.bet_units == 1  # negative TC floors at 1 unit
        rich = ShoeState(seen=(Rank.FIVE,) * 150, decks_total=6)
        _, _, model = process_frame(
            rich,
            fresh_shoe(),
            make_table(0, dealer=("Th",), hands=(("8s", "7h"),), hole=True),
            RULES,
            bet_cap=8,
        )
        assert model.bet_units == 8  # clamped at the cap

    def test_round_ended_frame_reports_count_without_advice(self):
        shoe, rs = fresh()
        deal = make_table(0, dealer=("Th",), hands=(("8s", "7h"),), hole=True)
        shoe, rs, _ = process_frame(shoe, rs, deal, RULES, bet_cap=8)
        surrender = make_table(1, dealer=("Th",), hands=(), hole=True)
        shoe, rs, model = process_frame(shoe, rs, surrender, RULES, bet_cap=8)
        assert Event.ROUND_ENDED in model.events
        assert model.per_hand == ()
        assert model.running_count == -1


class TestStatusModel:
    def test_status_reports_count_and_warnings_only(self):
        shoe = ShoeState(seen=(), decks_total=6)
        model = status_model(shoe, ("count reset — fresh shoe",))
        assert model.running_count == 0 and model.per_hand == ()
        assert model.warnings == ("count reset — fresh shoe",)
