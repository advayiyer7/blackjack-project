"""Count deviations: Illustrious 18 + Fab 4, boundary-exact (research §4/§5/§6).

Direction semantics (research discrepancy #7):
  at_or_above -> deviation fires at TC >= index
  below       -> deviation (Hit) fires STRICTLY below the index; at TC == index play basic.
Every entry is tested at index-1 / index / index+1.
"""

from __future__ import annotations

import pytest

from bjcounter.strategy.engine import decide
from bjcounter.types import Action, Rank, Rules, Surrender, TableContext

from .conftest import fab4_entries, i18_entries

R = Rank
CTX = TableContext(num_hands=1)
H17 = Rules(h17=True)
S17 = Rules(h17=False)
NO_SURR_H17 = Rules(h17=True, surrender=Surrender.NONE)
NO_SURR_S17 = Rules(h17=False, surrender=Surrender.NONE)

# Concrete non-pair hands for deviation hand labels.
DEV_HANDS = {
    "9": (R.FOUR, R.FIVE),
    "10": (R.FOUR, R.SIX),
    "11": (R.FIVE, R.SIX),
    "12": (R.FIVE, R.SEVEN),
    "13": (R.SIX, R.SEVEN),
    "14": (R.SIX, R.EIGHT),
    "15": (R.SEVEN, R.EIGHT),
    "16": (R.SEVEN, R.NINE),
    "T,T": (R.TEN, R.TEN),
}
PLAY_TO_ACTION = {
    "Stand": Action.STAND,
    "Hit": Action.HIT,
    "Double": Action.DOUBLE,
    "Split": Action.SPLIT,
    "Surrender": Action.SURRENDER,
}


def doc_ids(entry: dict) -> str:
    return f"{entry['play']}-{entry['hand']}-vs-{entry['dealer_up']}"


class TestIllustrious18:
    """Each I18 entry at its boundary, in the no-surrender config the I18 assumes.

    (With LS available, surrender precedence covers 15vT/16v9/16vT — tested separately below.)
    """

    @pytest.mark.parametrize("entry", i18_entries(), ids=doc_ids)
    def test_boundary(self, entry):
        if entry["play"] == "Insurance":
            pytest.skip("insurance flag covered in TestInsurance")
        hand, up, idx = DEV_HANDS[entry["hand"]], Rank(entry["dealer_up"]), entry["index"]
        # 11vA is basic strategy under H17 (carve-out) -> test S17 side for that entry.
        rules = NO_SURR_S17 if entry["hand"] == "11" else NO_SURR_H17
        dev_action = PLAY_TO_ACTION[entry["play"]]

        results = {
            tc: decide(cards=hand, upcard=up, tc=tc, rules=rules, ctx=CTX)
            for tc in (idx - 1, idx, idx + 1)
        }
        if entry["direction"] == "at_or_above":
            assert results[idx].action == dev_action
            assert results[idx + 1].action == dev_action
            assert results[idx - 1].action != dev_action
        else:  # "below" — strict: at the index itself play basic (Stand)
            assert results[idx].action == Action.STAND
            assert results[idx + 1].action == Action.STAND
            assert results[idx - 1].action == Action.HIT
            assert results[idx - 1].is_deviation

    def test_11_vs_a_is_basic_under_h17(self):
        # Carve-out: always Double under H17, at any TC, not a deviation.
        for tc in (-5, 0, 1, 5):
            advice = decide(cards=DEV_HANDS["11"], upcard=R.ACE, tc=tc, rules=H17, ctx=CTX)
            assert advice.action == Action.DOUBLE
            assert not advice.is_deviation

    def test_double_deviation_requires_two_cards(self):
        # 10vT double @ +4 can't fire on a 3-card 10; falls back to basic hit.
        advice = decide(
            cards=(R.TWO, R.THREE, R.FIVE), upcard=R.TEN, tc=5, rules=NO_SURR_H17, ctx=CTX
        )
        assert advice.action == Action.HIT

    def test_stand_deviations_apply_to_multicard_hands(self):
        # 16vT stand @ 0 applies to a 3-card 16 (surrender illegal there anyway).
        advice = decide(cards=(R.TEN, R.FOUR, R.TWO), upcard=R.TEN, tc=0, rules=H17, ctx=CTX)
        assert advice.action == Action.STAND
        assert advice.is_deviation

    def test_tt_split_deviation_marks_deviation(self):
        advice = decide(cards=(R.TEN, R.TEN), upcard=R.FIVE, tc=5, rules=H17, ctx=CTX)
        assert advice.action == Action.SPLIT
        assert advice.is_deviation
        # Mixed ten-values split the same way (ten-value pair grouping).
        advice = decide(cards=(R.KING, R.QUEEN), upcard=R.SIX, tc=4, rules=H17, ctx=CTX)
        assert advice.action == Action.SPLIT


class TestFab4AndSurrenderRegions:
    """Surrender-available behavior, per research §5 + §6 discrepancy #2."""

    @pytest.mark.parametrize("entry", fab4_entries(), ids=doc_ids)
    def test_fab4_boundary(self, entry):
        hand, up, idx = DEV_HANDS[entry["hand"]], Rank(entry["dealer_up"]), entry["index"]
        # 15vA: the published +1 index is S17-oriented; under H17 15vA is already basic
        # surrender (carve-out below). Test the S17 side for that entry.
        rules = S17 if entry["hand"] == "15" and entry["dealer_up"] == "A" else H17
        at = decide(cards=hand, upcard=up, tc=idx, rules=rules, ctx=CTX)
        above = decide(cards=hand, upcard=up, tc=idx + 1, rules=rules, ctx=CTX)
        below = decide(cards=hand, upcard=up, tc=idx - 1, rules=rules, ctx=CTX)
        assert at.action == Action.SURRENDER
        assert above.action == Action.SURRENDER
        assert below.action != Action.SURRENDER

    def test_15_vs_t_three_regions(self):
        # Hit < 0, Surrender 0..3, Stand >= 4 (research discrepancy #2).
        cases = {-1: Action.HIT, 0: Action.SURRENDER, 3: Action.SURRENDER, 4: Action.STAND}
        for tc, expected in cases.items():
            advice = decide(cards=DEV_HANDS["15"], upcard=R.TEN, tc=tc, rules=H17, ctx=CTX)
            assert advice.action == expected, f"15vT at TC={tc}: {advice}"
        # Hit below 0 and Stand at 4+ deviate from the chart's Rh.
        assert decide(cards=DEV_HANDS["15"], upcard=R.TEN, tc=-1, rules=H17, ctx=CTX).is_deviation
        assert decide(cards=DEV_HANDS["15"], upcard=R.TEN, tc=4, rules=H17, ctx=CTX).is_deviation

    def test_16_vs_9_stand_takes_over_at_5(self):
        # Research §4 note: 16v9 has the same structure — Surrender < 5, Stand >= 5.
        assert (
            decide(cards=DEV_HANDS["16"], upcard=R.NINE, tc=4, rules=H17, ctx=CTX).action
            == Action.SURRENDER
        )
        assert (
            decide(cards=DEV_HANDS["16"], upcard=R.NINE, tc=5, rules=H17, ctx=CTX).action
            == Action.STAND
        )

    def test_16_vs_t_surrenders_at_all_counts_when_legal(self):
        # No published stand-over-surrender crossover for 16vT in our sources; the I18
        # stand @ 0 is the stand-vs-hit index and applies only when surrender is illegal.
        for tc in (-3, 0, 3, 6):
            advice = decide(cards=DEV_HANDS["16"], upcard=R.TEN, tc=tc, rules=H17, ctx=CTX)
            assert advice.action == Action.SURRENDER, f"16vT at TC={tc}: {advice}"

    def test_pair_playing_as_hard_total_gets_surrender_deviations(self):
        # M1 review finding 1: 7,7 vs T is never split (pair cell H), so it plays as
        # hard 14 INCLUDING the Fab4 14vT surrender at TC>=3.
        for tc, expected in ((2, Action.HIT), (3, Action.SURRENDER), (10, Action.SURRENDER)):
            advice = decide(cards=(R.SEVEN, R.SEVEN), upcard=R.TEN, tc=tc, rules=H17, ctx=CTX)
            assert advice.action == expected, f"7,7 vs T at TC={tc}: {advice}"
        assert decide(cards=(R.SEVEN, R.SEVEN), upcard=R.TEN, tc=3, rules=H17, ctx=CTX).is_deviation

    def test_split_recommended_pair_never_surrenders(self):
        # 8,8 vs T is always plain P — the pair cell supersedes hard-16 surrender.
        for tc in (-3, 0, 5, 10):
            advice = decide(cards=(R.EIGHT, R.EIGHT), upcard=R.TEN, tc=tc, rules=H17, ctx=CTX)
            assert advice.action == Action.SPLIT, f"8,8 vs T at TC={tc}: {advice}"

    def test_15_vs_a_h17_carveout(self):
        # Under H17 15vA is basic surrender at any TC (published index is S17-only;
        # H17-specific number UNRESOLVED in research §6#4).
        for tc in (-3, 0, 3):
            advice = decide(cards=DEV_HANDS["15"], upcard=R.ACE, tc=tc, rules=H17, ctx=CTX)
            assert advice.action == Action.SURRENDER
            assert not advice.is_deviation


class TestInsurance:
    def test_insurance_flag_at_plus_3(self):
        for tc, expected in ((2, False), (3, True), (4, True)):
            advice = decide(cards=(R.FIVE, R.SEVEN), upcard=R.ACE, tc=tc, rules=H17, ctx=CTX)
            assert advice.insurance is expected

    def test_no_insurance_flag_without_ace(self):
        advice = decide(cards=(R.FIVE, R.SEVEN), upcard=R.TEN, tc=5, rules=H17, ctx=CTX)
        assert advice.insurance is False
