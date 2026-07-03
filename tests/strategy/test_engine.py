"""Engine behavior beyond chart cells: edge totals, legality fallbacks, purity."""

from __future__ import annotations

import sys

import pytest

from bjcounter.strategy.engine import decide
from bjcounter.types import Action, Rank, Rules, Surrender, TableContext

R = Rank
CTX = TableContext(num_hands=1)
H17 = Rules(h17=True)
S17 = Rules(h17=False)


class TestEdgeTotals:
    def test_21_stands(self):
        assert decide((R.ACE, R.TEN), R.ACE, 0, H17, CTX).action == Action.STAND
        assert decide((R.SEVEN, R.SEVEN, R.SEVEN), R.TEN, 0, H17, CTX).action == Action.STAND

    def test_bust_has_no_action(self):
        advice = decide((R.TEN, R.SIX, R.NINE), R.TEN, 0, H17, CTX)
        assert advice.action is None

    def test_hard_18_plus_vs_ace_stand_not_surrender(self):
        # Chart's "17+" row shows Rs vs A (H17), but that surrender applies to hard 17
        # ONLY — 18+ always stands (research §1.1 note + BJA chart).
        for cards in ((R.TEN, R.FOUR, R.FOUR), (R.TEN, R.FIVE, R.FOUR)):  # 18, 19
            for rules in (H17, S17):
                advice = decide(cards, R.ACE, 0, rules, CTX)
                assert advice.action == Action.STAND

    def test_hard_17_vs_ace_by_ruleset(self):
        h17_advice = decide((R.EIGHT, R.NINE), R.ACE, 0, H17, CTX)
        assert (h17_advice.action, h17_advice.fallback) == (Action.SURRENDER, Action.STAND)
        assert decide((R.EIGHT, R.NINE), R.ACE, 0, S17, CTX).action == Action.STAND

    def test_multicard_soft_totals_use_soft_chart_without_double(self):
        # A,2,4 = soft 17: chart says D vs 5 but 3 cards can't double -> falls back to HIT.
        advice = decide((R.ACE, R.TWO, R.FOUR), R.FIVE, 0, H17, CTX)
        assert advice.action == Action.HIT

    def test_soft_12_hits(self):
        # A,A when splitting is unavailable = soft 12 -> hit.
        advice = decide((R.ACE, R.ACE), R.SIX, 0, H17, TableContext(num_hands=2))
        assert advice.action == Action.HIT


class TestLegalityFallbacks:
    def test_ph_without_das_falls_to_hard_equivalent(self):
        no_das = Rules(das=False)
        # 2,2 vs 2 = Ph -> without DAS treat as hard 4 -> HIT.
        advice = decide((R.TWO, R.TWO), R.TWO, 0, no_das, CTX)
        assert advice.action == Action.HIT
        # 4,4 vs 5 = Ph -> without DAS hard 8 -> HIT.
        assert decide((R.FOUR, R.FOUR), R.FIVE, 0, no_das, CTX).action == Action.HIT
        # Plain P cells still split without DAS.
        assert decide((R.EIGHT, R.EIGHT), R.TEN, 0, no_das, CTX).action == Action.SPLIT

    def test_split_budget_exhausted_falls_through(self):
        maxed = TableContext(num_hands=4)
        # 8,8 with no split left = hard 16 vs T; surrender illegal post-split -> stand/hit path.
        advice = decide((R.EIGHT, R.EIGHT), R.TEN, 0, H17, maxed)
        assert advice.action in (Action.HIT, Action.STAND)  # 16vT stand dev fires at TC>=0
        assert advice.action == Action.STAND

    def test_rp_without_surrender_splits(self):
        advice = decide((R.EIGHT, R.EIGHT), R.ACE, 0, Rules(surrender=Surrender.NONE), CTX)
        assert advice.action == Action.SPLIT

    def test_rs_falls_back_to_stand_when_surrender_unavailable(self):
        # 17vA under H17 is Rs; with surrender NONE the fallback is plain STAND.
        advice = decide((R.EIGHT, R.NINE), R.ACE, 0, Rules(surrender=Surrender.NONE), CTX)
        assert advice.action == Action.STAND

    def test_rh_fallbacks_when_surrender_unavailable(self):
        # 16vT with surrender NONE at negative TC -> HIT (stand dev not fired below 0).
        advice = decide((R.SEVEN, R.NINE), R.TEN, -1, Rules(surrender=Surrender.NONE), CTX)
        assert advice.action == Action.HIT

    def test_surrender_not_vs_ace_rule(self):
        rules = Rules(surrender=Surrender.NOT_VS_ACE)
        advice = decide((R.SEVEN, R.EIGHT), R.ACE, 0, rules, CTX)  # 15vA, H17 basic = Rh
        assert advice.action != Action.SURRENDER

    def test_upcard_face_cards_normalize_to_ten(self):
        for up in (R.JACK, R.QUEEN, R.KING):
            advice = decide((R.SEVEN, R.NINE), up, 0, H17, CTX)
            assert advice.action == Action.SURRENDER  # 16 vs ten-value = Rh

    def test_three_card_hand_carries_caveat(self):
        advice = decide((R.FIVE, R.FOUR, R.THREE), R.SIX, 0, H17, CTX)
        assert advice.caveat is not None


class TestPurity:
    def test_strategy_modules_are_stdlib_pure(self):
        # M1 exit criterion proxy: no disallowed import statements in the strategy
        # package (pure-stdlib allowlist + bjcounter.types). Builtin-based escape
        # hatches (open/eval) are covered by code review, not this scan.
        allowed = {
            "dataclasses", "enum", "typing", "collections", "collections.abc",
            "functools", "itertools", "__future__",
        }
        import bjcounter.strategy.engine  # noqa: F401 — populates sys.modules for the scan

        for name in (
            "bjcounter.strategy.engine",
            "bjcounter.strategy.tables",
            "bjcounter.strategy.deviations",
            "bjcounter.strategy.hands",
        ):
            source = sys.modules[name].__file__
            with open(source, encoding="utf-8") as fh:
                text = fh.read()
            for line in text.splitlines():
                line = line.strip()
                if line.startswith(("import ", "from ")):
                    mod = line.split()[1].split(".")[0]
                    assert mod in allowed or mod == "bjcounter", f"{name}: {line}"
                    if mod == "bjcounter":
                        target = line.split()[1]
                        assert target.startswith(("bjcounter.types", "bjcounter.strategy")), (
                            f"{name} imports outside the pure core: {line}"
                        )


class TestDeterminism:
    def test_decide_is_deterministic(self):
        args = ((R.SEVEN, R.NINE), R.TEN, 0, H17, CTX)
        assert decide(*args) == decide(*args)


@pytest.mark.parametrize("tc", [-10, 10])
def test_extreme_counts_never_crash(tc):
    for cards in ((R.TWO, R.THREE), (R.ACE, R.SEVEN), (R.EIGHT, R.EIGHT)):
        for up in (R.TWO, R.SIX, R.TEN, R.ACE):
            advice = decide(cards, up, tc, H17, CTX)
            assert advice.action is not None
