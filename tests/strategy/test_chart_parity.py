"""Behavioral chart parity: every cell of both rule charts, driven through engine.decide().

At TC=0 with full legality (fresh 2-card hand, one hand, DAS on, LS-any-card, double-any-two)
the engine must reproduce the research chart cell exactly, with is_deviation False.
"""

from __future__ import annotations

import pytest

from bjcounter.strategy.engine import decide
from bjcounter.types import Action, Rank, Rules, TableContext

from .conftest import chart_cases

FULL_CTX = TableContext(num_hands=1)
RULES = {"h17": Rules(h17=True), "s17": Rules(h17=False)}

# Non-pair, ace-free two-card hands for each hard total (chart row "17+" tested as hard 17;
# hard 18-21 are covered separately in test_engine.py since the chart collapses them).
HARD_HANDS = {
    "5": ("2", "3"), "6": ("2", "4"), "7": ("3", "4"), "8": ("3", "5"),
    "9": ("4", "5"), "10": ("4", "6"), "11": ("5", "6"), "12": ("5", "7"),
    "13": ("6", "7"), "14": ("6", "8"), "15": ("7", "8"), "16": ("7", "9"),
    "17+": ("8", "9"),
}

# code -> (action, fallback) under full legality
CODE_TO_ACTION = {
    "H": (Action.HIT, None),
    "S": (Action.STAND, None),
    "D": (Action.DOUBLE, Action.HIT),
    "Ds": (Action.DOUBLE, Action.STAND),
    "P": (Action.SPLIT, None),
    "Ph": (Action.SPLIT, None),  # DAS is on in the parity config
    "Rh": (Action.SURRENDER, Action.HIT),
    "Rs": (Action.SURRENDER, Action.STAND),
    "Rp": (Action.SURRENDER, Action.SPLIT),
}


def hand_ranks(kind: str, hand_label: str) -> tuple[Rank, ...]:
    if kind == "hard":
        return tuple(Rank(c) for c in HARD_HANDS[hand_label])
    # soft "A,x" and pairs "x,x" labels are literal rank pairs
    a, b = hand_label.split(",")
    return (Rank(a), Rank(b))


@pytest.mark.parametrize(
    ("ruleset", "kind", "hand_label", "upcard", "code"),
    chart_cases(),
    ids=lambda v: str(v),
)
def test_chart_cell(ruleset, kind, hand_label, upcard, code):
    advice = decide(
        cards=hand_ranks(kind, hand_label),
        upcard=Rank(upcard),
        tc=0,
        rules=RULES[ruleset],
        ctx=FULL_CTX,
    )
    expected_action, expected_fallback = CODE_TO_ACTION[code]
    assert advice.action == expected_action, (
        f"{ruleset} {kind} {hand_label} vs {upcard}: chart={code} got={advice}"
    )
    assert advice.fallback == expected_fallback
    assert advice.is_deviation is False, (
        f"{ruleset} {kind} {hand_label} vs {upcard}: parity cell flagged as deviation: {advice}"
    )
