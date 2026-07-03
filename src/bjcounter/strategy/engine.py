"""Count-aware strategy engine. Pure, stdlib only (ARCHITECTURE §6).

Precedence: insurance flag -> surrender (Fab 4 / chart, with published stand-over-surrender
crossovers) -> split (T,T deviations, pair chart) -> hard/soft (I18 deviations, chart).
Deviations requiring an illegal action do not fire; hit/stand deviations apply to any
card count. A splittable pair consults ONLY the pair chart's own surrender cell (Rp) —
the pair cell supersedes the hand's hard-total surrender (8,8 vs T is P, never Rh).
"""

from __future__ import annotations

from dataclasses import replace

from bjcounter.strategy import tables
from bjcounter.strategy.deviations import (
    FAB4,
    I18,
    INSURANCE_INDEX,
    SURRENDER_STAND_OVERRIDES,
    DevEntry,
    lookup,
)
from bjcounter.strategy.hands import (
    HandView,
    can_double,
    can_split,
    can_surrender,
    hand_view,
    normalize_ten,
)
from bjcounter.types import Action, Advice, Rank, Rules, TableContext

MULTI_CARD_CAVEAT = "3+ cards: ignore if hand already doubled"


def decide(
    cards: tuple[Rank, ...], upcard: Rank, tc: int, rules: Rules, ctx: TableContext
) -> Advice:
    """Optimal action for one hand given the floored true count.

    Internally the upcard is normalized (ten-values collapse to "T") and passed to the
    advice paths as the chart-column string `up`.
    """
    norm_up = normalize_ten(upcard)
    up = str(norm_up)
    view = hand_view(cards)
    insurance = up == "A" and tc >= INSURANCE_INDEX
    caveat = MULTI_CARD_CAVEAT if view.n_cards >= 3 else None

    if view.total > 21:
        return Advice(action=None, insurance=insurance, caveat="busted")
    if view.total == 21:
        return Advice(action=Action.STAND, insurance=insurance, caveat=caveat)

    for path in (_surrender_advice, _split_advice, _hard_soft_advice):
        advice = path(view, up, tc, rules, ctx)
        if advice is not None:
            return replace(advice, insurance=insurance, caveat=caveat)
    raise AssertionError(  # pragma: no cover — _hard_soft_advice always returns
        "unreachable: chart interpretation always yields an action"
    )


def _surrender_advice(
    view: HandView, up: str, tc: int, rules: Rules, ctx: TableContext
) -> Advice | None:
    if not can_surrender(view, Rank(up), rules, ctx) or view.is_soft:
        return None
    if view.pair_rank is not None and can_split(view, rules, ctx):
        pair_cell = tables.pair_code(str(view.pair_rank), up, rules.h17)
        if pair_cell == "Rp":
            return Advice(action=Action.SURRENDER, fallback=Action.SPLIT)
        if pair_cell == "P" or (pair_cell == "Ph" and rules.das):
            return None  # split is the play; the pair cell supersedes hard-total surrender
        # Non-split pair cell (e.g. 7,7 vs T = H): the hand plays exactly as its hard
        # total, INCLUDING Fab4/override surrender logic below (M1 review finding 1).

    key = str(view.total)
    override_index = SURRENDER_STAND_OVERRIDES.get((key, up))
    if override_index is not None and tc >= override_index:
        return Advice(
            action=Action.STAND,
            is_deviation=True,
            deviation=f"I18: {key}v{up} stand @ TC>={override_index}",
        )

    chart_code = tables.hard_code(view.total, up, rules.h17)
    chart_fallback = Action.STAND if chart_code == "Rs" else Action.HIT
    fab = lookup(FAB4, rules, key, up)
    if fab is not None:
        if fab.fires(tc):
            return Advice(
                action=Action.SURRENDER,
                fallback=chart_fallback,
                is_deviation=chart_code not in ("Rh", "Rs"),
                deviation=fab.label if chart_code not in ("Rh", "Rs") else None,
            )
        if chart_code in ("Rh", "Rs"):
            # Chart surrenders, but below the Fab 4 index the play is the chart's own
            # non-surrender fallback (symmetric for Rh/Rs — M1 review).
            return Advice(
                action=chart_fallback,
                is_deviation=True,
                deviation=f"{fab.label} (decline surrender below index)",
            )
        return None
    if chart_code == "Rh":
        return Advice(action=Action.SURRENDER, fallback=Action.HIT)
    if chart_code == "Rs":
        return Advice(action=Action.SURRENDER, fallback=Action.STAND)
    return None


def _split_advice(
    view: HandView, up: str, tc: int, rules: Rules, ctx: TableContext
) -> Advice | None:
    if view.pair_rank is None or not can_split(view, rules, ctx):
        return None
    if view.pair_rank == Rank.TEN:
        dev = lookup(I18, rules, "T,T", up)
        if dev is not None and dev.fires(tc):
            return Advice(action=Action.SPLIT, is_deviation=True, deviation=dev.label)
    code = tables.pair_code(str(view.pair_rank), up, rules.h17)
    if code == "P" or (code == "Ph" and rules.das):
        return Advice(action=Action.SPLIT)
    if code == "Rp":
        return Advice(action=Action.SPLIT)  # surrender side handled in _surrender_advice
    return None  # H/S/D/Ds cells and DAS-less Ph: play the hand as its total


def _hard_soft_advice(
    view: HandView, up: str, tc: int, rules: Rules, ctx: TableContext
) -> Advice:
    if view.is_soft:
        return _interpret(tables.soft_code(view.total, up, rules.h17), view, up, rules, ctx)
    dev = lookup(I18, rules, str(view.total), up)
    if dev is not None and dev.fires(tc) and _dev_legal(dev, view, rules):
        fallback = Action.HIT if dev.action is Action.DOUBLE else None
        return Advice(
            action=dev.action, fallback=fallback, is_deviation=True, deviation=dev.label
        )
    return _interpret(tables.hard_code(view.total, up, rules.h17), view, up, rules, ctx)


def _dev_legal(dev: DevEntry, view: HandView, rules: Rules) -> bool:
    return dev.action != Action.DOUBLE or can_double(view, rules)


def _interpret(code: str, view: HandView, up: str, rules: Rules, ctx: TableContext) -> Advice:
    """Chart code -> Advice with legality-resolved fallbacks."""
    if code == "H":
        return Advice(action=Action.HIT)
    if code == "S":
        return Advice(action=Action.STAND)
    if code == "D":
        if can_double(view, rules):
            return Advice(action=Action.DOUBLE, fallback=Action.HIT)
        return Advice(action=Action.HIT)
    if code == "Ds":
        if can_double(view, rules):
            return Advice(action=Action.DOUBLE, fallback=Action.STAND)
        return Advice(action=Action.STAND)
    if code == "Rh":  # surrender already ruled out upstream
        return Advice(action=Action.HIT)
    if code == "Rs":
        return Advice(action=Action.STAND)
    raise AssertionError(  # pragma: no cover — defensive; chart data is closed-set
        f"unhandled chart code {code!r} for {view} vs {up}"
    )
