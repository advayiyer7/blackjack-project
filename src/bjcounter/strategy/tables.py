"""Basic-strategy chart data. Pure data, stdlib only.

Transcribed from docs/research/strategy-tables.md (6 decks, DAS, late surrender, double any
two). Tests re-parse that doc independently and assert cell-for-cell parity — do not edit
these tables without re-running the parity suite.

Codes: H hit / S stand / D double-else-hit / Ds double-else-stand / P split /
Ph split-if-DAS-else-hit / Rh surrender-else-hit / Rs surrender-else-stand /
Rp surrender-else-split.

The hard chart's "17+" row applies to hard 17 exactly; hard 18-21 always stand (the Rs vs A
overlay is a hard-17-only surrender — research §1.1 note).
"""

from __future__ import annotations

UPCARDS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "A")


def _row(codes: str) -> dict[str, str]:
    return dict(zip(UPCARDS, codes.split(), strict=True))


# H17 base chart (dealer hits soft 17). Keys: hard total / soft total / pair rank.
HARD_H17: dict[int, dict[str, str]] = {
    5: _row("H H H H H H H H H H"),
    6: _row("H H H H H H H H H H"),
    7: _row("H H H H H H H H H H"),
    8: _row("H H H H H H H H H H"),
    9: _row("H D D D D H H H H H"),
    10: _row("D D D D D D D D H H"),
    11: _row("D D D D D D D D D D"),
    12: _row("H H S S S H H H H H"),
    13: _row("S S S S S H H H H H"),
    14: _row("S S S S S H H H H H"),
    15: _row("S S S S S H H H Rh Rh"),
    16: _row("S S S S S H H Rh Rh Rh"),
    17: _row("S S S S S S S S S Rs"),
}

SOFT_H17: dict[int, dict[str, str]] = {  # keyed by soft total (A,2 = 13 ... A,9 = 20)
    13: _row("H H H D D H H H H H"),
    14: _row("H H H D D H H H H H"),
    15: _row("H H D D D H H H H H"),
    16: _row("H H D D D H H H H H"),
    17: _row("H D D D D H H H H H"),
    18: _row("Ds Ds Ds Ds Ds S S H H H"),
    19: _row("S S S S Ds S S S S S"),
    20: _row("S S S S S S S S S S"),
}

PAIRS_H17: dict[str, dict[str, str]] = {  # keyed by pair rank ("T" = any ten-value pair)
    "2": _row("Ph Ph P P P P H H H H"),
    "3": _row("Ph Ph P P P P H H H H"),
    "4": _row("H H H Ph Ph H H H H H"),
    "5": _row("D D D D D D D D H H"),
    "6": _row("Ph P P P P H H H H H"),
    "7": _row("P P P P P P H H H H"),
    "8": _row("P P P P P P P P P Rp"),
    "9": _row("P P P P P S P P S S"),
    "T": _row("S S S S S S S S S S"),
    "A": _row("P P P P P P P P P P"),
}

# The six cells that differ under S17 (dealer stands on all 17s) — research §1.4.
# (chart, key, upcard) -> s17 code
S17_DIFF: dict[tuple[str, int | str, str], str] = {
    ("hard", 11, "A"): "H",
    ("hard", 15, "A"): "H",
    ("hard", 17, "A"): "S",
    ("pairs", "8", "A"): "P",
    ("soft", 18, "2"): "S",
    ("soft", 19, "6"): "S",
}


def hard_code(total: int, upcard: str, h17: bool) -> str:
    if total <= 4:
        return "H"
    if total >= 18:
        return "S"  # 18-21 always stand; the Rs overlay is hard-17-only
    code = HARD_H17[total][upcard]
    if not h17:
        code = S17_DIFF.get(("hard", total, upcard), code)
    return code


def soft_code(total: int, upcard: str, h17: bool) -> str:
    if total <= 12:
        return "H"  # soft 12 (A,A unsplittable) always hits
    if total >= 21:
        return "S"
    code = SOFT_H17[total][upcard]
    if not h17:
        code = S17_DIFF.get(("soft", total, upcard), code)
    return code


def pair_code(pair_rank: str, upcard: str, h17: bool) -> str:
    code = PAIRS_H17[pair_rank][upcard]
    if not h17:
        code = S17_DIFF.get(("pairs", pair_rank, upcard), code)
    return code
