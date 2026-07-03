"""Illustrious 18 + Fab 4 count deviations. Pure data + rules-aware filtering.

Indices/directions per docs/research/strategy-tables.md §4/§5 (Schlesinger convention):
  AT_OR_ABOVE -> take the play when TC >= index
  BELOW       -> take the play (Hit) STRICTLY below the index (discrepancy #7)

H17 carve-outs (research §6): 11vA double and 15vA surrender are already basic strategy
under H17, so those entries exist only under S17. The H17-specific indices for 10vA and
12v6 are UNRESOLVED in the research; the published S17 values are used for both rule sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache

from bjcounter.types import Action, Rules

INSURANCE_INDEX = 3  # take insurance vs ace at TC >= +3


class Direction(StrEnum):
    AT_OR_ABOVE = "at_or_above"
    BELOW = "below"


@dataclass(frozen=True, slots=True)
class DevEntry:
    action: Action
    hand_key: str  # hard total ("9".."16") or pair key ("T,T")
    upcard: str  # "2".."9", "T", "A"
    index: int
    direction: Direction = Direction.AT_OR_ABOVE
    s17_only: bool = False
    label: str = ""

    def fires(self, tc: int) -> bool:
        if self.direction is Direction.AT_OR_ABOVE:
            return tc >= self.index
        return tc < self.index  # strict — at the index itself play basic


# Illustrious 18 (insurance handled separately via INSURANCE_INDEX).
I18: tuple[DevEntry, ...] = (
    DevEntry(Action.STAND, "16", "T", 0, label="I18: 16vT stand @ TC>=0"),
    DevEntry(Action.STAND, "15", "T", 4, label="I18: 15vT stand @ TC>=4"),
    DevEntry(Action.SPLIT, "T,T", "5", 5, label="I18: TTv5 split @ TC>=5"),
    DevEntry(Action.SPLIT, "T,T", "6", 4, label="I18: TTv6 split @ TC>=4"),
    DevEntry(Action.DOUBLE, "10", "T", 4, label="I18: 10vT double @ TC>=4"),
    DevEntry(Action.STAND, "12", "3", 2, label="I18: 12v3 stand @ TC>=2"),
    DevEntry(Action.STAND, "12", "2", 3, label="I18: 12v2 stand @ TC>=3"),
    DevEntry(Action.DOUBLE, "11", "A", 1, s17_only=True, label="I18: 11vA double @ TC>=1"),
    DevEntry(Action.DOUBLE, "9", "2", 1, label="I18: 9v2 double @ TC>=1"),
    DevEntry(Action.DOUBLE, "10", "A", 4, label="I18: 10vA double @ TC>=4"),
    DevEntry(Action.DOUBLE, "9", "7", 3, label="I18: 9v7 double @ TC>=3"),
    DevEntry(Action.STAND, "16", "9", 5, label="I18: 16v9 stand @ TC>=5"),
    DevEntry(Action.HIT, "13", "2", -1, Direction.BELOW, label="I18: 13v2 hit @ TC<-1"),
    DevEntry(Action.HIT, "12", "4", 0, Direction.BELOW, label="I18: 12v4 hit @ TC<0"),
    DevEntry(Action.HIT, "12", "5", -2, Direction.BELOW, label="I18: 12v5 hit @ TC<-2"),
    DevEntry(Action.HIT, "12", "6", -1, Direction.BELOW, label="I18: 12v6 hit @ TC<-1"),
    DevEntry(Action.HIT, "13", "3", -2, Direction.BELOW, label="I18: 13v3 hit @ TC<-2"),
)

# Fab 4 surrenders (with LS available). 15vA is S17-only (H17: already basic surrender).
FAB4: tuple[DevEntry, ...] = (
    DevEntry(Action.SURRENDER, "14", "T", 3, label="Fab4: 14vT surrender @ TC>=3"),
    DevEntry(Action.SURRENDER, "15", "T", 0, label="Fab4: 15vT surrender @ TC>=0"),
    DevEntry(Action.SURRENDER, "15", "9", 2, label="Fab4: 15v9 surrender @ TC>=2"),
    DevEntry(Action.SURRENDER, "15", "A", 1, s17_only=True, label="Fab4: 15vA surrender @ TC>=1"),
)

# Stand-over-surrender crossovers explicitly published in our research (§4 note / §6#2):
# 15vT stands at TC>=4 and 16v9 stands at TC>=5 even with surrender available. No such
# crossover is published for 16vT/16vA/15vA — those surrender at all counts when legal.
SURRENDER_STAND_OVERRIDES: dict[tuple[str, str], int] = {
    ("15", "T"): 4,
    ("16", "9"): 5,
}


@cache
def _index(entries: tuple[DevEntry, ...]) -> dict[tuple[str, str], DevEntry]:
    return {(e.hand_key, e.upcard): e for e in entries}


def lookup(
    entries: tuple[DevEntry, ...], rules: Rules, hand_key: str, upcard: str
) -> DevEntry | None:
    """O(1) deviation lookup, filtering S17-only entries out under H17 rules."""
    entry = _index(entries).get((hand_key, upcard))
    if entry is None or (entry.s17_only and rules.h17):
        return None
    return entry
