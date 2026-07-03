"""Hi-Lo tag values and running-count arithmetic."""

from __future__ import annotations

from bjcounter.counting.hilo import HILO_TAGS, running_count
from bjcounter.types import Rank


def test_tags_match_research_doc(doc_hilo_tags):
    assert {str(rank): tag for rank, tag in HILO_TAGS.items()} == doc_hilo_tags


def test_tags_are_balanced_over_one_deck():
    # 4 suits per rank: a full deck must sum to zero (balanced count).
    assert sum(4 * tag for tag in HILO_TAGS.values()) == 0


def test_running_count_examples():
    assert running_count(()) == 0
    assert running_count((Rank.TWO, Rank.SIX)) == 2
    assert running_count((Rank.TEN, Rank.ACE, Rank.KING)) == -3
    assert running_count((Rank.SEVEN, Rank.EIGHT, Rank.NINE)) == 0
    assert running_count((Rank.FIVE, Rank.TEN)) == 0
