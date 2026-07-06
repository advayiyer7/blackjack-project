"""Transition-classifier tests, including the adversarial sequences that the §13
review findings 1-5 demand. Every SUSPECT assertion also checks the count channel
stayed untouched (revealed empty, round state unchanged)."""

from __future__ import annotations

from bjcounter.counting.hilo import running_count
from bjcounter.tracker.reconcile import step
from bjcounter.tracker.rounds import fresh_shoe
from bjcounter.types import Event
from tests.tracker.conftest import make_table


def deal(frame_id=0, player=("8s", "7h"), upcard="Th"):
    return make_table(frame_id, dealer=(upcard,), hands=(player,), hole=True)


def ranks(cards):
    return sorted(str(c.rank) for c in cards)


class TestFreshShoe:
    def test_first_deal_starts_round_one_and_counts_all_three_faces(self):
        result = step(fresh_shoe(), deal())
        assert result.accepted
        assert result.events == (Event.NEW_ROUND,)
        assert ranks(result.revealed) == ["7", "8", "T"]
        assert result.round_state.round_index == 1
        assert not result.round_state.settled

    def test_mid_round_frame_at_fresh_shoe_is_suspect(self):
        mid = make_table(0, dealer=("Th", "9d"), hands=(("8s", "7h", "5c"),))
        result = step(fresh_shoe(), mid)
        assert not result.accepted
        assert result.revealed == ()
        assert result.round_state == fresh_shoe()

    def test_warned_frame_is_suspect_even_if_deal_shaped(self):
        warned = make_table(
            0, dealer=("Th",), hands=(("8s", "7h"),), hole=True,
            warnings=("low confidence 8s at (443, 416): 0.55",),
        )
        result = step(fresh_shoe(), warned)
        assert not result.accepted and result.revealed == ()


class TestNormalRound:
    """Deal -> hit -> settle -> next deal: the count is exact at every accepted frame."""

    def test_full_round_reveals_each_delta_exactly_once(self):
        r1 = step(fresh_shoe(), deal(0))

        hit = make_table(1, dealer=("Th",), hands=(("8s", "7h", "5c"),), hole=True)
        r2 = step(r1.round_state, hit)
        assert r2.accepted and r2.events == ()
        assert ranks(r2.revealed) == ["5"]

        settle = make_table(2, dealer=("Th", "9d", "6c"), hands=(("8s", "7h", "5c"),))
        r3 = step(r2.round_state, settle)
        assert r3.accepted and r3.events == (Event.HOLE_REVEALED,)
        assert ranks(r3.revealed) == ["6", "9"]
        assert r3.round_state.settled and r3.round_state.hole_was_revealed

        r4 = step(r3.round_state, deal(3, player=("2c", "3d"), upcard="4h"))
        assert r4.events == (Event.NEW_ROUND,)  # settled round -> no unsettled warning
        assert ranks(r4.revealed) == ["2", "3", "4"]
        assert r4.round_state.round_index == 2

        # Revealed across the sequence: 8,7,T,5,9,6,2,3,4 -> hi-lo 0+0-1+1+0+1+1+1+1
        seen = [c.rank for r in (r1, r2, r3, r4) for c in r.revealed]
        assert running_count(seen) == 4

    def test_capture_only_at_settle_still_counts_everything(self):
        r1 = step(fresh_shoe(), deal(0))
        settle = make_table(
            1, dealer=("Th", "9d", "6c"), hands=(("8s", "7h", "5c", "2d"),)
        )
        r2 = step(r1.round_state, settle)
        assert r2.accepted
        assert ranks(r2.revealed) == ["2", "5", "6", "9"]


class TestFinding1DroppedDetection:
    def test_vanished_card_mid_round_is_suspect_not_a_new_round(self):
        r1 = step(fresh_shoe(), deal(0))
        dropped = make_table(1, dealer=("Th",), hands=(("8s",),), hole=True)  # 7h lost
        result = step(r1.round_state, dropped)
        assert not result.accepted
        assert result.revealed == ()
        assert result.round_state == r1.round_state  # count channel untouched
        assert any("no longer visible" in w for w in result.warnings)

    def test_dropped_hit_card_making_the_frame_deal_shaped_is_suspect(self):
        # Nastiest variant: after a hit, losing the third card makes the frame
        # deal-shaped again — the persistence rule must refuse the fake new round.
        r1 = step(fresh_shoe(), deal(0))
        hit = make_table(1, dealer=("Th",), hands=(("8s", "7h", "5c"),), hole=True)
        r2 = step(r1.round_state, hit)
        dropped = make_table(2, dealer=("Th",), hands=(("8s", "7h"),), hole=True)
        result = step(r2.round_state, dropped)
        assert not result.accepted
        assert result.round_state == r2.round_state
        assert any("not a credible new deal" in w for w in result.warnings)


class TestFinding2CapturedSurrender:
    def test_surrender_frame_is_round_ended_and_recounts_nothing(self):
        r1 = step(fresh_shoe(), deal(0))
        surrender = make_table(1, dealer=("Th",), hands=(), hole=True)
        result = step(r1.round_state, surrender)
        assert result.accepted
        assert result.events == (Event.ROUND_ENDED,)
        assert result.revealed == ()  # the persisting upcard is NOT double-counted
        assert result.round_state.settled

    def test_next_deal_after_surrender_is_a_clean_new_round(self):
        r1 = step(fresh_shoe(), deal(0))
        surrender = make_table(1, dealer=("Th",), hands=(), hole=True)
        r2 = step(r1.round_state, surrender)
        r3 = step(r2.round_state, deal(2, player=("9c", "9d"), upcard="Ah"))
        assert r3.events == (Event.NEW_ROUND,)
        assert ranks(r3.revealed) == ["9", "9", "A"]

    def test_full_table_clear_is_round_ended(self):
        r1 = step(fresh_shoe(), deal(0))
        settle = make_table(1, dealer=("Th", "9d"), hands=(("8s", "7h"),))
        r2 = step(r1.round_state, settle)
        cleared = make_table(2, dealer=(), hands=())
        r3 = step(r2.round_state, cleared)
        assert r3.accepted and r3.events == (Event.ROUND_ENDED,)
        assert r3.revealed == ()


class TestFinding3DuplicateAndRedeal:
    def test_identical_frame_is_a_duplicate_no_op(self):
        r1 = step(fresh_shoe(), deal(0))
        result = step(r1.round_state, deal(1))
        assert result.accepted
        assert result.events == (Event.DUPLICATE_FRAME,)
        assert result.revealed == ()
        assert result.round_state == r1.round_state
        assert any("recapture" in w for w in result.warnings)  # §12 residual risk

    def test_different_deal_shaped_frame_is_a_second_round(self):
        r1 = step(fresh_shoe(), deal(0))
        redeal = deal(1, player=("2c", "3d"), upcard="4h")
        result = step(r1.round_state, redeal)
        assert Event.NEW_ROUND in result.events
        assert Event.PREV_ROUND_UNSETTLED in result.events  # round 1 never settled
        assert result.round_state.round_index == 2
        assert ranks(result.revealed) == ["2", "3", "4"]

    def test_redeal_repeating_one_card_at_an_anchor_is_still_a_new_deal(self):
        # A six-deck shoe repeats an exact card at a reused anchor ~2% of rounds; a
        # single persisting face must NOT stall the tracker (the card is physically
        # new and must be counted again).
        r1 = step(fresh_shoe(), deal(0))  # 8s,7h vs Th
        settle = make_table(1, dealer=("Th", "9d"), hands=(("8s", "7h"),))
        r2 = step(r1.round_state, settle)
        redeal = deal(2, player=("8s", "2d"), upcard="6h")  # 8s repeats at slot 0
        result = step(r2.round_state, redeal)
        assert result.accepted
        assert Event.NEW_ROUND in result.events
        assert ranks(result.revealed) == ["2", "6", "8"]


class TestFinding4LabelFlicker:
    def test_suit_flicker_on_a_deal_shaped_frame_is_suspect(self):
        # The flickered frame is still deal-shaped; the persistence rule (2 of 3 faces
        # carry over) must keep it from double-counting as a second round.
        r1 = step(fresh_shoe(), deal(0))  # player 8s,7h
        flicker = make_table(1, dealer=("Th",), hands=(("8h", "7h"),), hole=True)
        result = step(r1.round_state, flicker)
        assert not result.accepted
        assert result.round_state == r1.round_state
        assert any("not a credible new deal" in w for w in result.warnings)

    def test_suit_flicker_mid_round_is_suspect(self):
        r1 = step(fresh_shoe(), deal(0))
        hit = make_table(1, dealer=("Th",), hands=(("8s", "7h", "5c"),), hole=True)
        r2 = step(r1.round_state, hit)
        flicker = make_table(2, dealer=("Th",), hands=(("8s", "7d", "5c"),), hole=True)
        result = step(r2.round_state, flicker)
        assert not result.accepted
        assert result.round_state == r2.round_state
        assert any("label changed" in w for w in result.warnings)

    def test_double_flicker_on_a_deal_shaped_recapture_is_suspect(self):
        # Review finding: with only a >=2-persistence bar, misreading TWO of the three
        # cards on a recapture of a still-showing deal dropped persistence to 1 and
        # silently recounted the round. Deal-shaped -> deal-shaped transitions share
        # all three anchors, so ANY surviving label must refuse NEW_DEAL.
        r1 = step(fresh_shoe(), deal(0, player=("As", "Kh")))  # natural, settled
        assert r1.round_state.settled
        recapture = deal(1, player=("Ad", "Kc"))  # both player suits misread; Th persists
        result = step(r1.round_state, recapture)
        assert not result.accepted
        assert result.revealed == ()
        assert result.round_state == r1.round_state

    def test_double_flicker_mid_hand_deal_recapture_is_suspect(self):
        r1 = step(fresh_shoe(), deal(0))  # 8s,7h vs Th — still live
        recapture = deal(1, player=("8d", "7c"))  # both suits misread; Th persists
        result = step(r1.round_state, recapture)
        assert not result.accepted
        assert result.round_state == r1.round_state

    def test_quick_redeal_repeating_one_card_stalls_as_suspect(self):
        # Deal -> deal with one exact repeat is indistinguishable from a double
        # flicker, so it stalls loudly (rare compound case: skipped settle AND an
        # exact-card repeat). The settle -> deal path keeps the lenient threshold —
        # see test_redeal_repeating_one_card_at_an_anchor_is_still_a_new_deal.
        r1 = step(fresh_shoe(), deal(0))  # 8s,7h vs Th
        redeal = deal(1, player=("8s", "2d"), upcard="6h")  # 8s repeats at its anchor
        result = step(r1.round_state, redeal)
        assert not result.accepted
        assert result.round_state == r1.round_state

    def test_rank_flicker_on_dealer_upcard_is_suspect(self):
        r1 = step(fresh_shoe(), deal(0))  # dealer Th
        flicker = make_table(1, dealer=("7h",), hands=(("8s", "7h", "5c"),), hole=True)
        result = step(r1.round_state, flicker)
        assert not result.accepted
        assert any("label changed" in w for w in result.warnings)


class TestFinding5SkippedSettle:
    def test_new_deal_after_unsettled_round_flags_short_count(self):
        r1 = step(fresh_shoe(), deal(0))
        hit = make_table(1, dealer=("Th",), hands=(("8s", "7h", "5c"),), hole=True)
        r2 = step(r1.round_state, hit)
        # settle never captured; next capture is already the next deal
        r3 = step(r2.round_state, deal(2, player=("6c", "6d"), upcard="2s"))
        assert Event.PREV_ROUND_UNSETTLED in r3.events
        assert any("count may be short" in w for w in r3.warnings)

    def test_natural_settles_without_hole_reveal(self):
        r1 = step(fresh_shoe(), deal(0, player=("As", "Kh")))
        assert r1.round_state.settled
        r2 = step(r1.round_state, deal(1, player=("2c", "3d"), upcard="4h"))
        assert Event.PREV_ROUND_UNSETTLED not in r2.events


class TestSplits:
    def test_split_relocation_counts_only_the_drawn_cards(self):
        r1 = step(fresh_shoe(), deal(0, player=("8s", "8h")))
        split = make_table(
            1, dealer=("Th",), hands=(("8s", "5c"), ("8h", "9d")), hole=True
        )
        result = step(r1.round_state, split)
        assert result.accepted
        assert Event.SPLIT in result.events
        assert ranks(result.revealed) == ["5", "9"]  # the 8s/8h are already counted

    def test_split_before_draws_reveals_nothing(self):
        r1 = step(fresh_shoe(), deal(0, player=("8s", "8h")))
        split = make_table(1, dealer=("Th",), hands=(("8s",), ("8h",)), hole=True)
        result = step(r1.round_state, split)
        assert result.accepted
        assert result.events == (Event.SPLIT,)
        assert result.revealed == ()

    def test_double_split_between_captures_emits_two_split_events(self):
        r1 = step(fresh_shoe(), deal(0, player=("8s", "8h")))
        resplit = make_table(
            1, dealer=("Th",), hands=(("8s", "4c"), ("8h", "8d"), ("8d", "2c")), hole=True
        )
        result = step(r1.round_state, resplit)
        assert result.accepted
        assert result.events.count(Event.SPLIT) == 2

    def test_hand_count_decrease_without_clear_is_suspect(self):
        r1 = step(fresh_shoe(), deal(0, player=("8s", "8h")))
        split = make_table(1, dealer=("Th",), hands=(("8s", "5c"), ("8h", "9d")), hole=True)
        r2 = step(r1.round_state, split)
        merged = make_table(2, dealer=("Th",), hands=(("8s", "5c"),), hole=True)
        result = step(r2.round_state, merged)
        assert not result.accepted
        assert result.round_state == r2.round_state


class TestDealerIntegrity:
    def test_back_appearing_mid_round_is_suspect(self):
        r1 = step(fresh_shoe(), deal(0))
        settle = make_table(1, dealer=("Th", "9d"), hands=(("8s", "7h"),))
        r2 = step(r1.round_state, settle)
        ghost = make_table(2, dealer=("Th", "9d"), hands=(("8s", "7h"),), hole=True,
                           hole_slot=2)
        result = step(r2.round_state, ghost)
        assert not result.accepted
        assert any("hole card appeared" in w for w in result.warnings)

    def test_hole_flag_dropping_without_new_face_is_suspect(self):
        r1 = step(fresh_shoe(), deal(0))
        vanished = make_table(1, dealer=("Th",), hands=(("8s", "7h"),))
        result = step(r1.round_state, vanished)
        assert not result.accepted
        assert any("hole card vanished" in w for w in result.warnings)

    def test_moved_dealer_card_is_suspect(self):
        r1 = step(fresh_shoe(), deal(0))
        # Same multiset, but the upcard sits one fan slot to the right.
        moved = make_table(1, dealer=("Th",), hands=(("8s", "7h"),), hole=True,
                           hole_slot=0)
        result = step(r1.round_state, moved)
        assert not result.accepted


class TestMultiplicity:
    def test_duplicate_rank_suit_pairs_count_with_multiplicity(self):
        # Six-deck shoe: the same physical card label can appear twice in one round.
        r1 = step(fresh_shoe(), deal(0, player=("8h", "8h")))
        assert ranks(r1.revealed) == ["8", "8", "T"]
        hit = make_table(1, dealer=("Th",), hands=(("8h", "8h", "8h"),), hole=True)
        r2 = step(r1.round_state, hit)
        assert r2.accepted
        assert ranks(r2.revealed) == ["8"]  # exactly one new 8h, not three
