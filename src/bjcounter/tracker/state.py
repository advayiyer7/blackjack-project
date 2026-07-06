"""TableState assembly: zone/hand assignment with per-frame self-calibration (§4.3).

Pure, stdlib only. Geometry constants are trainer facts transcribed from
docs/research/trainer-notes.md §4/§6 (the same source vision/synthesize.py cites —
tracker must not import vision, so the numbers are stated in both places; a drift test
in tests/tracker/test_state.py asserts the two transcriptions stay identical).

Any condition that makes the frame untrustworthy (low confidence, a card that fits no
anchor, a fan whose extent disagrees with its card count) lands in TableState.warnings;
reconcile.step() treats a warned frame as SUSPECT and changes nothing (§4.2 step 0).
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import median

from bjcounter.types import Card, Detection, Hand, Rank, Suit, TableState

CARD_W, CARD_H = 67, 94  # sprite size at scale 1.0
FAN_DX = 24
PLAYER_FAN_DY = -4
DEALER_ANCHOR = (422, 10)
# First-card top-left per hand when N hands are in play (player.js Split() table).
PLAYER_ANCHORS: Mapping[int, tuple[tuple[int, int], ...]] = {
    1: ((443, 416),),
    2: ((443, 416), (638, 377)),
    3: ((196, 358), (443, 416), (638, 377)),
    4: ((50, 250), (280, 395), (500, 415), (675, 345)),
}
DEALER_ZONE_MAX_Y = 160  # unscaled; dealer fan sits at y=10, player hands at y>=234
CONF_THRESHOLD = 0.80
AMBIGUOUS_DIST = CARD_W  # a card further than ~1 card width from every slot is suspect
BACK_LABEL = "back"

Position = tuple[float, float]  # unscaled trainer-canvas coords


def _parse_label(label: str) -> tuple[Rank, Suit]:
    try:
        return Rank(label[0]), Suit(label[1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unparseable card label: {label!r}") from exc


def _card(det: Detection) -> Card:
    rank, suit = _parse_label(det.label)
    return Card(rank=rank, suit=suit, bbox=det.bbox, conf=det.conf)


def _fan_cardinality_warning(xs: list[float], n_cards: int, what: str) -> str | None:
    """A fan's x-extent must agree with its card count — catches NMS swallowing one
    card of a tight fan AND duplicate boxes (§4.3). round() ties-to-even is fine here:
    real jitter is a few px against a 24px step, so ties never occur in practice."""
    if n_cards < 2:
        return None
    expected = round((max(xs) - min(xs)) / FAN_DX) + 1
    if expected != n_cards:
        return f"{what} fan extent implies {expected} cards, detected {n_cards}"
    return None


def _fit_layout(
    positions: list[Position], n_hands: int
) -> tuple[float, list[tuple[int, int]], float] | None:
    """Assign each card position to its best (hand, fan-slot) in an n-hand layout.

    Returns (total_cost, [(hand, k) per card], worst_card_cost), or None when the
    layout is impossible: some hand ends up empty, or a hand's leftmost card is not at
    its anchor (every visible hand's first card sits exactly at the anchor — cards are
    never removed individually mid-round).
    """
    anchors = PLAYER_ANCHORS[n_hands]
    assignment: list[tuple[int, int]] = []
    total = 0.0
    worst = 0.0
    per_hand_ks: dict[int, list[int]] = {h: [] for h in range(n_hands)}
    for ux, uy in positions:
        best: tuple[float, int, int] | None = None
        for hand, (ox, oy) in enumerate(anchors):
            k = max(0, round((ux - ox) / FAN_DX))
            ex, ey = ox + k * FAN_DX, oy + k * PLAYER_FAN_DY
            cost = abs(ux - ex) + abs(uy - ey)
            if best is None or cost < best[0]:
                best = (cost, hand, k)
        cost, hand, k = best  # anchors is never empty
        assignment.append((hand, k))
        per_hand_ks[hand].append(k)
        total += cost
        worst = max(worst, cost)
    if any(not ks for ks in per_hand_ks.values()):
        return None
    if any(min(ks) != 0 for ks in per_hand_ks.values()):
        return None
    return total, assignment, worst


def _assemble_dealer(
    dets: list[Detection], positions: list[Position]
) -> tuple[tuple[Card, ...], bool, list[str]]:
    """Dealer fan in x order; the hole (a back) occupies a fan slot but is no face."""
    order = sorted(range(len(dets)), key=lambda i: dets[i].bbox[0])
    has_hole = any(d.label == BACK_LABEL for d in dets)
    cards = tuple(_card(dets[i]) for i in order if dets[i].label != BACK_LABEL)
    warning = _fan_cardinality_warning([p[0] for p in positions], len(dets), "dealer")
    return cards, has_hole, [warning] if warning else []


def _assemble_player(
    dets: list[Detection], positions: list[Position]
) -> tuple[tuple[Hand, ...], list[str]]:
    """Fit the best 1..4-hand layout and group cards into hands, x-ordered."""
    warnings: list[str] = []
    fits = {n: _fit_layout(positions, n) for n in PLAYER_ANCHORS}
    valid = {n: f for n, f in fits.items() if f is not None}
    if not valid:
        warnings.append("player cards fit no hand layout")
        ordered = sorted(dets, key=lambda d: d.bbox[0])
        return (Hand(slot=0, cards=tuple(_card(d) for d in ordered)),), warnings

    n_best = min(valid, key=lambda n: (valid[n][0], n))
    _, assignment, worst = valid[n_best]
    if worst > AMBIGUOUS_DIST:
        warnings.append(f"card {worst:.0f}px from its nearest hand slot (ambiguous)")
    grouped: dict[int, list[tuple[Detection, Position]]] = {}
    for det, pos, (hand, _) in zip(dets, positions, assignment, strict=True):
        grouped.setdefault(hand, []).append((det, pos))
    hands = []
    for slot in sorted(grouped):
        members = sorted(grouped[slot], key=lambda m: m[0].bbox[0])
        warning = _fan_cardinality_warning(
            [pos[0] for _, pos in members], len(members), f"hand {slot}"
        )
        if warning:
            warnings.append(warning)
        hands.append(Hand(slot=slot, cards=tuple(_card(d) for d, _ in members)))
    return tuple(hands), warnings


def assemble_table(
    frame_id: int,
    detections: tuple[Detection, ...],
    *,
    table_origin: tuple[int, int] = (0, 0),
    conf_threshold: float = CONF_THRESHOLD,
) -> TableState:
    """Build a TableState from raw detections (capture-region pixel coords).

    `table_origin` is where the trainer's 960x640 canvas starts inside the captured
    frame — (0, bar_height) when the capture includes the count bar above the felt.
    """
    if not detections:
        return TableState(
            frame_id=frame_id, player_hands=(), dealer_cards=(), dealer_has_hole=False
        )

    warnings = [
        f"low confidence {d.label} at {d.bbox[:2]}: {d.conf:.2f}"
        for d in detections
        if d.conf < conf_threshold
    ]
    scale = median(d.bbox[2] for d in detections) / CARD_W
    ox, oy = table_origin

    def unscaled(det: Detection) -> Position:
        return (det.bbox[0] - ox) / scale, (det.bbox[1] - oy) / scale

    dealer, player = [], []
    for det in detections:
        pos = unscaled(det)
        (dealer if pos[1] < DEALER_ZONE_MAX_Y else player).append((det, pos))

    dealer_cards, dealer_has_hole, dealer_warnings = _assemble_dealer(
        [d for d, _ in dealer], [p for _, p in dealer]
    )
    warnings += dealer_warnings

    player_hands: tuple[Hand, ...] = ()
    if any(d.label == BACK_LABEL for d, _ in player):
        warnings.append("face-down card in the player zone")
        player = [(d, p) for d, p in player if d.label != BACK_LABEL]
    if player:
        player_hands, player_warnings = _assemble_player(
            [d for d, _ in player], [p for _, p in player]
        )
        warnings += player_warnings

    return TableState(
        frame_id=frame_id,
        player_hands=player_hands,
        dealer_cards=dealer_cards,
        dealer_has_hole=dealer_has_hole,
        warnings=tuple(warnings),
    )
