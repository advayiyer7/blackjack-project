"""Synthetic frame compositor: trainer sprites onto the felt at trainer-exact geometry.

Real capture yields ~175 card instances per session across 53 classes, so reaching the
40-instance M3 training gate organically needs ~10 more manual sessions. The trainer
renders fixed sprites (deck.png) at fixed anchors on a fixed felt (table.png) and scales
the whole canvas as one unit (CSS transform), so compositing reproduces captured frames
faithfully: paste sprites at unscaled coords, resize the whole canvas, prepend the count
bar exactly as scripts/capture_session.py captures it. Synthetic frames form the M4
TRAINING pool; real captured frames are reserved for val/test.

Known domain gap (accepted, revisit at M4 if val false-positives appear): real frames
also contain chips, action buttons, and rule-strip text; synthetic frames are bare felt
plus the count bar. Backgrounds only suppress false positives — card recall is untouched.

Geometry source: docs/research/trainer-notes.md §4/§6 (player.js/dealer.js coordinates).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from bjcounter.vision.autolabel import (
    BACK_CLASS,
    BACK_Y,
    CARD_H,
    CARD_W,
    CLASS_NAMES,
    CardHit,
    yolo_lines,
)

TABLE_W, TABLE_H = 960, 640
COUNT_BAR_CSS_PX = 36  # keep in sync with scripts/capture_session.py
DEALER_ANCHOR = (422, 10)
FAN_DX = 24
PLAYER_FAN_DY = -4
# First-card top-left per hand when N hands are in play (player.js Split() table).
PLAYER_ANCHORS: Mapping[int, tuple[tuple[int, int], ...]] = {
    1: ((443, 416),),
    2: ((443, 416), (638, 377)),
    3: ((196, 358), (443, 416), (638, 377)),
    4: ((50, 250), (280, 395), (500, 415), (675, 345)),
}
# Whole-canvas scales seen in practice: CSS breakpoint 0.9 and DPI-zoomed captures.
SCALES = (0.9, 1.0, 1.12, 1.25)
HAND_COUNT_WEIGHTS = {1: 15, 2: 35, 3: 25, 4: 25}  # splits first: real data covers 1-hand
BAR_TEXT = "Running Count: 0  |  Decks Left: 6.00  |  True Count: 0"


@dataclass(frozen=True, slots=True)
class Placement:
    class_id: int
    x: int  # unscaled table coords, top-left of the full card
    y: int


@dataclass(frozen=True, slots=True)
class FrameSpec:
    dealer: tuple[int, ...]  # class ids in fan order; BACK_CLASS = face-down hole
    hands: tuple[tuple[int, ...], ...]  # 1..4 player hands, class ids in fan order
    scale: float


def placements(spec: FrameSpec) -> tuple[Placement, ...]:
    """Card positions for a spec, dealer fan first then hands, in paint order."""
    if not 1 <= len(spec.hands) <= 4:
        raise ValueError(f"player hands must number 1..4, got {len(spec.hands)}")
    out = [
        Placement(card, DEALER_ANCHOR[0] + i * FAN_DX, DEALER_ANCHOR[1])
        for i, card in enumerate(spec.dealer)
    ]
    for hand_idx, hand in enumerate(spec.hands):
        ox, oy = PLAYER_ANCHORS[len(spec.hands)][hand_idx]
        out += [
            Placement(card, ox + i * FAN_DX, oy + i * PLAYER_FAN_DY) for i, card in enumerate(hand)
        ]
    return tuple(out)


def _sprite(deck_bgr: np.ndarray, class_id: int) -> np.ndarray:
    y = BACK_Y if class_id == BACK_CLASS else class_id * CARD_H
    return deck_bgr[y : y + CARD_H, 0:CARD_W]


def render(
    spec: FrameSpec,
    table_bgr: np.ndarray,
    deck_bgr: np.ndarray,
    bar_text: str = BAR_TEXT,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Composite a frame exactly as capture_session saves one: count bar + scaled table.

    Returns (frame_bgr, YOLO label lines in final frame coordinates).
    """
    canvas = table_bgr.copy()
    placed = placements(spec)
    for p in placed:
        canvas[p.y : p.y + CARD_H, p.x : p.x + CARD_W] = _sprite(deck_bgr, p.class_id)

    s = spec.scale
    scaled_w, scaled_h = round(TABLE_W * s), round(TABLE_H * s)
    if s != 1.0:
        interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
        canvas = cv2.resize(canvas, (scaled_w, scaled_h), interpolation=interp)

    bar_h = round(COUNT_BAR_CSS_PX * s)
    bar = np.zeros((bar_h, scaled_w, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        bar_text,
        (round(8 * s), round(24 * s)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45 * s,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    frame = np.vstack([bar, canvas])

    hits = [CardHit(p.class_id, round(p.x * s), round(p.y * s) + bar_h, 1.0) for p in placed]
    labels = yolo_lines(hits, frame.shape[1], frame.shape[0], s)
    return frame, tuple(labels)


def plan_frames(
    targets: Mapping[str, int], rng: random.Random, cards_per_hand: tuple[int, int] = (2, 5)
) -> tuple[FrameSpec, ...]:
    """Deficit-driven frame plan: keep emitting frames until every class meets its target.

    Faces are drawn weighted by remaining deficit (uniform once all deficits are met);
    "back" deficit is burned down by dealing hole cards. Deterministic for a given rng
    seed. Hands need not be blackjack-legal — the detector only learns appearance and
    layout; legality matters for tracker fixtures, which script their own specs.
    """
    unknown = set(targets) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"unknown class names in targets: {sorted(unknown)}")
    deficit = {name: max(0, int(targets.get(name, 0))) for name in CLASS_NAMES}
    face_names = [n for n in CLASS_NAMES if n != "back"]

    def draw_face() -> int:
        open_names = [n for n in face_names if deficit[n] > 0]
        pool = open_names or face_names
        weights = [deficit[n] for n in pool] if open_names else None
        name = rng.choices(pool, weights=weights)[0]
        deficit[name] = max(0, deficit[name] - 1)
        return CLASS_NAMES.index(name)

    frames: list[FrameSpec] = []
    while any(v > 0 for v in deficit.values()):
        counts, weights = zip(*HAND_COUNT_WEIGHTS.items(), strict=True)
        n_hands = rng.choices(counts, weights=weights)[0]
        hands = tuple(
            tuple(draw_face() for _ in range(rng.randint(*cards_per_hand))) for _ in range(n_hands)
        )
        if deficit["back"] > 0 or rng.random() < 0.4:
            dealer = (draw_face(), BACK_CLASS)  # deal-shaped: upcard + hole
            deficit["back"] = max(0, deficit["back"] - 1)
        else:
            dealer = tuple(draw_face() for _ in range(rng.randint(2, 5)))  # settled fan
        frames.append(FrameSpec(dealer=dealer, hands=hands, scale=rng.choice(SCALES)))
    return tuple(frames)
