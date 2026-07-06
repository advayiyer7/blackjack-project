"""M3 auto-labeler: match trainer sprites against captured frames -> detections/YOLO labels.

The trainer renders cards as fixed sprites (deck.png, 67x94 per card, verified pixel-identical
every deal), so ground-truth labels come from template-matching each card's top-left CORNER
strip — the only part guaranteed visible in a fan (cards overlap at +24px offsets). A match at
(x, y) implies the full-card bbox (x, y, 67*s, 94*s).

Sprite sheet layout (verified visually + against live frames): 4 blocks of 13 ranks
(2..9,T,J,Q,K,A) in suit order clubs, diamonds, hearts, spades; card back at y=5358.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CARD_W, CARD_H = 67, 94
BACK_Y = 5358
CORNER_W, CORNER_H = 22, 60  # top-left strip: rank glyph + suit pip, visible even in a fan
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUITS = ("c", "d", "h", "s")  # sprite block order, verified
CLASS_NAMES: tuple[str, ...] = tuple(
    f"{rank}{suit}" for suit in SUITS for rank in RANKS
) + ("back",)
BACK_CLASS = 52

MATCH_THRESHOLD = 0.88
MIN_GREEN_FRACTION = 0.25  # validity gate: editor/rules-panel/junk frames have no felt


@dataclass(frozen=True, slots=True)
class CardHit:
    class_id: int
    x: int  # top-left of the full card bbox, frame coords
    y: int
    score: float

    @property
    def name(self) -> str:
        return CLASS_NAMES[self.class_id]


def load_templates(deck_png: Path, scale: float) -> list[np.ndarray]:
    """53 templates resized to the session's capture scale.

    Faces use the top-left corner strip (survives fan overlap). The back uses the FULL
    sprite: its ornate corner is too low-texture to clear the threshold after resampling
    (~0.81), while the full card scores ~0.93 — and a hole card is dealt last in the
    dealer fan, so it is never occluded.
    """
    deck = cv2.imread(str(deck_png))
    if deck is None:
        raise FileNotFoundError(deck_png)
    corner_size = (round(CORNER_W * scale), round(CORNER_H * scale))
    templates = [
        cv2.resize(
            deck[i * CARD_H : i * CARD_H + CORNER_H, 0:CORNER_W],
            corner_size,
            interpolation=cv2.INTER_CUBIC,
        )
        for i in range(52)
    ]
    back = deck[BACK_Y : BACK_Y + CARD_H, 0:CARD_W]
    back_size = (round(CARD_W * scale), round(CARD_H * scale))
    templates.append(cv2.resize(back, back_size, interpolation=cv2.INTER_CUBIC))
    return templates


def is_table_frame(frame_bgr: np.ndarray) -> bool:
    """Cheap validity gate: a real trainer frame is dominated by green felt."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 80, 40), (90, 255, 255))
    return float(np.count_nonzero(green)) / green.size >= MIN_GREEN_FRACTION


def detect_cards(
    frame_bgr: np.ndarray,
    templates: list[np.ndarray],
    scale: float,
    threshold: float = MATCH_THRESHOLD,
) -> list[CardHit]:
    """All card corners in the frame, deduplicated by position (best score wins)."""
    # Radii must stay well UNDER the fan offset (24px * scale) or the second card of a
    # fan is swallowed as a duplicate of the first — distinct cards are >= 24*scale apart,
    # duplicate template hits for the SAME card land within a couple of px.
    raw: list[CardHit] = []
    suppress = max(round(10 * scale), 6)
    for class_id, template in enumerate(templates):
        result = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
        while True:
            _, score, _, loc = cv2.minMaxLoc(result)
            if score < threshold:
                break
            raw.append(CardHit(class_id, int(loc[0]), int(loc[1]), float(score)))
            x, y = loc
            result[
                max(0, y - suppress) : y + suppress, max(0, x - suppress) : x + suppress
            ] = -1.0

    dedup_r = max(round(10 * scale), 6)
    kept: list[CardHit] = []
    for hit in sorted(raw, key=lambda h: -h.score):
        if all(abs(hit.x - k.x) > dedup_r or abs(hit.y - k.y) > dedup_r for k in kept):
            kept.append(hit)
    return kept


def yolo_lines(hits: list[CardHit], frame_w: int, frame_h: int, scale: float) -> list[str]:
    """YOLO detection format: class cx cy w h, normalized, full-card extent (clipped)."""
    lines = []
    for hit in hits:
        w = min(CARD_W * scale, frame_w - hit.x)
        h = min(CARD_H * scale, frame_h - hit.y)
        cx, cy = (hit.x + w / 2) / frame_w, (hit.y + h / 2) / frame_h
        lines.append(f"{hit.class_id} {cx:.6f} {cy:.6f} {w / frame_w:.6f} {h / frame_h:.6f}")
    return lines


def player_hand_clusters(hits: list[CardHit], scale: float, frame_h: int) -> int:
    """Rough count of distinct player hands (split coverage metric): cluster player-zone
    hits by x-gaps larger than ~1.5 card widths."""
    ys = [h.y for h in hits if h.class_id != BACK_CLASS]
    if not ys:
        return 0
    player = sorted(
        h.x for h in hits if h.class_id != BACK_CLASS and h.y > frame_h * 0.42
    )
    if not player:
        return 0
    clusters = 1
    for a, b in zip(player, player[1:], strict=False):
        if b - a > CARD_W * scale * 1.5:
            clusters += 1
    return clusters


def analyze_session(
    session_dir: Path, deck_png: Path, limit: int | None = None
) -> dict:
    """Detections + stats for one capture session; returns a report dict."""
    meta = json.loads((session_dir / "session_meta.json").read_text())
    scale = float(meta["scale"])
    templates = load_templates(deck_png, scale)
    frames = sorted(session_dir.glob("frame_*.png"))[:limit]

    per_frame: dict[str, list[CardHit]] = {}
    skipped = 0
    for path in frames:
        frame = cv2.imread(str(path))
        if frame is None or not is_table_frame(frame):
            skipped += 1
            continue
        per_frame[path.name] = detect_cards(frame, templates, scale)

    class_counts = dict.fromkeys(CLASS_NAMES, 0)
    hand_hist = dict.fromkeys((0, 1, 2, 3, 4), 0)
    scores = []
    sample_h = None
    for hits in per_frame.values():
        for hit in hits:
            class_counts[hit.name] += 1
            scores.append(hit.score)
        frame_h = meta["region"][3]
        clusters = min(player_hand_clusters(hits, scale, frame_h), 4)
        hand_hist[clusters] += 1
        sample_h = frame_h

    return {
        "session": session_dir.name,
        "scale": scale,
        "frames_total": len(frames),
        "frames_valid": len(per_frame),
        "frames_skipped": skipped,
        "class_counts": class_counts,
        "hand_histogram": hand_hist,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "min_score": float(np.min(scores)) if scores else 0.0,
        "frame_height": sample_h,
        "detections": {n: [(h.class_id, h.x, h.y, round(h.score, 4)) for h in v]
                       for n, v in per_frame.items()},
    }
