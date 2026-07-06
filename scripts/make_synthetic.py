"""Generate the synthetic YOLO training pool (images + labels) from trainer assets.

Usage: python scripts/make_synthetic.py [--per-class 80] [--seed 7] [--background-pct 5]

Plans deficit-driven frames until every one of the 53 classes has at least --per-class
instances, renders them at capture-faithful geometry (count bar included), and writes:

    data/synthetic/images/frame_NNNNN.png
    data/synthetic/labels/frame_NNNNN.txt      (YOLO: class cx cy w h, normalized)
    data/synthetic/classes.txt
    data/synthetic/synth_meta.json             (seed/targets — regeneration recipe)

Everything under data/synthetic/ is gitignored (regenerable, third-party art).
Real captured frames in data/raw/ are reserved for val/test at M4.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from bjcounter.vision.autolabel import CLASS_NAMES
from bjcounter.vision.synthesize import SCALES, FrameSpec, plan_frames, render

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "data" / "assets"
OUT = REPO / "data" / "synthetic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=80, help="min instances per class")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--background-pct", type=int, default=5,
        help="extra empty-felt frames (%% of planned frames) for false-positive suppression",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def render_all(
    frames: list[FrameSpec],
    table: np.ndarray,
    deck: np.ndarray,
    out_dir: Path,
    rng: random.Random,
) -> dict[str, int]:
    """Render every frame to images/ + labels/; returns per-class instance counts."""
    images_dir, labels_dir = out_dir / "images", out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    counts = dict.fromkeys(CLASS_NAMES, 0)
    for i, spec in enumerate(frames):
        bar = (
            f"Running Count: {rng.randint(-12, 12):+d}  |  "
            f"Decks Left: {rng.uniform(1, 6):.2f}  |  "
            f"True Count: {rng.randint(-6, 6):+d}"
        )
        frame, labels = render(spec, table, deck, bar_text=bar)
        cv2.imwrite(str(images_dir / f"frame_{i:05d}.png"), frame)
        (labels_dir / f"frame_{i:05d}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )
        for line in labels:
            counts[CLASS_NAMES[int(line.split()[0])]] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frames)} frames rendered...", flush=True)
    return counts


def main() -> None:
    args = parse_args()
    deck = cv2.imread(str(ASSETS / "deck.png"))
    table = cv2.imread(str(ASSETS / "table.png"))
    if deck is None or table is None:
        raise SystemExit(f"missing trainer assets in {ASSETS} — see data/assets/README.md")

    rng = random.Random(args.seed)
    targets = dict.fromkeys(CLASS_NAMES, args.per_class)
    frames = list(plan_frames(targets, rng))
    n_background = len(frames) * args.background_pct // 100
    frames += [
        FrameSpec(dealer=(), hands=((),), scale=rng.choice(SCALES)) for _ in range(n_background)
    ]
    rng.shuffle(frames)

    counts = render_all(frames, table, deck, args.out, rng)

    (args.out / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    meta = {
        "seed": args.seed,
        "per_class_target": args.per_class,
        "background_pct": args.background_pct,
        "frames": len(frames),
        "class_counts": counts,
        "specs": [asdict(s) for s in frames],
    }
    (args.out / "synth_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    below = {n: c for n, c in counts.items() if c < args.per_class}
    print(f"\n{len(frames)} frames -> {args.out}")
    print(f"instances: {sum(counts.values())}, min class count: {min(counts.values())}")
    if below:
        print(f"WARNING: below target: {below}")
    print("Run scripts/dataset_report.py to refresh data/REPORT.md.")


if __name__ == "__main__":
    main()
