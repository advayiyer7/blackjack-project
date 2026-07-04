"""M3 dataset capture: locate the trainer table once, then timed deduplicated capture.

Usage (from repo root, venv active):
    python scripts/capture_session.py            # auto-locate the table, then capture
    python scripts/capture_session.py --region X,Y,W,H   # skip auto-location

Flow: switch to the browser showing the trainer, the script finds the 960x640 felt on
screen (any scale), saves a preview crop for you to confirm, then captures the region every
--interval seconds, saving only frames it hasn't seen before (pixel-exact dedup) into
data/raw/session_<timestamp>/. Ctrl+C to stop. Raw frames are gitignored and never leave
this machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from bjcounter.vision.capture import Grabber, find_table, set_dpi_awareness

REPO = Path(__file__).resolve().parents[1]
TABLE_PNG = REPO / "data" / "assets" / "table.png"
LOCATE_COUNTDOWN_S = 5
START_COUNTDOWN_S = 5
MIN_MATCH_SCORE = 0.45
COUNT_BAR_CSS_PX = 36  # the #countinfo bar sits directly above the felt — capture it too


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between grabs")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--region", type=str, default=None, help="manual X,Y,W,H (absolute px)")
    return parser.parse_args()


def locate_region(grabber: Grabber) -> tuple[tuple[int, int, int, int], float]:
    print(f"Switch to the trainer browser window now — locating table in {LOCATE_COUNTDOWN_S}s...")
    for i in range(LOCATE_COUNTDOWN_S, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    screen, (offset_x, offset_y) = grabber.virtual_screen()
    print("Searching for the table (one-time, ~10-30s)...")
    match = find_table(screen, cv2.imread(str(TABLE_PNG)))
    x, y, w, h = match.region
    # Extend upward to include the count bar — per-frame ground truth for evals.
    bar = round(COUNT_BAR_CSS_PX * match.scale)
    region = (x + offset_x, max(y + offset_y - bar, 0), w, h + bar)
    print(f"Found: region={region} scale={match.scale:.2f} score={match.score:.3f}")
    if match.score < MIN_MATCH_SCORE:
        print("WARNING: low match confidence — check the preview carefully or pass --region.")
    return region, match.scale


def confirm_preview(grabber: Grabber, region, session_dir: Path) -> bool:
    preview_path = session_dir / "region_preview.png"
    cv2.imwrite(str(preview_path), grabber.grab(region))
    print(f"\nPreview written to: {preview_path}")
    answer = input("Open it — is the whole table (and nothing but the table) inside? [y/n] ")
    return answer.strip().lower().startswith("y")


def capture_loop(grabber: Grabber, region, session_dir: Path, interval: float) -> int:
    seen_hashes: set[str] = set()
    saved = 0
    print(f"\nSwitch to the trainer window NOW — capture starts in {START_COUNTDOWN_S}s...")
    for i in range(START_COUNTDOWN_S, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("\nCapturing — play now. Ctrl+C to stop.\n")
    try:
        while True:
            frame = grabber.grab(region)
            digest = hashlib.md5(frame.tobytes()).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                cv2.imwrite(str(session_dir / f"frame_{saved:05d}.png"), frame)
                saved += 1
                if saved % 25 == 0:
                    print(f"  {saved} unique frames saved...")
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    return saved


def main() -> None:
    args = parse_args()
    set_dpi_awareness()  # must precede Grabber creation
    if not TABLE_PNG.exists() and args.region is None:
        raise SystemExit(f"missing {TABLE_PNG} — see data/assets/README.md, or pass --region")

    grabber = Grabber()
    session_dir = args.out / f"session_{datetime.now():%Y%m%d_%H%M%S}"
    session_dir.mkdir(parents=True, exist_ok=True)

    if args.region:
        x, y, w, h = (int(v) for v in args.region.split(","))
        region, scale = (x, y, w, h), w / 960
    else:
        region, scale = locate_region(grabber)

    while not confirm_preview(grabber, region, session_dir):
        raw = input("Enter region manually as X,Y,W,H (or blank to retry auto-locate): ").strip()
        if raw:
            x, y, w, h = (int(v) for v in raw.split(","))
            region, scale = (x, y, w, h), w / 960
        else:
            region, scale = locate_region(grabber)

    meta = {
        "region": list(region),
        "scale": scale,
        "interval_s": args.interval,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (session_dir / "session_meta.json").write_text(json.dumps(meta, indent=2))

    saved = capture_loop(grabber, region, session_dir, args.interval)
    grabber.close()
    print(f"\nDone: {saved} unique frames in {session_dir}")
    print("Raw frames are gitignored and stay on this machine.")


if __name__ == "__main__":
    main()
