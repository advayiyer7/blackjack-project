"""Analyze all capture sessions and write the dataset progress report (data/REPORT.md).

Usage: python scripts/dataset_report.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from bjcounter.vision.autolabel import CLASS_NAMES, analyze_session

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
DECK = REPO / "data" / "assets" / "deck.png"
TARGET_PER_CLASS = 40
TARGET_FRAMES = 1500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="frames per session cap")
    args = parser.parse_args()

    reports = []
    for session_dir in sorted(RAW.glob("session_*")):
        if not (session_dir / "session_meta.json").exists():
            continue
        if not any(session_dir.glob("frame_*.png")):
            continue
        print(f"analyzing {session_dir.name}...", flush=True)
        report = analyze_session(session_dir, DECK, limit=args.limit)
        reports.append(report)
        (session_dir / "detections.json").write_text(
            json.dumps({k: v for k, v in report.items() if k != "detections"} |
                       {"detections": report["detections"]}, indent=1)
        )
        print(
            f"  valid {report['frames_valid']}/{report['frames_total']} "
            f"(skipped {report['frames_skipped']}), mean score {report['mean_score']:.3f}"
        )

    totals = dict.fromkeys(CLASS_NAMES, 0)
    hands = dict.fromkeys((0, 1, 2, 3, 4), 0)
    valid = 0
    for r in reports:
        valid += r["frames_valid"]
        for name, count in r["class_counts"].items():
            totals[name] += count
        for k, v in r["hand_histogram"].items():
            hands[k] += v

    below = {n: c for n, c in totals.items() if c < TARGET_PER_CLASS}
    lines = [
        "# Dataset progress report",
        f"\nGenerated {date.today()} by scripts/dataset_report.py.\n",
        f"- Sessions analyzed: {len(reports)}",
        f"- Valid frames (cumulative): **{valid}** / target ~{TARGET_FRAMES}",
        f"- Card instances total: {sum(totals.values())}",
        f"- Classes below the {TARGET_PER_CLASS}-instance gate: **{len(below)}** of 53",
        f"- Player-hand layout coverage (frames): 1 hand={hands[1]}, 2 hands={hands[2]}, "
        f"3 hands={hands[3]}, 4 hands={hands[4]}",
        "\n## Per-session\n",
        "| Session | Scale | Frames | Valid | Skipped | Mean score | Min score |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r['session']} | {r['scale']:.2f} | {r['frames_total']} | "
            f"{r['frames_valid']} | {r['frames_skipped']} | {r['mean_score']:.3f} | "
            f"{r['min_score']:.3f} |"
        )
    lines += [
        "\n## Instances per class\n",
        "| Class | Count | | Class | Count |",
        "|---|---|---|---|---|",
    ]
    names = list(CLASS_NAMES)
    half = (len(names) + 1) // 2
    for i in range(half):
        left = f"| {names[i]} | {totals[names[i]]} |"
        j = i + half
        right = f" | {names[j]} | {totals[names[j]]} |" if j < len(names) else " | | |"
        lines.append(left + right)
    if below:
        lines += ["\n## Below target\n", ", ".join(f"{n} ({c})" for n, c in sorted(below.items()))]

    (REPO / "data" / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport -> data/REPORT.md | valid frames {valid}, "
          f"classes below target: {len(below)}")


if __name__ == "__main__":
    main()
