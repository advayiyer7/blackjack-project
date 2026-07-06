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
SYNTHETIC = REPO / "data" / "synthetic"
DECK = REPO / "data" / "assets" / "deck.png"
TARGET_PER_CLASS = 40
TARGET_FRAMES = 1500


def tally_synthetic() -> tuple[int, dict[str, int]]:
    """Frame and per-class instance counts from the synthetic pool's label files."""
    counts = dict.fromkeys(CLASS_NAMES, 0)
    label_files = sorted((SYNTHETIC / "labels").glob("frame_*.txt")) if SYNTHETIC.exists() else []
    for path in label_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return len(label_files), counts


def analyze_real_sessions(limit: int | None) -> list[dict]:
    """Template-match every capture session; persist per-session detections.json."""
    reports = []
    for session_dir in sorted(RAW.glob("session_*")):
        if not (session_dir / "session_meta.json").exists():
            continue
        if not any(session_dir.glob("frame_*.png")):
            continue
        print(f"analyzing {session_dir.name}...", flush=True)
        report = analyze_session(session_dir, DECK, limit=limit)
        reports.append(report)
        (session_dir / "detections.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8"
        )
        print(
            f"  valid {report['frames_valid']}/{report['frames_total']} "
            f"(skipped {report['frames_skipped']}), mean score {report['mean_score']:.3f}"
        )
    return reports


def aggregate(reports: list[dict]) -> tuple[int, dict[str, int], dict[int, int]]:
    totals = dict.fromkeys(CLASS_NAMES, 0)
    hands = dict.fromkeys((0, 1, 2, 3, 4), 0)
    valid = 0
    for r in reports:
        valid += r["frames_valid"]
        for name, count in r["class_counts"].items():
            totals[name] += count
        for k, v in r["hand_histogram"].items():
            hands[k] += v
    return valid, totals, hands


def render_report(
    reports: list[dict],
    valid: int,
    totals: dict[str, int],
    hands: dict[int, int],
    synth_frames: int,
    synth_counts: dict[str, int],
) -> str:
    train_below = {n: c for n, c in synth_counts.items() if c < TARGET_PER_CLASS}
    below = {n: c for n, c in totals.items() if c < TARGET_PER_CLASS}
    lines = [
        "# Dataset progress report",
        f"\nGenerated {date.today()} by scripts/dataset_report.py.\n",
        "## Training pool (synthetic, data/synthetic/)\n",
        f"- Frames: **{synth_frames}** (regenerable — see synth_meta.json for the seed)",
        f"- Card instances: {sum(synth_counts.values())}",
        f"- Classes below the {TARGET_PER_CLASS}-instance train gate: **{len(train_below)}**"
        f" of 53" + (f" — {sorted(train_below)}" if train_below else ""),
        "\n## Val/test pool (real captures, data/raw/)\n",
        f"- Sessions analyzed: {len(reports)}",
        f"- Valid frames (cumulative): **{valid}** / target ~{TARGET_FRAMES}",
        f"- Card instances total: {sum(totals.values())}",
        f"- Classes below the {TARGET_PER_CLASS}-instance gate: **{len(below)}** of 53",
        f"- Player-hand layout coverage (frames): 1 hand={hands[1]}, 2 hands={hands[2]}, "
        f"3 hands={hands[3]}, 4 hands={hands[4]}",
        "\n## Per-session (real captures)\n",
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
        "\n## Instances per class (real / synthetic)\n",
        "| Class | Real | Synth | | Class | Real | Synth |",
        "|---|---|---|---|---|---|---|",
    ]
    names = list(CLASS_NAMES)
    half = (len(names) + 1) // 2
    for i in range(half):
        left = f"| {names[i]} | {totals[names[i]]} | {synth_counts[names[i]]} |"
        j = i + half
        right = (
            f" | {names[j]} | {totals[names[j]]} | {synth_counts[names[j]]} |"
            if j < len(names)
            else " | | | |"
        )
        lines.append(left + right)
    if below:
        lines += [
            "\n## Real captures below target\n",
            ", ".join(f"{n} ({c})" for n, c in sorted(below.items())),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="frames per session cap")
    args = parser.parse_args()

    reports = analyze_real_sessions(args.limit)
    valid, totals, hands = aggregate(reports)
    synth_frames, synth_counts = tally_synthetic()
    train_below = sum(1 for c in synth_counts.values() if c < TARGET_PER_CLASS)

    report_md = render_report(reports, valid, totals, hands, synth_frames, synth_counts)
    (REPO / "data" / "REPORT.md").write_text(report_md, encoding="utf-8")
    print(f"\nreport -> data/REPORT.md | real valid frames {valid}, "
          f"synthetic frames {synth_frames}, train classes below gate: {train_below}")


if __name__ == "__main__":
    main()
