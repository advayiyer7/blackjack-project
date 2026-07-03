"""Fixture source: docs/research/strategy-tables.md parsed independently of the engine.

The research doc is the ground truth (ARCHITECTURE §6/§9). The engine embeds its own tables;
these fixtures re-parse the doc so every chart cell and deviation index is asserted against
the cross-checked research, cell for cell.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "research" / "strategy-tables.md"

UPCARDS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "A")


def _fenced_blocks(text: str, kind: str) -> list[str]:
    return re.findall(rf"```{kind}\n(.*?)```", text, re.DOTALL)


def _parse_chart(block: str) -> dict[tuple[str, str], str]:
    """Chart CSV -> {(hand_label, upcard): action_code}."""
    rows = list(csv.reader(io.StringIO(block.strip())))
    header = rows[0]
    assert header[0] == "hand" and tuple(header[1:]) == UPCARDS, f"unexpected header: {header}"
    chart: dict[tuple[str, str], str] = {}
    for row in rows[1:]:
        hand = row[0]
        for up, code in zip(UPCARDS, row[1:], strict=True):
            chart[(hand, up)] = code
    return chart


def _load_doc() -> dict:
    text = DOC_PATH.read_text(encoding="utf-8")
    csv_blocks = _fenced_blocks(text, "csv")
    json_blocks = _fenced_blocks(text, "json")
    # Positional layout of the research doc: hard, soft, pairs, s17-diff, hilo / i18, fab4.
    assert len(csv_blocks) == 5, f"expected 5 csv blocks, found {len(csv_blocks)}"
    assert len(json_blocks) == 2, f"expected 2 json blocks, found {len(json_blocks)}"
    hard = _parse_chart(csv_blocks[0])
    soft = _parse_chart(csv_blocks[1])
    pairs = _parse_chart(csv_blocks[2])

    s17_rows = list(csv.reader(io.StringIO(csv_blocks[3].strip())))
    assert s17_rows[0] == ["hand", "upcard", "h17_action", "s17_action"]
    s17_diff = {(r[0], r[1]): (r[2], r[3]) for r in s17_rows[1:]}

    # Sanity anchors from the research doc's own sanity-check list.
    assert hard[("16", "T")] == "Rh"
    assert hard[("11", "A")] == "D"
    assert s17_diff[("11", "A")] == ("D", "H")
    assert pairs[("8,8", "A")] == "Rp"

    i18 = json.loads(json_blocks[0])
    fab4 = json.loads(json_blocks[1])
    # Anchor the JSON blocks too — both share a schema, so a doc reorder would
    # otherwise swap them silently.
    assert len(i18) == 18 and i18[0]["play"] == "Insurance"
    assert len(fab4) == 4 and all(e["play"] == "Surrender" for e in fab4)

    return {
        "hard": hard,
        "soft": soft,
        "pairs": pairs,
        "s17_diff": s17_diff,
        "i18": i18,
        "fab4": fab4,
    }


_DOC = _load_doc()


def h17_chart() -> dict[tuple[str, str, str], str]:
    """{(kind, hand_label, upcard): code} for the H17 rule set."""
    merged: dict[tuple[str, str, str], str] = {}
    for kind in ("hard", "soft", "pairs"):
        for (hand, up), code in _DOC[kind].items():
            merged[(kind, hand, up)] = code
    return merged


def s17_chart() -> dict[tuple[str, str, str], str]:
    """H17 chart with the 6-cell S17 diff applied."""
    merged = h17_chart()
    for (hand, up), (h17_code, s17_code) in _DOC["s17_diff"].items():
        if hand in ("8,8",):
            kind = "pairs"
        elif hand.startswith("A,"):
            kind = "soft"
        else:
            kind = "hard"
        key = (kind, hand if hand != "17" else "17+", up)
        assert merged[key] == h17_code, f"S17 diff base mismatch at {key}"
        merged[key] = s17_code
    return merged


def chart_cases() -> list[tuple[str, str, str, str, str]]:
    """(ruleset, kind, hand_label, upcard, expected_code) for every cell of both charts."""
    cases = []
    for ruleset, chart in (("h17", h17_chart()), ("s17", s17_chart())):
        for (kind, hand, up), code in sorted(chart.items()):
            cases.append((ruleset, kind, hand, up, code))
    return cases


def i18_entries() -> list[dict]:
    return _DOC["i18"]


def fab4_entries() -> list[dict]:
    return _DOC["fab4"]
