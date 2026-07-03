"""Hi-Lo tag fixture parsed from docs/research/strategy-tables.md §2 (ground truth)."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pytest

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "research" / "strategy-tables.md"


@pytest.fixture(scope="session")
def doc_hilo_tags() -> dict[str, int]:
    text = DOC_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```csv\n(.*?)```", text, re.DOTALL)
    tag_block = next(b for b in blocks if b.startswith("rank,tag"))
    rows = list(csv.reader(io.StringIO(tag_block.strip())))
    assert rows[0] == ["rank", "tag"]
    tags = {rank: int(tag) for rank, tag in rows[1:]}
    assert len(tags) == 13
    return tags
