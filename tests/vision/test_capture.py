"""Table-region location, tested synthetically (no screen required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from bjcounter.vision.capture import (  # noqa: E402
    TABLE_H,
    TABLE_W,
    TableMatch,
    find_table,
    region_with_count_bar,
)

TABLE_PNG = Path(__file__).resolve().parents[2] / "data" / "assets" / "table.png"

def synthetic_screen(table: np.ndarray, scale: float, pos: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(3)
    screen = rng.integers(0, 255, size=(1200, 2000, 3), dtype=np.uint8)
    w, h = int(TABLE_W * scale), int(TABLE_H * scale)
    resized = cv2.resize(table, (w, h), interpolation=cv2.INTER_AREA)
    x, y = pos
    screen[y : y + h, x : x + w] = resized
    return screen


@pytest.mark.parametrize(
    ("scale", "pos"), [(0.9, (300, 180)), (0.74, (40, 60)), (1.35, (500, 250))]
)
@pytest.mark.skipif(
    not TABLE_PNG.exists(), reason="table.png asset not fetched (see data/assets/README.md)"
)
def test_find_table_recovers_scale_and_position(scale, pos):
    table = cv2.imread(str(TABLE_PNG))
    match = find_table(synthetic_screen(table, scale, pos), table)
    assert match.score > 0.9
    assert abs(match.scale - scale) <= 0.02
    x, y, w, h = match.region
    assert abs(x - pos[0]) <= 4 and abs(y - pos[1]) <= 4
    assert abs(w - TABLE_W * scale) <= TABLE_W * 0.02


def test_screen_too_small_raises():
    table = np.zeros((TABLE_H, TABLE_W, 3), dtype=np.uint8)
    tiny = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        find_table(tiny, table)


def test_find_table_accepts_a_screen_exactly_the_size_of_the_table():
    rng = np.random.default_rng(7)
    table = rng.integers(0, 255, size=(TABLE_H, TABLE_W, 3), dtype=np.uint8)

    match = find_table(table.copy(), table, coarse_scales=(1.0,))

    assert match.region == (0, 0, TABLE_W, TABLE_H)
    assert match.scale == 1.0
    assert match.score > 0.99


def test_count_bar_region_keeps_negative_virtual_screen_coordinates():
    match = TableMatch(region=(150, 100, 960, 640), scale=1.0, score=0.99)
    assert region_with_count_bar(match, (-1920, -1080), 36) == (-1770, -1016, 960, 676)


def test_count_bar_region_rejects_a_bar_outside_the_virtual_screen():
    match = TableMatch(region=(150, 20, 960, 640), scale=1.0, score=0.99)
    with pytest.raises(ValueError, match="count bar"):
        region_with_count_bar(match, (-1920, -1080), 36)
