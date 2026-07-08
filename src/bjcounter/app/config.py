"""Frozen app configuration, persisted to a gitignored config.json (ARCHITECTURE §7).

The persisted file contains machine-specific screen coordinates; config.example.json
(committed) documents the shape. The capture region INCLUDES the trainer's count bar
(count_bar_px CSS px above the felt, same convention as scripts/capture_session.py),
so the tracker's table origin sits bar-height below the region top.

Loading is schema-tolerant in both directions: unknown keys (file written by a newer
version) are dropped, missing keys (older file) fall back to the dataclass defaults,
and a structurally unreadable file returns None — triggering the first-run relocate
flow instead of a raw traceback at startup.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from pathlib import Path

from bjcounter.types import BBox, DoubleRule, Rules, Surrender

CONFIG_PATH = Path("config.json")
COUNT_BAR_CSS_PX = 36  # keep in sync with scripts/capture_session.py


@dataclass(frozen=True, slots=True)
class AppConfig:
    region: BBox  # absolute virtual-screen coords, count bar included
    scale: float  # trainer canvas scale within the region (region width / 960)
    rules: Rules = Rules()
    detector: str = "onnx"  # "onnx" | "template"
    onnx_path: str = "models/best.onnx"
    deck_png: str = "data/assets/deck.png"  # template-fallback sprites
    conf_threshold: float = 0.80
    count_bar_px: int = COUNT_BAR_CSS_PX
    bet_cap: int = 8
    hotkey_capture: str = "<ctrl>+<alt>+c"
    hotkey_reset: str = "<ctrl>+<alt>+r"
    hotkey_quit: str = "<ctrl>+<alt>+q"

    @property
    def table_origin(self) -> tuple[int, int]:
        """Where the 960x640 canvas starts inside a captured frame."""
        return (0, round(self.count_bar_px * self.scale))


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    payload = asdict(config)
    payload["region"] = list(config.region)
    payload["rules"] = {
        k: v.value if isinstance(v, Enum) else v for k, v in payload["rules"].items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rules_from(raw: dict) -> Rules:
    """Rules from persisted JSON; missing keys keep the dataclass defaults."""
    defaults = Rules()
    return Rules(
        decks=int(raw.get("decks", defaults.decks)),
        h17=bool(raw.get("h17", defaults.h17)),
        das=bool(raw.get("das", defaults.das)),
        surrender=Surrender(raw.get("surrender", defaults.surrender.value)),
        peek=bool(raw.get("peek", defaults.peek)),
        double=DoubleRule(raw.get("double", defaults.double.value)),
        max_hands=int(raw.get("max_hands", defaults.max_hands)),
        max_ace_hands=int(raw.get("max_ace_hands", defaults.max_ace_hands)),
        hit_split_aces=bool(raw.get("hit_split_aces", defaults.hit_split_aces)),
    )


def load_config(path: Path = CONFIG_PATH) -> AppConfig | None:
    """The persisted config, or None when absent OR unreadable (first-run flow).

    Unreadable configs are reported, not raised — the caller relocates and rewrites.
    """
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rules = _rules_from(raw.pop("rules", {}))
        region = tuple(int(v) for v in raw.pop("region"))
        scale = float(raw.pop("scale"))
        known = {f.name for f in fields(AppConfig)}
        overrides = {k: v for k, v in raw.items() if k in known}
        return replace(AppConfig(region=region, scale=scale, rules=rules), **overrides)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"warning: {path} is unreadable ({exc!r}) — falling back to first-run setup")
        return None
