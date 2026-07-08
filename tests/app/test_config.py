"""AppConfig persistence round-trip and derived geometry."""

from __future__ import annotations

from dataclasses import replace

from bjcounter.app.config import AppConfig, load_config, save_config
from bjcounter.types import Rules, Surrender


class TestConfigRoundTrip:
    def test_save_then_load_reproduces_the_config(self, tmp_path):
        config = AppConfig(region=(349, 135, 1200, 845), scale=1.25)
        path = tmp_path / "config.json"
        save_config(config, path)
        assert load_config(path) == config

    def test_non_default_rules_and_overrides_survive(self, tmp_path):
        config = replace(
            AppConfig(region=(0, 0, 960, 676), scale=1.0),
            rules=Rules(decks=2, h17=False, surrender=Surrender.NOT_VS_ACE),
            detector="template",
            bet_cap=4,
            hotkey_quit="<ctrl>+<alt>+x",
        )
        path = tmp_path / "config.json"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded == config
        assert loaded.rules.surrender is Surrender.NOT_VS_ACE

    def test_missing_file_returns_none(self, tmp_path):
        assert load_config(tmp_path / "nope.json") is None


class TestSchemaTolerance:
    """A persisted config must survive version drift in both directions (review
    finding: a raw KeyError/TypeError at startup forced manual config deletion)."""

    def test_unknown_keys_from_a_newer_version_are_dropped(self, tmp_path):
        import json

        path = tmp_path / "config.json"
        save_config(AppConfig(region=(0, 0, 960, 676), scale=1.0), path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["field_added_in_v2"] = "whatever"
        path.write_text(json.dumps(raw), encoding="utf-8")
        loaded = load_config(path)
        assert loaded == AppConfig(region=(0, 0, 960, 676), scale=1.0)

    def test_missing_rules_keys_fall_back_to_defaults(self, tmp_path):
        import json

        path = tmp_path / "config.json"
        save_config(AppConfig(region=(0, 0, 960, 676), scale=1.0), path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        del raw["rules"]["hit_split_aces"]  # a field an older version didn't have
        del raw["bet_cap"]
        path.write_text(json.dumps(raw), encoding="utf-8")
        loaded = load_config(path)
        assert loaded is not None
        assert loaded.rules == Rules()
        assert loaded.bet_cap == 8

    def test_unreadable_file_falls_back_to_first_run(self, tmp_path, capsys):
        path = tmp_path / "config.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_config(path) is None
        assert "unreadable" in capsys.readouterr().out

    def test_missing_required_region_falls_back_to_first_run(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text('{"scale": 1.0}', encoding="utf-8")
        assert load_config(path) is None


class TestDerivedGeometry:
    def test_table_origin_scales_the_count_bar(self):
        assert AppConfig(region=(0, 0, 960, 676), scale=1.0).table_origin == (0, 36)
        assert AppConfig(region=(0, 0, 1200, 845), scale=1.25).table_origin == (0, 45)
