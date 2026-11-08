"""build_config plumbing that runs without a screen: region parsing and overrides."""

from __future__ import annotations

import argparse

import pytest

from bjcounter.app.main import parse_region


class TestParseRegion:
    def test_valid_region_derives_scale_from_width(self):
        region, scale = parse_region("349,135,1200,845")
        assert region == (349, 135, 1200, 845)
        assert scale == 1.25

    @pytest.mark.parametrize("raw", ["1,2,3", "a,b,c,d", "1;2;3;4", ""])
    def test_malformed_region_exits_with_a_message(self, raw):
        with pytest.raises(SystemExit, match="--region"):
            parse_region(raw)

    def test_nonpositive_dimensions_exit(self):
        with pytest.raises(SystemExit, match="positive"):
            parse_region("0,0,-960,676")


def test_template_flag_overrides_detector(tmp_path, monkeypatch):
    from bjcounter.app import config as config_module
    from bjcounter.app.main import build_config

    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr("bjcounter.app.main.CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(
        "bjcounter.app.main.save_config",
        lambda cfg, path=None: config_module.save_config(cfg, tmp_path / "config.json"),
    )
    monkeypatch.setattr(
        "bjcounter.app.main.load_config",
        lambda: config_module.load_config(tmp_path / "config.json"),
    )
    args = argparse.Namespace(relocate=False, template=True, region="0,0,960,676")
    config = build_config(args)
    assert config.detector == "template"
    assert config.region == (0, 0, 960, 676)
    # Persisted config keeps the default detector; --template is per-invocation.
    assert config_module.load_config(tmp_path / "config.json").detector == "onnx"


def test_region_flag_replaces_saved_region(tmp_path, monkeypatch):
    from bjcounter.app.config import AppConfig, load_config, save_config
    from bjcounter.app.main import build_config

    path = tmp_path / "config.json"
    save_config(AppConfig(region=(0, 0, 960, 676), scale=1.0), path)
    monkeypatch.setattr("bjcounter.app.main.load_config", lambda: load_config(path))
    monkeypatch.setattr("bjcounter.app.main.save_config", lambda cfg: save_config(cfg, path))

    args = argparse.Namespace(relocate=False, template=False, region="100,200,1200,845")
    config = build_config(args)

    assert config.region == (100, 200, 1200, 845)
    assert config.scale == 1.25
    assert load_config(path) == config
