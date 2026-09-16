"""test_config_user_bootstrap.py

验证 ensure_user_config()：pip/开发模式首启落盘内置模板、已有配置不覆盖、
exe（frozen）模式零干预。
"""

import json
import sys
from pathlib import Path

import pytest

from core import config as config_mod


def _set_global_cfg(tmp_path, monkeypatch) -> Path:
    """把 GLOBAL_CONFIG 指向 tmp 隔离路径，并确保 sys.frozen 不存在。"""
    cfg_path = tmp_path / ".config" / "super-code" / "super-code.json"
    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG", cfg_path)
    monkeypatch.delattr(sys, "frozen", raising=False)
    return cfg_path


def test_creates_template_when_missing(tmp_path, monkeypatch):
    cfg_path = _set_global_cfg(tmp_path, monkeypatch)
    assert not cfg_path.exists()

    result = config_mod.ensure_user_config()

    assert result == cfg_path
    assert cfg_path.exists()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["api_key"] == "YOUR_API_KEY_HERE"
    assert "model" in data


def test_keeps_existing_config_untouched(tmp_path, monkeypatch):
    cfg_path = _set_global_cfg(tmp_path, monkeypatch)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({"model": "custom"}), encoding="utf-8")

    result = config_mod.ensure_user_config()

    assert result is None
    assert json.loads(cfg_path.read_text(encoding="utf-8")) == {"model": "custom"}


def test_skips_when_frozen(tmp_path, monkeypatch):
    cfg_path = _set_global_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    result = config_mod.ensure_user_config()

    assert result is None
    assert not cfg_path.exists()
