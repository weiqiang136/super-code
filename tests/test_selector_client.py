"""Tests for extract_model selector client building (记忆 selector 独立 client 构建).

Covers:
- str form → returns (None, model): reuse main client (old behavior)
- empty / absent → falls back to main model
- dict form → independent LLMClient with overridden base_url / api_key
- dict partial fields → fall back to main-conversation values
- dict + extra_body → merged into model_profiles exact key (overrides outer profile,
  solves same-model-name deadlock)
- empty dict → all fall back to main conversation, no profile pollution
"""
import json
import sys
sys.path.insert(0, "src")

from argparse import Namespace
from pathlib import Path

from core.config import load_app_config
from core.llm import LLMClient
from tui.app import _build_selector_client


def _load(tmp_path: Path, cfg_dict: dict):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg_dict, ensure_ascii=False), encoding="utf-8")
    return load_app_config(
        Namespace(
            provider=None,
            model=None,
            max_tokens=None,
            api_key=None,
            base_url=None,
            config=str(cfg_file),
        )
    )


def _base() -> dict:
    return {
        "model": "deepseek-v4-flash",
        "api_key": "MAIN_KEY",
        "base_url": "https://api.deepseek.com",
        "timeout": 300,
        "model_profiles": {
            "deepseek-v4-flash": {
                "extra_body": {
                    "reasoning_effort": "max",
                    "thinking": {"type": "enabled"},
                }
            },
            "glm": {"extra_body": {"thinking": {"type": "disabled"}}},
        },
    }


def test_str_form_reuses_main_client(tmp_path):
    """字符串形态 → client 为 None（main 复用），model 用 extract_model。"""
    cfg = _base()
    cfg["extract_model"] = "glm-4-flash"
    client, model = _build_selector_client(_load(tmp_path, cfg))
    assert client is None
    assert model == "glm-4-flash"


def test_str_empty_falls_back_to_main_model(tmp_path):
    """未配置 / 空串 → model 回退主模型。"""
    client, model = _build_selector_client(_load(tmp_path, _base()))
    assert client is None
    assert model == "deepseek-v4-flash"


def test_dict_full_override(tmp_path):
    """dict 全字段 → 独立 client，base_url/api_key/timeout/model 全部用 dict 值。"""
    cfg = _base()
    cfg["extract_model"] = {
        "model": "glm-4-flash",
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key": "ZAI_KEY",
        "timeout": 30,
    }
    client, model = _build_selector_client(_load(tmp_path, cfg))
    assert isinstance(client, LLMClient)
    assert model == "glm-4-flash"
    assert "z.ai" in str(client._client.base_url)
    # 未提供 extra_body → 不往 model_profiles 加精确 key，外层 profile 原样保留
    assert client._model_profiles == cfg["model_profiles"]


def test_dict_partial_falls_back_to_main(tmp_path):
    """dict 部分字段 → 省略项回退主对话值（base_url 用主对话的）。"""
    cfg = _base()
    cfg["extract_model"] = {"model": "deepseek-v4-flash", "api_key": "ZAI_KEY"}
    client, model = _build_selector_client(_load(tmp_path, cfg))
    assert isinstance(client, LLMClient)
    assert model == "deepseek-v4-flash"
    assert "deepseek.com" in str(client._client.base_url)


def test_dict_extra_body_overrides_same_name(tmp_path):
    """dict + extra_body + 与主模型同名 → 精确 key 覆盖外层 profile（解死结）。"""
    cfg = _base()
    cfg["extract_model"] = {
        "model": "deepseek-v4-flash",
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    client, model = _build_selector_client(_load(tmp_path, cfg))
    assert isinstance(client, LLMClient)
    assert model == "deepseek-v4-flash"
    # 同名精确 key 注入 → _lookup_extra_body 按最长 key 优先必然命中它
    assert client._model_profiles["deepseek-v4-flash"] == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


def test_empty_dict_all_fallback(tmp_path):
    """空 dict → 全部回退主对话值，且不污染 model_profiles。"""
    cfg = _base()
    cfg["extract_model"] = {}
    client, model = _build_selector_client(_load(tmp_path, cfg))
    assert isinstance(client, LLMClient)
    assert model == "deepseek-v4-flash"
    assert "deepseek.com" in str(client._client.base_url)
    assert client._model_profiles == cfg["model_profiles"]
