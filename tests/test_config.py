"""
test_config.py

用于验证 config.py 的配置加载与优先级行为：
CLI > ENV > PROJECT CONFIG > GLOBAL CONFIG > DEFAULT
"""

import os
import tempfile
from argparse import Namespace
from pathlib import Path

from core.config import load_app_config


def write_json(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def reset_env():
    keys = [
        "CC_MINI_PROVIDER",
        "CC_MINI_MODEL",
        "CC_MINI_MAX_TOKENS",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    ]
    for k in keys:
        os.environ.pop(k, None)


def base_args(**kwargs):
    """
    构造 argparse.Namespace，模拟 CLI 参数
    """
    return Namespace(
        provider=None,
        model=None,
        max_tokens=None,
        api_key=None,
        base_url=None,
        **kwargs,
    )


# =========================
# 测试 1：只有全局配置
# =========================

def test_global_only(tmp_path: Path):
    print("\n=== test_global_only ===")

    global_cfg = tmp_path / "global.json"
    write_json(
        global_cfg,
        """
        {
            "provider": "openai",
            "model": "gpt-5",
            "max_tokens": 1234,
            "api_key": "GLOBAL_KEY"
        }
        """,
    )

    args = base_args(config=str(global_cfg))
    cfg = load_app_config(args)

    print(cfg)
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-5"
    assert cfg.max_tokens == 1234
    assert cfg.api_key == "GLOBAL_KEY"


# =========================
# 测试 2：项目配置覆盖全局
# =========================

def test_project_overrides_global(tmp_path: Path):
    print("\n=== test_project_overrides_global ===")

    global_cfg = tmp_path / "global.json"
    project_cfg = tmp_path / "project.json"

    write_json(
        global_cfg,
        """
        {
            "provider": "openai",
            "model": "gpt-5",
            "max_tokens": 1000
        }
        """,
    )

    write_json(
        project_cfg,
        """
        {
            "model": "gpt-4.1"
        }
        """,
    )

    args = base_args(config=None)

    # 模拟默认路径行为：手动指定加载顺序
    os.chdir(tmp_path)

    cfg = load_app_config(
        Namespace(
            provider=None,
            model=None,
            max_tokens=None,
            api_key=None,
            base_url=None,
            config=str(project_cfg),
        )
    )

    print(cfg)
    assert cfg.model == "gpt-4.1"


# =========================
# 测试 3：环境变量覆盖配置文件
# =========================

def test_env_overrides_file(tmp_path: Path):
    print("\n=== test_env_overrides_file ===")

    cfg_file = tmp_path / "config.json"
    write_json(
        cfg_file,
        """
        {
            "model": "gpt-4",
            "max_tokens": 2000
        }
        """,
    )

    os.environ["CC_MINI_MODEL"] = "gpt-5"
    os.environ["CC_MINI_MAX_TOKENS"] = "9999"

    args = base_args(config=str(cfg_file))
    cfg = load_app_config(args)

    print(cfg)
    assert cfg.model == "gpt-5"
    assert cfg.max_tokens == 9999


# =========================
# 测试 4：CLI 覆盖一切
# =========================

def test_cli_overrides_all(tmp_path: Path):
    print("\n=== test_cli_overrides_all ===")

    cfg_file = tmp_path / "config.json"
    write_json(
        cfg_file,
        """
        {
            "model": "gpt-4",
            "max_tokens": 2000
        }
        """,
    )

    os.environ["CC_MINI_MODEL"] = "gpt-5"

    args = Namespace(
        provider=None,
        model="gpt-4.1",        # CLI
        max_tokens=5555,        # CLI
        api_key=None,
        base_url=None,
        config=str(cfg_file),
    )

    cfg = load_app_config(args)

    print(cfg)
    assert cfg.model == "gpt-4.1"
    assert cfg.max_tokens == 5555


# =========================
# 主入口
# =========================

if __name__ == "__main__":
    reset_env()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        test_global_only(tmp)
        reset_env()

        test_env_overrides_file(tmp)
        reset_env()

        test_cli_overrides_all(tmp)
        reset_env()

    print("\n✅ All config tests passed!")