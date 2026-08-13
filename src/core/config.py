from __future__ import annotations

import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
from dotenv import load_dotenv

load_dotenv()

# =========================
# 常量 & 默认值
# =========================

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.1-codex"

GLOBAL_CONFIG = Path.home() / ".config" / "super-code" / "super-code.json"
PROJECT_CONFIG = Path.cwd() / ".super-code.json"

# 便携目录：exe 所在目录，PyInstaller 打包后用 sys.executable，开发模式用 __file__ 推导项目根
if getattr(sys, "frozen", False):
    _EXE_DIR = Path(sys.executable).parent
else:
    _EXE_DIR = Path(__file__).resolve().parent.parent.parent
PORTABLE_CONFIG = _EXE_DIR / "super-code.json"


def get_portable_dir() -> Path:
    """返回 exe 所在目录（便携分发根目录），供 mcp、skills 等模块复用。"""
    return _EXE_DIR


# =========================
# 最终配置对象
# =========================

@dataclass(frozen=True)
class AppConfig:
    provider: str
    api_key: str | None
    base_url: str | None
    model: str
    max_tokens: int                     # 模型的最大输出tokens
    config_paths: tuple[Path, ...]
    auto_dream: bool = True              # 是否启用自动 dream 整合
    dream_interval_hours: float = 24.0   # 两次整合之间的最小间隔（小时）
    dream_min_sessions: int = 5          # 触发整合所需的最少新会话数
    # Step 6 性能优化：记忆相关性 selector 用的小模型（如 gpt-4o-mini / haiku）。
    # 空字符串 / None → 回退到 model 字段。仅影响 find_relevant_memories.build_relevant_memories_prefix
    # 这一处侧查询；主对话仍用 model。
    extract_model: str = ""
    # 协调者模式开关。CLI --coordinator 优先；其次读配置文件 coordinator 字段；
    # 默认 False（此时 features/coordinator.py 仍会从 SUPER_CODE_COORDINATOR env 兜底，
    # 保持向后兼容）。
    coordinator: bool = False
    # HTTP 读超时（秒）。GLM / Qwen 等思考模型首 token 延迟极长，建议 ≥ 300
    timeout: float = 300.0
    # 按模型名子串匹配的额外请求参数。key 为模型名子串（大小写不敏感），
    # value 含可选的 extra_body dict。最长 key 优先匹配。
    # 示例（JSON）：
    #   "model_profiles": {
    #     "glm": {
    #       "extra_body": {
    #         "thinking": { "type": "disabled" }
    #       }
    #     }
    #   }
    model_profiles: dict = field(default_factory=dict)
    # 沙箱配置（原始 dict，由 sandbox 模块自己解析。None 表示未启用）
    sandbox: dict | None = None
    # ── 语音模式（全双工语音对话）──
    voice_enabled: bool = False                     # 是否启用语音功能
    voice_stt_provider: str = "volcengine"          # STT 服务商: volcengine / aliyun_nls / openai_whisper
    voice_stt_api_key: str = ""                     # STT API key（空则复用主 api_key，仅 openai_whisper 用）
    voice_stt_params: dict = field(default_factory=dict)  # STT 厂商私有参数（各 provider 自己解析）
    voice_tts_voice: str = "zh-CN-YunxiNeural"      # Edge-TTS 音色（默认云希男声）
    voice_vad_sensitivity: int = 2                  # VAD 灵敏度 (0-3, 0最敏感)
    voice_interrupt_threshold: float = 0.3          # 打断检测阈值（秒）
    voice_speaker_verification: bool = False        # 是否启用声纹身份验证
    voice_speaker_threshold: float = 0.75           # 声纹相似度阈值 (0-1)
    voice_wake_word_enabled: bool = False           # 是否启用唤醒词始终在线
    voice_porcupine_access_key: str = ""            # Picovoice AccessKey（唤醒词必需）
    voice_wake_word_ppn_path: str = ""              # 自定义唤醒词 .ppn 模型路径
    voice_persona_style: str = "butler"             # 对话人格: butler / concise / casual
    voice_auto_silence_timeout: int = 30            # 静默超时（秒，0=不自动退出）
    voice_model: str = ""                           # 语音模式专用模型（空=复用主 model）


# =========================
# 核心入口
# =========================

def load_app_config(args: Namespace) -> AppConfig:
    """
    配置优先级：
    CLI > 当前目录(项目) > 便携目录(exe同级) > HOME目录(全局) > 默认值
    """

    # 1️⃣ 读取配置文件（全局 → 便携 → 项目。cfg.update 后加载覆盖前，项目最高）
    file_cfg, paths = _load_files(args.config)

    # 2️⃣ 读取环境变量
    # env_cfg = _load_env()

    # 3️⃣ provider 决策
    provider = (
        args.provider
        # or env_cfg.get("provider")
        or file_cfg.get("provider")
        or DEFAULT_PROVIDER
    )

    # 4️⃣ model 决策
    model = (
        args.model
        # or env_cfg.get("model")
        or file_cfg.get("model")
        or DEFAULT_MODEL
    )

    # 5️⃣ max_tokens 决策
    max_tokens = int(
        args.max_tokens
        # or env_cfg.get("max_tokens")
        or file_cfg.get("max_tokens")
        or 131072
    )

    # 6️⃣ api_key / base_url（provider 相关）
    api_key = (
        args.api_key
        # or env_cfg.get(f"{provider}_api_key")
        or file_cfg.get("api_key")
    )

    base_url = (
        args.base_url
        # or env_cfg.get(f"{provider}_base_url")
        or file_cfg.get("base_url")
    )

    # 7️⃣ auto-dream 配置
    auto_dream = not getattr(args, "auto_dream", True)
    dream_interval_hours = float(
        getattr(args, "dream_interval", None)
        or file_cfg.get("dream_interval_hours")
        or 24.0
    )
    dream_min_sessions = int(
        getattr(args, "dream_min_sessions", None)
        or file_cfg.get("dream_min_sessions")
        or 5
    )

    # 8️⃣ extract_model（Step 6 性能优化用的小模型）
    # 仅支持从配置文件读取——CLI 暂不暴露，避免参数膨胀。空字符串 / 缺省视作未配置。
    extract_model = str(file_cfg.get("extract_model") or "").strip()

    # 9️⃣ coordinator 模式开关：CLI > 文件 > False（env 兜底由 coordinator.py 自己处理）
    coordinator = bool(
        getattr(args, "coordinator", False)
        or file_cfg.get("coordinator")
        or False
    )

    # 🔟 timeout / model_profiles（仅从配置文件读取）
    timeout = float(file_cfg.get("timeout") or 300.0)
    model_profiles = dict(file_cfg.get("model_profiles") or {})

    # 1️⃣1️⃣ 沙箱配置（仅从配置文件读取，or None 表示未启用）
    sandbox: dict | None = file_cfg.get("sandbox")

    # 1️⃣2️⃣ 语音模式配置（从配置文件的 voice 节读取，全部有默认值）
    voice_cfg: dict = file_cfg.get("voice") or {}
    voice_enabled = bool(voice_cfg.get("enabled") or False)
    voice_stt_provider = str(voice_cfg.get("stt_provider") or "volcengine")
    voice_stt_api_key = str(voice_cfg.get("stt_api_key") or "")
    voice_stt_params = dict(voice_cfg.get("stt_params") or {})
    voice_tts_voice = str(voice_cfg.get("tts_voice") or "zh-CN-YunxiNeural")
    voice_vad_sensitivity = int(voice_cfg.get("vad_sensitivity") or 2)
    voice_interrupt_threshold = float(voice_cfg.get("interrupt_threshold") or 0.3)
    voice_speaker_verification = bool(voice_cfg.get("speaker_verification") or False)
    voice_speaker_threshold = float(voice_cfg.get("speaker_threshold") or 0.75)
    voice_wake_word_enabled = bool(voice_cfg.get("wake_word_enabled") or False)
    voice_porcupine_access_key = str(voice_cfg.get("porcupine_access_key") or "")
    voice_wake_word_ppn_path = str(voice_cfg.get("wake_word_ppn_path") or "")
    voice_persona_style = str(voice_cfg.get("persona_style") or "butler")
    voice_auto_silence_timeout = int(voice_cfg.get("auto_silence_timeout") or 30)
    voice_model = str(voice_cfg.get("model") or "")

    return AppConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        config_paths=paths,
        auto_dream=auto_dream,
        dream_interval_hours=dream_interval_hours,
        dream_min_sessions=dream_min_sessions,
        extract_model=extract_model,
        coordinator=coordinator,
        timeout=timeout,
        model_profiles=model_profiles,
        sandbox=sandbox,
        voice_enabled=voice_enabled,
        voice_stt_provider=voice_stt_provider,
        voice_stt_api_key=voice_stt_api_key,
        voice_stt_params=voice_stt_params,
        voice_tts_voice=voice_tts_voice,
        voice_vad_sensitivity=voice_vad_sensitivity,
        voice_interrupt_threshold=voice_interrupt_threshold,
        voice_speaker_verification=voice_speaker_verification,
        voice_speaker_threshold=voice_speaker_threshold,
        voice_wake_word_enabled=voice_wake_word_enabled,
        voice_porcupine_access_key=voice_porcupine_access_key,
        voice_wake_word_ppn_path=voice_wake_word_ppn_path,
        voice_persona_style=voice_persona_style,
        voice_auto_silence_timeout=voice_auto_silence_timeout,
        voice_model=voice_model,
    )


# =========================
# 配置文件
# =========================

def _load_files(explicit: str | None) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """
    显式 --config > 默认路径
    """
    cfg: dict[str, Any] = {}
    loaded: list[Path] = []

    def load(path: Path):
        nonlocal cfg                # 修改外层函数的cfg变量，而不是新建一个局部变量
        with path.open("r", encoding="utf-8") as f:
            cfg.update(json.load(f))
        loaded.append(path)     # 表示这个配置文件我确实加载过

    if explicit:                                # 判断用户是否在CLI命令行显示传了配置路径，比如 --config xxxx.json
        path = Path(explicit).expanduser()      # 把字符串变成Path对象，expanduser把~展开为用户目录
        if not path.exists():
            raise ValueError(f"Config not found: {path}")
        load(path)
        return cfg, tuple(loaded)

    # 便携目录（exe 同级 super-code.json）：优先于全局配置，方便一键分发
    if GLOBAL_CONFIG.exists():
        load(GLOBAL_CONFIG)

    if PORTABLE_CONFIG.exists():
        load(PORTABLE_CONFIG)

    if PROJECT_CONFIG.exists():
        load(PROJECT_CONFIG)

    return cfg, tuple(loaded)


# =========================
# 环境变量
# =========================

def _load_env() -> dict[str, Any]:
    return {
        "provider": os.getenv("SUPER_CODE_PROVIDER"),
        "model": os.getenv("SUPER_CODE_MODEL"),
        "max_tokens": os.getenv("SUPER_CODE_MAX_TOKENS"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "anthropic_base_url": os.getenv("ANTHROPIC_BASE_URL"),
    }