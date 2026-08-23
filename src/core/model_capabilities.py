"""模型能力表：静态判断常见模型是否支持视觉输入（supports_vision）。

业界 Aider 同款做法（model-metadata.json 的 supports_vision 标记）。本表只做
"确定支持 / 确定不支持"的保守判断；**未知模型返回 None，由调用方默认放行**
（试一次，API 400 再提示），避免误伤新模型。

匹配规则：子串匹配 + 长度倒序（先精确后宽松）。大小写不敏感。
"""
from __future__ import annotations

# 确定支持视觉的模型关键词（按长度倒序匹配，长 key 优先）
_VISION_KEYS = sorted([
    # 智谱 GLM
    "glm-4v", "glm-4.5v", "glm-4.6v", "glm-5v",
    # 阿里 Qwen-VL
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl",
    # DeepSeek
    "deepseek-vl", "deepseek-v4-flash-vision",
    # 兜底：型号名自带 vision / vl 特征（实验版、定制名）
    "vision", "-vl",
], key=len, reverse=True)

# 确定不支持视觉的模型关键词（仅列明确文本为主的老型号）
_NON_VISION_KEYS = sorted([
    "gpt-3.5",
    "deepseek-chat", "deepseek-reasoner", "deepseek-v3",
], key=len, reverse=True)


def supports_vision(model: str) -> bool | None:
    """判断模型是否支持视觉输入。

    Returns:
        True 支持 / False 不支持 / None 未知（调用方默认放行）。
    """
    if not model:
        return None
    m = model.lower()
    for key in _VISION_KEYS:
        if key in m:
            return True
    for key in _NON_VISION_KEYS:
        if key in m:
            return False
    return None
