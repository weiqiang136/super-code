import os
import sys
sys.path.insert(0, "src")
from dotenv import load_dotenv

from core.llm import LLMClient

load_dotenv()

def test_streaming_output():
    llm = LLMClient(
        provider='openai',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "解释什么是streaming LLM输出"},
    ]

    print("\n>>> streaming start\n")

    with llm.stream(
            model="glm-4",
            messages=messages,
            temperature=0.7,
    ) as stream:
        for token in stream:
            print(token, end="", flush=True)

        final_text = stream.final()

    print("\n\n>>> streaming end")
    print(">>> final text:")
    print(final_text)

    assert final_text, "Streaming output should not be empty"
    assert isinstance(final_text, str), "Final text should be a string"


def test_non_streaming_output():
    llm = LLMClient(
        provider='openai',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用一句话解释什么是机器学习"},
    ]

    response = llm.create(
        model="glm-4",
        messages=messages,
        temperature=0.7,
    )

    assert response.text, "Response text should not be empty"
    assert isinstance(response.text, str), "Response text should be a string"
    assert response.raw is not None, "Raw response should not be None"


# ── DeepSeek reasoning_content 空占位（方案 A）───────────────────────────────
# 背景：deepseek-flash thinking 模式要求 assistant tool_calls 消息必须携带
# reasoning_content 字段，缺字段直接 400；补空串即可放行。2026-10-10 实测。

def _tool_call_msg(rc=None):
    content = [{"type": "tool_use", "id": "call_1", "name": "Read",
                "input": {"file_path": "a.txt"}}]
    msg = {"role": "assistant", "content": content}
    if rc is not None:
        msg["reasoning_content"] = rc
    return msg


def test_pad_helper_scope():
    from core.llm import _needs_reasoning_padding
    assert _needs_reasoning_padding("https://api.deepseek.com", "glm-5.3-flash") is True
    assert _needs_reasoning_padding("https://open.bigmodel.cn/api/paas/v4", "deepseek-flash") is True
    assert _needs_reasoning_padding("https://open.bigmodel.cn/api/paas/v4", "glm-5.3-flash") is False
    assert _needs_reasoning_padding(None, "gpt-4o") is False


def test_pad_adds_empty_rc_for_deepseek_tool_calls():
    from core.llm import _to_openai_messages
    msgs = [{"role": "user", "content": "hi"}, _tool_call_msg()]
    out = _to_openai_messages(None, msgs, pad_reasoning=True)
    asst = [m for m in out if m.get("role") == "assistant"][0]
    assert asst.get("reasoning_content") == ""
    assert asst["tool_calls"]


def test_pad_preserves_real_rc():
    from core.llm import _to_openai_messages
    msgs = [_tool_call_msg(rc="真实思考链")]
    out = _to_openai_messages(None, msgs, pad_reasoning=True)
    assert out[0]["reasoning_content"] == "真实思考链"


def test_pad_off_keeps_missing_rc_absent():
    from core.llm import _to_openai_messages
    msgs = [_tool_call_msg()]
    out = _to_openai_messages(None, msgs, pad_reasoning=False)
    assert "reasoning_content" not in out[0]


def test_pad_covers_plain_assistant_without_tool_calls():
    # DeepSeek 官方契约（2026-10 文档实测）：带 tools 的请求中，纯文本 assistant
    # 回复也必须携带 reasoning_content——即使该轮模型未实际调用工具
    from core.llm import _to_openai_messages
    msgs = [{"role": "assistant", "content": "纯文本回复"}]
    out = _to_openai_messages(None, msgs, pad_reasoning=True)
    assert out[0].get("reasoning_content") == ""


def test_pad_not_applied_for_non_deepseek():
    # 非 DeepSeek（默认 pad_reasoning=False）不动任何消息
    from core.llm import _to_openai_messages
    msgs = [_tool_call_msg(), {"role": "assistant", "content": "纯文本"}]
    out = _to_openai_messages(None, msgs)
    assert all("reasoning_content" not in m for m in out)
