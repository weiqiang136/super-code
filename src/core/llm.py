import json
from dataclasses import dataclass
from typing import Any, Optional, Iterator

from openai import OpenAI

# Provider，目前只实现openai，后续接入其他厂商
OPENAI = 'openai'
VALID_PROVIDERS = [OPENAI]


def validate_provider(provider):
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Invalid provider: {provider}")
    return provider


@dataclass
class StreamMessage:
    """Represents the final message from a stream."""
    content: Any
    usage: Any | None = None
    reasoning_content: str | None = None  # DeepSeek 等思考模型的推理内容，需原样透传


class OpenAIStream:

    def __init__(self, *,
                 client: Any,
                 model: str,
                 messages: list[dict[str, Any]],
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 tools: Optional[list[dict]] = None,
                 extra_body: Optional[dict] = None):
        self._client = client
        self._params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        # 添加工具参数
        if tools:
            self._params["tools"] = tools
            self._params["tool_choice"] = "auto"

        # 模型特定额外参数（如 GLM thinking 控制）
        if extra_body:
            self._params["extra_body"] = extra_body

        # 请求流式 usage 数据（OpenAI 需要显式开启）
        self._params["stream_options"] = {"include_usage": True}

        self._stream = None
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []  # 捕获 reasoning_content（DeepSeek 思考模型）
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._usage: dict[str, int] | None = None

    def __enter__(self) -> "OpenAIStream":
        self._stream = self._client.chat.completions.create(**self._params)
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if self._stream and hasattr(self._stream, "close"):
            self._stream.close()

    def __iter__(self) -> Iterator[str]:
        _toolgen_sent = False
        for chunk in self._stream:
            # 从流式响应中捕获 usage 数据
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._usage = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                }
            for choice in _value(chunk, "choices", []) or []:
                delta = _value(choice, "delta", {}) or {}
                content = _value(delta, "content")
                if content:
                    self._text_parts.append(content)
                    yield content
                # 捕获 reasoning_content（DeepSeek 思考模型），yield 事件供 TUI 显示进度
                reasoning = _value(delta, "reasoning_content")
                if reasoning:
                    self._reasoning_parts.append(reasoning)
                    yield f"\x00thinking\x00{reasoning}"
                for tool_call in _value(delta, "tool_calls", []) or []:
                    # 首次生成 tool_call 时发送 sentinel，让上层提前显示等待指示
                    if not _toolgen_sent:
                        _toolgen_sent = True
                        yield "\x00toolgen\x00"
                    index = int(_value(tool_call, "index", 0) or 0)
                    entry = self._tool_calls.setdefault(index, {
                        "id": "",
                        "name": "",
                        "arguments": "",
                    })
                    tool_id = _value(tool_call, "id")
                    if tool_id:
                        entry["id"] = tool_id
                    function = _value(tool_call, "function", {}) or {}
                    name = _value(function, "name")
                    if name:
                        entry["name"] = name
                    arguments = _value(function, "arguments")
                    if arguments:
                        entry["arguments"] += arguments

    # 获取完整响应
    def final(self) -> StreamMessage:
        content: list[dict[str, Any]] = []
        text = "".join(self._text_parts)
        if text:
            content.append({"type": "text", "text": text})
        for index in sorted(self._tool_calls):
            tool_call = self._tool_calls[index]
            raw_args = tool_call.get("arguments", "").strip()
            parsed_args: Any = {}
            if raw_args:
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {}
            content.append({
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": tool_call.get("name", ""),
                "input": parsed_args if isinstance(parsed_args, dict) else {},
            })
        return StreamMessage(content=content, usage=self._usage,
                             reasoning_content="".join(self._reasoning_parts) or None)


def lookup_extra_body(model: str, profiles: dict) -> Optional[dict]:
    """按模型名子串匹配 model_profiles，最长 key 优先（精确优先），大小写不敏感。"""
    for key in sorted(profiles, key=len, reverse=True):
        if key.lower() in model.lower():
            eb = profiles[key].get("extra_body")
            return eb if eb else None
    return None


# 构建客户端
class LLMClient:
    def __init__(self, *, provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 300.0, model_profiles: Optional[dict] = None):
        self.provider = validate_provider(provider)
        self._base_url = base_url
        self._model_profiles: dict = model_profiles or {}
        if self.provider == OPENAI:
            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # 非流式输出（用于 compact 摘要等场景）
    def create(self, *, model: str, max_tokens: int, messages: list[dict[str, Any]],
               system: str | None = None, strip_thinking: bool = False) -> "StreamMessage":
        if self.provider == OPENAI:
            params: dict[str, Any] = {
                "model": model,
                "messages": _to_openai_messages(
                    system, messages,
                    pad_reasoning=_needs_reasoning_padding(self._base_url, model)),
                "max_tokens": max_tokens,
            }
            # 摘要等场景不需要思考：跳过 model_profiles（extra_body 均为思考控制参数），
            # 走 API 默认（非思考）路径——思考模型非流式会全量输出到 reasoning_content、
            # content 为空 → 摘要为空。
            if not strip_thinking:
                eb = lookup_extra_body(model, self._model_profiles)
                if eb:
                    params["extra_body"] = eb
            resp = self._client.chat.completions.create(**params)
            choice = resp.choices[0].message if resp.choices else None
            text = (choice.content or "") if choice else ""
            # 非流式响应同样捕获 usage（DeepSeek/OpenAI 均返回 prompt_tokens），
            # 归一化字段名与流式路径（__iter__）保持一致；无 usage 时保持 None。
            usage = None
            raw_usage = getattr(resp, "usage", None)
            if raw_usage is not None:
                usage = {
                    "input_tokens": getattr(raw_usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(raw_usage, "completion_tokens", 0) or 0,
                }
            return StreamMessage(content=[{"type": "text", "text": text}], usage=usage)
        raise NotImplementedError(f"create not implemented for provider: {self.provider}")

    # 流式输出
    def stream(self, *, model: str, system_prompt: str, messages: list[dict[str, Any]], temperature: float = 0.7,
               max_tokens: Optional[int] = None, tools: Optional[list[dict]] = None) -> "OpenAIStream":
        if self.provider == OPENAI:
            return OpenAIStream(
                client=self._client,
                model=model,
                max_tokens=max_tokens,
                messages=_to_openai_messages(
                    system_prompt, messages,
                    pad_reasoning=_needs_reasoning_padding(self._base_url, model)),
                temperature=temperature,
                tools=tools,
                extra_body=lookup_extra_body(model, self._model_profiles),
            )
        return NotImplementedError

def _value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _needs_reasoning_padding(base_url: str | None, model: str) -> bool:
    """是否需要给 assistant 消息补 reasoning_content 空占位（方案 A：仅 DeepSeek）。

    DeepSeek 官方契约（2026-10 文档 + 实测）：携带 tools 的请求，后续所有请求
    必须完整回传 reasoning_content——即使该轮模型未实际调用工具；历史含
    tool_calls 的无 tools 请求（compact 摘要路径）同样拦截。缺字段直接 400
    （"The `reasoning_content` in the thinking mode must be passed back to
    the API"），实测补空串即可解除。模型自适应思考导致部分响应无
    reasoning_content，回放时便缺字段。仅对 DeepSeek 生效（base_url 或模型名
    命中），不影响其它服务商。
    """
    if model and model.startswith("deepseek"):
        return True
    return bool(base_url and "deepseek" in base_url)

def _to_openai_messages(system: str | None, messages: list[dict[str, Any]],
                        pad_reasoning: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user" and isinstance(content, list):
            tool_results = [
                block for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            if tool_results and len(tool_results) == len(content):
                for block in tool_results:
                    out.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _tool_result_to_text(block.get("content", "")),
                    })
                continue

            out.append({
                "role": "user",
                "content": _user_content_blocks_to_openai(content),
            })
            continue

        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    })
            text = "".join(text_parts)
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if text:
                assistant_message["content"] = text
            elif not tool_calls:
                assistant_message["content"] = ""
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            # 透传 reasoning_content（DeepSeek 要求下一轮必须带回）
            if message.get("reasoning_content"):
                assistant_message["reasoning_content"] = message["reasoning_content"]
            elif pad_reasoning:
                # DeepSeek 官方契约（2026-10 文档实测）：带 tools 的请求中所有 assistant
                # 消息都必须带 reasoning_content——含未调用工具的纯文本回复；历史含
                # tool_calls 的无 tools 请求同样拦截。缺字段一律补空串占位（实测放行）。
                assistant_message["reasoning_content"] = ""
            out.append(assistant_message)
            continue

        out.append({
            "role": role,
            "content": content,
        })
        # DeepSeek 官方契约：纯字符串 assistant 回复（无 block 结构）同样要带
        # reasoning_content，缺字段补空串占位
        if role == "assistant" and pad_reasoning and not message.get("reasoning_content"):
            out[-1]["reasoning_content"] = ""

    return out


def _user_content_blocks_to_openai(content: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            source = block.get("source", {})
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })
    if not parts:
        return [{"type": "text", "text": ""}]
    return parts

def _tool_result_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)