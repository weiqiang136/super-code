"""
test_engine.py

用于验证 Engine 的 submit 方法功能
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from core.llm import LLMClient
from core.engine import Engine


def test_submit_basic():
    """测试基本的 submit 流式输出功能"""
    print("\n=== test_submit_basic ===")
    
    llm = LLMClient(
        provider='openai',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    engine = Engine(
        llm=llm,
        tools=[],
        model=os.getenv("CC_MINI_MODEL", "glm-4"),
        system_prompt="You are a helpful assistant."
    )
    
    # 收集所有事件
    events = []
    text_chunks = []
    
    for event in engine.submit("用一句话解释什么是人工智能"):
        events.append(event)
        if event[0] == 'text':
            text_chunks.append(event[1])
            print(event[1], end="", flush=True)
        elif event[0] == 'waiting':
            print(f"\n[Received waiting event]")
    
    print("\n")
    
    # 验证结果
    assert len(events) > 0, "Should receive at least one event"
    assert any(e[0] == 'text' for e in events), "Should receive text events"
    assert any(e[0] == 'waiting' for e in events), "Should receive waiting event"
    
    final_text = "".join(text_chunks)
    assert final_text, "Final text should not be empty"
    assert isinstance(final_text, str), "Final text should be a string"
    
    # 验证消息历史
    messages = engine.get_messages()
    assert len(messages) == 2, f"Should have 2 messages (user + assistant), got {len(messages)}"
    assert messages[0]['role'] == 'user', "First message should be from user"
    assert messages[1]['role'] == 'assistant', "Second message should be from assistant"
    assert messages[1]['content'] == final_text, "Assistant message should match streamed text"
    
    print(f"✓ Received {len(events)} events")
    print(f"✓ Final text length: {len(final_text)} characters")
    print(f"✓ Messages count: {len(messages)}")


def test_submit_multiple_turns():
    """测试多轮对话"""
    print("\n=== test_submit_multiple_turns ===")
    
    llm = LLMClient(
        provider='openai',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    engine = Engine(
        llm=llm,
        tools=[],
        model=os.getenv("CC_MINI_MODEL", "glm-4"),
        system_prompt="You are a helpful assistant."
    )
    
    # 第一轮对话
    print("\n--- Turn 1 ---")
    for event in engine.submit("你好，请介绍一下自己"):
        if event[0] == 'text':
            print(event[1], end="", flush=True)
    
    messages_after_first = len(engine.get_messages())
    print(f"\nMessages after first turn: {messages_after_first}")
    
    # 第二轮对话
    print("\n--- Turn 2 ---")
    for event in engine.submit("你能帮我做什么？"):
        if event[0] == 'text':
            print(event[1], end="", flush=True)
    
    messages_after_second = len(engine.get_messages())
    print(f"\nMessages after second turn: {messages_after_second}")
    
    # 验证消息历史正确累积
    assert messages_after_first == 2, "Should have 2 messages after first turn"
    assert messages_after_second == 4, "Should have 4 messages after second turn"
    
    print(f"✓ Multi-turn conversation works correctly")


def test_get_set_messages():
    """测试 get_messages 和 set_messages 方法"""
    print("\n=== test_get_set_messages ===")
    
    llm = LLMClient(
        provider='openai',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    engine = Engine(
        llm=llm,
        tools=[],
        model=os.getenv("CC_MINI_MODEL", "glm-4")
    )
    
    # 初始状态
    messages = engine.get_messages()
    assert messages == [], "Initial messages should be empty"
    
    # 设置消息
    test_messages = [
        {"role": "system", "content": "Test system prompt"},
        {"role": "user", "content": "Test user message"}
    ]
    engine.set_messages(test_messages)
    
    # 验证设置成功
    retrieved = engine.get_messages()
    assert len(retrieved) == 2, "Should have 2 messages"
    assert retrieved[0]['role'] == 'system', "First message should be system"
    assert retrieved[1]['role'] == 'user', "Second message should be user"
    
    # 验证返回的是副本，不是引用
    retrieved.append({"role": "assistant", "content": "test"})
    assert len(engine.get_messages()) == 2, "Original messages should not be modified"
    
    print(f"✓ get_messages and set_messages work correctly")


def test_system_prompt_property():
    """测试 system_prompt 属性"""
    print("\n=== test_system_prompt_property ===")
    
    llm = LLMClient(
        provider='openai',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    engine = Engine(
        llm=llm,
        tools=[],
        model=os.getenv("CC_MINI_MODEL", "glm-4"),
        system_prompt="Initial prompt"
    )
    
    # 读取 system_prompt
    assert engine.system_prompt == "Initial prompt", "Should return initial prompt"
    
    # 修改 system_prompt
    engine.system_prompt = "Updated prompt"
    assert engine.system_prompt == "Updated prompt", "Should return updated prompt"
    
    print(f"✓ system_prompt property works correctly")


if __name__ == "__main__":
    test_submit_basic()
    test_submit_multiple_turns()
    test_get_set_messages()
    test_system_prompt_property()
    print("\n✅ All engine tests passed!")
