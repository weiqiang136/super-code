"""
test_engine_with_tools.py

用于验证 Engine 的工具调用功能
"""
import os
from pathlib import Path
from core.llm import LLMClient
from core.engine import Engine
from core.permissions import PermissionChecker
from tools.file_read import FileReadTool

from dotenv import load_dotenv

load_dotenv()

def test_submit_with_file_read():
    """测试使用 Read 工具读取文件"""
    print("\n=== test_submit_with_file_read ===")
    
    # 创建一个测试文件
    test_file = Path("test_sample.txt")
    test_file.write_text("Hello, this is a test file.\nLine 2\nLine 3", encoding="utf-8")
    
    try:
        llm = LLMClient(
            provider='openai',
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        # 创建引擎，传入 Read 工具
        read_tool = FileReadTool()
        engine = Engine(
            permission_checker=PermissionChecker(auto_approve = False),
            tools=[read_tool],
            model=os.getenv("CC_MINI_MODEL", "glm-4"),
            system_prompt="You are a helpful assistant. You can use the Read tool to read files."
        )
        
        # 收集所有事件
        events = []
        text_chunks = []
        
        # 询问模型读取文件
        absolute_path = str(test_file.absolute())
        prompt = f"请读取文件 {absolute_path} 的内容并告诉我里面有什么"
        
        print(f"\nPrompt: {prompt}\n")
        
        for event in engine.submit(prompt):
            events.append(event)
            event_type = event[0]
            
            if event_type == 'text':
                text_chunks.append(event[1])
                print(event[1], end="", flush=True)
            elif event_type == 'waiting':
                print(f"\n[Waiting for tool decision]")
            elif event_type == 'tool_call':
                tool_name, tool_input, activity = event[1], event[2], event[3]
                print(f"\n[Tool call: {tool_name}]")
                print(f"  Input: {tool_input}")
                print(f"  Activity: {activity}")
            elif event_type == 'tool_executing':
                tool_name, tool_input, activity = event[1], event[2], event[3]
                print(f"\n[Executing: {tool_name}]")
            elif event_type == 'tool_result':
                tool_name, tool_input, result = event[1], event[2], event[3]
                print(f"\n[Tool result: {tool_name}]")
                print(f"  Error: {result.is_error}")
                print(f"  Content preview: {result.content[:100]}...")
        
        print("\n")
        
        # # 验证结果
        # assert len(events) > 0, "Should receive at least one event"
        # assert any(e[0] == 'text' for e in events), "Should receive text events"
        #
        # # 检查是否有工具调用
        # tool_calls = [e for e in events if e[0] == 'tool_call']
        # tool_results = [e for e in events if e[0] == 'tool_result']
        #
        # print(f"✓ Received {len(events)} events")
        # print(f"✓ Tool calls: {len(tool_calls)}")
        # print(f"✓ Tool results: {len(tool_results)}")
        #
        # if tool_calls:
        #     print(f"✓ Tools were invoked successfully")
        #     # 验证工具结果
        #     for result_event in tool_results:
        #         result = result_event[3]
        #         assert not result.is_error, f"Tool should not error: {result.content}"
        #
        # # 验证消息历史
        # messages = engine.get_messages()
        # print(f"✓ Messages count: {len(messages)}")
        #
        # # 应该有 user -> assistant -> user(tool_result)
        # assert len(messages) >= 2, "Should have at least 2 messages"
        
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()


def test_submit_without_tools():
    """测试不使用工具的普通对话"""
    print("\n=== test_submit_without_tools ===")
    
    llm = LLMClient(
        provider='openai',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    engine = Engine(
        llm=llm,
        tools=[],  # 不传工具
        model=os.getenv("CC_MINI_MODEL", "glm-4"),
        system_prompt="You are a helpful assistant."
    )
    
    events = []
    text_chunks = []
    
    for event in engine.submit("用一句话解释什么是Python"):
        events.append(event)
        if event[0] == 'text':
            text_chunks.append(event[1])
            print(event[1], end="", flush=True)
    
    print("\n")
    
    final_text = "".join(text_chunks)
    assert final_text, "Should receive text response"
    assert len(events) > 0, "Should receive events"
    
    print(f"✓ Non-tool conversation works correctly")
    print(f"✓ Response length: {len(final_text)} characters")


if __name__ == "__main__":
    test_submit_without_tools()
    test_submit_with_file_read()
    print("\n✅ All engine tool tests passed!")
