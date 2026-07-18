import os
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
