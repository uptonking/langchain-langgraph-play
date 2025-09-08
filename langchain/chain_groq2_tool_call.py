import getpass
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langchain.chat_models import init_chat_model


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


# [How to pass tool outputs to chat models](https://python.langchain.com/docs/how_to/tool_results_pass_to_model/)
# 👀 同一模型少数时候会执行失败，看异常信息是大模型生成tool-call的参数有时结构错误
def main():
    # 📌 print("Hello, init chat model")
    # model = init_chat_model("qwen/qwen3-32b", model_provider="groq")
    model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
    tools = [add, multiply]
    llm_with_tools = model.bind_tools(tools)

    query = "What is 11 * 22? Also, what is 33 + 44?"
    messages = [HumanMessage(query)]
    res_with_tools = llm_with_tools.invoke(messages)
    print("\n------res_with_tools")
    print(res_with_tools.tool_calls)
    messages.append(res_with_tools)

    for tool_call in res_with_tools.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    print("\n------tools_with_args")
    print(messages)

    result = llm_with_tools.invoke(messages)

    print("\n------result")
    print(result)


if __name__ == "__main__":
    main()
