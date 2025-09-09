import json
import os
from typing import Any, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from prettyformatter import pprint

from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# import pprint


class WeatherResponse(TypedDict):
    conditions: str


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always freezing cold in {city}"


def main():
    # 📌 print("Hello, init chat model")
    # model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
    agent = create_react_agent(
        # model="groq:llama-3.1-8b-instant",
        model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
        tools=[get_weather],
        response_format=WeatherResponse,  # pyright: ignore[reportArgumentType]
        # prompt=prompt,
    )

    res = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in Shenzhen"}]}
    )

    pprint(res, json=True)


if __name__ == "__main__":
    main()
