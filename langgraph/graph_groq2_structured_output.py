import json
import os
from typing import Any, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from prettyformatter import pprint

from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState


class WeatherResponse(TypedDict):
    weatherDetails: str


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always freezing cold in {city}"


# def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
#     user_name = config["configurable"].get("user_name")
#     system_msg = f"You are a helpful assistant. Address the user as {user_name}."
#     return [{"role": "system", "content": system_msg}] + state["messages"]


# 👀 调用tool失败概率很高
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

    # {"messages": [{"role": "user", "content": "what is the weather in Shenzhen? what can i do for outdoor activities?"}]},
    # res = agent.invoke(
    #     {
    #         "messages": [
    #             {"role": "user", "content": "what is the weather in Shenzhen? "}
    #         ]
    #     },
    #     # config={"configurable": {"user_name": "yaoo"}},
    # )

    res = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in Shenzhen"}]}
    )

    # "structured_response"   : {"weatherDetails": "Shenzhen"}
    pprint(res, json=True)


if __name__ == "__main__":
    main()
