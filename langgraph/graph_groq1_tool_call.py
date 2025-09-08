import json
import os
from typing import Any

from prettyformatter import pprint

from langgraph.prebuilt import create_react_agent

# import pprint


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}"


def main():
    # 📌 print("Hello, init chat model")
    # model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
    agent = create_react_agent(
        model="groq:llama-3.1-8b-instant",
        tools=[get_weather],
        prompt="You are a helpful assistant",
    )

    res = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in Guangzhou"}]}
    )

    pprint(res, json=True)
    # pp = pprint.PrettyPrinter(depth=10)
    # pp.pprint(res)
    # print(json.dumps(res, indent=4, sort_keys=True))
    print(f"Type of res: {type(res)}")


if __name__ == "__main__":
    main()
