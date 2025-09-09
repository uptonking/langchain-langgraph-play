from typing import Annotated, Any

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model


tool = TavilySearch(max_results=2)
tools = [tool]

# llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
llm = init_chat_model(
    model="qwen/qwen3-4b-2507",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[Any], add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# Create a function to run the tools
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


count = 0

while True:
    try:
        if count > 0:
            break
        # user_input = input("User: ")
        user_input = "when did deepseek v3.1 model release ?"
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
        count += 1
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
