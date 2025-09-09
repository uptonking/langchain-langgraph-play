from typing import Annotated, Any, TypedDict

from prettyformatter import pprint

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

llm = init_chat_model(
    model="google/gemma-3-4b",
    base_url="http://localhost:1234/v1/",
    model_provider="openai",
    api_key="not-needed",
    # temperature=0.1,
)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list[Any], add_messages]


def chatBot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatBot)

graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()


# res = llm.invoke("give a brief intro to reactjs in less than 80 words.")
# pprint(res, json=True)


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
        user_input = "give a brief intro to reactjs in less than 80 words."
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
        count += 1
    except Exception as e:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
