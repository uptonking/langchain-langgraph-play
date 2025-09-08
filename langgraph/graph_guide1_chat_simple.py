import os
from typing import Annotated, Any, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


# The State includes the graph's schema and reducer functions that handle state updates
# Each node can receive the current State as input and output an update to the state.
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # Updates to messages will be appended to the existing list rather than overwriting it, thanks to the prebuilt reducer function.
    messages: Annotated[list[dict[str, Any]], add_messages]


# A StateGraph object defines the structure of our chatbot as a "state machine".
# We'll add nodes to represent the llm and functions our chatbot can call and edges to specify how the bot should transition between these functions.
graph_builder = StateGraph(State)


llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")


# Nodes represent units of work and are typically regular functions.
# Node takes the current State as input and returns a dictionary containing an updated messages list under the key "messages".
# This is the basic pattern for all LangGraph node functions.
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever the node is used.
graph_builder.add_node("chatbot", chatbot)

# Add an entry point to tell the graph where to start its work each time it is run
graph_builder.add_edge(START, "chatbot")
# Add an exit point to indicate where the graph should finish execution.
graph_builder.add_edge("chatbot", END)

graph: CompiledStateGraph[State, None, State, State] = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "\\q"]:
            print(
                "Bye, AI is temporarily closed, but chat is available anytime when restarting. "
            )
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
