import json
import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import ToolMessage
from langchain_tavily import TavilySearch
from prettyformatter import pprint

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


tool = TavilySearch(max_results=2)
tools = [tool]
# tool.invoke("What's a 'node' in LangGraph?")


# checks the most recent message in the state and calls tools if the message contains `tool_calls`.
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict[str, list[Any]]):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# The State includes the graph's schema and reducer functions that handle state updates
# Each node can receive the current State as input and output an update to the state.
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # Updates to messages will be appended to the existing list rather than overwriting it, thanks to the prebuilt reducer function.
    messages: Annotated[list[dict[str, Any]], add_messages]


# checks for tool_calls in the chatbot's output
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]  # pyright: ignore[reportArgumentType, reportAssignmentType]
    elif messages := state.get("messages", []):
        ai_message: dict[str, Any] = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:  # pyright: ignore[reportAttributeAccessIssue]
        return "search"
    return END


# A StateGraph object defines the structure of our chatbot as a "state machine".
# We'll add nodes to represent the llm and functions our chatbot can call and edges to specify how the bot should transition between these functions.
graph_builder = StateGraph(State)


llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")

llm_with_tools = llm.bind_tools(tools)


# Nodes represent units of work and are typically regular functions.
# Node takes the current State as input and returns a dictionary containing an updated messages list under the key "messages".
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever the node is used.
graph_builder.add_node("chatbot", chatbot)


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("search", tool_node)  # pyright: ignore[reportArgumentType]

# Conditional edges start from a single node and usually contain "if" statements to route to different nodes by state
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else e.g., "tools": "my_tools"
    {"search": "search", END: END},
)

graph_builder.add_edge("search", "chatbot")


# Add an entry point to tell the graph where to start its work each time it is run
graph_builder.add_edge(START, "chatbot")

# Add an exit point to indicate where the graph should finish execution.
# graph_builder.add_edge("chatbot", END)

graph: CompiledStateGraph[State, None, State, State] = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            # print("Assistant:", value["messages"][-1].content)
            print("Assistant:")
            pprint(value["messages"][-1].content, json=True)
            # pprint(value["messages"][-1].content)


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
