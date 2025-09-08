from typing import TypedDict

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prettyformatter import pprint

from langchain import hub
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# 🧑‍🏫 [Build a RAG App with chat history: Part 2](https://python.langchain.com/docs/tutorials/rag/)
# > This tutorial will focus on adding logic for incorporating historical messages.
# Conversational experiences can be naturally represented using a sequence of messages


llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
# llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


vector_store = InMemoryVectorStore(embeddings)

web_parser_bs4 = bs4.SoupStrainer(
    # class_=("post-content", "post-title", "post-header")
    class_=(
        # 👀 网页源码的class类名只有1个且后面跟着空格，这里的选择器必须要加上空格才生效, 源码为 <div class="crayons-article__main ">
        # 尝试选择子元素 <div class="crayons-article__body text-styles spec__body">, 但这里只写 rayons-article__body 时不生效，需要将3个类名都写上
        "crayons-article__main ",
        # "crayons-article__body text-styles spec__body",
        # "crayons-article__body ", # 不生效
        "crayons-article__header__meta",
    )
)
# Load and chunk contents of the blog
loader = WebBaseLoader(
    # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    web_paths=(
        "https://dev.to/nyxtom/introduction-to-crdts-for-realtime-collaboration-2eb1",
    ),
    bs_kwargs={"parse_only": web_parser_bs4},
)
docs = loader.load()
print(f"🔢 Total characters: {len(docs[0].page_content)}")
# print(docs[0].page_content[:420])

# Even for those models that could fit the full post in their context window, models can struggle to find information in very long inputs.
# split the Document into chunks. help to retrieve only the most relevant parts of the blog post at run time.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    # track index in original document
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")


# embed the contents of each document split and insert these embeddings into a vector store
document_ids = vector_store.add_documents(documents=all_splits)
print(f"document_ids: {len(document_ids)}")


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval. ToolNode executes the tool and adds the result as a ToolMessage to the state.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


# connect nodes/steps as graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
# allow the first query_or_respond step to "short-circuit" and respond directly to the user if it does not generate a tool call.
# 👀👇 ai判断执行一次或不执行而结束，最多执行一次
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


# a user message that does not require an additional retrieval step
input_message = "Hello"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},  # pyright: ignore[reportArgumentType]
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# response = graph.invoke({"question": "What is Task Decomposition?"})  # pyright: ignore[reportArgumentType]
# response = graph.invoke({"question": "What is CmRDTs  ?"})  # pyright: ignore[reportArgumentType]
# response = graph.invoke(
#     {"messages": [{"role": "user", "content": "what is Sequence CRDTs ?"}]}  # pyright: ignore[reportArgumentType]
# )
# print("\n#### -------- ####")
# print(f"Type of res: {type(response)}")
# pprint(response, json=True)

input_message = "What is Sequence CRDT ?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},  # pyright: ignore[reportArgumentType]
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


# 👀👇 注意此时ai理解不了it，回复是 Please tell me what "it" refers to
input_message = "What are the common use cases for it in development or products ?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},  # pyright: ignore[reportArgumentType]
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
