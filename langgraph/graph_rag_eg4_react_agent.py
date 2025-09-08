import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prettyformatter import pprint

from langchain import hub
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition

# 🧑‍🏫 [Build a RAG App with chat history: Part 2](https://python.langchain.com/docs/tutorials/rag/)
# we give an LLM discretion to execute multiple retrieval steps.


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


# In production, the Q&A application will usually persist the chat history into a database, and be able to read and update it appropriately.
# the nodes in our graph are appending messages to the state, we will retain a consistent chat history across invocations.
memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)


graph = create_react_agent(llm, [retrieve], checkpointer=memory)

# 💡 Specify an ID for the thread
config = {"configurable": {"thread_id": "rag_by_react"}}

# a user message that does not require an additional retrieval step
input_message = "Hello"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},  # pyright: ignore[reportArgumentType]
    stream_mode="values",
    config=config,  # pyright: ignore[reportArgumentType]
):
    step["messages"][-1].pretty_print()


input_message = "What is Sequence CRDT ?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},  # pyright: ignore[reportArgumentType]
    stream_mode="values",
    config=config,  # pyright: ignore[reportArgumentType]
):
    step["messages"][-1].pretty_print()


# 👀👇 注意此时ai因为带有memory而能理解it，并会通过tool-call执行retrival, 此时可能执行多次tool-call查找相关信息
# 🤔 执行次数不确定可能造成风险
# input_message = "What are the common use cases for it in development or products ?"
input_message = (
    "are there any products or company or team that uses it as a core technology ?"
)

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},  # pyright: ignore[reportArgumentType]
    stream_mode="values",
    config=config,  # pyright: ignore[reportArgumentType]
):
    step["messages"][-1].pretty_print()
