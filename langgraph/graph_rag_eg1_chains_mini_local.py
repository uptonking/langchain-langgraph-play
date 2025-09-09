from typing import TypedDict

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prettyformatter import pprint

from langchain import hub
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

# 🧑‍🏫 [Build a RAG App: Part 1](https://python.langchain.com/docs/tutorials/rag/)
# > This tutorial will show how to build a simple Q&A application over a text data source.
# we represented the user question, retrieved context, and generated answer as separate keys in the state


# llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
llm = init_chat_model(
    model="qwen/qwen3-4b-2507",
    # because LM Studio mimics OpenAI's API
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    # LM Studio accepts any string for api key
    api_key="not-needed",
)

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embeddings = OpenAIEmbeddings(
    model="text-embedding-qwen3-embedding-0.6b",
    base_url="http://127.0.0.1:1234/v1",
    # Whether to check the token length of inputs and automatically split inputs longer than embedding_ctx_length.
    check_embedding_ctx_length=False,
    api_key="not-needed",  # pyright: ignore[reportArgumentType]
)


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
    chunk_size=500,
    chunk_overlap=100,
    # track index in original document
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")


# embed the contents of each document split and insert these embeddings into a vector store
document_ids = vector_store.add_documents(documents=all_splits)
print(f"document_ids: {len(document_ids)}")


# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify api_url="https://api.smith.langchain.com" in hub.pull.
# prompt = hub.pull("rlm/rag-prompt")


template = """You are an assistant for question-answering tasks. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking" at the end of the answer.

Question: {question}

Context: {context}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)


example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()
print("\n### region-prompt-eg")
print(f"prompt-eg: {len(example_messages)}")
print(example_messages[0].content)
print("### endregion-prompt-eg\n")


# Define state for application
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


# Our retrieval step simply runs a similarity search using the input question
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# generation step formats the retrieved context and original question into a prompt for the chat model.
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# connect nodes/steps as graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# response = graph.invoke({"question": "What is Task Decomposition?"})  # pyright: ignore[reportArgumentType]
# response = graph.invoke({"question": "What is Sequence CRDTs  ?"})  # pyright: ignore[reportArgumentType]
response = graph.invoke({"question": "What is CmRDTs  ?"})  # pyright: ignore[reportArgumentType]
print("\n#### -------- ####")
# response is type <class 'dict'>
# response['answer'] is type <class 'str'>
print(f"Type of res: {type(response)}")
# print(response["answer"])
pprint(response, json=True)
