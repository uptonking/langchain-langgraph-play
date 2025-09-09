from typing import Literal

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents.base import Document
from langchain_core.messages import convert_to_messages
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prettyformatter import pprint
from pydantic import BaseModel, Field

from langchain import hub
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition

# 🧑‍🏫 [Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
# Retrieval agents are useful when you want an LLM to make a decision about whether to retrieve context from a vectorstore or respond to the user directly.


# llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

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
    check_embedding_ctx_length=False,
    api_key="not-needed",  # pyright: ignore[reportArgumentType]
)


vector_store = InMemoryVectorStore(embeddings)

# 1️⃣ Preprocess documents, by WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    # "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]
docs = [WebBaseLoader(url).load() for url in urls]
print(f"🔢 characters: {len(docs[0][0].page_content)}")
# print(docs[0].page_content[:420])
# docs[0][0].page_content.strip()[:1000]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    # chunk_size=100,
    # chunk_overlap=50,
    chunk_size=1000,
    chunk_overlap=200,
    # track index in original document
    # add_start_index=True,
)
doc_splits = text_splitter.split_documents(docs_list)

print(f"Split blog post into {len(doc_splits)} sub-documents.")
# doc_splits[0].page_content.strip()


# embed the contents of each document split and insert these embeddings into a vector store
document_ids = vector_store.add_documents(documents=doc_splits)
print(f"document_ids: {len(document_ids)}")

# 2️⃣ Create a retriever tool

retriever = vector_store.as_retriever()


retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)
# Test the retriever_tool
# retriever_tool.invoke({"query": "types of reward hacking"})


# 3️⃣ Generate query or respond directly


# might call an LLM to generate a response based on the current graph state (list of messages)
def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    # response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# input = {"messages": [{"role": "user", "content": "hello!"}]}
# generate_query_or_respond(input)["messages"][-1].pretty_print()


# 4️⃣ Grade documents - determine whether the retrieved documents are relevant to the question.

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


# structured output schema: determine whether the retrieved documents are relevant to the question.
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


# grader_model = init_chat_model("openai:gpt-4.1", temperature=0)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        # grader_model
        llm.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "types of reward hacking"},
#                     }
#                 ],
#             },
#             {"role": "tool", "content": "meow", "tool_call_id": "1"},
#         ]
#     )
# }
# grade_documents(input)


# 5️⃣ Rewrite question if irrelevant documents retrived

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


# The retriever tool can return potentially irrelevant documents, which indicates a need to improve the original user question.
def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    # response = response_model.invoke([{"role": "user", "content": prompt}])
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}


# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "types of reward hacking"},
#                     }
#                 ],
#             },
#             {"role": "tool", "content": "meow", "tool_call_id": "1"},
#         ]
#     )
# }

# response = rewrite_question(input)
# print(response["messages"][-1]["content"])

# 6️⃣ Generate answer

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


# generate the final answer based on the original question and the retrieved context
def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    # response = response_model.invoke([{"role": "user", "content": prompt}])
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "types of reward hacking"},
#                     }
#                 ],
#             },
#             {
#                 "role": "tool",
#                 "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
#                 "tool_call_id": "1",
#             },
#         ]
#     )
# }

# response = generate_answer(input)
# response["messages"][-1].pretty_print()


# 7️⃣ Assemble the graph


workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# Start with a generate_query_or_respond
workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()


# -------------

for chunk in graph.stream(
    {  # pyright: ignore[reportArgumentType]
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
