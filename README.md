# langchain-langgraph-playgrounds
> quickstart boilerplate for langchain/langgraph in python

# quickstart
- examples(mostly with langgraph)
  - llm with groq api
  - ✨ llm with local llm, tested with LM Studio
  - tool call: tavily api
  - structured output

- RAG
  - simple
  - generate_query_or_respond
  - memory: chat history
  - docs grading

```sh
# local llm is tested with LM Studio
#  🤔 it seems that when using local LM Studio, you have to turn off global mode for proxy like clash, you can use rule/direct mode

uv sync

# no api keys required for local llm
uv run langgraph/graph_eg_chat_local_mini.py

# rag with local qwen3-4b/qwen3-embedding-0.6b-gguf
uv run langgraph/graph_rag_eg1_chains_mini_local.py

```

```sh
# config api keys, like openai/groq
cp .env.example .env

uv sync

# chat with groq/llama-4
uv run --env-file .env -- langgraph/graph_guide1_chat_simple.py

# rag with gemini-2.5-flash/gemini-embedding-001
uv run --env-file .env -- langgraph/graph_rag_eg1_chains_mini.py

```

# roadmap
- [ ] `graph.stream` not work with local llm
# notes
- examples in js/ts: https://github.com/uptonking/langchainjs-langgraphjs-play

# license
MIT