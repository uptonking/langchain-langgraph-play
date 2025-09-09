# langchain-langgraph-playgrounds
> quickstart boilerplate for langchain/langgraph in python

# quickstart
- examples(mostly with langgraph)
  - llm with groq api
  - llm with local llm, tested with LM Studio
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

```

```sh
# config api keys, like openai/groq
cp .env.example .env

uv sync

uv run --env-file .env -- langgraph/graph_rag_eg1_chains_mini.py

```

# roadmap
- [ ] graph.stream not work with local llm
# notes
- examples in js/ts: https://github.com/uptonking/langchainjs-langgraphjs-play

# license
MIT