# langchain-langgraph-playgrounds
> quickstart boilerplate for langchain/langgraph in python

# quickstart
- examples(mostly with langgraph)
  - llm with groq api
  - tool call: tavily api
  - structured output

- RAG
  - simple
  - generate_query_or_respond
  - memory: chat history
  - docs grading

```sh
# config api keys
cp .env.example .env

uv sync

uv run --env-file .env -- langgraph/graph_rag_eg1_chains_mini

```

# notes
- examples in js/ts: https://github.com/uptonking/langchainjs-langgraphjs-play
# license
MIT